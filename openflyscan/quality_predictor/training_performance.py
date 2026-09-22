"""Bounded feature caches and opt-in, exclusive pipeline timings."""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import time

import torch


_active_profiler = ContextVar("head_pipeline_profiler", default=None)


@dataclass(frozen=True)
class PerformanceConfig:
    builder_cache_scenes: int = 0
    teacher_cache_scenes: int = 0
    cpu_feature_cache_mib: int = 1024
    gpu_feature_cache_mib: int = 256
    gpu_feature_cache_entries: int = 0
    cpu_feature_cache_policy: str = "scene"
    prefetch_features: bool = True
    profile_stages: bool = False

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "cpu_feature_cache_policy":
                if value not in ("global", "scene"):
                    raise ValueError("cpu_feature_cache_policy must be global or scene")
            elif name in ("profile_stages", "prefetch_features"):
                if type(value) is not bool:
                    raise ValueError(f"{name} must be boolean")
            elif type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    @classmethod
    def legacy(cls, *, profile_stages=False):
        return cls(builder_cache_scenes=2, teacher_cache_scenes=2,
                   cpu_feature_cache_mib=0, gpu_feature_cache_mib=1024,
                   gpu_feature_cache_entries=128, cpu_feature_cache_policy="global", prefetch_features=False,
                   profile_stages=profile_stages)


class TensorLRU:
    def __init__(self, max_bytes, max_entries=0):
        if max_bytes < 0 or max_entries < 0:
            raise ValueError("cache limits must be nonnegative")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = self.evictions = 0
        self.admission_skips = 0
        self.protected = set()

    def get(self, key):
        if key not in self.entries:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return self.entries[key][0]

    def __contains__(self, key):
        return key in self.entries

    def put(self, key, tensors):
        size = sum(tensor.numel()*tensor.element_size() for tensor in tensors)
        if key in self.entries:
            self.bytes -= self.entries.pop(key)[1]
        if size > self.max_bytes or self.max_bytes == 0:
            self.admission_skips += 1
            return
        protected_bytes = sum(self.entries[entry][1] for entry in self.protected if entry in self.entries)
        protected_count = sum(entry in self.entries for entry in self.protected)
        if protected_bytes+size > self.max_bytes or (self.max_entries and protected_count >= self.max_entries):
            self.admission_skips += 1
            return
        while self.entries and (self.bytes+size > self.max_bytes or
                                (self.max_entries and len(self.entries) >= self.max_entries)):
            oldest = next(entry for entry in self.entries if entry not in self.protected)
            self.bytes -= self.entries.pop(oldest)[1]
            self.evictions += 1
        self.entries[key] = (tensors, size)
        self.bytes += size

    @contextmanager
    def protect(self, keys):
        previous = self.protected
        self.protected = previous | {key for key in keys if key in self.entries}
        try:
            yield
        finally:
            self.protected = previous

    @contextmanager
    def batch(self, keys):
        yield

    def snapshot(self):
        return dict(entries=len(self.entries), bytes=self.bytes, max_bytes=self.max_bytes,
                    hits=self.hits, misses=self.misses, evictions=self.evictions,
                    admission_skips=self.admission_skips)


class SceneTensorCache:
    def __init__(self, max_bytes, scenes):
        scenes = list(scenes)
        if max_bytes < 0 or not scenes or len(set(scenes)) != len(scenes):
            raise ValueError("scene cache requires a nonnegative budget and distinct scenes")
        self.max_bytes = max_bytes
        quota, remainder = divmod(max_bytes, len(scenes))
        self.scenes = {scene: TensorLRU(quota+(index < remainder)) for index, scene in enumerate(scenes)}

    def get(self, key):
        return self.scenes[key[0]].get(key)

    def __contains__(self, key):
        return key in self.scenes[key[0]]

    def put(self, key, tensors):
        self.scenes[key[0]].put(key, tensors)

    @contextmanager
    def batch(self, keys):
        from contextlib import ExitStack

        grouped = {}
        for key in keys:
            grouped.setdefault(key[0], []).append(key)
        with ExitStack() as stack:
            for scene, selected in grouped.items():
                stack.enter_context(self.scenes[scene].protect(selected))
            yield

    def snapshot(self):
        per_scene = {scene: cache.snapshot() for scene, cache in self.scenes.items()}
        totals = {name: sum(row[name] for row in per_scene.values())
                  for name in ("entries", "bytes", "hits", "misses", "evictions", "admission_skips")}
        return dict(**totals, max_bytes=self.max_bytes, policy="scene", per_scene=per_scene)


class FeaturePrefetcher:
    def __init__(self):
        self.executor = None
        self.tasks = OrderedDict()
        self.hits = self.submitted_photos = 0
        self.wait_s = 0.0

    def submit(self, token, keys, loader):
        if token in self.tasks:
            raise ValueError("duplicate prefetch token")
        if len(self.tasks) >= 2:
            raise RuntimeError("only current and next photo groups may be prefetched")
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="head-feature-reader")
        keys = tuple(keys)
        self.submitted_photos += len(keys)
        def load():
            return {key: loader(key[1]) for key in keys}
        self.tasks[token] = (set(keys), self.executor.submit(load))

    def take(self, key):
        for keys, future in self.tasks.values():
            if key not in keys:
                continue
            started = time.perf_counter()
            values = future.result()
            self.wait_s += time.perf_counter()-started
            keys.remove(key)
            self.hits += 1
            return values.pop(key)
        return None

    def retire(self, token):
        task = self.tasks.pop(token, None)
        if task is not None:
            task[1].result().clear()

    def close(self):
        try:
            for token in list(self.tasks):
                self.retire(token)
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
                self.executor = None

    def snapshot(self):
        return dict(pending_batches=len(self.tasks), hits=self.hits,
                    submitted_photos=self.submitted_photos, wait_s=self.wait_s,
                    max_pending_batches=2, workers=1)


class StageProfiler:
    def __init__(self, enabled=False, synchronize=None):
        self.enabled = enabled
        self.synchronize = synchronize or (lambda: None)
        self.rows = {}
        self.stack = []

    def reset(self):
        if self.stack:
            raise RuntimeError("cannot reset an active stage profiler")
        self.rows.clear()

    @contextmanager
    def activate(self):
        if not self.enabled:
            yield
            return
        token = _active_profiler.set(self)
        try:
            yield
        finally:
            _active_profiler.reset(token)

    @contextmanager
    def measure(self, name):
        self.synchronize()
        frame = dict(start=time.perf_counter(), children=0.0)
        self.stack.append(frame)
        try:
            yield
        finally:
            self.synchronize()
            elapsed = time.perf_counter()-frame["start"]
            self.stack.pop()
            if self.stack:
                self.stack[-1]["children"] += elapsed
            row = self.rows.setdefault(name, dict(calls=0, inclusive_s=0.0, exclusive_s=0.0))
            row["calls"] += 1
            row["inclusive_s"] += elapsed
            row["exclusive_s"] += max(0.0, elapsed-frame["children"])

    def snapshot(self):
        return {name: dict(row) for name, row in self.rows.items()}


@contextmanager
def stage(name):
    profiler = _active_profiler.get()
    if profiler is None:
        yield
    else:
        with profiler.measure(name):
            yield


def timed(name, *, source=False):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            profiler = getattr(args[0], "profiler", None) if source else None
            if profiler is not None:
                with profiler.activate(), stage(name):
                    return function(*args, **kwargs)
            with stage(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def synchronize_model(model):
    if model is not None:
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
