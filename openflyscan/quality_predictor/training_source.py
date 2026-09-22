"""Rolling frozen-Full cache; fresh Missing inference and observation-loss labels."""

import dataclasses
import hashlib
import json
from collections import OrderedDict

import numpy as np
import torch

from .observation_source import sample_window
from .paired_observation_source import SourceV2
from .footprint_window_sampler import sample_footprint_window
from .base_supervision import full_region_supervision
from .model import LocalQueryInputs
from .sparse_supervision import (
    ObservationPolicy, corresponding_missing_labels, dense_source_patch_index,
)
from .training_performance import stage, timed


def stable_seed(*values):
    return int.from_bytes(hashlib.sha256(json.dumps(values).encode()).digest()[:4], "little")


def resample_missing(record, template, rng):
    footprints = record["footprint_data"]
    centers = footprints["centers"]
    window = sample_footprint_window(template["anchor"], centers, footprints["bbox_mins"], footprints["bbox_maxs"])
    stems = list(map(str, footprints["stems"]))
    if [stems[index] for index in window.stems_indices] != template["full"]:
        raise ValueError("cached Full no longer matches its shifted footprint window")
    core = np.asarray(window.core_indices)
    low, high = np.quantile(centers[core], [.1, .9], axis=0)
    relative = rng.uniform(.25, .75, 2)
    target = low+relative*(high-low)
    count = int(rng.integers(10, 13))
    ordered = core[np.argsort(np.linalg.norm(centers[core]-target, axis=1), kind="stable")]
    deleted = [stems[index] for index in ordered[:count]]
    return {**template, "deleted": deleted, "missing": [stem for stem in template["full"] if stem not in deleted],
            "relative_location": relative.tolist(), "target_xy": target.tolist()}


def combine_inputs(inputs):
    packed = {}
    for field in dataclasses.fields(LocalQueryInputs):
        values = [getattr(value, field.name) for value in inputs]
        if field.name != "query_features":
            maximum = max(value.shape[1] for value in values)
            values = [torch.cat([value, value.new_zeros((len(value), maximum-value.shape[1], *value.shape[2:]))], 1)
                      for value in values]
        packed[field.name] = torch.cat(values, 0)
    return LocalQueryInputs(**packed)


class SourceV3(SourceV2):
    def __init__(self, *args, training_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        if training_config is None or self.full_label_policy is None:
            raise ValueError("V3 requires explicit training and Full supervision policies")
        self.training_config = training_config
        if training_config.exclude_outer_chunks:
            from .chunk_filter import prepare_record
            for scene in self.train:
                prepare_record(self.records[scene])
        self.observation_policy = ObservationPolicy(**self.plan.get("observation_policy_v3", {}))
        self.full_cache = OrderedDict()
        self.cache_capacity = len(self.records)*training_config.chunks_per_rank
        self.v3_counts = dict(full_hits=0, full_misses=0, full_evictions=0, missing_inferences=0,
                              affected=0, unaffected=0, unknown=0)

    def choice_for_step(self, scene, step, rank, ordinal, world):
        scene_index = self.train.index(scene)
        first = (scene_index-ordinal*world-rank) % len(self.train)
        visit = (step-first)//len(self.train)
        generation = visit//self.training_config.full_reuse_visits
        sample_seed = stable_seed(self.training_config.seed, scene, rank, ordinal, generation)
        full_rng = np.random.default_rng(sample_seed)
        if self.training_config.exclude_outer_chunks:
            from .chunk_filter import sample_interior_window
            template = sample_interior_window(self.records[scene], full_rng, weak=bool(full_rng.random() < .5))
        else:
            template = sample_window(self.records[scene], full_rng, weak=bool(full_rng.random() < .5))
        rng = np.random.default_rng(stable_seed(self.training_config.seed, "missing", step, rank, ordinal))
        return {**resample_missing(self.records[scene], template, rng), "full_seed": self.training_config.seed,
                "full_sample_seed": sample_seed,
                "cache_slot": ordinal, "generation": generation, "visit": visit}

    def _full_entry(self, scene, choice, seed):
        key = (scene, int(choice.get("cache_slot", 0)))
        full_seed = int(choice.get("full_seed", seed))
        identity = (tuple(choice["full"]), full_seed)
        entry = self.full_cache.get(key)
        if entry is not None and entry["identity"] == identity:
            self.full_cache.move_to_end(key)
            self.v3_counts["full_hits"] += 1
            return entry, True
        self.v3_counts["full_misses"] += 1
        with stage("full_extract"):
            inputs, full, proposal = self.extract(scene, choice["full"], full_seed)
        values = self.full_pixel_labels(scene, full)
        with stage("v3_full_supervision"):
            if "teacher_mapping" in self.records[scene]:
                values, cells, tracks = full["full_teacher_patches"]
                labels = full_region_supervision(full, proposal, values, cells, tracks, self.full_label_policy)
            else:
                target, mask = self.region_labels(full, proposal, values)
                labels = dict(target=target, mask=mask, rank_mask=mask.copy(), weight=mask.astype(np.float32),
                              audit=dict(original_valid=int(mask.sum()), recovered=0, single_view=0,
                                         unknown=int((~mask).sum()), rows=[],
                                         boundary="Frozen legacy raw-to-rectified Full labels; six patches/two views; no recovery."))
            evaluation_regions = None
            if self.training_config.directional_quality:
                from .directional_quality import observed_view_labels
                labels, evaluation_regions = observed_view_labels(full, proposal, values, labels)
            if self.training_config.view_auxiliary:
                from .view_supervision import feature_matched_view_labels
                view_labels, view_audit = feature_matched_view_labels(full, values, inputs.view_mask.shape)
                labels = {**labels, **view_labels, 'audit': {**labels['audit'], 'view_auxiliary': view_audit}}
            reference, reference_labels = dense_source_patch_index(
                full, proposal, values, directional_quality=self.training_config.directional_quality)
        cpu_inputs = LocalQueryInputs(**{field.name: getattr(inputs, field.name).detach().cpu().clone()
                                       for field in dataclasses.fields(inputs)})
        entry = dict(identity=identity, inputs=cpu_inputs, labels=labels, reference=reference,
                     reference_labels=reference_labels,
                     evidence=full["geometry_evidence_audit"], evaluation_regions=evaluation_regions,
                     source_recovery=full.get('source_recovery_audit'))
        self.full_cache[key] = entry
        self.full_cache.move_to_end(key)
        while len(self.full_cache) > self.cache_capacity:
            self.full_cache.popitem(last=False)
            self.v3_counts["full_evictions"] += 1
        return entry, False

    @timed("pair_v3", source=True)
    def pair(self, scene, rng, seed=20260907, choice=None):
        if self.training_config.exclude_outer_chunks and scene in self.train:
            from .chunk_filter import check_choice, sample_interior_window
            if choice is None:
                choice = {**sample_interior_window(self.records[scene], rng, weak=bool(rng.random() < .5)),
                          'full_seed': seed, 'cache_slot': -1}
            if not check_choice(self.records[scene], choice)['passed']:
                raise ValueError('training pair attempted to bypass outer-chunk rejection')
        choice = choice or {**sample_window(self.records[scene], rng, weak=bool(rng.random() < .5)),
                            "full_seed": seed, "cache_slot": -1}
        entry, hit = self._full_entry(scene, choice, seed)
        with stage("missing_extract"):
            missing_inputs, missing, proposal = self.extract(scene, choice["missing"], seed)
        self.v3_counts["missing_inferences"] += 1
        with stage("v3_observation_supervision"):
            weak = corresponding_missing_labels(entry["reference"], entry["reference_labels"], missing, proposal,
                                                choice["deleted"], self.quality_limits, self.observation_policy)
        for name in ("affected", "unaffected", "unknown"):
            self.v3_counts[name] += weak["audit"][name]
        device = missing_inputs.point_features.device
        full_inputs = LocalQueryInputs(**{field.name: getattr(entry["inputs"], field.name).to(device)
                                         for field in dataclasses.fields(LocalQueryInputs)})
        inputs = combine_inputs([full_inputs, missing_inputs])
        full_labels = entry["labels"]
        full_count, missing_count = len(full_labels["target"]), len(weak["target"])
        full_normalized = (full_labels["target"]-self.quality_limits[0])/np.diff(self.quality_limits)[0]
        labels = dict(total=np.r_[np.clip(full_normalized, 0, 1), weak["target"]].astype(np.float32),
                      rank_target=np.r_[full_normalized, weak["rank_target"]].astype(np.float32),
                      total_mask=np.r_[full_labels["mask"], weak["mask"]],
                      total_rank_mask=np.r_[full_labels["rank_mask"], weak["rank_mask"]],
                      total_weight=np.r_[full_labels["weight"], weak["weight"]].astype(np.float32),
                      affected=np.r_[np.zeros(full_count, bool), weak["affected"]],
                      groups=np.r_[np.zeros(full_count, np.int64), np.ones(missing_count, np.int64)])
        if self.training_config.directional_quality:
            shape = (missing_count, inputs.view_mask.shape[1])
            normalized = (full_labels['view_raw']-self.quality_limits[0])/np.diff(self.quality_limits)[0]
            labels.update(view_target=np.r_[np.clip(normalized, 0, 1), np.zeros(shape)].astype(np.float32),
                          view_rank_target=np.r_[normalized, np.zeros(shape)].astype(np.float32),
                          view_target_mask=np.r_[full_labels['view_mask'], np.zeros(shape, bool)],
                          legacy_total=np.r_[np.clip((full_labels['legacy_target']-self.quality_limits[0])/
                                                   np.diff(self.quality_limits)[0], 0, 1), weak['target']].astype(np.float32),
                          legacy_rank_mask=np.r_[full_labels['legacy_rank_mask'], weak['rank_mask']])
        if self.training_config.view_auxiliary:
            shape = (missing_count, inputs.view_mask.shape[1])
            normalized = (full_labels['view_raw']-self.quality_limits[0])/np.diff(self.quality_limits)[0]
            labels.update(view_target=np.r_[normalized, np.zeros(shape)].astype(np.float32),
                          view_rank_target=np.r_[normalized, np.zeros(shape)].astype(np.float32),
                          view_target_mask=np.r_[full_labels['view_mask'], np.zeros(shape, bool)])
        summary = dict(scene=scene, choice=choice, full_regions=full_count, missing_regions=missing_count,
                       full_valid=int(full_labels["mask"].sum()), missing_valid=int(weak["mask"].sum()),
                       full_cache_hit=hit, cache_counters=dict(self.v3_counts),
                       pair_hash=hashlib.sha256(json.dumps(choice, sort_keys=True).encode()).hexdigest(),
                       label_audit=dict(full=full_labels["audit"], missing=weak["audit"]),
                       evidence_audit=dict(full=entry["evidence"], missing=missing["geometry_evidence_audit"]),
                       target_policy="Full measured exposure-corrected GS; Missing observation-loss pseudo quality",
                       geometry_delta_computed=False)
        if self.training_config.source_patch_recovery:
            summary['source_recovery'] = dict(full=entry['source_recovery'], missing=missing['source_recovery_audit'])
        if self.training_config.directional_quality:
            if int(choice.get('cache_slot', -1)) < 0:
                summary['evaluation_regions'] = entry['evaluation_regions']
            summary['label_audit']['full'] = {**summary['label_audit']['full'],
                'view_labels_with_features': int((full_labels['view_mask'] & entry['inputs'].view_mask.numpy()).sum()),
                'view_targets_clipped': int((full_labels['view_mask'] & ((normalized < 0) | (normalized > 1))).sum()),
                'regional_targets_clipped': int((full_labels['mask'] & ((full_normalized < 0) | (full_normalized > 1))).sum())}
            summary['target_policy'] = 'Full observed-view quality and worst-two regional tail; Missing regional weak tail, no per-view pseudo truth'
        return inputs, labels, summary, None

    def smoke_cache_requirements(self):
        counts = dict(self.v3_counts)
        return dict(passed=all(counts[name] > 0 for name in ("full_hits", "full_misses", "affected", "unaffected")),
                    counts=counts, entries=len(self.full_cache), capacity=self.cache_capacity)


def concatenate_pairs(pairs, rank=0, world_size=1):
    inputs = combine_inputs([pair[0] for pair in pairs])
    labels = {}
    for key in pairs[0][1]:
        if key == 'groups':
            continue
        values = [pair[1][key] for pair in pairs]
        if values[0].ndim == 2:
            values = [np.pad(value, ((0, 0), (0, inputs.view_mask.shape[1]-value.shape[1]))) for value in values]
        labels[key] = np.concatenate(values)
    labels["state"] = np.concatenate([pair[1]["groups"] for pair in pairs])
    labels["groups"] = np.concatenate([pair[1]["groups"]+2*(ordinal*world_size+rank)
                                       for ordinal, pair in enumerate(pairs)])
    labels["pair_ids"] = labels["groups"]//2
    return inputs, labels
