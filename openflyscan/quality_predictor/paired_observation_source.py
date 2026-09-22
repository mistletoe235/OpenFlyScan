"""Versioned data adapter retaining the frozen v1 sampling and teacher contract."""

import dataclasses
from collections import OrderedDict

import numpy as np
import torch

from .observation_source import Source
from .geometry_evidence import augment_current_inputs, label_support_audit, self_projection_audit
from .model import LocalQueryInputs
from .base_supervision import FullLabelPolicy, FullTeacherLookup, full_region_supervision
from .training_performance import stage, timed


class SourceV2(Source):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        configured = self.plan.get("full_label_policy")
        self.full_label_policy = FullLabelPolicy(**configured) if configured else None
        self.full_teacher_lookups = OrderedDict()

    @timed('full_pixel_labels_v2')
    def full_pixel_labels(self, scene, cloud):
        if "teacher_mapping" not in self.records[scene]:
            return super().full_pixel_labels(scene, cloud)
        record = self.records[scene]
        contracted = "teacher_mapping_contract" in record
        cache_key = (scene, record["teacher_mapping"], str(record.get("source")), str(record.get("evaluation")),
                     self.full_label_policy is not None, int(cloud.get("sample_stride", 14)),
                     tuple(cloud["stems"]), tuple(cloud["maps"].shape[:3]))
        if "full_teacher_patches" not in cloud or (contracted and cloud.get("full_teacher_cache_key") != cache_key):
            if scene not in self.full_teacher_lookups:
                self.cache_counts['teacher_misses'] += 1
                with stage('teacher_lookup_build'):
                    self.full_teacher_lookups[scene] = FullTeacherLookup(self.root, self.records[scene])
                limit = self.performance.teacher_cache_scenes or len(self.records)
                while len(self.full_teacher_lookups) > limit:
                    self.full_teacher_lookups.popitem(last=False)
                    self.cache_counts['teacher_evictions'] += 1
            else:
                self.cache_counts['teacher_hits'] += 1
            self.full_teacher_lookups.move_to_end(scene)
            with stage('teacher_pixel_mapping'):
                cloud["full_teacher_patches"] = self.full_teacher_lookups[scene].patches(
                    cloud, with_tracks=self.full_label_policy is not None)
            cloud["full_teacher_cache_key"] = cache_key
        return cloud["full_teacher_patches"][0]

    def extract(self, scene, stems, seed):
        inputs, cloud, proposal = super().extract(scene, stems, seed)
        with stage('v2_geometry_evidence'):
            inputs, audit = augment_current_inputs(inputs, cloud, proposal)
        cloud["geometry_evidence_audit"] = audit
        return inputs, cloud, proposal

    @timed('pair_v2', source=True)
    def pair(self, scene, rng, seed=20260907, choice=None):
        inputs, labels, summary, clouds = super().pair(scene, rng, seed, choice)
        full, missing, full_proposal, missing_proposal, errors = clouds
        full_count = len(full_proposal["centers"])
        with stage('v2_label_support'):
            full_labels = label_support_audit(full, full_proposal, self.full_pixel_labels(scene, full),
                                             inputs.view_mask[:full_count].cpu().numpy())
            missing_labels = label_support_audit(missing, missing_proposal, errors,
                                                inputs.view_mask[full_count:].cpu().numpy())
        for name, audit, expected in [("full", full_labels, labels["total_mask"][:full_count]),
                                      ("missing", missing_labels, labels["structure_mask"][full_count:])]:
            if not np.array_equal([row["valid"] for row in audit["rows"]], expected):
                raise RuntimeError(f"{name} label audit disagrees with frozen teacher mask")
        summary["label_audit"] = dict(full=full_labels, missing=missing_labels)
        with stage('v2_self_projection'):
            summary["self_projection"] = dict(full=self_projection_audit(full), missing=self_projection_audit(missing))
        summary["evidence_audit"] = dict(full=full["geometry_evidence_audit"], missing=missing["geometry_evidence_audit"])
        labels["total_rank_mask"] = labels["total_mask"].copy()
        labels["total_weight"] = labels["total_mask"].astype(np.float32)
        if self.full_label_policy is not None and "teacher_mapping" in self.records[scene]:
            values, cells, tracks = full["full_teacher_patches"]
            with stage('v2_full_supervision'):
                result = full_region_supervision(full, full_proposal, values, cells, tracks, self.full_label_policy)
            if not np.array_equal(result["rank_mask"], labels["total_mask"][:full_count]):
                raise RuntimeError("Full supervision changed the original valid set")
            labels["total"][:full_count] = np.clip(
                (result["target"]-self.quality_limits[0])/np.diff(self.quality_limits)[0], 0, 1)
            labels["total_mask"][:full_count] = result["mask"]
            labels["total_weight"][:full_count] = result["weight"]
            summary["full_supervision"] = {key: value for key, value in result["audit"].items() if key != "rows"}
            summary["full_valid"] = int(result["mask"].sum())
            summary["label_audit"]["full_recovery"] = result["audit"]
        return inputs, labels, summary, clouds


def concatenate_pairs(pairs, rank=0, world_size=1):
    if not pairs:
        raise ValueError("at least one Full/Missing pair is required")
    inputs = LocalQueryInputs(**{
        field.name: torch.cat([getattr(pair[0], field.name) for pair in pairs], dim=0)
        for field in dataclasses.fields(LocalQueryInputs)
    })
    labels = {key: np.concatenate([pair[1][key] for pair in pairs])
              for key in ["total", "total_mask", "structure", "structure_mask"]}
    labels["total_weight"] = np.concatenate([
        pair[1].get("total_weight", pair[1]["total_mask"].astype(np.float32)) for pair in pairs])
    labels["total_rank_mask"] = np.concatenate([
        pair[1].get("total_rank_mask", pair[1]["total_mask"]) for pair in pairs])
    labels["groups"] = np.concatenate([
        pair[1]["groups"] + 2*(ordinal*world_size+rank) for ordinal, pair in enumerate(pairs)
    ])
    labels["state"] = np.concatenate([pair[1]["groups"] for pair in pairs])
    labels["pair_ids"] = labels["groups"] // 2
    return inputs, labels
