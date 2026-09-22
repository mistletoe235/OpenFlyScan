"""Frozen-input sensitivity audit for the shared risk, not a performance claim."""

import dataclasses

import numpy as np
from scipy.stats import spearmanr
import torch

from .model import LocalQueryInputs


def grouped_mean_inputs(inputs, groups, fields, quality_slice=None):
    values = {field.name: getattr(inputs, field.name) for field in dataclasses.fields(inputs)}
    for name in fields:
        value = values[name].clone()
        for group in torch.unique(groups):
            selected = groups == group
            current = value[selected].clone()
            if name == "query_features":
                current[:] = current.mean(0)
            else:
                valid = inputs.view_mask[selected].bool()
                if not valid.any():
                    continue
                if quality_slice is None or name != "quality_geometry_features":
                    current[valid] = current[valid].mean(0)
                else:
                    component = current[..., quality_slice].clone()
                    component[valid] = component[valid].mean(0)
                    current[..., quality_slice] = component
            value[selected] = current
        values[name] = value
    return LocalQueryInputs(**values)


@torch.no_grad()
def modality_audit(head, inputs, labels):
    baseline = head(inputs)["total"]
    conditions = {
        "dino": (["dino_features"], None),
        "point_decoder": (["point_features"], None),
        "confidence_decoder": (["confidence_features"], None),
        "confidence_scalar": (["quality_geometry_features"], slice(3, 4)),
        "explicit_quality_first_four": (["quality_geometry_features"], slice(0, 4)),
        "pose_geometry": (["pose_geometry_features"], None),
        "rgb": (["rgb_stats"], None),
        "all_geometry": (["point_features", "confidence_features", "pose_geometry_features",
                          "quality_geometry_features", "query_features"], None),
    }
    rows = {}
    for name, (fields, quality_slice) in conditions.items():
        changed = head(grouped_mean_inputs(inputs, labels["groups"], fields, quality_slice))["total"]
        rows[name] = {}
        for state, state_name in ((0, "full_real"), (1, "missing_pseudo")):
            mask = labels["total_mask"].bool() & (labels["groups"] % 2 == state)
            rows[name][state_name] = dict(
                regions=int(mask.sum()),
                mean_prediction_change=float((changed[mask]-baseline[mask]).abs().mean()) if mask.any() else None,
                baseline_mae=float((baseline[mask]-labels["total"][mask]).abs().mean()) if mask.any() else None,
                ablated_mae=float((changed[mask]-labels["total"][mask]).abs().mean()) if mask.any() else None)
            rank_mask = mask & labels["total_rank_mask"].bool()
            target = labels["rank_target"][rank_mask].cpu().numpy()
            metrics = {}
            for condition, prediction in (("baseline", baseline), ("ablated", changed)):
                scores = prediction[rank_mask].cpu().numpy()
                budget = int(np.ceil(.2*len(target)))
                correlation = float(spearmanr(scores, target).statistic) if (
                    len(target) > 1 and np.std(target) > 1e-8 and np.std(scores) > 1e-8) else None
                metrics[condition] = dict(spearman=correlation, worst20_recall=(
                    len(set(np.argsort(scores)[-budget:]) & set(np.argsort(target)[-budget:]))/budget) if budget else None)
            rows[name][state_name]["ranking"] = metrics
    return dict(conditions=rows, boundary="Within-chunk means preserve scale and masks. Missing MAE uses pseudo labels; sensitivity is not independent GS accuracy.")
