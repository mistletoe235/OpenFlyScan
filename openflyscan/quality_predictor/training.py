"""Regional error prediction with Base and weak Sparse supervision."""

import dataclasses

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from torch.distributed.nn.functional import all_gather as differentiable_all_gather

from .training_base import TrainingConfigV2, TrainingHeadV2, grouped_regression, ranking_loss


@dataclasses.dataclass(frozen=True)
class QualityPredictorTrainingConfig(TrainingConfigV2):
    schema: str = "exposure_head_v3"
    full_reuse_visits: int = 8
    learning_rate_schedule_steps: int = 10000
    validation_cases: int = 32
    missing_structure_regression_weight: float = 0.0
    missing_structure_local_ranking_weight: float = 0.0
    missing_structure_cross_ranking_weight: float = 0.0
    missing_total_regression_weight: float = 0.1
    missing_total_local_ranking_weight: float = 0.025
    missing_total_cross_ranking_weight: float = 0.025
    directional_quality: bool = False
    full_view_regression_weight: float = 0.0
    full_view_ranking_weight: float = 0.0
    exclude_outer_chunks: bool = False
    view_auxiliary: bool = False
    source_patch_recovery: bool = False

    def __post_init__(self):
        base = {field.name: getattr(self, field.name) for field in dataclasses.fields(TrainingConfigV2)}
        base["schema"] = "exposure_head_v2"
        TrainingConfigV2(**base)
        if self.schema != "exposure_head_v3" or self.full_reuse_visits < 2:
            raise ValueError("V3 needs its own schema and at least two Full reuse visits")
        if self.learning_rate_schedule_steps < self.steps:
            raise ValueError("learning-rate horizon must cover all steps")
        if self.validation_cases < 1:
            raise ValueError("validation case count must be positive")
        if any(getattr(self, name) != 0 for name in base if name.startswith("missing_structure_")):
            raise ValueError("V3 must not enable geometry-delta supervision")
        if not 0 < self.missing_total_regression_weight <= self.full_regression_weight:
            raise ValueError("Missing weak regression must be positive and no stronger than Full")
        if type(self.directional_quality) is not bool:
            raise ValueError('directional_quality must be boolean')
        if type(self.exclude_outer_chunks) is not bool or (self.exclude_outer_chunks and self.directional_quality):
            raise ValueError('outer-chunk experiment must use the unchanged V3 regional model')
        weights = (self.full_view_regression_weight, self.full_view_ranking_weight)
        if any(not torch.isfinite(torch.tensor(value)) or value < 0 for value in weights):
            raise ValueError('view loss weights must be finite and nonnegative')
        if type(self.view_auxiliary) is not bool or type(self.source_patch_recovery) is not bool:
            raise ValueError('auxiliary and recovery flags must be boolean')
        if self.view_auxiliary and (self.directional_quality or not self.source_patch_recovery):
            raise ValueError('view auxiliary requires source recovery and unchanged regional output')
        enabled = self.directional_quality or self.view_auxiliary
        if enabled != all(value > 0 for value in weights) or (not enabled and any(weights)):
            raise ValueError('directional mode requires both view losses, legacy mode requires neither')


class QualityPredictor(TrainingHeadV2):
    def __init__(self, config):
        if config.directional_quality:
            torch.nn.Module.__init__(self)
            from .directional_quality import DirectionalQualityHead
            self.head = DirectionalQualityHead(config.model_config())
        else:
            super().__init__(config)
        self.directional_quality = config.directional_quality
        self.view_auxiliary = config.view_auxiliary
        if self.view_auxiliary:
            hidden = config.hidden_dim
            self.view_head = torch.nn.Sequential(torch.nn.Linear(3*hidden, hidden), torch.nn.GELU(),
                                                torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, 1))

    def forward(self, inputs):
        if self.directional_quality:
            return self.head(inputs)
        if self.view_auxiliary:
            output = self.head(inputs, return_view_context=True)
            return dict(total=output.total_risk, view_risk=self.view_head(output.view_context).squeeze(-1),
                        view_mask=inputs.view_mask)
        return {"total": self.head(inputs).total_risk}


def gather_total(prediction, labels):
    if not distributed.is_initialized() or distributed.get_world_size() == 1:
        return prediction, labels
    size = torch.tensor([len(prediction)], device=prediction.device, dtype=torch.long)
    sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(sizes, size)
    lengths = [int(value.item()) for value in sizes]
    maximum = max(lengths)
    padded = functional.pad(prediction, (0, maximum-len(prediction)))
    merged = torch.cat([value[:length] for value, length in zip(differentiable_all_gather(padded), lengths)])
    merged_labels = {}
    for name in sorted(labels):
        padded_label = functional.pad(labels[name], (0, maximum-len(labels[name])))
        outputs = [torch.empty_like(padded_label) for _ in lengths]
        distributed.all_gather(outputs, padded_label)
        merged_labels[name] = torch.cat([value[:length] for value, length in zip(outputs, lengths)])
    return merged, merged_labels


def cross_ranking_loss(prediction, target, mask, groups, config, scene_ids=None):
    active = torch.unique(groups[mask])
    if len(active) < 2:
        return prediction.sum()*0, dict(pairs=0, group_pairs=0, cross_scene_pairs=0)
    indices = [torch.nonzero(mask & (groups == group), as_tuple=False).flatten() for group in active]
    maximum = max(len(index) for index in indices)
    padded = torch.stack([functional.pad(index, (0, maximum-len(index))) for index in indices])
    valid = torch.arange(maximum, device=prediction.device)[None, :] < torch.tensor(
        [len(index) for index in indices], device=prediction.device)[:, None]
    pairs = torch.triu_indices(len(indices), len(indices), offset=1, device=prediction.device)
    losses = []
    pair_count = group_count = cross_count = 0
    for start in range(0, pairs.shape[1], 256):
        left, right = pairs[:, start:start+256]
        left_ids, right_ids = padded[left], padded[right]
        difference = target[left_ids][:, :, None]-target[right_ids][:, None, :]
        selected = valid[left, :, None] & valid[right, None, :] & (difference.abs() >= config.minimum_target_gap)
        counts = selected.sum((1, 2))
        delta = prediction[left_ids][:, :, None]-prediction[right_ids][:, None, :]
        residual = functional.relu(config.ranking_margin-difference.sign()*delta)
        eligible = counts > 0
        losses.append(((residual*selected).sum((1, 2))/counts.clamp_min(1))[eligible].sum())
        pair_count += int(counts.sum())
        group_count += int(eligible.sum())
        if scene_ids is not None:
            cross_count += int((selected & (scene_ids[left_ids][:, :, None] != scene_ids[right_ids][:, None, :])).sum())
    loss = torch.stack(losses).sum()/max(group_count, 1)
    return loss, dict(pairs=pair_count, group_pairs=group_count, cross_scene_pairs=cross_count)


def training_loss(predictions, labels, config):
    pieces, statistics = {}, {}
    prediction = predictions["total"]
    for state, prefix in [(0, "full"), (1, "missing_total")]:
        mask = labels["total_mask"].bool() & (labels["state"] == state)
        pieces[prefix+"_regression"] = grouped_regression(
            prediction, labels["total"], mask, labels["groups"], labels["total_weight"])
        pieces[prefix+"_local_ranking"], statistics[prefix+"_local_ranking"] = ranking_loss(
            prediction, labels["rank_target"], mask & labels["total_rank_mask"].bool(), labels["groups"], config)
    if config.full_cross_ranking_weight or config.missing_total_cross_ranking_weight:
        global_prediction, global_labels = gather_total(prediction, {name: value for name, value in labels.items()
                                                                    if value.ndim == 1})
        for state, prefix in [(0, "full"), (1, "missing_total")]:
            mask = global_labels["total_rank_mask"].bool() & (global_labels["state"] == state)
            pieces[prefix+"_cross_ranking"], statistics[prefix+"_cross_ranking"] = cross_ranking_loss(
                global_prediction, global_labels["rank_target"], mask, global_labels["groups"], config,
                scene_ids=global_labels.get("scene_ids"))
    if config.directional_quality or config.view_auxiliary:
        from .directional_quality import view_losses
        pieces['full_view_regression'], pieces['full_view_ranking'], statistics['full_view_ranking'] = view_losses(
            predictions, labels, config)
    loss = sum(value*getattr(config, name+"_weight") for name, value in pieces.items())
    return loss, pieces, statistics


def smoke_requirements(config, counts, missing_gradient_verified):
    required = [field.name.removesuffix("_weight") for field in dataclasses.fields(config)
                if field.name.endswith("ranking_weight") and getattr(config, field.name) > 0]
    return dict(passed=bool(missing_gradient_verified and all(counts.get(name, 0) > 0 for name in required)),
                required_ranking_terms=required, missing_gradient_verified=missing_gradient_verified)
