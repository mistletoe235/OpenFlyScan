"""V2 loss, distributed ranking, and explicitly defined modality controls."""

import dataclasses
import itertools
import math
from dataclasses import dataclass

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from torch import nn
from torch.distributed.nn.functional import all_gather as differentiable_all_gather

from .geometry_evidence import REGION_EVIDENCE_NAMES, VIEW_EVIDENCE_NAMES
from .model import LocalQueryInputs, LocalQueryQualityConfig, LocalQueryQualityHead


@dataclass(frozen=True)
class TrainingConfigV2:
    schema: str = "exposure_head_v2"
    seed: int = 20260907
    steps: int = 10000
    chunks_per_rank: int = 1
    expected_world_size: int = 8
    allow_repeated_scenes_per_step: bool = False
    max_regions: int = 64
    hidden_dim: int = 128
    learning_rate: float = 0.0002
    weight_decay: float = 0.0001
    warmup_steps: int = 100
    save_every: int = 500
    validation_every: int = 200
    modality: str = "full"
    dino_dropout: float = 0.05
    confidence_dropout: float = 0.05
    minimum_target_gap: float = 0.15
    ranking_margin: float = 0.05
    full_regression_weight: float = 0.5
    missing_structure_regression_weight: float = 0.5
    full_local_ranking_weight: float = 0.125
    missing_structure_local_ranking_weight: float = 0.125
    missing_total_local_ranking_weight: float = 0.125
    full_cross_ranking_weight: float = 0.125
    missing_total_cross_ranking_weight: float = 0.125
    missing_structure_cross_ranking_weight: float = 0.0625

    def __post_init__(self):
        if self.schema != "exposure_head_v2":
            raise ValueError("unsupported v2 configuration schema")
        if type(self.allow_repeated_scenes_per_step) is not bool:
            raise ValueError("allow_repeated_scenes_per_step must be a bool")
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        for name in ("steps", "chunks_per_rank", "expected_world_size", "max_regions", "hidden_dim", "save_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.modality not in ("full", "dino_only", "confidence_only", "point_only", "explicit_geometry_only"):
            raise ValueError("unknown modality control")
        if min(self.dino_dropout, self.confidence_dropout) < 0 or self.dino_dropout+self.confidence_dropout > 1:
            raise ValueError("modality dropout probabilities must sum to at most one")
        for field in dataclasses.fields(self):
            if field.name.endswith("_weight") and getattr(self, field.name) < 0:
                raise ValueError("loss weights cannot be negative")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.warmup_steps < 0:
            raise ValueError("invalid optimizer configuration")
        if self.validation_every < 0:
            raise ValueError("validation_every cannot be negative")
        if self.minimum_target_gap <= 0 or self.ranking_margin <= 0:
            raise ValueError("ranking gap and margin must be positive")

    def model_config(self):
        return LocalQueryQualityConfig(hidden_dim=self.hidden_dim,
                                       quality_geometry_dim=4+len(VIEW_EVIDENCE_NAMES),
                                       query_dim=10+len(REGION_EVIDENCE_NAMES))


def minimum_training_scenes(config):
    global_pairs = config.expected_world_size*config.chunks_per_rank
    return min(8, global_pairs) if config.allow_repeated_scenes_per_step else global_pairs


class TrainingHeadV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = LocalQueryQualityHead(config.model_config())

    def forward(self, inputs):
        output = self.head(inputs)
        return {"total": output.total_risk, "structure": output.structure_risk}


def smoke_requirements(config, counts, missing_gradient_verified):
    required = [field.name.removesuffix("_weight") for field in dataclasses.fields(config)
                if field.name.endswith("ranking_weight") and getattr(config, field.name) > 0]
    gradient_required = config.missing_total_local_ranking_weight > 0
    passed = all(counts.get(name, 0) > 0 for name in required)
    return dict(passed=passed and (missing_gradient_verified or not gradient_required),
                required_ranking_paths=required, missing_local_gradient_required=gradient_required)


def modality_inputs(inputs, pair_ids, config, generator=None, training=True):
    values = {field.name: getattr(inputs, field.name) for field in dataclasses.fields(inputs)}
    content = set(values)-{"view_mask"}
    retained = {
        "dino_only": {"dino_features"},
        "confidence_only": {"confidence_features"},
        "point_only": {"point_features"},
        "explicit_geometry_only": {"pose_geometry_features", "quality_geometry_features", "query_features"},
    }
    dropped = {"dino": 0, "confidence": 0, "pairs": 0}
    if config.modality != "full":
        for name in content-retained[config.modality]:
            values[name] = torch.zeros_like(values[name])
    elif training:
        for pair in torch.unique(pair_ids).tolist():
            selected = pair_ids == pair
            draw = float(torch.rand((), generator=generator))
            dropped["pairs"] += 1
            name = None
            if draw < config.dino_dropout:
                name = "dino_features"
                dropped["dino"] += 1
            elif draw < config.dino_dropout+config.confidence_dropout:
                name = "confidence_features"
                dropped["confidence"] += 1
            if name:
                values[name] = values[name].clone()
                values[name][selected] = 0
    return LocalQueryInputs(**values), dropped


def labels_to_device(labels, device):
    return {name: torch.as_tensor(value, device=device) for name, value in labels.items()}


def grouped_regression(prediction, target, mask, groups, reliability=None):
    losses = []
    for group in torch.unique(groups):
        selected = mask & (groups == group)
        if not selected.any():
            continue
        truth = target[selected]
        weights = 1+2*truth.square()
        residual = functional.smooth_l1_loss(prediction[selected], truth, beta=0.1, reduction="none")
        confidence = reliability[selected] if reliability is not None else torch.ones_like(truth)
        losses.append((residual*weights*confidence).sum()/weights.sum())
    return torch.stack(losses).mean() if losses else prediction.sum()*0


def ranking_loss(prediction, target, mask, groups, config, *, cross=False, scene_ids=None):
    active = torch.unique(groups[mask]).tolist()
    group_pairs = itertools.combinations(active, 2) if cross else ((group, group) for group in active)
    losses = []
    pair_count = cross_scene_count = 0
    for first, second in group_pairs:
        left = mask & (groups == first)
        right = mask & (groups == second)
        difference = target[left, None]-target[right][None, :]
        selected = difference.abs() >= config.minimum_target_gap
        if not cross:
            selected &= torch.triu(torch.ones_like(selected), diagonal=1)
        count = int(selected.sum().item())
        if not count:
            continue
        delta = prediction[left, None]-prediction[right][None, :]
        losses.append(functional.relu(config.ranking_margin-difference.sign()*delta)[selected].mean())
        pair_count += count
        if cross and scene_ids is not None:
            cross_scene_count += int((selected & (scene_ids[left, None] != scene_ids[right][None, :])).sum().item())
    loss = torch.stack(losses).mean() if losses else prediction.sum()*0
    return loss, dict(pairs=pair_count, group_pairs=len(losses), cross_scene_pairs=cross_scene_count)


def gather_predictions_and_labels(predictions, labels):
    if not distributed.is_initialized() or distributed.get_world_size() == 1:
        return predictions, labels
    world = distributed.get_world_size()
    local = torch.stack([predictions["total"], predictions["structure"]], -1)
    size = torch.tensor([len(local)], device=local.device, dtype=torch.long)
    sizes = [torch.empty_like(size) for _ in range(world)]
    distributed.all_gather(sizes, size)
    maximum = max(int(value.item()) for value in sizes)
    padded = functional.pad(local, (0, 0, 0, maximum-len(local)))
    gathered = differentiable_all_gather(padded)
    merged = torch.cat([value[:int(length.item())] for value, length in zip(gathered, sizes)])
    gathered_labels = {}
    for name in sorted(labels):
        value = labels[name]
        padded_label = functional.pad(value, (0, maximum-len(value)))
        outputs = [torch.empty_like(padded_label) for _ in range(world)]
        distributed.all_gather(outputs, padded_label)
        gathered_labels[name] = torch.cat([output[:int(length.item())] for output, length in zip(outputs, sizes)])
    return {"total": merged[:, 0], "structure": merged[:, 1]}, gathered_labels


def training_loss(predictions, labels, config):
    pieces, statistics = {}, {}
    groups = labels["groups"]
    full = labels["total_mask"].bool() & (labels["state"] == 0)
    full_rank = full & labels.get("total_rank_mask", labels["total_mask"]).bool()
    missing = labels["structure_mask"].bool() & (labels["state"] == 1)
    pieces["full_regression"] = grouped_regression(predictions["total"], labels["total"], full, groups, labels.get("total_weight"))
    pieces["missing_structure_regression"] = grouped_regression(predictions["structure"], labels["structure"], missing, groups)
    for name, output, target, mask in [
        ("full_local_ranking", "total", "total", full_rank),
        ("missing_structure_local_ranking", "structure", "structure", missing),
        ("missing_total_local_ranking", "total", "structure", missing),
    ]:
        pieces[name], statistics[name] = ranking_loss(predictions[output], labels[target], mask, groups, config)
    need_cross = any(getattr(config, name) > 0 for name in (
        "full_cross_ranking_weight", "missing_total_cross_ranking_weight", "missing_structure_cross_ranking_weight"))
    if need_cross:
        global_predictions, global_labels = gather_predictions_and_labels(predictions, labels)
        for name, output, target, mask_name, state in [
            ("full_cross_ranking", "total", "total", "total_mask", 0),
            ("missing_total_cross_ranking", "total", "structure", "structure_mask", 1),
            ("missing_structure_cross_ranking", "structure", "structure", "structure_mask", 1),
        ]:
            mask = global_labels[mask_name].bool() & (global_labels["state"] == state)
            if state == 0:
                mask &= global_labels.get("total_rank_mask", global_labels[mask_name]).bool()
            pieces[name], statistics[name] = ranking_loss(
                global_predictions[output], global_labels[target], mask, global_labels["groups"],
                config, cross=True, scene_ids=global_labels.get("scene_ids"))
    total = sum(value*getattr(config, name+"_weight") for name, value in pieces.items())
    return total, pieces, statistics


def initialize_head(head, checkpoint):
    state = checkpoint["head"]
    if not all(name.startswith("head.") for name in state):
        state = {"head."+name: value for name, value in state.items()}
    current = head.state_dict()
    unexpected = set(state)-set(current)
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {sorted(unexpected)}")
    copied = {name: value for name, value in state.items() if name in current and value.shape == current[name].shape}
    reset = sorted(set(current)-set(copied))
    allowed = ("head.query.", "head.quality_geometry.")
    if any(not name.startswith(allowed) for name in reset):
        raise ValueError("checkpoint differs outside the expanded input projectors")
    head.load_state_dict({**current, **copied})
    return dict(source_step=checkpoint.get("step"), copied_tensors=len(copied), reset_tensors=reset,
                optimizer_restored=False, boundary="Warm start only; expanded projectors are not prediction-equivalent to v1.")
