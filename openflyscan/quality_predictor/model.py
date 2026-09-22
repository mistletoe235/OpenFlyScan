"""Single-SLRF-chunk, query-conditioned reconstruction-risk model.

The model deliberately does not own chunking, camera registration, route
planning, or a long-term memory.  It consumes observations of one spatial
query gathered from one already-formed SLRF chunk.  GS measurements are
teacher-only and must never appear in :class:`LocalQueryInputs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class LocalQueryQualityConfig:
    point_feature_dim: int = 1024
    confidence_feature_dim: int = 1024
    dino_feature_dim: int = 1024
    pose_geometry_dim: int = 10
    quality_geometry_dim: int = 4
    rgb_stats_dim: int = 6
    query_dim: int = 10
    hidden_dim: int = 128
    attention_heads: int = 4
    attention_layers: int = 1
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        for name in (
            "point_feature_dim",
            "confidence_feature_dim",
            "dino_feature_dim",
            "pose_geometry_dim",
            "quality_geometry_dim",
            "rgb_stats_dim",
            "query_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class LocalQueryInputs:
    """Model-visible input for one padded batch.

    The leading shapes are ``[B,V,...]`` where ``V`` is the number of
    observations of the queried spatial region, not the number of images fed
    to Pi3X.  Pi3X itself must have already processed the complete 30-view
    SLRF chunk.
    """

    point_features: Tensor
    confidence_features: Tensor
    dino_features: Tensor
    pose_geometry_features: Tensor
    quality_geometry_features: Tensor
    rgb_stats: Tensor
    query_features: Tensor
    view_mask: Tensor


@dataclass(frozen=True)
class LocalQueryQualityOutput:
    structure_risk: Tensor
    appearance_risk: Tensor
    total_risk: Tensor
    appearance_gate: Tensor
    structure_embedding: Tensor
    appearance_embedding: Tensor
    structure_attention: Tensor
    appearance_attention: Tensor
    view_context: Optional[Tensor] = None


@dataclass(frozen=True)
class LocalQueryTargets:
    structure: Tensor
    appearance: Tensor
    total: Tensor
    structure_mask: Tensor
    appearance_mask: Tensor
    total_mask: Tensor


def _projector(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
    )


class _QueryConditionedPool(nn.Module):
    """Permutation-invariant observation encoder and query attention pool."""

    def __init__(self, hidden_dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            4 * hidden_dim,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            encoder_layer, layers, enable_nested_tensor=False
        )
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.scale = hidden_dim**-0.5

    def forward(self, observations: Tensor, query: Tensor, mask: Tensor, return_context=False):
        encoded = self.context(observations, src_key_padding_mask=~mask)
        scores = torch.einsum(
            "bvd,bd->bv", self.key(encoded), self.query(query)
        ) * self.scale
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bv,bvd->bd", weights, self.value(encoded))
        result = (self.out(pooled + query), weights)
        return (*result, encoded) if return_context else result


class LocalQueryQualityHead(nn.Module):
    """Separate structure and RGB evidence before a learned late fusion.

    The Point and Confidence decoder features are task-specific Pi3X features
    taken after their five-layer decoders and before the convolutional heads.
    DINO features come from the per-image encoder before geometric priors and
    multi-view decoding.  No old shared-decoder feature is accepted.
    """

    def __init__(self, config: LocalQueryQualityConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.point = _projector(config.point_feature_dim, d)
        self.confidence = _projector(config.confidence_feature_dim, d)
        self.dino = _projector(config.dino_feature_dim, d)
        self.pose_geometry = _projector(config.pose_geometry_dim, d)
        self.quality_geometry = _projector(config.quality_geometry_dim, d)
        self.rgb = _projector(config.rgb_stats_dim, d)
        self.query = _projector(config.query_dim, d)

        self.structure_fuse = nn.Sequential(
            nn.Linear(4 * d, d), nn.GELU(), nn.LayerNorm(d)
        )
        self.appearance_fuse = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(), nn.LayerNorm(d)
        )
        self.structure_pool = _QueryConditionedPool(
            d, config.attention_heads, config.attention_layers, config.dropout
        )
        self.appearance_pool = _QueryConditionedPool(
            d, config.attention_heads, config.attention_layers, config.dropout
        )
        self.structure_head = nn.Linear(d, 1)
        self.appearance_head = nn.Linear(d, 1)
        self.gate = nn.Sequential(
            nn.Linear(3 * d + 1, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.total_fuse = nn.Sequential(
            nn.Linear(3 * d + 4, 2 * d),
            nn.GELU(),
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, 1),
        )
        self.empty_structure = nn.Parameter(torch.zeros(d))
        self.empty_appearance = nn.Parameter(torch.zeros(d))

    def _validate(self, inputs: LocalQueryInputs) -> tuple[int, int]:
        tensors = {
            "point_features": (inputs.point_features, self.config.point_feature_dim),
            "confidence_features": (
                inputs.confidence_features,
                self.config.confidence_feature_dim,
            ),
            "dino_features": (inputs.dino_features, self.config.dino_feature_dim),
            "pose_geometry_features": (
                inputs.pose_geometry_features, self.config.pose_geometry_dim,
            ),
            "quality_geometry_features": (
                inputs.quality_geometry_features, self.config.quality_geometry_dim,
            ),
            "rgb_stats": (inputs.rgb_stats, self.config.rgb_stats_dim),
        }
        batch = views = None
        for name, (tensor, feature_dim) in tensors.items():
            if tensor.ndim != 3 or tensor.shape[-1] != feature_dim:
                raise ValueError(f"{name} must have shape [B,V,{feature_dim}]")
            if batch is None:
                batch, views = tensor.shape[:2]
            elif tensor.shape[:2] != (batch, views):
                raise ValueError(f"{name} has inconsistent B,V dimensions")
        if inputs.query_features.shape != (batch, self.config.query_dim):
            raise ValueError(
                f"query_features must have shape [B,{self.config.query_dim}]"
            )
        if inputs.view_mask.shape != (batch, views) or inputs.view_mask.dtype != torch.bool:
            raise ValueError("view_mask must be bool [B,V]")
        for name, (tensor, _) in tensors.items():
            valid_values = tensor[inputs.view_mask]
            if not torch.isfinite(valid_values).all():
                raise ValueError(f"{name} contains non-finite visible values")
        if not torch.isfinite(inputs.query_features).all():
            raise ValueError("query_features contains non-finite values")
        return int(batch), int(views)

    def forward(self, inputs: LocalQueryInputs, return_view_context=False) -> LocalQueryQualityOutput:
        self._validate(inputs)
        visible = inputs.view_mask.unsqueeze(-1)
        point_features = inputs.point_features.masked_fill(~visible, 0.0)
        confidence_features = inputs.confidence_features.masked_fill(~visible, 0.0)
        dino_features = inputs.dino_features.masked_fill(~visible, 0.0)
        pose_geometry_features = inputs.pose_geometry_features.masked_fill(~visible, 0.0)
        quality_geometry_features = inputs.quality_geometry_features.masked_fill(~visible, 0.0)
        rgb_stats = inputs.rgb_stats.masked_fill(~visible, 0.0)
        q = self.query(inputs.query_features)
        pose_geometry = self.pose_geometry(pose_geometry_features)
        quality_geometry = self.quality_geometry(quality_geometry_features)
        structure_tokens = self.structure_fuse(
            torch.cat(
                [
                    self.point(point_features), self.confidence(confidence_features),
                    pose_geometry, quality_geometry,
                ],
                dim=-1,
            )
        )
        appearance_tokens = self.appearance_fuse(
            torch.cat(
                [self.dino(dino_features), self.rgb(rgb_stats), pose_geometry],
                dim=-1,
            )
        )
        pool_mask = inputs.view_mask.clone()
        empty = ~pool_mask.any(dim=1)
        if torch.any(empty):
            pool_mask[empty, 0] = True
            structure_tokens = structure_tokens.clone()
            appearance_tokens = appearance_tokens.clone()
            structure_tokens[empty, 0] = self.empty_structure
            appearance_tokens[empty, 0] = self.empty_appearance
        structure_result = self.structure_pool(structure_tokens, q, pool_mask, return_view_context)
        appearance_result = self.appearance_pool(appearance_tokens, q, pool_mask, return_view_context)
        structure, structure_attention = structure_result[:2]
        appearance, appearance_attention = appearance_result[:2]
        view_context = None
        if return_view_context:
            view_context = torch.cat([structure_result[2], appearance_result[2],
                                      q[:, None].expand_as(structure_result[2])], -1)
        structure_risk = torch.sigmoid(self.structure_head(structure).squeeze(-1))
        appearance_risk = torch.sigmoid(self.appearance_head(appearance).squeeze(-1))

        support_log = torch.log1p(inputs.view_mask.sum(dim=1, keepdim=True).float())
        appearance_gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [structure, appearance, q, support_log], dim=-1
                )
            ).squeeze(-1)
        )
        total_risk = torch.sigmoid(self.total_fuse(torch.cat([
            structure, appearance, q, structure_risk[:, None],
            appearance_risk[:, None], appearance_gate[:, None], support_log,
        ], dim=-1)).squeeze(-1))
        return LocalQueryQualityOutput(
            structure_risk=structure_risk,
            appearance_risk=appearance_risk,
            total_risk=total_risk,
            appearance_gate=appearance_gate,
            structure_embedding=structure,
            appearance_embedding=appearance,
            structure_attention=structure_attention,
            appearance_attention=appearance_attention,
            view_context=view_context,
        )


def _weighted_huber(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    tail_weight: float,
) -> Tensor:
    if not torch.any(mask):
        return prediction.sum() * 0.0
    element = F.smooth_l1_loss(prediction, target, beta=0.1, reduction="none")
    # High-risk examples matter, but retain continuous supervision everywhere.
    weight = 1.0 + tail_weight * target.square()
    return (element[mask] * weight[mask]).sum() / weight[mask].sum().clamp_min(1e-6)


def groupwise_ranking_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    group_ids: Tensor,
    *,
    minimum_target_gap: float = 0.15,
    margin: float = 0.05,
) -> Tensor:
    """Rank queries only against other queries from the same current chunk."""

    losses = []
    for group in torch.unique(group_ids):
        selected = (group_ids == group) & mask
        if int(selected.sum()) < 2:
            continue
        p = prediction[selected]
        t = target[selected]
        difference = t[:, None] - t[None, :]
        separated = difference.abs() >= minimum_target_gap
        separated.fill_diagonal_(False)
        if not torch.any(separated):
            continue
        direction = torch.sign(difference)
        pred_difference = p[:, None] - p[None, :]
        losses.append(F.relu(margin - direction * pred_difference)[separated].mean())
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def local_query_quality_loss(
    current: LocalQueryQualityOutput,
    targets: LocalQueryTargets,
    group_ids: Tensor,
    *,
    reference: Optional[LocalQueryQualityOutput] = None,
    pair_delta: Optional[Tensor] = None,
    pair_mask: Optional[Tensor] = None,
    absolute_weight: float = 1.0,
    ranking_weight: float = 0.25,
    pair_weight: float = 0.5,
    tail_weight: float = 2.0,
) -> dict[str, Tensor]:
    """Risk supervision; no candidate-action gain is trained here."""

    structure = _weighted_huber(
        current.structure_risk,
        targets.structure,
        targets.structure_mask,
        tail_weight=tail_weight,
    )
    appearance = _weighted_huber(
        current.appearance_risk,
        targets.appearance,
        targets.appearance_mask,
        tail_weight=tail_weight,
    )
    total = _weighted_huber(
        current.total_risk,
        targets.total,
        targets.total_mask,
        tail_weight=tail_weight,
    )
    ranking = groupwise_ranking_loss(
        current.total_risk, targets.total, targets.total_mask, group_ids
    )

    pair_regression = current.total_risk.sum() * 0.0
    pair_order = current.total_risk.sum() * 0.0
    if reference is not None:
        if pair_delta is None or pair_mask is None:
            raise ValueError("paired reference requires pair_delta and pair_mask")
        if pair_delta.shape != current.total_risk.shape or pair_mask.shape != pair_delta.shape:
            raise ValueError("pair_delta and pair_mask must be [B]")
        predicted_delta = current.total_risk - reference.total_risk
        pair_regression = _weighted_huber(
            predicted_delta.clamp(-1.0, 1.0),
            pair_delta.clamp(-1.0, 1.0),
            pair_mask,
            tail_weight=tail_weight,
        )
        positive = pair_mask & (pair_delta >= 0.05)
        if torch.any(positive):
            pair_order = F.relu(0.03 - predicted_delta[positive]).mean()

    active_absolute = []
    if torch.any(targets.structure_mask):
        active_absolute.append(structure)
    if torch.any(targets.appearance_mask):
        active_absolute.append(appearance)
    if torch.any(targets.total_mask):
        active_absolute.append(total)
    absolute = (
        torch.stack(active_absolute).mean()
        if active_absolute else current.total_risk.sum() * 0.0
    )
    pair = pair_regression + pair_order
    loss = absolute_weight * absolute + ranking_weight * ranking + pair_weight * pair
    return {
        "loss": loss,
        "absolute": absolute,
        "structure": structure,
        "appearance": appearance,
        "total": total,
        "ranking": ranking,
        "pair": pair,
        "pair_regression": pair_regression,
        "pair_order": pair_order,
    }


def dynamic_mask_monotonic_loss(
    masked_risk: Tensor,
    full_risk: Tensor,
    removed_support_fraction: Tensor,
    *,
    maximum_margin: float = 0.02,
) -> Tensor:
    """Weak augmentation loss; it never invents an exact GS degradation value."""
    if masked_risk.shape != full_risk.shape or removed_support_fraction.shape != full_risk.shape:
        raise ValueError("dynamic monotonic inputs must all have shape [B]")
    importance = removed_support_fraction.detach().clamp(0.0, 1.0)
    required = maximum_margin * importance
    return F.relu(required - (masked_risk - full_risk)).mean()
