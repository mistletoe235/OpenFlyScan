"""Observed-view quality with one conservative bad-view-tail risk per region."""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
import torch.nn.functional as functional

from .model import _projector, _QueryConditionedPool


def worst_two_numpy(values):
    values = np.asarray(values)
    valid = np.isfinite(values)
    ordered = np.sort(np.where(valid, values, -np.inf), axis=-1)[..., -2:]
    selected = np.isfinite(ordered)
    return np.where(selected, ordered, 0).sum(-1)/selected.sum(-1).clip(1)


def worst_two_tensor(values, mask):
    ordered = values.masked_fill(~mask, -torch.inf).topk(min(2, values.shape[1]), dim=1).values
    selected = torch.isfinite(ordered)
    return ordered.masked_fill(~selected, 0).sum(1)/selected.sum(1).clamp_min(1)


def observed_view_labels(cloud, proposal, values, original):
    points = np.asarray(cloud['maps']).reshape(-1, 3)
    values = np.asarray(values).reshape(-1)
    views, height, width = cloud['maps'].shape[:3]
    valid = np.isfinite(points).all(1) & np.isfinite(values)
    indices = np.flatnonzero(valid)
    tree = cKDTree(points[valid])
    centers = np.asarray(proposal['centers'])
    targets = np.full((len(centers), views), np.nan, np.float32)
    counts = np.zeros((len(centers), views), np.int64)
    evidence = []
    for region, center in enumerate(centers):
        found = indices[tree.query_ball_point(center, proposal['cell_size'])]
        view_ids = found//(height*width)
        image_evidence = []
        for view in np.unique(view_ids):
            patches = found[view_ids == view]
            targets[region, view] = np.median(values[patches])
            counts[region, view] = len(patches)
            pixel = patches % (height*width)
            image_evidence.append(dict(stem=cloud['stems'][view], patch_hw=[height, width],
                                       patch_pixels_yx=np.stack([pixel//width, pixel % width], -1).tolist()))
        evidence.append(dict(xyz=center.tolist(), radius=float(proposal['cell_size']),
                             image_evidence=image_evidence, image_space_id='pi3x_source_patch_grid',
                             alignment_verified=False))
    view_mask = np.isfinite(targets)
    mask = original['mask'] & (view_mask.sum(1) > 0)
    rank_mask = original['rank_mask'] & (view_mask.sum(1) >= 2) & (counts.sum(1) >= 6)
    audit = {**original['audit'],
             'directional': True, 'view_labels': int(view_mask.sum()),
             'directional_rank_regions': int(rank_mask.sum()),
             'original_rank_regions': int(original['rank_mask'].sum()),
             'region_rule': 'mean of worst two measured view log-MSE medians; no angular bins',
             'unknown_views_are_not_good': True}
    return dict(target=worst_two_numpy(targets).astype(np.float32), mask=mask, rank_mask=rank_mask,
                weight=original['weight'], audit=audit, view_raw=np.nan_to_num(targets), view_mask=view_mask,
                legacy_target=original['target'], legacy_rank_mask=original['rank_mask']), evidence


class DirectionalQualityHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_dim
        for name, dimension in [('point', config.point_feature_dim), ('confidence', config.confidence_feature_dim),
                                ('dino', config.dino_feature_dim), ('pose_geometry', config.pose_geometry_dim),
                                ('quality_geometry', config.quality_geometry_dim), ('rgb', config.rgb_stats_dim),
                                ('query', config.query_dim)]:
            setattr(self, name, _projector(dimension, hidden))
        self.structure_fuse = nn.Sequential(nn.Linear(4*hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.appearance_fuse = nn.Sequential(nn.Linear(3*hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.structure_pool = _QueryConditionedPool(hidden, config.attention_heads, config.attention_layers, config.dropout).context
        self.appearance_pool = _QueryConditionedPool(hidden, config.attention_heads, config.attention_layers, config.dropout).context
        self.total_fuse = nn.Sequential(nn.Linear(3*hidden, 2*hidden), nn.GELU(), nn.LayerNorm(2*hidden),
                                        nn.Linear(2*hidden, 1))
        self.empty_risk = nn.Linear(hidden, 1)

    def forward(self, inputs):
        mask = inputs.view_mask
        if mask.ndim != 2 or mask.shape[1] == 0:
            raise ValueError('observations require a nonempty padded view axis')
        visible = mask[..., None]
        pose = self.pose_geometry(inputs.pose_geometry_features.masked_fill(~visible, 0))
        structure = self.structure_fuse(torch.cat([
            self.point(inputs.point_features.masked_fill(~visible, 0)),
            self.confidence(inputs.confidence_features.masked_fill(~visible, 0)), pose,
            self.quality_geometry(inputs.quality_geometry_features.masked_fill(~visible, 0))], -1))
        appearance = self.appearance_fuse(torch.cat([
            self.dino(inputs.dino_features.masked_fill(~visible, 0)),
            self.rgb(inputs.rgb_stats.masked_fill(~visible, 0)), pose], -1))
        query = self.query(inputs.query_features)
        safe_mask = mask.clone()
        empty = ~mask.any(1)
        safe_mask[empty, 0] = True
        structure = self.structure_pool(structure+query[:, None], src_key_padding_mask=~safe_mask)
        appearance = self.appearance_pool(appearance+query[:, None], src_key_padding_mask=~safe_mask)
        per_view = self.total_fuse(torch.cat([structure, appearance, query[:, None].expand_as(structure)], -1))
        per_view = torch.sigmoid(per_view.squeeze(-1))
        regional = worst_two_tensor(per_view, mask)
        regional = torch.where(empty, torch.sigmoid(self.empty_risk(query).squeeze(-1)), regional)
        return dict(total=regional, view_risk=per_view, view_mask=mask)


def view_losses(predictions, labels, config):
    scores = predictions['view_risk']
    target = labels['view_target']
    raw_target = labels['view_rank_target']
    mask = labels['view_target_mask'].bool() & predictions['view_mask'] & (labels['state'] == 0)[:, None]
    residual = functional.smooth_l1_loss(scores, target, beta=.1, reduction='none')
    weights = 1+2*target.square()
    regional = (residual*weights*mask).sum(1)/(weights*mask).sum(1).clamp_min(1)
    valid_regions = mask.any(1)
    regression = []
    ranking = []
    difference = raw_target[:, :, None]-raw_target[:, None, :]
    pairs = mask[:, :, None] & mask[:, None, :] & (difference >= config.minimum_target_gap)
    pair_counts = pairs.sum((1, 2))
    hinge = functional.relu(config.ranking_margin-(scores[:, :, None]-scores[:, None, :]))
    region_rank = (hinge*pairs).sum((1, 2))/pair_counts.clamp_min(1)
    for group in torch.unique(labels['groups'][labels['state'] == 0]):
        selected = (labels['groups'] == group) & valid_regions
        if selected.any():
            regression.append(regional[selected].mean())
        selected = (labels['groups'] == group) & (pair_counts > 0)
        if selected.any():
            ranking.append(region_rank[selected].mean())
    zero = scores.sum()*0
    return (torch.stack(regression).mean() if regression else zero,
            torch.stack(ranking).mean() if ranking else zero,
            dict(pairs=int(pair_counts.sum()), regions=int(valid_regions.sum()), views=int(mask.sum())))
