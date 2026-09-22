"""Weak current-quality targets from source-patch correspondence, not geometry delta."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ObservationPolicy:
    cap_db: float = 2.2
    saturation_fraction: float = 0.06
    minimum_patches: int = 6
    minimum_views: int = 2
    minimum_dominance: float = 0.6

    def __post_init__(self):
        if not np.isfinite(self.cap_db) or not 0 < self.cap_db <= 2.2:
            raise ValueError("penalty cap must be in (0, 2.2] dB")
        if not np.isfinite(self.saturation_fraction) or self.saturation_fraction <= 0:
            raise ValueError("saturation fraction must be positive")
        if self.minimum_patches < 6 or self.minimum_views < 2:
            raise ValueError("Missing correspondence requires six patches and two views")
        if not 0.5 < self.minimum_dominance <= 1:
            raise ValueError("invalid correspondence policy")


def pseudo_targets(raw_full, fraction, affected, limits, policy=ObservationPolicy()):
    raw_full, fraction = np.asarray(raw_full), np.asarray(fraction)
    if not np.isfinite(raw_full).all() or not np.isfinite(fraction).all():
        raise ValueError("pseudo targets require finite baseline and support fraction")
    if np.any((fraction < 0) | (fraction > 1)):
        raise ValueError("support fraction must be in [0, 1]")
    low, high = limits
    if not np.isfinite(limits).all() or high <= low:
        raise ValueError("invalid training quality limits")
    penalty = policy.cap_db / 10 * np.asarray(affected) * (-np.expm1(-fraction / policy.saturation_fraction))
    raw = raw_full + penalty
    normalized = (raw-low)/(high-low)
    return dict(raw=raw, rank_target=normalized, target=np.clip(normalized, 0, 1), penalty_db=10*penalty)


def source_patch_index(cloud, proposal):
    points = cloud["maps"].reshape(-1, 3)
    valid = np.isfinite(points).all(1)
    centers = np.asarray(proposal["centers"])
    owners = np.full(len(points), -1, np.int32)
    support = np.zeros((len(centers), len(cloud["stems"])), bool)
    if len(centers) and valid.any():
        distance, nearest = cKDTree(centers).query(points[valid])
        indices = np.flatnonzero(valid)[distance <= proposal["cell_size"]]
        owners[indices] = nearest[distance <= proposal["cell_size"]]
        tree = cKDTree(points[valid])
        valid_indices = np.flatnonzero(valid)
        per_view = int(np.prod(cloud["maps"].shape[1:3]))
        for region, center in enumerate(centers):
            found = valid_indices[tree.query_ball_point(center, proposal["cell_size"])]
            support[region, np.unique(found // per_view)] = True
    return dict(stems=tuple(cloud["stems"]), shape=tuple(cloud["maps"].shape[:3]),
                owners=owners, support=support)


def dense_source_patch_index(cloud, proposal, values, *, directional_quality=False):
    points = cloud["maps"].reshape(-1, 3)
    values = np.asarray(values).reshape(-1)
    if len(values) != len(points):
        raise ValueError("Full teacher must cover the source patch grid")
    valid = np.isfinite(points).all(1)
    owners = np.full(len(points), -1, np.int32)
    keys = np.floor((points[valid]-proposal["origin"])/proposal["cell_size"]).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    owners[valid] = inverse
    views = len(cloud["stems"])
    per_view = int(np.prod(cloud["maps"].shape[1:3]))
    support = np.zeros((len(unique), views), bool)
    indices = np.flatnonzero(valid)
    support[inverse, indices//per_view] = True
    supervised = valid & np.isfinite(values)
    indices = np.flatnonzero(supervised)
    group = owners[indices]*views+indices//per_view
    order = np.lexsort((values[indices], group))
    group_ids, starts, counts = np.unique(group[order], return_index=True, return_counts=True)
    sorted_values = values[indices[order]]
    medians = (sorted_values[starts+(counts-1)//2]+sorted_values[starts+counts//2])/2
    per_view_values = np.full((len(unique), views), np.nan, np.float32)
    per_view_values.reshape(-1)[group_ids] = medians
    labelled_views = np.isfinite(per_view_values).sum(1)
    patch_counts = np.bincount(owners[indices], minlength=len(unique))
    mask = (patch_counts >= 6) & (labelled_views >= 2)
    target = np.zeros(len(unique), np.float32)
    target[mask] = np.nanmedian(per_view_values[mask], axis=1)
    if directional_quality:
        from .directional_quality import worst_two_numpy
        target[mask] = worst_two_numpy(per_view_values[mask])
    labels = dict(target=target, mask=mask, rank_mask=mask.copy(), weight=mask.astype(np.float32))
    reference = dict(stems=tuple(cloud["stems"]), shape=tuple(cloud["maps"].shape[:3]),
                     owners=owners, support=support)
    return reference, labels


def corresponding_missing_labels(reference, full_labels, missing, proposal, deleted, limits,
                                 policy=ObservationPolicy()):
    full_stems, missing_stems = reference["stems"], tuple(missing["stems"])
    if set(deleted) & set(missing_stems) or set(full_stems) != set(deleted) | set(missing_stems):
        raise ValueError("deletion list must exactly partition Full into deleted and Missing")
    if tuple(missing["maps"].shape[1:3]) != reference["shape"][1:]:
        raise ValueError("source patch sampling changed between Full and Missing")
    per_view = int(np.prod(reference["shape"][1:]))
    missing_to_full = np.array([full_stems.index(stem) for stem in missing_stems])
    removed = np.isin(full_stems, deleted)
    points = missing["maps"].reshape(-1, 3)
    valid = np.isfinite(points).all(1)
    valid_indices = np.flatnonzero(valid)
    tree = cKDTree(points[valid])
    count = len(proposal["centers"])
    baseline = np.zeros(count, np.float32)
    fraction = np.zeros(count, np.float32)
    mask = np.zeros(count, bool)
    rank_mask = np.zeros(count, bool)
    weights = np.zeros(count, np.float32)
    matched_region = np.full(count, -1, np.int64)
    rows = []
    for region, center in enumerate(proposal["centers"]):
        found = valid_indices[tree.query_ball_point(center, proposal["cell_size"])]
        views = found // per_view
        row = dict(region=region, status="unknown", reason="insufficient_current_support",
                   current_patches=len(found), current_views=len(np.unique(views)))
        rows.append(row)
        if len(found) < policy.minimum_patches or len(np.unique(views)) < policy.minimum_views:
            continue
        full_patches = missing_to_full[views]*per_view + found % per_view
        owners = reference["owners"][full_patches]
        known = owners >= 0
        known[known] &= full_labels["mask"][owners[known]]
        candidates, votes = np.unique(owners[known], return_counts=True)
        if not len(candidates):
            row["reason"] = "unknown_full_quality"
            continue
        winner = int(candidates[np.argmax(votes)])
        coverage = float(known.sum()/len(found))
        row.update(full_region=winner, full_cells=len(candidates), coverage=coverage,
                   matched_views=len(np.unique(views[known])))
        if coverage < policy.minimum_dominance or len(np.unique(views[known])) < policy.minimum_views:
            row["reason"] = "ambiguous_region"
            continue
        row["full_quality_log_mse_range"] = float(np.ptp(full_labels["target"][candidates]))
        observed = reference["support"][candidates].any(0)
        remaining_count = int((observed & ~removed).sum())
        if remaining_count < policy.minimum_views:
            row["reason"] = "insufficient_retained_support"
            continue
        deleted_count = int((observed & removed).sum())
        baseline[region] = np.median([np.median(full_labels["target"][owners[known & (views == view)]])
                                      for view in np.unique(views[known])])
        fraction[region] = deleted_count/int(observed.sum())
        mask[region] = True
        rank_mask[region] = full_labels["rank_mask"][candidates].all()
        weights[region] = np.min(full_labels["weight"][candidates])*coverage
        matched_region[region] = winner
        row.update(status="affected" if deleted_count else "unaffected", reason="source_patch_match",
                   full_views=int(observed.sum()), deleted_views=deleted_count,
                   retained_views=remaining_count, fraction=float(fraction[region]))
    targets = pseudo_targets(baseline, fraction, fraction > 0, limits, policy)
    return dict(**targets, mask=mask, rank_mask=rank_mask & mask, weight=weights,
                fraction=fraction, affected=mask & (fraction > 0), matched_region=matched_region,
                audit=dict(affected=int((mask & (fraction > 0)).sum()),
                           unaffected=int((mask & (fraction == 0)).sum()), unknown=int((~mask).sum()),
                           clipped=int((mask & ((targets["rank_target"] < 0) | (targets["rank_target"] > 1))).sum()),
                           rows=rows))
