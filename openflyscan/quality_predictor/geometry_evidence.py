"""Current-chunk geometric evidence and label-availability audits, never GT."""

from collections import Counter
from dataclasses import replace

import numpy as np
import torch
from scipy.spatial import cKDTree

from .surface_reprojection_check import check_surface_observations


VIEW_EVIDENCE_NAMES = (
    "log_relative_xyz_disagreement",
    "log_relative_lateral_disagreement",
    "log_anchor_roundtrip_image_fraction",
    "roundtrip_available",
)
REGION_EVIDENCE_NAMES = (
    "projectable_fraction", "log_projectable_views", "compared_fraction",
    "depth_boundary_fraction", "behind_fraction", "outside_fraction",
    "invalid_fraction", "consistent_fraction_1pct", "consistent_fraction_2pct",
    "consistent_fraction_5pct", "behind_surface_conflict_fraction_5pct",
    "foreground_or_occlusion_fraction_5pct", "log_depth_error_median",
    "log_depth_error_p90", "log_xyz_error_median", "log_xyz_error_p90",
    "log_roundtrip_median", "log_roundtrip_p90", "consistent_direction_spread",
    "consistent_baseline_over_depth", "has_comparable_views", "has_anchor",
)


def _project(point, pose, intrinsic):
    local = pose[:3, :3].T @ (point - pose[:3, 3])
    if not np.isfinite(local).all() or local[2] <= 1e-6:
        return None
    homogeneous = intrinsic @ local
    return homogeneous[:2] / homogeneous[2]


def _quantiles(values):
    if not values:
        return [0.0, 0.0]
    return np.log1p(np.quantile(values, [0.5, 0.9])).tolist()


def current_geometry_evidence(cloud, proposal):
    maps = np.asarray(cloud["maps"])
    poses = np.asarray(cloud["camera_poses"])
    intrinsics = np.asarray(cloud["intrinsics"])
    stride = int(cloud.get("sample_stride", 14))
    views, height, width = maps.shape[:3]
    centers = np.asarray(proposal["centers"])
    flat = maps.reshape(-1, 3)
    finite_ids = np.flatnonzero(np.isfinite(flat).all(axis=1))
    tree = cKDTree(flat[finite_ids]) if len(finite_ids) else None
    per_view = np.zeros((len(centers), views, len(VIEW_EVIDENCE_NAMES)), np.float32)
    region = np.zeros((len(centers), len(REGION_EVIDENCE_NAMES)), np.float32)
    audit = []
    image_diagonal = max(float(np.hypot(width * stride, height * stride)), 1.0)
    for query_index, center in enumerate(centers):
        observation = check_surface_observations(center, maps, poses, intrinsics, stride)
        counts = Counter(row["status"] for row in observation["observations"])
        anchor = None
        anchor_uv = None
        if tree is not None:
            nearest = int(finite_ids[tree.query(center)[1]])
            anchor = nearest // (height * width)
            anchor_uv = _project(center, poses[anchor], intrinsics[anchor])
        depth_errors, xyz_errors, roundtrips = [], [], []
        consistent_views, consistent_depths = [], []
        for row in observation["observations"]:
            if row["status"] not in ("compared", "depth_boundary"):
                continue
            view = row["view"]
            pixel = np.asarray(row["uv_fraction"]) * [width, height] - 0.5
            column, line = np.floor(pixel).astype(int)
            horizontal, vertical = pixel - [column, line]
            weights = np.array([(1-horizontal)*(1-vertical), horizontal*(1-vertical),
                                (1-horizontal)*vertical, horizontal*vertical])
            matched = weights @ maps[view, line:line+2, column:column+2].reshape(4, 3)
            delta_camera = poses[view, :3, :3].T @ (matched - center)
            depth = row["expected_depth"]
            xyz_error = float(np.linalg.norm(delta_camera) / depth)
            lateral = float(np.linalg.norm(delta_camera[:2]) / depth)
            return_uv = _project(matched, poses[anchor], intrinsics[anchor]) if anchor is not None else None
            roundtrip = float(np.linalg.norm(return_uv-anchor_uv) / image_diagonal) if return_uv is not None and anchor_uv is not None else None
            per_view[query_index, view] = [*np.log1p([xyz_error, lateral, roundtrip or 0.0]), roundtrip is not None]
            if row["status"] != "compared":
                continue
            depth_errors.append(abs(row["relative_depth_error"]))
            xyz_errors.append(xyz_error)
            if roundtrip is not None:
                roundtrips.append(roundtrip)
            if abs(row["relative_depth_error"]) <= 0.02:
                consistent_views.append(view)
                consistent_depths.append(depth)
        direction_spread = baseline = 0.0
        if len(consistent_views) >= 2:
            cameras = poses[consistent_views, :3, 3]
            directions = cameras - center
            directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-9)
            direction_spread = float(np.clip(1-np.linalg.norm(directions.mean(0)), 0, 1))
            distances = np.linalg.norm(cameras[:, None]-cameras[None, :], axis=-1)
            upper = distances[np.triu_indices(len(cameras), 1)]
            baseline = float(np.quantile(upper, 0.9) / max(float(np.median(consistent_depths)), 1e-6))
        denominator = max(counts["compared"], 1)
        projectable = counts["compared"] + counts["depth_boundary"]
        positive_conflicts = sum(row.get("relative_depth_error", 0) > 0.05 for row in observation["observations"] if row["status"] == "compared")
        region[query_index] = [
            projectable / views, np.log1p(projectable), counts["compared"] / views,
            counts["depth_boundary"] / views, counts["behind"] / views,
            counts["outside"] / views, counts["invalid"] / views,
            *[observation["depth_support"][str(threshold)] / denominator for threshold in (0.01, 0.02, 0.05)],
            positive_conflicts / denominator, observation["occluded_at_5pct"] / denominator,
            *_quantiles(depth_errors), *_quantiles(xyz_errors), *_quantiles(roundtrips),
            direction_spread, np.log1p(baseline), bool(counts["compared"]), anchor_uv is not None,
        ]
        audit.append(dict(projectable=projectable, compared=counts["compared"],
                          consistent_2pct=len(consistent_views), status_counts=dict(counts)))
    if not np.isfinite(per_view).all() or not np.isfinite(region).all():
        raise ValueError("non-finite current geometric evidence")
    return per_view, region, audit


def augment_current_inputs(inputs, cloud, proposal):
    per_view, region, audit = current_geometry_evidence(cloud, proposal)
    device = inputs.query_features.device
    evidence = torch.as_tensor(per_view, device=device)
    evidence = evidence.masked_fill(~inputs.view_mask[..., None], 0)
    result = replace(inputs,
                     quality_geometry_features=torch.cat([inputs.quality_geometry_features, evidence], -1),
                     query_features=torch.cat([inputs.query_features, torch.as_tensor(region, device=device)], -1))
    return result, audit


def self_projection_audit(cloud):
    maps, poses, intrinsics = cloud["maps"], cloud["camera_poses"], cloud["intrinsics"]
    stride = int(cloud.get("sample_stride", 14))
    height, width = maps.shape[1:3]
    rows, columns = np.indices((height, width))
    expected = np.stack([columns*stride+stride//2, rows*stride+stride//2], -1)
    errors = []
    total = len(maps)*height*width
    for world, pose, intrinsic in zip(maps, poses, intrinsics):
        local = (world-pose[:3, 3]) @ pose[:3, :3]
        valid = np.isfinite(local).all(-1) & (local[..., 2] > 1e-6)
        homogeneous = local @ intrinsic.T
        projected = homogeneous[valid, :2] / homogeneous[valid, 2, None]
        errors.extend(np.linalg.norm(projected-expected[valid], axis=-1).tolist())
    return dict(valid_fraction=len(errors)/max(total, 1),
                median_pixels=float(np.median(errors)) if errors else None,
                p90_pixels=float(np.quantile(errors, 0.9)) if errors else None,
                boundary="Self-consistency diagnostic, not independent camera or geometry ground truth.")


def label_support_audit(cloud, proposal, pixel_values, view_mask):
    points = cloud["maps"].reshape(-1, 3)
    values = pixel_values.reshape(-1)
    finite = np.isfinite(points).all(1) & np.isfinite(values)
    indices = np.flatnonzero(finite)
    tree = cKDTree(points[finite]) if len(indices) else None
    pixels_per_view = int(np.prod(cloud["maps"].shape[1:3]))
    rows = []
    for query_index, center in enumerate(proposal["centers"]):
        found = indices[tree.query_ball_point(center, proposal["cell_size"])] if tree is not None else np.array([], dtype=int)
        views = found // pixels_per_view
        unique_views = np.unique(views)
        valid = len(found) >= 6 and len(unique_views) >= 2
        reasons = []
        if len(found) < 6:
            reasons.append("fewer_than_six_teacher_patches")
        if len(unique_views) < 2:
            reasons.append("fewer_than_two_teacher_views")
        per_view = [float(np.median(values[found[views == view]])) for view in unique_views]
        rows.append(dict(query=query_index, valid=valid, label_patches=len(found),
                         label_views=len(unique_views), projectable_views=int(view_mask[query_index].sum()),
                         reasons=reasons, median=float(np.median(per_view)) if per_view else None,
                         worst_view=float(max(per_view)) if per_view else None))
    return dict(regions=len(rows), valid=sum(row["valid"] for row in rows),
                unavailable_despite_two_projectable_views=sum(not row["valid"] and row["projectable_views"] >= 2 for row in rows),
                reasons=dict(Counter(reason for row in rows for reason in row["reasons"])), rows=rows,
                boundary="Unavailable labels remain unknown; no confidence/prediction-derived GT or automatic label recovery.")
