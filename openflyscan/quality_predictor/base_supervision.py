"""Offline Full GS targets with source pixels and independent SfM track support."""

from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

from .teacher_grid import lookup_pixels, mapping_models


@dataclass(frozen=True)
class FullLabelPolicy:
    schema: str = "sfm_supported_full_v2"
    minimum_shared_tracks: int = 3
    maximum_view_log_mse_range: float = 0.3
    recovered_weight: float = 0.5
    source_track_recovery: bool = True
    single_view_supervision: bool = False
    single_view_weight: float = 0.25
    maximum_single_view_projection_patches: float = 1.0
    maximum_single_view_relative_depth_error: float = 0.05

    def __post_init__(self):
        if self.schema != "sfm_supported_full_v2":
            raise ValueError("unsupported Full label policy")
        if type(self.minimum_shared_tracks) is not int or self.minimum_shared_tracks < 3:
            raise ValueError("at least three independent SfM tracks are required")
        if not np.isfinite(self.maximum_view_log_mse_range) or not 0 < self.maximum_view_log_mse_range <= 0.3:
            raise ValueError("view disagreement limit must be in (0, 0.3]")
        if not np.isfinite(self.recovered_weight) or not 0 < self.recovered_weight <= 0.5:
            raise ValueError("recovered supervision must have weight in (0, 0.5]")
        if type(self.source_track_recovery) is not bool:
            raise ValueError("source_track_recovery must be boolean")
        if type(self.single_view_supervision) is not bool:
            raise ValueError("single_view_supervision must be boolean")
        for name, maximum in (("single_view_weight", 0.25),
                              ("maximum_single_view_projection_patches", 1.0),
                              ("maximum_single_view_relative_depth_error", 0.05)):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"{name} must be in (0, {maximum}]")
        if self.single_view_supervision and self.single_view_weight > self.recovered_weight:
            raise ValueError("single-view weight cannot exceed multiview recovery weight")


class FullTeacherLookup:
    def __init__(self, root, record):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
        from map_rgb_to_rectified_uv import load_pair

        sys.path.insert(0, str(Path(root)/"open-lixel-h3dgs-color11-20260808/preprocess"))
        import read_write_model as model_io

        self.mapping = record["teacher_mapping"]
        if self.mapping not in ("rectified_resize", "raw_to_rectified"):
            raise ValueError("Full teacher needs an explicitly verified pixel mapping")
        source_model, teacher_model = mapping_models(record)
        if self.mapping == "raw_to_rectified":
            raw_cameras, rectified_cameras, raw_images, rectified_images = load_pair(
                source_model, teacher_model, model_io)
        else:
            rectified_cameras = model_io.read_cameras_binary(str(teacher_model/"cameras.bin"))
            rectified_images = {image.name: image for image in
                                model_io.read_images_binary(str(teacher_model/"images.bin")).values()}
            raw_cameras, raw_images = rectified_cameras, rectified_images
        self.source_cameras = rectified_cameras if self.mapping == "rectified_resize" else raw_cameras
        self.source_images = {Path(name).stem: image for name, image in
                              (rectified_images if self.mapping == "rectified_resize" else raw_images).items()}
        self.target_cameras = rectified_cameras
        self.target_images = {Path(name).stem: image for name, image in rectified_images.items()}
        self.teacher = {**record["teacher_data"], "index": record["teacher_index"]}
        self.track_cache = {}

    def patches(self, cloud, *, with_tracks=True):
        from map_rgb_to_rectified_uv import map_uv

        views, height, width = cloud["maps"].shape[:3]
        stride = int(cloud.get("sample_stride", 14))
        vertical, horizontal = np.indices((height, width))
        pixels = np.stack([horizontal*stride+stride//2, vertical*stride+stride//2], -1)
        input_wh = np.array([width*stride, height*stride])
        values = np.full((views, height, width), np.nan, np.float32)
        cells = np.full((views, height, width, 2), -1, np.int64)
        tracks = {}
        for view, stem in enumerate(cloud["stems"]):
            if stem not in self.source_images or stem not in self.target_images:
                continue
            source_image = self.source_images[stem]
            source_camera = self.source_cameras[source_image.camera_id]
            target_camera = self.target_cameras[self.target_images[stem].camera_id]
            source_wh = np.array([source_camera.width, source_camera.height])

            def mapper(coordinates):
                raw = (coordinates+.5)*source_wh/input_wh-.5
                return map_uv(raw, source_camera, target_camera)

            values[view], _, cells[view] = lookup_pixels(
                self.teacher, stem, pixels, input_wh, (target_camera.width, target_camera.height),
                mapping=self.mapping, mapper=mapper, return_cells=True)
            if not with_tracks:
                continue
            key = (stem, height, width)
            if key not in self.track_cache:
                xy = np.asarray(source_image.xys)
                point_ids = np.asarray(source_image.point3D_ids)
                valid = (point_ids >= 0) & np.isfinite(xy).all(1)
                patch_xy = np.floor(xy[valid]/source_wh*[width, height]).astype(np.int64)
                indexed = {}
                for (column, row), track in zip(patch_xy, point_ids[valid]):
                    if 0 <= column < width and 0 <= row < height:
                        indexed.setdefault(int(row*width+column), set()).add(int(track))
                self.track_cache[key] = indexed
            for patch, identities in self.track_cache[key].items():
                tracks[view*height*width+patch] = identities
        return values, cells, tracks


def _recovery_vote(found, values, cells, patch_tracks, stems, per_view, policy):
    views = found//per_view
    observed = np.unique(views)
    result = dict(supervised_patches=found.tolist(), supervised_views=len(observed))
    if len(observed) < 2:
        return None, {**result, "reason": "fewer_than_two_labelled_views"}
    votes, track_sets, unique_cells = [], {}, []
    for view in observed:
        members = found[views == view]
        cell_votes, identities = {}, set()
        for patch in members:
            if (cells[patch] >= 0).all():
                cell_votes.setdefault(tuple(cells[patch]), []).append(values[patch])
            identities.update(patch_tracks.get(int(patch), ()))
        track_sets[int(view)] = identities
        votes.append(float(np.median([np.median(vote) for vote in cell_votes.values()])) if cell_votes else np.nan)
        unique_cells.extend([dict(stem=stems[int(view)], cell=list(map(int, cell))) for cell in cell_votes])
    links, connected = [], {int(observed[0])}
    for ordinal, first in enumerate(observed):
        for second in observed[ordinal+1:]:
            shared = sorted(track_sets[int(first)] & track_sets[int(second)])
            if len(shared) >= policy.minimum_shared_tracks:
                links.append(dict(views=[int(first), int(second)], tracks=shared))
    for _ in observed:
        for link in links:
            if connected.intersection(link["views"]):
                connected.update(link["views"])
    spread = float(np.ptp(votes)) if np.isfinite(votes).all() else None
    result.update(teacher_cells=unique_cells, sfm_links=links, view_log_mse_range=spread)
    if len(connected) != len(observed):
        return None, {**result, "reason": "insufficient_independent_sfm_correspondence"}
    if spread is None or spread > policy.maximum_view_log_mse_range:
        return None, {**result, "reason": "teacher_views_disagree_or_unavailable"}
    return float(np.median(votes)), result


def _source_track_matches(seeds, center, points, values, cells, patch_tracks, track_patches, per_view, policy):
    anchors = [int(patch) for patch in seeds if np.isfinite(points[patch]).all()
               and np.isfinite(values[patch]) and (cells[patch] >= 0).all()
               and len(patch_tracks.get(int(patch), ())) >= policy.minimum_shared_tracks]
    if not anchors:
        return np.array([], dtype=np.int64), None
    anchor = min(anchors, key=lambda patch: (np.linalg.norm(points[patch]-center), patch))
    identities = patch_tracks[anchor]
    overlaps = {}
    for identity in identities:
        for patch in track_patches.get(identity, ()):
            if patch//per_view != anchor//per_view:
                overlaps[patch] = overlaps.get(patch, 0)+1
    matches = {anchor//per_view: (len(identities), anchor)}
    for patch, support in sorted(overlaps.items()):
        if support < policy.minimum_shared_tracks or not np.isfinite(values[patch]) or (cells[patch] < 0).any():
            continue
        view = patch//per_view
        if view not in matches or support > matches[view][0]:
            matches[view] = (support, patch)
    return np.array([matches[view][1] for view in sorted(matches)], dtype=np.int64), anchor


def _single_view_vote(found, seeds, center, cloud, values, cells, policy):
    points = cloud["maps"].reshape(-1, 3)
    _, height, width = cloud["maps"].shape[:3]
    per_view = height*width
    observed = np.unique(found//per_view)
    evidence = dict(supervised_patches=found.tolist(), supervised_views=len(observed))
    if len(observed) != 1:
        return None, {**evidence, "reason": "not_single_view"}
    seeds = np.asarray(seeds)
    if seeds.ndim != 1 or (len(seeds) and (not np.issubdtype(seeds.dtype, np.integer)
                                         or (seeds < 0).any() or (seeds >= len(points)).any())):
        raise ValueError("source patch indices must belong to the current cloud")
    candidates = np.intersect1d(found, seeds)
    if not len(candidates):
        return None, {**evidence, "reason": "no_direct_source_patch"}
    if (cells[found] < 0).any():
        return None, {**evidence, "reason": "invalid_teacher_cell"}
    if "camera_poses" not in cloud or "intrinsics" not in cloud:
        return None, {**evidence, "reason": "source_camera_unavailable"}
    anchor = int(min(candidates, key=lambda patch: (np.linalg.norm(points[patch]-center), patch)))
    view, pixel = divmod(anchor, per_view)
    row, column = divmod(pixel, width)
    stride = int(cloud.get("sample_stride", 14))
    pose, intrinsic = np.asarray(cloud["camera_poses"][view]), np.asarray(cloud["intrinsics"][view])
    if stride <= 0 or not np.isfinite(pose).all() or not np.isfinite(intrinsic).all():
        return None, {**evidence, "reason": "invalid_source_camera"}
    local = (np.stack([points[anchor], center])-pose[:3, 3])@pose[:3, :3]
    evidence.update(source_anchor_patch=anchor, source_stem=str(cloud["stems"][view]))
    if not np.isfinite(local).all() or (local[:, 2] <= 1e-6).any():
        return None, {**evidence, "reason": "source_or_center_behind_camera"}
    projected = local@intrinsic.T
    if (np.abs(projected[:, 2]) <= 1e-9).any():
        return None, {**evidence, "reason": "invalid_source_projection"}
    projected = projected[:, :2]/projected[:, 2:]
    expected = np.array([column*stride+stride//2, row*stride+stride//2])
    projection_errors = np.linalg.norm(projected-expected, axis=1)/stride
    depth_error = float(abs(local[0, 2]-local[1, 2])/local[1, 2])
    evidence.update(source_projection_error_patches=float(projection_errors[0]),
                    center_projection_error_patches=float(projection_errors[1]),
                    relative_depth_error=depth_error)
    if not np.isfinite(projected).all() or not ((projected >= -.5).all()
            and (projected < np.array([width, height])*stride-.5).all()):
        return None, {**evidence, "reason": "source_or_center_outside_image"}
    if projection_errors.max() > policy.maximum_single_view_projection_patches:
        return None, {**evidence, "reason": "source_projection_mismatch"}
    if depth_error > policy.maximum_single_view_relative_depth_error:
        return None, {**evidence, "reason": "source_depth_mismatch"}
    cell_votes = {}
    for patch in found:
        cell_votes.setdefault(tuple(cells[patch]), []).append(values[patch])
    votes = [float(np.median(vote)) for vote in cell_votes.values()]
    spread = float(np.ptp(votes))
    evidence.update(teacher_cells=[dict(stem=str(cloud["stems"][view]), cell=list(map(int, cell)))
                                   for cell in cell_votes], teacher_cell_log_mse_range=spread)
    if spread > policy.maximum_view_log_mse_range:
        return None, {**evidence, "reason": "single_view_teacher_cells_disagree"}
    return float(np.median(votes)), {**evidence, "reason": None,
        "reliability_scope": "Known source pixels and Pi3X self-consistency only; not independent geometry truth or multiview quality."}


def full_region_supervision(cloud, proposal, pixel_values, teacher_cells, patch_tracks, policy=None):
    policy = policy or FullLabelPolicy()
    points = cloud["maps"].reshape(-1, 3)
    values = np.asarray(pixel_values).reshape(-1)
    cells = np.asarray(teacher_cells).reshape(-1, 2)
    if len(points) != len(values) or len(cells) != len(values):
        raise ValueError("teacher patches must match the current cloud")
    valid = np.isfinite(points).all(1) & np.isfinite(values)
    indices = np.flatnonzero(valid)
    tree = cKDTree(points[valid])
    per_view = int(np.prod(cloud["maps"].shape[1:3]))
    count = len(proposal["centers"])
    targets = np.zeros(count, np.float32)
    weights = np.zeros(count, np.float32)
    original_mask = np.zeros(count, bool)
    track_patches = {}
    if policy.source_track_recovery:
        for patch, identities in patch_tracks.items():
            if type(patch) is not int or not 0 <= patch < len(values):
                raise ValueError("SfM patch support must belong to the current views")
            for identity in identities:
                track_patches.setdefault(identity, []).append(patch)
    rows = []
    for query, center in enumerate(proposal["centers"]):
        found = indices[tree.query_ball_point(center, proposal["cell_size"])]
        views = found//per_view
        observed = np.unique(views)
        original_mask[query] = len(found) >= 6 and len(observed) >= 2
        row = dict(query=query, original_valid=bool(original_mask[query]), patches=len(found),
                   views=len(observed), status="unknown", weight=0., source_patches=found.tolist())
        if original_mask[query]:
            targets[query] = np.median([np.median(values[found[views == view]]) for view in observed])
            weights[query] = 1
            row.update(status="original", weight=1.)
        else:
            target, evidence = _recovery_vote(found, values, cells, patch_tracks, cloud["stems"], per_view, policy)
            method = "original_neighborhood_sfm_check"
            spread = evidence.get("view_log_mse_range")
            disagreement = spread is not None and spread > policy.maximum_view_log_mse_range
            if target is None and not disagreement and policy.source_track_recovery and "source_patch_indices" in proposal:
                row["neighborhood_rejection"] = evidence
                matched, anchor = _source_track_matches(proposal["source_patch_indices"][query], center,
                    points, values, cells, patch_tracks, track_patches, per_view, policy)
                target, evidence = _recovery_vote(matched, values, cells, patch_tracks, cloud["stems"], per_view, policy)
                evidence["source_anchor_patch"] = anchor
                if evidence.get("reason") == "fewer_than_two_labelled_views":
                    evidence["reason"] = "fewer_than_two_sfm_matched_views"
                method = "source_pixel_sfm_tracks_without_3d_radius"
            row.update(evidence, recovery_method=method)
            if target is not None:
                targets[query] = target
                weights[query] = policy.recovered_weight
                row.update(status="recovered", weight=float(weights[query]))
            elif policy.single_view_supervision and len(observed) == 1 and evidence.get("supervised_views", 0) < 2 and not disagreement:
                seeds = proposal.get("source_patch_indices", [[] for _ in range(count)])[query]
                single_target, single_evidence = _single_view_vote(found, seeds, center, cloud, values, cells, policy)
                row["single_view_evidence"] = single_evidence
                if single_target is not None:
                    targets[query] = single_target
                    weights[query] = policy.single_view_weight
                    row.update(status="single_view", weight=float(weights[query]),
                               multiview_rejection=dict(reason=row.get("reason"), recovery_method=method),
                               reason=None, recovery_method="single_view_source_projection")
        row["target_log10_mse"] = float(targets[query]) if weights[query] else None
        rows.append(row)
    return dict(target=targets, mask=weights > 0, weight=weights, rank_mask=original_mask,
                audit=dict(policy=asdict(policy), regions=count, original_valid=int(original_mask.sum()),
                           recovered=sum(row["status"] == "recovered" for row in rows),
                           single_view=sum(row["status"] == "single_view" for row in rows),
                           unknown=int((weights == 0).sum()), rows=rows,
                           boundary="Full GS error only. SfM tracks check multiview correspondence; optional single-view labels check source projection, not geometry truth. Missing unchanged. New labels are regression-only."))
