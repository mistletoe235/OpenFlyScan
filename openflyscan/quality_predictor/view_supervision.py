"""Current-source observation recovery and feature-matched Full-only teachers."""

import numpy as np
import torch
from scipy.spatial import cKDTree

from .region_features import pack_current_region_inputs


def pack_source_supported_inputs(cloud, proposal, point, confidence, dino, rgb):
    inputs = pack_current_region_inputs(cloud, proposal, point, confidence, dino, rgb)
    maps = np.asarray(cloud['maps'])
    views, height, width = maps.shape[:3]
    centers = np.asarray(proposal['centers'])
    original = inputs.view_mask.detach().cpu().numpy().copy()
    support = []
    for region, view in np.argwhere(original):
        pose = cloud['camera_poses'][view]
        camera = (centers[region]-pose[:3, 3]) @ pose[:3, :3]
        pixel = cloud['intrinsics'][view] @ camera
        pixel = pixel[:2]/pixel[2]/14-.5
        column, row = np.floor(pixel).astype(int)
        horizontal, vertical = pixel-[column, row]
        patches = view*height*width + (row+np.array([0, 0, 1, 1]))*width + column+np.array([0, 1, 0, 1])
        weights = np.array([(1-horizontal)*(1-vertical), horizontal*(1-vertical),
                            (1-horizontal)*vertical, horizontal*vertical])
        support.append((int(region), int(view), patches, weights))
    flat_points = maps.reshape(-1, 3)
    finite = np.isfinite(flat_points).all(1) & np.isfinite(np.asarray(cloud['conf']).reshape(-1))
    reliable = np.zeros(len(flat_points), bool)
    reprojection_error = np.full(len(flat_points), np.inf)
    depths = np.zeros(len(flat_points))
    for view in range(views):
        indices = np.flatnonzero(finite[view*height*width:(view+1)*height*width])+view*height*width
        pose = cloud['camera_poses'][view]
        camera = (flat_points[indices]-pose[:3, 3]) @ pose[:3, :3]
        pixels = camera @ cloud['intrinsics'][view].T
        projected = pixels[:, :2]/np.maximum(pixels[:, 2:], 1e-8)
        local = indices % (height*width)
        source = np.stack([(local % width+.5)*14, (local//width+.5)*14], -1)
        error = np.linalg.norm(projected-source, axis=1)
        reliable[indices] = (camera[:, 2] > 0) & (error <= 14.)
        depths[indices] = camera[:, 2]
        reprojection_error[indices] = error
    indices = np.flatnonzero(reliable)
    tree = cKDTree(flat_points[indices])
    neighborhoods = tree.query_ball_point(centers, proposal['cell_size'])
    scale = max(float(np.median(np.linalg.norm(proposal['valid_points']-proposal['origin'], axis=1))), 1e-6)
    recovered, errors, unsupported = [], [], 0
    feature_sources = [point, confidence, dino, rgb]
    feature_outputs = [inputs.point_features, inputs.confidence_features, inputs.dino_features, inputs.rgb_stats]
    for region, neighborhood in enumerate(neighborhoods):
        found = indices[neighborhood]
        view_ids = found//(height*width)
        for view in range(views):
            if original[region, view]:
                continue
            candidates = found[view_ids == view]
            if not len(candidates):
                unsupported += 1
                continue
            distances = np.linalg.norm(flat_points[candidates]-centers[region], axis=1)
            selected = candidates[np.argsort(distances, kind='stable')[:4]]
            pose = cloud['camera_poses'][view]
            expected = float(((centers[region]-pose[:3, 3]) @ pose[:3, :3])[2])
            if expected <= 0:
                unsupported += 1
                continue
            local = selected % (height*width)
            rows, columns = local//width, local % width
            samples = [source[view, rows, columns].float() for source in feature_sources]
            if not all(bool(torch.isfinite(sample).all()) for sample in samples):
                unsupported += 1
                continue
            for output, sample in zip(feature_outputs, samples):
                output[region, view] = sample.mean(0)
            representative = flat_points[selected].mean(0)
            relative = pose[:3, 3]-representative
            distance = max(float(np.linalg.norm(relative)), 1e-8)
            direction = -relative/distance
            uv = [(columns.mean()+.5)/width, (rows.mean()+.5)/height]
            pose_values = np.r_[relative/scale, direction, distance/scale, np.dot(pose[:3, 2], direction), uv]
            error = (float(np.mean(depths[selected]))-expected)/expected
            spread = float(np.ptp(depths[selected]))/expected
            scalar = float(np.median(np.asarray(cloud['conf']).reshape(-1)[selected]))
            quality = [np.sign(error)*np.log1p(abs(error)), np.log1p(abs(error)),
                       np.log1p(spread), 1/(1+np.exp(-np.clip(scalar, -60, 60)))]
            inputs.pose_geometry_features[region, view] = torch.as_tensor(pose_values, device=point.device)
            inputs.quality_geometry_features[region, view] = torch.as_tensor(quality, device=point.device)
            inputs.view_mask[region, view] = True
            support.append((region, view, selected, np.full(len(selected), 1/len(selected))))
            recovered.append((region, view))
            errors.extend(reprojection_error[selected].tolist())
    cloud['region_feature_support'] = support
    cloud['source_recovery_audit'] = dict(original_views=int(original.sum()), recovered_views=len(recovered),
        unsupported_views=unsupported, recovered_slots=recovered,
        maximum_reprojection_error_pixels=max(errors, default=0.), maximum_allowed_error_pixels=14.,
        current_views_only=True, teacher_used_for_selection=False)
    return inputs


def feature_matched_view_labels(cloud, values, shape):
    values = np.asarray(values).reshape(-1)
    targets = np.zeros(shape, np.float32)
    mask = np.zeros(shape, bool)
    for region, view, patches, weights in cloud['region_feature_support']:
        selected = values[patches]
        valid = np.isfinite(selected) & (weights > 1e-6)
        mass = weights[valid].sum()
        if mass < .5:
            continue
        targets[region, view] = np.sum(selected[valid]*weights[valid])/mass
        mask[region, view] = True
    recovered_labels = sum(bool(mask[region, view]) for region, view in cloud['source_recovery_audit']['recovered_slots'])
    return dict(view_raw=targets, view_mask=mask), dict(feature_matched_labels=int(mask.sum()),
        recovered_view_labels=recovered_labels, clipping=False, source_patch_identity_matched=True)
