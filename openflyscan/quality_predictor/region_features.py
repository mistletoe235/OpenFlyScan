"""Pack current-point regions without a deleted/reference query camera.

Geometry stays relative to the current chunk. No deletion IDs, GS targets,
Full-state query or reference-photo pose is accepted by this adapter.
"""

import numpy as np
import torch

from .model import LocalQueryInputs
from .surface_reprojection_check import check_surface_observations


def pack_current_region_inputs(cloud, proposal, point_features, confidence_features, dino_features, rgb_stats):
    maps, poses, intrinsics = cloud['maps'], cloud['camera_poses'], cloud['intrinsics']
    queries, origin, radius = proposal['centers'], proposal['origin'], proposal['cell_size']
    points = proposal['valid_points']
    scale = max(float(np.median(np.linalg.norm(points-origin, axis=1))), 1e-6)
    views, height, width = maps.shape[:3]
    count = len(queries)
    device = point_features.device
    assert point_features.shape == confidence_features.shape == dino_features.shape == (views, height, width, 1024)
    assert rgb_stats.shape == (views, height, width, 6)
    pose_values = np.zeros((count, views, 10), np.float32)
    quality_values = np.zeros((count, views, 4), np.float32)
    query_values = np.zeros((count, 10), np.float32)
    masks = np.zeros((count, views), bool)
    observations_to_pool = []
    interpolation_weights = []
    for query_index, center in enumerate(queries):
        local = points[np.linalg.norm(points-center, axis=1) <= radius]-center
        if len(local) >= 3:
            values, axes = np.linalg.eigh(np.cov(local.T))
            values = np.maximum(values, 0)
            moments = values/max(float(values.sum()), 1e-12)
            normal = axes[:, 0] if (values[1]-values[0])/max(float(values.sum()), 1e-12) > 1e-6 else np.zeros(3)
            if normal[np.argmax(abs(normal))] < 0:
                normal = -normal
        else:
            moments = np.zeros(3)
            normal = np.zeros(3)
        query_values[query_index] = np.r_[(center-origin)/scale, moments, normal, radius/scale]
        observations = check_surface_observations(center, maps, poses, intrinsics)['observations']
        for observation in observations:
            if observation['status'] not in ('compared', 'depth_boundary'):
                continue
            view = observation['view']
            horizontal, vertical = observation['uv_fraction']
            pixel_x, pixel_y = horizontal*width-.5, vertical*height-.5
            column, row = int(np.floor(pixel_x)), int(np.floor(pixel_y))
            offset_x, offset_y = pixel_x-column, pixel_y-row
            if not (0 <= column < width-1 and 0 <= row < height-1):
                continue
            observations_to_pool.append((query_index, view, row, column))
            interpolation_weights.append([(1-offset_x)*(1-offset_y), offset_x*(1-offset_y),
                                          (1-offset_x)*offset_y, offset_x*offset_y])
            relative = poses[view, :3, 3]-center
            distance = max(float(np.linalg.norm(relative)), 1e-8)
            direction = -relative/distance
            optical = poses[view, :3, 2]
            pose_values[query_index, view] = np.r_[relative/scale, direction, distance/scale,
                                                   np.dot(optical, direction), horizontal, vertical]
            error = observation['relative_depth_error']
            spread = observation['neighborhood_relative_depth_range']
            confidence = float(np.median(cloud['conf'][view, row:row+2, column:column+2]))
            quality_values[query_index, view] = [np.sign(error)*np.log1p(abs(error)), np.log1p(abs(error)),
                                                np.log1p(max(spread, 0)), 1/(1+np.exp(-np.clip(confidence, -60, 60)))]
            masks[query_index, view] = True
    outputs = [torch.zeros((count, views, channels), device=device) for channels in (1024, 1024, 1024, 6)]
    if observations_to_pool:
        indices = torch.as_tensor(observations_to_pool, dtype=torch.long, device=device)
        weights = torch.as_tensor(interpolation_weights, dtype=torch.float32, device=device)
        rows = indices[:, 2, None]+torch.tensor([0, 0, 1, 1], device=device)
        columns = indices[:, 3, None]+torch.tensor([0, 1, 0, 1], device=device)
        for output, features in zip(outputs, (point_features, confidence_features, dino_features, rgb_stats)):
            samples = features[indices[:, 1, None], rows, columns].float()
            output[indices[:, 0], indices[:, 1]] = (samples*weights[..., None]).sum(1)
    point_output, confidence_output, dino_output, rgb_output = outputs
    return LocalQueryInputs(point_output, confidence_output, dino_output,
                            torch.as_tensor(pose_values, device=device), torch.as_tensor(quality_values, device=device),
                            rgb_output, torch.as_tensor(query_values, device=device), torch.as_tensor(masks, device=device))
