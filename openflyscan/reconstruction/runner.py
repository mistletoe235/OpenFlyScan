"""Pi3X inference and Quality Predictor using only current observations."""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from openflyscan.inference import load_predictor
from openflyscan.quality_predictor.footprint_window_sampler import sample_footprint_window
from openflyscan.quality_predictor.geometry_evidence import augment_current_inputs
from openflyscan.quality_predictor.native_pi3x_geometry import native_cloud
from openflyscan.quality_predictor.observation_source import Source
from openflyscan.quality_predictor.pi3x_features import FrozenPi3XQueryFeatureBuilder
from openflyscan.quality_predictor.region_selection import allocate_retake_budget


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def progress(output, stage, done, total):
    record = dict(stage=stage, completed=done, total=total)
    write_json(output / 'progress.json', record)
    print(json.dumps(record), flush=True)


def initialize(config, policy):
    if sys.version_info < (3, 11):
        import typing
        import typing_extensions
        if not hasattr(typing, 'Self'):
            typing.Self = typing_extensions.Self
    geoff3d = Path(config['geoff3d_root']).resolve()
    sys.path.insert(0, str(geoff3d))
    os.chdir(geoff3d)
    from geoff3d.slrf.model_runner import init_model_from_hydra, apply_runtime_prior_policy, build_prior_overrides
    model, _ = init_model_from_hydra('pi3x', 'aws', ['model.model_config.load_pretrained_weights=false'] +
                                    build_prior_overrides('pi3x', policy), torch.device(config.get('device', 'cuda:0')))
    checkpoint = torch.load(config['pi3x_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
    del checkpoint
    apply_runtime_prior_policy(model, policy)
    return model.eval().requires_grad_(False)


def cache_features(model, source, cache, checkpoint_hash, output):
    from geoff3d.slrf.scene_io import build_views_from_scene, load_chunk_views_from_scene
    device = next(model.parameters()).device
    lightweight, metadata = build_views_from_scene(source, max_image_size=518, patch_size=14, device=device, show_progress=False)
    entries = {}
    for index, stem in enumerate(map(str, metadata['stems'])):
        path = cache / checkpoint_hash / f'{digest(metadata["image_paths"][stem])}_{metadata["target_h"]}x{metadata["target_w"]}.npz'
        if not path.is_file():
            views, _ = load_chunk_views_from_scene(lightweight, metadata, [index],
                                                  {'pose': 'none', 'ray': 'none', 'depth': 'none'}, device,
                                                  num_workers=0, norm_type='identity')
            images = torch.cat([view['img'] for view in views])
            normalized = (images - model.model.image_mean) / model.model.image_std
            with torch.inference_mode():
                tokens = model.model.encoder(normalized, is_training=True)['x_norm_patchtokens']
            patch_height, patch_width = images.shape[-2] // 14, images.shape[-1] // 14
            patches = images.reshape(1, 3, patch_height, 14, patch_width, 14).permute(0, 2, 4, 1, 3, 5)
            statistics = torch.cat([patches.mean((-1, -2)), patches.std((-1, -2))], dim=-1)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, feature=tokens.reshape(patch_height, patch_width, 1024).float().cpu().numpy(),
                                    rgb_stats=statistics[0].float().cpu().numpy().astype(np.float16))
            temporary.replace(path)
        entries[stem] = dict(stem=stem, cache=str(path))
        if (index + 1) % 8 == 0 or index + 1 == len(metadata['stems']):
            progress(output, 'image_features', index + 1, len(metadata['stems']))
    return entries


def spatial_groups(centers):
    leaves = []
    def split(indices):
        if len(indices) <= 30:
            leaves.append(indices)
            return
        axis = int(np.argmax(np.ptp(centers[indices, :2], axis=0)))
        indices = indices[np.argsort(centers[indices, axis])]
        split(indices[:len(indices) // 2])
        split(indices[len(indices) // 2:])
    split(np.arange(len(centers)))
    tree = cKDTree(centers[:, :2])
    groups = []
    for indices in leaves:
        nearby = np.atleast_1d(tree.query(np.mean(centers[indices, :2], axis=0), k=min(90, len(centers)))[1])
        extra = [int(index) for index in nearby if index not in set(indices)]
        groups.append((list(map(int, indices)), list(map(int, indices)) + extra[:30 - len(indices)]))
    return groups


class CurrentObservationSource:
    def __init__(self, model, builder, training_config, max_regions):
        self.model = model
        self.feature_builder = builder
        self.training_config = training_config
        self.max_regions = max_regions

    def builder(self, scene):
        return self.feature_builder

    def extract(self, scene, stems, seed):
        inputs, cloud, proposal = Source._extract(self, scene, stems, seed)
        inputs, audit = augment_current_inputs(inputs, cloud, proposal)
        cloud['geometry_evidence_audit'] = audit
        return inputs, cloud, proposal


def save_dense(model, builder, cloud, predictions, path, stride):
    from geoff3d.slrf.geometry_align import estimate_similarity_umeyama
    sensor = np.stack([builder._camera(stem)[0] for stem in cloud['stems']])
    centers = np.stack([prediction['cam_trans'][0].float().cpu().numpy() for prediction in predictions])
    scale, rotation, translation, valid, note = estimate_similarity_umeyama(centers, sensor[:, :3, 3])
    if not valid:
        raise ValueError(note)
    grid_y, grid_x = np.meshgrid(np.arange(stride // 2, builder.target_h, stride),
                                  np.arange(stride // 2, builder.target_w, stride), indexing='ij')
    dense = np.stack([prediction['pts3d'][0, grid_y, grid_x].float().cpu().numpy() for prediction in predictions])
    dense = (scale * dense.reshape(-1, 3) @ rotation.T + translation).reshape(dense.shape).astype(np.float32)
    confidence = np.stack([prediction['conf'][0, grid_y, grid_x, 0].float().cpu().numpy() for prediction in predictions])
    arrays = {key: cloud[key] for key in ('stems', 'maps', 'conf', 'rgb', 'camera_poses', 'intrinsics', 'sample_stride')}
    np.savez_compressed(path, **arrays, dense_maps=dense, dense_conf=confidence, dense_y=grid_y[:, 0],
                        dense_x=grid_x[0], sensor_poses=sensor, target_hw=[builder.target_h, builder.target_w],
                        alignment_scale=scale, alignment_R=rotation, alignment_t=translation)


def run(manifest_path, config, output, cache, budget, preview=False):
    started = time.time()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('GS_or_SfM_inputs') is not False:
        raise ValueError('sensor-only manifest required')
    stems = [record['stem'] for record in manifest['records']]
    centers = np.asarray([record['enu'] for record in manifest['records']])
    if len(stems) < 30 or len(stems) != len(set(stems)):
        raise ValueError('at least 30 distinct images are required')
    policy = dict(model='pi3x', pose=manifest['pose_prior'], ray='input', depth='none',
                  translation='none', rotation='none', bootstrap_ray=False, bootstrap_depth=False)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = digest(config['pi3x_checkpoint'])
    model = initialize(config, policy)
    entries = cache_features(model, Path(manifest['source']), cache, checkpoint_hash, output)
    builder = FrozenPi3XQueryFeatureBuilder(model, Path(manifest['source']), entries, np.zeros(3))
    seed = config.get('seed', 20260907)
    footprints = np.zeros((len(stems), 2))
    lower, upper = np.zeros_like(footprints), np.zeros_like(footprints)
    cloud_xyz, cloud_rgb, fits = [], [], []
    groups = spatial_groups(centers)
    for chunk, (core, indices) in enumerate(groups):
        names = [stems[index] for index in indices]
        builder.recenter = np.mean(centers[indices], axis=0)
        cloud = native_cloud(model, builder, names, seed)
        residual = np.linalg.norm(cloud['camera_poses'][:, :3, 3] - centers[indices], axis=1)
        fits.append(dict(chunk=chunk, rmse=float(np.sqrt(np.mean(residual ** 2)))))
        for index in core:
            view = indices.index(index)
            points, confidence = cloud['maps'][view], cloud['conf'][view]
            valid = np.isfinite(points).all(-1) & np.isfinite(confidence)
            if not valid.any():
                raise ValueError('no finite geometry for ' + stems[index])
            supported = valid & (confidence >= np.quantile(confidence[valid], .25))
            footprints[index] = np.median(points[supported, :2], axis=0)
            lower[index], upper[index] = np.quantile(points[supported, :2], [.02, .98], axis=0)
            cloud_xyz.append(points[valid])
            cloud_rgb.append(cloud['rgb'][view][valid])
            if manifest['records'][index].get('camera_ypr_deg') is None:
                forward = cloud['camera_poses'][view, :3, 2]
                manifest['records'][index]['camera_ypr_deg'] = [float(np.degrees(np.arctan2(forward[0], forward[1]))),
                    float(np.degrees(np.arctan2(forward[2], np.linalg.norm(forward[:2])))), 0.]
                manifest['records'][index]['attitude_source'] = 'Pi3X_prediction_for_planning'
        temporary = output / 'bootstrap_cloud.tmp'
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, xyz=np.concatenate(cloud_xyz).astype(np.float32),
                                rgb=np.concatenate(cloud_rgb).astype(np.float32))
        temporary.replace(output / 'bootstrap_cloud.npz')
        progress(output, 'pi3x_geometry', chunk + 1, len(groups))
    write_json(output / 'planning_manifest.json', manifest)
    if preview:
        write_json(output / 'complete.json', dict(preview=True, images=len(stems), seconds=time.time() - started))
        return
    predictor = load_predictor(config['quality_checkpoint'], config.get('device', 'cuda:0'))
    source = CurrentObservationSource(model, builder, predictor.training_config, config.get('max_regions_per_chunk', 64))
    windows, pending = [], set(range(len(stems)))
    while pending:
        anchor = min(pending)
        window = sample_footprint_window(footprints[anchor], footprints, lower, upper)
        indices = list(window.stems_indices)
        if anchor not in indices:
            raise ValueError('spatial window did not cover its anchor')
        windows.append(indices)
        pending.difference_update(indices)
    geometry = output / 'surface' / 'geometry'
    geometry.mkdir(parents=True, exist_ok=True)
    rows, geometry_records = [], []
    image_space_id = digest(manifest_path)
    for chunk, indices in enumerate(windows):
        names = [stems[index] for index in indices]
        captured = {}
        hook = model.register_forward_hook(lambda module, inputs, result: captured.__setitem__('predictions', result))
        try:
            inputs, cloud, proposal = source.extract(manifest['scene'], names, seed)
        finally:
            hook.remove()
        with torch.inference_mode():
            scores = predictor(inputs)['total'].float().cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite quality predictions')
        path = geometry / f'chunk_{chunk:04d}.npz'
        save_dense(model, builder, cloud, captured.pop('predictions'), path, config.get('dense_stride', 4))
        geometry_records.append(dict(chunk=chunk, path=str(path), sha256=digest(path)))
        flat = cloud['maps'].reshape(-1, 3)
        valid = np.isfinite(flat).all(1)
        flat_indices = np.flatnonzero(valid)
        tree = cKDTree(flat[valid])
        pixel_count = builder.patch_h * builder.patch_w
        for region, (center, score) in enumerate(zip(proposal['centers'], scores)):
            members = flat_indices[tree.query_ball_point(center, proposal['cell_size'])]
            view_indices, pixels = members // pixel_count, members % pixel_count
            evidence = []
            for view in np.unique(view_indices):
                current = pixels[view_indices == view]
                evidence.append(dict(stem=names[int(view)], patch_pixels_yx=np.stack(
                    [current // builder.patch_w, current % builder.patch_w], axis=1).tolist(),
                    points=len(current), patch_hw=[builder.patch_h, builder.patch_w]))
            rows.append(dict(chunk=chunk, region=region, xyz=center.tolist(), score=float(score),
                             radius=float(proposal['cell_size']), view_support=int(inputs.view_mask[region].sum()),
                             within_chunk_rank=float(np.mean(scores <= score)), scene=manifest['scene'],
                             image_space_id=image_space_id, alignment_verified=False,
                             image_evidence=sorted(evidence, key=lambda entry: entry['points'], reverse=True)))
        if chunk == 0:
            np.savez_compressed(output / 'sample_inputs.npz', **{
                field.name: getattr(inputs, field.name).detach().cpu().numpy() for field in dataclasses.fields(inputs)})
            np.save(output / 'sample_scores.npy', scores)
        del inputs, cloud
        progress(output, 'quality_prediction', chunk + 1, len(windows))
    allocation = allocate_retake_budget(rows, budget)
    selection = dict(status='ranking_frozen', input_images=len(stems), policy=policy, queries=rows,
                     selected=allocation['selected'], merged_regions=allocation['merged_regions'],
                     windows=[[stems[index] for index in indices] for indices in windows],
                     bootstrap_alignment=fits, pi3x_checkpoint_sha256=checkpoint_hash,
                     quality_checkpoint_sha256=digest(config['quality_checkpoint']),
                     source_patch_recovery=predictor.training_config.source_patch_recovery,
                     geometry_protocol='Pi3X local groups aligned to uploaded GPS by Sim(3)',
                     selection_schema='retake_budget_v2', GS_or_SfM_inputs=False)
    write_json(output / 'selection_frozen.json', selection)
    write_json(geometry / 'manifest.json', dict(status='complete', chunks=geometry_records,
               input_selection_sha256=digest(output / 'selection_frozen.json'), selection_unchanged=True))
    write_json(output / 'complete.json', dict(preview=False, images=len(stems), seconds=time.time() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=12)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    run(args.manifest.resolve(), json.loads(args.config.read_text()), args.output.resolve(),
        args.cache.resolve(), args.budget, args.preview)


if __name__ == '__main__':
    main()
