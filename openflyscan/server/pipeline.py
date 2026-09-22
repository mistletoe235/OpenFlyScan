"""Session state, cancellable jobs and the deployed reconstruction workflow."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

from openflyscan.missions.export import export_mission
from openflyscan.missions.geometry_export import geo_to_enu
from openflyscan.planning.recapture_budget_core import BudgetPolicy
from openflyscan.reconstruction.cameras import dji_xmp_telemetry, prepare_scene

PIPELINE_VERSION = 'openflyscan-pi3x-quality-directional-v1'
_LOCKS = {}
_GUARD = threading.RLock()
_PROCESSES = {}
_INTERRUPTED = set()


class PipelineCancelled(RuntimeError):
    pass


def state_lock(session_dir):
    with _GUARD:
        return _LOCKS.setdefault(str(Path(session_dir).resolve()), threading.RLock())


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(path)


def load_state(session_dir):
    with state_lock(session_dir):
        return json.loads((Path(session_dir) / 'state.json').read_text())


def mutate_state(session_dir, mutator):
    with state_lock(session_dir):
        state = load_state(session_dir)
        mutator(state)
        state['updated_at_epoch_ms'] = int(time.time() * 1000)
        atomic_json(Path(session_dir) / 'state.json', state)
        return state


def update_state(session_dir, **changes):
    return mutate_state(session_dir, lambda state: state.update(changes))


def terminate(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass


def cancel_session_processes(session_dir):
    with state_lock(session_dir):
        update_state(session_dir, cancelled=True, sealed=True)
        processes = list(_PROCESSES.get(str(Path(session_dir).resolve()), []))
    for process in processes:
        terminate(process)
    return [process.pid for process in processes]


def interrupt_session_processes(session_dir):
    with state_lock(session_dir):
        key = str(Path(session_dir).resolve())
        _INTERRUPTED.add(key)
        processes = list(_PROCESSES.get(key, []))
    for process in processes:
        terminate(process)
    update_state(session_dir, running=False, phase='interrupted', message='Service restarted; retry processing')


def ensure_active(session_dir):
    if load_state(session_dir).get('cancelled') or str(Path(session_dir).resolve()) in _INTERRUPTED:
        raise PipelineCancelled('Session cancelled')


def workstation_config():
    path = Path(os.environ['OPENFLYSCAN_CONFIG']).resolve()
    config = json.loads(path.read_text())
    for key in ('geoff3d_root', 'pi3x_checkpoint', 'quality_checkpoint'):
        if not Path(config[key]).is_absolute() or not Path(config[key]).exists():
            raise ValueError(f'{key} must name an existing absolute path')
    return path, config


def write_cloud(artifacts, cloud_path, maximum):
    with np.load(cloud_path, allow_pickle=False) as cloud:
        xyz, rgb = cloud['xyz'], cloud['rgb']
    valid = np.isfinite(xyz).all(1) & np.isfinite(rgb).all(1)
    xyz, rgb = xyz[valid], rgb[valid]
    if not len(xyz):
        raise ValueError('empty point cloud')
    indices = np.linspace(0, len(xyz) - 1, min(maximum, len(xyz))).astype(int)
    xyz = xyz[indices].astype('<f4')
    colors = np.clip(rgb[indices] * 255, 0, 255).astype('u1')
    packed = np.empty(len(xyz), dtype=[('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))])
    packed['xyz'], packed['rgb'] = xyz, colors
    temporary = artifacts / 'point_cloud.ply.tmp'
    with temporary.open('wb') as stream:
        stream.write(('ply\nformat binary_little_endian 1.0\nelement vertex ' + str(len(xyz)) +
                      '\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\n'
                      'property uchar green\nproperty uchar blue\nend_header\n').encode())
        stream.write(packed.tobytes())
    temporary.replace(artifacts / 'point_cloud.ply')
    origin = np.median(xyz, axis=0).astype('<f4')
    viewer = np.empty(len(xyz), dtype=[('xyz', '<f4', (3,)), ('rgb', 'u1', (4,))])
    viewer['xyz'] = xyz - origin
    viewer['rgb'][:, :3], viewer['rgb'][:, 3] = colors, 255
    temporary = artifacts / 'cloud.bin.tmp'
    with temporary.open('wb') as stream:
        stream.write(b'V86C' + struct.pack('<I', len(xyz)) + origin.tobytes() + viewer.tobytes())
    temporary.replace(artifacts / 'cloud.bin')
    atomic_json(artifacts / 'cloud_meta.json', dict(points=len(xyz), origin=origin.tolist(),
                geometry_kind='Pi3X', coordinate_frame='local_metric_xyz_aligned_to_uploaded_gps'))
    return len(xyz)


def publish_preview(session_dir, output, config):
    cloud = output / 'bootstrap_cloud.npz'
    if not cloud.exists():
        return
    artifacts = session_dir / 'artifacts'
    try:
        count = write_cloud(artifacts, cloud, config.get('max_preview_points', 180000))
    except (OSError, ValueError, EOFError):
        return
    def publish(state):
        state['preview_ready'] = True
        state['point_count'] = count
        state.setdefault('artifacts', {}).update(
            point_cloud_ply=f'/api/sessions/{session_dir.name}/artifacts/point_cloud.ply',
            viewer_data=f'/api/sessions/{session_dir.name}/artifacts/viewer.json',
            viewer=f'/s/{session_dir.name}/viewer')
    mutate_state(session_dir, publish)


def run_command(session_dir, stage, arguments, progress_output=None, config=None):
    ensure_active(session_dir)
    update_state(session_dir, phase=stage, message=stage.replace('_', ' '))
    log_path = session_dir / 'logs' / (stage + '.log')
    with log_path.open('a') as log:
        with state_lock(session_dir):
            ensure_active(session_dir)
            worker_python = workstation_config()[1].get('worker_python', sys.executable)
            process = subprocess.Popen([worker_python, '-m', *arguments], stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=os.environ.copy())
            key = str(session_dir.resolve())
            _PROCESSES.setdefault(key, set()).add(process)
        previous = None
        try:
            while process.poll() is None:
                ensure_active(session_dir)
                if progress_output and (progress_output / 'progress.json').exists():
                    try:
                        event = json.loads((progress_output / 'progress.json').read_text())
                        if event != previous:
                            update_state(session_dir, phase=event['stage'], message=f"{event['stage']}: {event['completed']}/{event['total']}",
                                progress=.1 + .6 * event['completed'] / max(event['total'], 1),
                                scal3r_lane_target_windows=event['total'], scal3r_lane_completed_windows=event['completed'],
                                scal3r_lane_message='Pi3X ' + event['stage'])
                            if event['stage'] == 'pi3x_geometry':
                                publish_preview(session_dir, progress_output, config)
                            previous = event
                    except (OSError, ValueError, KeyError):
                        pass
                time.sleep(.5)
            ensure_active(session_dir)
            if process.returncode:
                raise RuntimeError(f'{stage} failed; see logs/{stage}.log')
        finally:
            terminate(process)
            with state_lock(session_dir):
                _PROCESSES[key].discard(process)


def plan_mission(session_dir, output, config):
    state = load_state(session_dir)
    selection_path = output / 'selection_frozen.json'
    selection = json.loads(selection_path.read_text())
    manifest_path = output / 'planning_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not selection['selected']:
        return None
    takeoff = state['config'].get('takeoff_absolute_altitude_m')
    if takeoff is None or state['config'].get('test_only'):
        update_state(session_dir, planning_status='requires_takeoff_reference')
        return None
    observations = output / 'surface' / 'observations'
    if not (observations / 'observations.json').exists():
        if observations.exists():
            observations.rename(output / 'surface' / ('observations_failed_' + uuid.uuid4().hex))
        run_command(session_dir, 'surface_evidence', ['openflyscan.planning.analyze_frozen_region_observations',
                    '--selection', str(selection_path), '--geometry', str(output / 'surface' / 'geometry'), '--output', str(observations)])
    policy_path = output / 'planning_policy.json'
    atomic_json(policy_path, dataclasses.asdict(BudgetPolicy()))
    reference = manifest['reference_wgs84']
    takeoff_enu = geo_to_enu([reference[0], reference[1], takeoff], reference)[2]
    ceiling = float(state['config'].get('max_relative_altitude_m', 80))
    planning_config = dict(scene=manifest['scene'], selection=str(selection_path),
        surface_root=str(output / 'surface'), manifest=str(manifest_path), cloud=str(output / 'bootstrap_cloud.npz'),
        output=str(output / 'candidate_strips'), shared_policy=str(policy_path),
        mission=dict(ceiling_mode='explicit_enu', ceiling_enu_z_m=takeoff_enu + ceiling),
        geometry=dict(target_budget=len(selection['selected']), photo_budget=0, height_sampling_policy='stable_grid',
                      height_step_m=10, height_search_max_m=120, budget_atomic_strips=True,
                      interval_s=float(state['config'].get('minimum_capture_interval_s', 2)),
                      speed_mps=float(state['config'].get('speed_mps', 4))))
    atomic_json(output / 'planning_config.json', planning_config)
    candidates = output / 'candidate_strips'
    if not (candidates / 'selection_cache.npz').exists():
        if candidates.exists():
            candidates.rename(output / ('candidate_strips_failed_' + uuid.uuid4().hex))
        run_command(session_dir, 'candidate_strips', ['openflyscan.planning.run_budgeted_recapture_trial',
                    '--config', str(output / 'planning_config.json'), '--export-whole-surveys'])
    route = output / 'reacquisition'
    if not (route / 'route_plan.json').exists():
        if route.exists():
            route.rename(output / ('reacquisition_failed_' + uuid.uuid4().hex))
        run_command(session_dir, 'strip_selection', ['openflyscan.planning.run_directional_surface_cover',
                    '--evidence-run', str(candidates), '--full-plan', str(candidates), '--output', str(route)])
    plan = json.loads((route / 'route_plan.json').read_text())
    record = manifest['records'][0]
    width, height = record['image_size']
    intrinsics = np.asarray(record['intrinsics'])
    camera = dict(horizontal_fov_deg=float(np.degrees(2 * np.arctan(width / (2 * intrinsics[0, 0])))),
                  vertical_fov_deg=float(np.degrees(2 * np.arctan(height / (2 * intrinsics[1, 1])))))
    export_config = dict(state['config'], camera_profile=dict(id=state['config'].get('camera_model', 'uploaded-camera'),
                       image_width_px=width, image_height_px=height, **camera,
                       minimum_capture_interval_s=planning_config['geometry']['interval_s']))
    mission, mapping = export_mission(plan, manifest, camera, selection, export_config)
    atomic_json(session_dir / 'artifacts' / 'mission.json', mission)
    atomic_json(session_dir / 'artifacts' / 'mission_provenance.json', mapping)
    update_state(session_dir, mission_schema_version=mission['schema_version'], planning_status='preview_ready')
    return plan


def process_session(session_dir, final):
    session_dir = Path(session_dir).resolve()
    ensure_active(session_dir)
    path, config = workstation_config()
    state = load_state(session_dir)
    rows = list(state['images'])
    reconstruction_only = bool(state['config'].get('test_only'))
    minimum = config.get('minimum_images', 30)
    if len(rows) < minimum:
        update_state(session_dir, phase='waiting_for_images', message=f'At least {minimum} images are required', running=False)
        return
    identity = dict(version=PIPELINE_VERSION, config=config, session_config=state['config'],
                    images=rows, final=final)
    package = Path(__file__).resolve().parents[1]
    identity['source_digest'] = hashlib.sha256(b''.join(path.read_bytes() for path in sorted(package.rglob('*.py')))).hexdigest()
    identity['weight_files'] = {key: [Path(config[key]).stat().st_size, Path(config[key]).stat().st_mtime_ns]
                              for key in ('pi3x_checkpoint', 'quality_checkpoint')}
    stamp = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    workspace = session_dir / 'workstation' / stamp
    output = workspace / 'inference'
    workspace.mkdir(parents=True, exist_ok=True)
    update_state(session_dir, running=True, error=None, completed=False, pipeline_version=PIPELINE_VERSION,
                 phase='preparing', processed_images=len(rows), message='Preparing current observations')
    atomic_json(session_dir / 'artifacts' / 'viewer.json', dict(geometry_kind='Pi3X', candidates=[], tasks=[]))
    def clear_previous(current):
        current.setdefault('artifacts', {}).pop('mission', None)
        current['mission_schema_version'] = None
    mutate_state(session_dir, clear_previous)
    try:
        if not (output / 'complete.json').exists():
            prepare_scene(session_dir, rows, state['config'], workspace)
            arguments = ['openflyscan.reconstruction.runner', '--manifest', str(workspace / 'sensor_manifest.json'),
                         '--config', str(path), '--output', str(output), '--cache', str(session_dir / 'feature_cache'),
                         '--budget', str(state['config'].get('maximum_tasks', 12))]
            if not final or reconstruction_only:
                arguments.append('--preview')
            run_command(session_dir, 'reconstruction', arguments, output, config)
        publish_preview(session_dir, output, config)
        if final and not reconstruction_only:
            selection = json.loads((output / 'selection_frozen.json').read_text())
            plan = plan_mission(session_dir, output, config)
            ensure_active(session_dir)
            atomic_json(session_dir / 'artifacts' / 'quality.json', selection)
            atomic_json(session_dir / 'artifacts' / 'viewer.json', dict(geometry_kind='Pi3X',
                candidates=[dict(center_xyz_m=region['xyz'], score=region['score'], views=region['view_support'],
                                 detector='Quality Predictor') for region in selection['selected']],
                tasks=[dict(camera_xyz_m=[point['position_enu_m'] for point in row['waypoints']])
                       for row in plan['actions']] if plan else []))
            def publish(current):
                current['artifacts']['predictions'] = f'/api/sessions/{session_dir.name}/artifacts/quality.json'
                if plan:
                    current['artifacts']['mission'] = f'/api/sessions/{session_dir.name}/artifacts/mission.json'
            mutate_state(session_dir, publish)
        ensure_active(session_dir)
        update_state(session_dir, running=False, completed=final, phase='complete' if final else 'receiving',
                     progress=1. if final else 0., message='Processing complete' if final else 'Pi3X preview ready; receiving images',
                     last_preview_images=len(rows), fast_sfm_completed_images=len(rows))
    except PipelineCancelled:
        raise
    except Exception as error:
        update_state(session_dir, running=False, completed=False, phase='failed', error=str(error), message=str(error))
        raise
