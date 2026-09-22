"""Uploaded camera metadata and sensor-coordinate scene preparation."""

import json
import hashlib
import math
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

from openflyscan.missions.geometry_export import geo_to_enu


def finite(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def dji_xmp_telemetry(path):
    with Path(path).open('rb') as stream:
        payload = stream.read(256 * 1024)
    values = {key.decode(): float(value) for key, value in re.findall(
        rb'drone-dji:([A-Za-z]+)="([-+0-9.eE]+)"', payload)}
    with Image.open(path) as image:
        try:
            comment = image.getexif().get_ifd(34665).get(37510)
        except (KeyError, AttributeError):
            comment = None
    if isinstance(comment, bytes) and comment.startswith(b'ASCII\0\0\0'):
        comment = comment[8:].decode('utf-8', 'replace')
    try:
        record = json.loads(comment) if isinstance(comment, str) else {}
    except ValueError:
        record = {}
    if isinstance(record, dict) and record.get('source') == 'dji_video_downlink':
        for source, target in [('yaw_deg', 'FlightYawDegree'), ('gimbal_pitch_deg', 'GimbalPitchDegree'),
                               ('relative_altitude_m', 'RelativeAltitude')]:
            value = finite(record.get(source))
            if value is not None:
                values[target] = value
        values['pose_metadata_source'] = 'openfly_exif_user_comment'
    return values


def upload_metadata(headers, metadata):
    raw = headers.get('X-OpenFly-Metadata')
    if raw:
        if len(raw) > 4096:
            raise ValueError('camera metadata header exceeds 4096 characters')
        record = json.loads(raw)
        if not isinstance(record, dict) or record.get('schema_version') != 1:
            raise ValueError('unsupported camera metadata schema')
        for key in ('camera_ypr_deg', 'intrinsics', 'intrinsics_image_size'):
            if key in record:
                values = np.asarray(record[key], dtype=float)
                shape = {'camera_ypr_deg': (3,), 'intrinsics': (3, 3), 'intrinsics_image_size': (2,)}[key]
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError('invalid ' + key)
                metadata[key] = values.tolist()
        if 'intrinsics' in record and 'intrinsics_image_size' not in record:
            raise ValueError('intrinsics_image_size [width, height] is required with intrinsics')
        metadata['camera_metadata_schema'] = 1
    for header, field in [('X-Camera-Yaw', 'GimbalYawDegree'), ('X-Camera-Pitch', 'GimbalPitchDegree'),
                          ('X-Camera-Roll', 'GimbalRollDegree')]:
        if header in headers:
            value = finite(headers[header])
            if value is None:
                raise ValueError('invalid ' + header)
            metadata[field] = value
    return metadata


def camera_rotation(ypr):
    yaw, pitch, roll = np.radians(ypr)
    forward = np.array([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    right = np.array([np.cos(yaw), -np.sin(yaw), 0.])
    down = np.cross(forward, right)
    return np.column_stack((np.cos(roll) * right + np.sin(roll) * down,
                            -np.sin(roll) * right + np.cos(roll) * down, forward))


def camera_ypr(row):
    if 'camera_ypr_deg' in row:
        return list(row['camera_ypr_deg'])
    if all(row.get(key) == 0 for key in ('GimbalYawDegree', 'GimbalPitchDegree', 'GimbalRollDegree')):
        return None
    yaw = finite(row.get('GimbalYawDegree'))
    if yaw is None:
        yaw = finite(row.get('FlightYawDegree'))
    pitch = finite(row.get('GimbalPitchDegree'))
    roll = finite(row.get('GimbalRollDegree'))
    if yaw is None or pitch is None:
        return None
    return [yaw, pitch, roll if roll is not None else 0.]


def prepare_scene(session_dir, rows, config, output):
    output = Path(output)
    source = output / 'sensor_scene'
    (source / 'images').mkdir(parents=True, exist_ok=True)
    (source / 'cams').mkdir(exist_ok=True)
    test_only = config.get('altitude_mode') == 'relative_height_test'
    positions = np.asarray([[row['latitude'], row['longitude'],
                             row.get('relative_altitude_m') if test_only else row['altitude_m']] for row in rows], dtype=float)
    if not np.isfinite(positions).all() or not (np.abs(positions[:, 0]) <= 90).all() or not (np.abs(positions[:, 1]) <= 180).all():
        raise ValueError('finite geographic camera positions are required')
    reference = np.median(positions, axis=0).tolist()
    if test_only:
        reference[2] = 0.
    records = []
    all_attitudes = all(camera_ypr(row) is not None for row in rows)
    for row, geo in zip(rows, positions):
        image_path = (Path(session_dir) / 'images' / row['stored_name']).resolve()
        with Image.open(image_path) as image:
            width, height = image.size
        stem = f"frame_{int(row['sequence']):06d}"
        ypr = camera_ypr(row)
        pose = np.eye(4)
        pose[:3, 3] = geo_to_enu(geo, reference)
        if ypr is not None:
            pose[:3, :3] = camera_rotation(ypr)
        focal = width / (2 * math.tan(math.radians(config['horizontal_fov_deg']) / 2))
        intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]])
        if 'intrinsics' in row:
            original_width, original_height = row['intrinsics_image_size']
            if min(original_width, original_height) <= 0:
                raise ValueError('invalid intrinsics image dimensions')
            intrinsics = np.asarray(row['intrinsics'], float).copy()
            intrinsics[0] *= width / original_width
            intrinsics[1] *= height / original_height
            if min(intrinsics[0, 0], intrinsics[1, 1]) <= 0 or not np.allclose(intrinsics[2], [0, 0, 1]):
                raise ValueError('invalid pinhole camera intrinsics')
        text = 'extrinsic opencv(x Right, y Down, z Forward) world2camera\n'
        text += '\n'.join(' '.join(map(str, line)) for line in np.linalg.inv(pose))
        text += '\n\nintrinsic: fx fy cx cy (pixel)\n'
        text += '\n'.join(' '.join(map(str, line)) for line in intrinsics)
        text += f'\n\nh w hfov\n{height} {width} {config["horizontal_fov_deg"]}\n'
        (source / 'cams' / (stem + '.txt')).write_text(text)
        link = source / 'images' / (stem + image_path.suffix.lower())
        if not link.exists():
            os.link(image_path, link)
        if row.get('sha256'):
            hasher = hashlib.sha256()
            with link.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    hasher.update(block)
            if hasher.hexdigest() != row['sha256']:
                raise ValueError('image changed after the job snapshot; retry the current session')
        records.append(dict(stem=stem, image=str(image_path), lat=float(geo[0]), lon=float(geo[1]),
                            alt=float(geo[2]), enu=pose[:3, 3].tolist(), camera_ypr_deg=ypr,
                            attitude_source='uploaded_sensor' if ypr is not None else 'unavailable',
                            roll_assumed_zero='camera_ypr_deg' not in row and finite(row.get('GimbalRollDegree')) is None,
                            image_size=[width, height], intrinsics=intrinsics.tolist()))
    manifest = dict(scene=Path(session_dir).name, raw_images=len(rows), source=str(source), records=records,
                    reference_wgs84=reference, pose_prior='input' if all_attitudes else 'none',
                    ray_prior='input', depth_prior='none', GS_or_SfM_inputs=False,
                    coordinate_frame='local_east_north_relative_to_takeoff_m_TEST_ONLY' if test_only else 'ENU',
                    test_only=test_only, takeoff_absolute_altitude_m=config.get('takeoff_absolute_altitude_m'),
                    attitude_policy='Use camera priors only when every frame has attitude; otherwise infer rotation with Pi3X.')
    (output / 'sensor_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest
