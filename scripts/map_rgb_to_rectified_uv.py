#!/usr/bin/env python3
"""Map ORIGINAL RGB pixel indices to COLMAP-undistorted pixel indices.

No GS errors, published dataset calibration, or 3D point coordinates are used.
Models come from this run's rgb_sfm and published_scene_source/sparse/0.
Input/output UV use array convention: top-left pixel center is (0, 0).
COLMAP camera coordinates put that center at (0.5, 0.5), so conversion is explicit.
Pi3X resized/cropped UV must first be inverted to ORIGINAL RGB coordinates.
Model formulas: https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def calibration(camera):
    p = np.asarray(camera.params, dtype=np.float64)
    model = camera.model
    if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
        fx = fy = p[0]
        cx, cy = p[1:3]
        d = np.zeros(8)
        if model != 'SIMPLE_PINHOLE': d[0] = p[3]
        if model == 'RADIAL': d[1] = p[4]
    elif model in ('PINHOLE', 'OPENCV', 'FULL_OPENCV'):
        fx, fy, cx, cy = p[:4]
        d = np.zeros(8)
        if model != 'PINHOLE': d[:len(p)-4] = p[4:]
    else:
        raise ValueError(f'Unsupported camera {model}; do not approximate fisheye as pinhole')
    k = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    if not np.isfinite(k).all() or min(fx, fy) <= 0: raise ValueError('Invalid calibration')
    return k, d


def map_uv(uv, source_camera, target_camera, pixel_center_offset=0.5):
    """Preserve leading dimensions; return UV and valid target-pixel-center mask."""
    uv = np.asarray(uv, dtype=np.float64)
    if uv.shape[-1] != 2: raise ValueError('Expected UV[...,2]')
    shape = uv.shape
    p = uv.reshape(-1, 2)
    finite = np.isfinite(p).all(axis=1)
    inverse_valid = np.zeros(len(p),dtype=bool)
    result = np.full_like(p, np.nan)
    if finite.any():
        ks, ds = calibration(source_camera)
        kt, dt = calibration(target_camera)
        xy = cv2.undistortPointsIter((p[finite]+pixel_center_offset).reshape(-1, 1, 2),
             ks, ds, None, None, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-14)).reshape(-1, 2)
        rays = np.column_stack([xy, np.ones(len(xy))])
        # A finite inverse is not necessarily a converged inverse for strong
        # radial distortion. Reject inaccurate source reprojection generically.
        reprojection, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), ks, ds)
        residual=np.linalg.norm(reprojection.reshape(-1,2)-(p[finite]+pixel_center_offset),axis=1)
        inverse_valid[finite]=np.isfinite(residual)&(residual<=1e-3)
        projected, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), kt, dt)
        result[finite] = projected.reshape(-1, 2)-pixel_center_offset
    valid = finite & inverse_valid & np.isfinite(result).all(axis=1)
    valid &= (p[:, 0] >= 0) & (p[:, 0] <= source_camera.width-1)
    valid &= (p[:, 1] >= 0) & (p[:, 1] <= source_camera.height-1)
    valid &= (result[:, 0] >= 0) & (result[:, 0] <= target_camera.width-1)
    valid &= (result[:, 1] >= 0) & (result[:, 1] <= target_camera.height-1)
    return result.reshape(shape), valid.reshape(shape[:-1])


def load_pair(raw_model, rectified_model, rw):
    rc = rw.read_cameras_binary(str(raw_model/'cameras.bin'))
    tc = rw.read_cameras_binary(str(rectified_model/'cameras.bin'))
    ri = {im.name: im for im in rw.read_images_binary(str(raw_model/'images.bin')).values()}
    ti = {im.name: im for im in rw.read_images_binary(str(rectified_model/'images.bin')).values()}
    if set(ri) != set(ti): raise ValueError('Raw and rectified registered image sets differ')
    for name in ri:
        a, b = ri[name], ti[name]
        if not np.allclose(a.qvec2rotmat(), b.qvec2rotmat(), atol=1e-10, rtol=0):
            raise ValueError(f'{name}: orientation changed; intrinsics-only UV warp invalid')
        if not np.allclose(a.tvec, b.tvec, atol=1e-10, rtol=0):
            raise ValueError(f'{name}: translation changed; intrinsics-only UV warp invalid')
    return rc, tc, ri, ti


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-model', type=Path, required=True)
    p.add_argument('--rectified-model', type=Path, required=True)
    p.add_argument('--read-write-model-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--raw-images', type=Path)
    p.add_argument('--rectified-images', type=Path)
    p.add_argument('--check-image')
    p.add_argument('--image-name')
    p.add_argument('--uv-npy', type=Path)
    p.add_argument('--mapped-output', type=Path)
    a = p.parse_args()
    sys.path.insert(0, str(a.read_write_model_root))
    import read_write_model as rw
    rc, tc, ri, ti = load_pair(a.raw_model, a.rectified_model, rw)
    pairs = sorted({(ri[n].camera_id, ti[n].camera_id) for n in ri})
    results = []
    for sid, tid in pairs:
        s, t = rc[sid], tc[tid]
        u, v = np.meshgrid(np.linspace(0, s.width-1, 33), np.linspace(0, s.height-1, 25))
        uv = np.stack([u, v], axis=-1)
        mapped, valid = map_uv(uv, s, t)
        back, back_valid = map_uv(mapped, t, s)
        errors = np.linalg.norm(back-uv, axis=-1)
        shift = mapped-uv
        checked=valid&back_valid
        maximum = float(np.max(errors[checked])) if checked.any() else None
        if maximum is not None and maximum > 1e-3: raise ValueError(f'Valid roundtrip failed {maximum}px')
        results.append(dict(raw_id=sid, rectified_id=tid,
            raw=dict(model=s.model, width=s.width, height=s.height, params=s.params.tolist()),
            rectified=dict(model=t.model, width=t.width, height=t.height, params=t.params.tolist()),
            roundtrip_max_px=maximum, roundtrip_median_px=float(np.median(errors[checked])) if checked.any() else None,
            invalid_or_outside_points=int((~checked).sum()),
            median_shift_uv_px=np.median(shift.reshape(-1, 2), axis=0).tolist(),
            identity_mapping_error_max_px=float(np.linalg.norm(shift, axis=-1).max()),
            sampled_inside_fraction=float(valid.mean())))
    # Independent check against COLMAP's already-undistorted feature coordinates.
    errors = []
    for name in sorted(ri):
        s, t = ri[name], ti[name]
        # COLMAP preserves POINT2D_IDX. Point3D IDs can repeat within a GLOMAP
        # image, so matching a dict keyed only by point3D_id silently pairs the
        # wrong image observations (found in the c835 audit).
        if not np.array_equal(s.point3D_ids, t.point3D_ids):
            raise ValueError(f'{name}: POINT2D_IDX/track ordering changed; audit needs explicit correspondence')
        indices = np.flatnonzero(s.point3D_ids >= 0)
        if not len(indices): continue
        indices = indices[::max(1, len(indices)//32)][:32]
        original, expected = s.xys[indices], t.xys[indices]
        actual, valid = map_uv(original, rc[s.camera_id], tc[t.camera_id], pixel_center_offset=0.)
        errors.extend(np.linalg.norm(actual[valid]-expected[valid], axis=-1).tolist())
    report = dict(schema='raw-rgb-to-colmap-rectified-uv-v1', images=len(ri),
        coordinate_convention='array pixel centers at integers; camera projection uses +0.5 then -0.5',
        raw_model=str(a.raw_model), rectified_model=str(a.rectified_model),
        calibration_pairs=results, poses_identical=True, gs_errors_used=False,
        original_to_pi3x_preprocessing='not part of this warp; invert cache resize/crop before calling',
        independent_colmap_tracks=dict(count=len(errors),
            median_px=float(np.median(errors)) if errors else None,
            max_px=float(np.max(errors)) if errors else None))
    if errors and np.max(errors) > 1e-3: raise ValueError(f'COLMAP keypoint warp disagrees: max {np.max(errors)}')
    if a.check_image:
        if not a.raw_images or not a.rectified_images: raise ValueError('Both image directories required')
        from PIL import Image
        raw = np.asarray(Image.open(a.raw_images/a.check_image).convert('RGB'))
        actual = np.asarray(Image.open(a.rectified_images/a.check_image).convert('RGB'))
        s, t = rc[ri[a.check_image].camera_id], tc[ti[a.check_image].camera_id]
        assert raw.shape[:2] == (s.height, s.width) and actual.shape[:2] == (t.height, t.width)
        x, y = np.meshgrid(np.arange(t.width), np.arange(t.height))
        original_uv, valid = map_uv(np.stack([x,y], axis=-1), t, s)
        warped = cv2.remap(raw, original_uv[...,0].astype('float32'), original_uv[...,1].astype('float32'), cv2.INTER_LINEAR)
        diff = np.abs(warped.astype(float)-actual.astype(float))[valid]
        report['rectified_rgb_reproduction'] = dict(image=a.check_image, valid_pixels=int(valid.sum()),
            single_pass_mean_absolute_8bit=float(diff.mean()), p99_absolute_8bit=float(np.quantile(diff,.99)),
            note='COLMAP warp.cc warps at raw resolution then bilinear resizes; one-pass remap is not pixel-identical')
        # COLMAP uses two resampling stages to limit aliasing. Reproduce that
        # structure for a content check without changing the geometric UV map.
        from types import SimpleNamespace
        kt, dt = calibration(t)
        if np.any(dt): raise ValueError('RGB reproduction expects rectified pinhole target')
        sx, sy = s.width/t.width, s.height/t.height
        scaled = SimpleNamespace(model='PINHOLE', width=s.width, height=s.height,
            params=np.array([kt[0,0]*sx,kt[1,1]*sy,kt[0,2]*sx,kt[1,2]*sy]))
        x, y = np.meshgrid(np.arange(s.width), np.arange(s.height))
        coords, _ = map_uv(np.stack([x,y],axis=-1), scaled, s)
        intermediate = cv2.remap(raw,coords[...,0].astype('float32'),coords[...,1].astype('float32'),cv2.INTER_LINEAR)
        reproduced = np.asarray(Image.fromarray(intermediate).resize((t.width,t.height),Image.Resampling.BILINEAR))
        d2 = np.abs(reproduced.astype(float)-actual.astype(float))[valid]
        report['rectified_rgb_reproduction'].update(two_pass_mean_absolute_8bit=float(d2.mean()),
            two_pass_p99_absolute_8bit=float(np.quantile(d2,.99)),
            two_pass_note='Same geometry/two-stage sampling; PIL/OpenCV vs FreeImage kernels may differ')
    if a.uv_npy:
        if not a.image_name or not a.mapped_output: raise ValueError('Mapping needs image-name and mapped-output')
        uv, valid = map_uv(np.load(a.uv_npy), rc[ri[a.image_name].camera_id], tc[ti[a.image_name].camera_id])
        np.savez_compressed(a.mapped_output, uv=uv, valid=valid)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
