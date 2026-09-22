"""Training-only outer footprint-window rejection; never an image or quality mask."""

import numpy as np
from scipy.spatial import ConvexHull

from .footprint_window_sampler import sample_footprint_window


def window_clearance(centers, equations, anchor, core_indices):
    anchor = np.asarray(anchor, dtype=float)
    centers = np.asarray(centers, dtype=float)
    equations = np.asarray(equations, dtype=float)
    radius = float(np.median(np.linalg.norm(centers[np.asarray(core_indices)]-anchor, axis=1)))
    clearance = float(-(equations[:, :2]@anchor+equations[:, 2]).max())
    tolerance = max(float(np.linalg.norm(np.ptp(centers, axis=0)))*1e-8, 1e-10)
    return dict(passed=bool(clearance > radius+tolerance), boundary_distance=clearance,
                core_radius=radius, margin=clearance-radius)


def build_policy(record):
    footprints = record.get('footprint_data')
    if footprints is None:
        footprints = dict(np.load(record['footprints']))
    stems = list(map(str, footprints['stems']))
    centers = np.asarray(footprints['centers'], dtype=float)
    if centers.shape != (len(stems), 2) or not np.isfinite(centers).all():
        raise ValueError('finite footprint centers required')
    hull = ConvexHull(centers)
    eligible = []
    for index, center in enumerate(footprints['centers']):
        window = sample_footprint_window(center, footprints['centers'], footprints['bbox_mins'], footprints['bbox_maxs'])
        if window_clearance(centers, hull.equations, center, window.core_indices)['passed']:
            eligible.append(index)
    policy = dict(schema='interior_footprint_chunk_v5', coordinate_frame='existing training footprint XY',
        footprints=str(record['footprints']), anchor_count=len(stems),
        margin_rule='distance from shifted anchor to scene footprint-center hull must exceed median radius of 21 core footprint centers',
        eligible_anchor_stems=[stems[index] for index in eligible],
        footprint_hull_xy=centers[hull.vertices].tolist(), hull_equations=hull.equations.tolist(),
        reference_stems=stems, reference_centers=centers.tolist(),
        core_count=21, overlap_count=9,
        scope='training windows only; check the actual shifted anchor and core; no camera or individual query removal',
        boundary='Operational chunk-sampling boundary from existing footprints, not a certified GS map ROI. Convex envelope does not exclude internal holes or concavities. No PSNR-based filtering.')
    if not eligible:
        raise ValueError(f"{record['scene']}: no interior 30-view windows; do not drop scene or relax boundary silently")
    return policy


def prepare_record(record):
    policy = record.get('outer_chunk_filter')
    if not policy or policy.get('schema') != 'interior_footprint_chunk_v5':
        raise ValueError('missing frozen interior-footprint policy')
    stems = list(map(str, record['footprint_data']['stems']))
    if stems != policy['reference_stems']:
        raise ValueError('footprint identities changed after boundary preparation')
    if not np.array_equal(record['footprint_data']['centers'], np.asarray(policy['reference_centers'])):
        raise ValueError('footprint coordinates changed after boundary preparation')
    eligible = set(policy['eligible_anchor_stems'])
    if not eligible or not eligible <= set(stems):
        raise ValueError('invalid eligible anchor identities')
    record['interior_anchor_indices'] = np.array([index for index, stem in enumerate(stems) if stem in eligible], dtype=int)
    record['footprint_stem_indices'] = {stem: index for index, stem in enumerate(stems)}


def check_choice(record, choice):
    if len(choice['full']) != 30 or len(set(choice['full'])) != 30:
        raise ValueError('interior policy requires an unchanged 30-view window')
    core = [record['footprint_stem_indices'][stem] for stem in choice['full'][:21]]
    return window_clearance(record['footprint_data']['centers'], record['outer_chunk_filter']['hull_equations'],
                            choice['anchor'], core)


def sample_interior_window(record, rng, weak):
    from .observation_source import sample_window

    eligible = record['interior_anchor_indices']
    weak_eligible = np.intersect1d(record['weak_indices'], eligible)
    effective_weak = bool(weak and len(weak_eligible))
    for attempt in range(256):
        choice = sample_window(record, rng, weak=effective_weak, anchor_indices=eligible)
        receipt = check_choice(record, choice)
        if receipt['passed']:
            return {**choice, 'outer_chunk_filter': dict(**receipt, rejected_proposals=attempt,
                weak_requested=bool(weak), weak_fallback_to_uniform=bool(weak and not effective_weak),
                definition='shifted anchor clearance greater than median core radius')}
    raise RuntimeError(f"{record['scene']}: no valid shifted interior window after 256 proposals; no outer fallback")
