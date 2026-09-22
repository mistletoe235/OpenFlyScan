"""Conservative cross-chunk deduplication using geometry AND shared image evidence."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeduplicationConfig:
    radius_fraction: float = 0.5
    minimum_patch_overlap: float = 0.3
    minimum_shared_patches: int = 3
    normal_cosine: float = 0.9
    minimum_shared_views: int = 2
    image_overlap: float = 0.6


def _patches(region):
    result = {}
    for evidence in region.get("image_evidence", []):
        shape = tuple(evidence.get("patch_hw", ()))
        if len(shape) != 2 or any(not isinstance(value, int) or value <= 0 for value in shape):
            raise ValueError("patch_hw must contain two positive integers")
        key = (evidence["stem"], shape)
        pixels = set()
        for pixel in evidence["patch_pixels_yx"]:
            if len(pixel) != 2 or any(not isinstance(value, int) or not 0 <= value < limit for value, limit in zip(pixel, shape)):
                raise ValueError("patch evidence must index the stated image grid")
            pixels.add(tuple(pixel))
        result.setdefault(key, set()).update(pixels)
    return result


def same_surface(first, second, config):
    if first["chunk"] == second["chunk"]:
        return False
    if not first.get("alignment_verified") or not second.get("alignment_verified"):
        return False
    if not first.get("frame_id") or first["frame_id"] != second.get("frame_id"):
        return False
    if not first.get("scene") or first["scene"] != second.get("scene"):
        return False
    radius = min(first["radius"], second["radius"])
    if np.linalg.norm(np.asarray(first["xyz"])-second["xyz"]) > config.radius_fraction*radius:
        return False
    if "normal" in first and "normal" in second:
        left, right = np.asarray(first["normal"]), np.asarray(second["normal"])
        denominator = np.linalg.norm(left)*np.linalg.norm(right)
        if denominator <= 1e-8 or abs(float(left@right))/denominator < config.normal_cosine:
            return False
    left_patches, right_patches = _patches(first), _patches(second)
    for key in left_patches.keys() & right_patches.keys():
        shared = len(left_patches[key] & right_patches[key])
        overlap = shared / max(min(len(left_patches[key]), len(right_patches[key])), 1)
        if shared >= config.minimum_shared_patches and overlap >= config.minimum_patch_overlap:
            return True
    return False


def shared_image_surface(first, second, config, first_patches=None, second_patches=None):
    if not first.get("scene") or first["scene"] != second.get("scene"):
        return False
    if not first.get("image_space_id") or first["image_space_id"] != second.get("image_space_id"):
        return False
    left = _patches(first) if first_patches is None else first_patches
    right = _patches(second) if second_patches is None else second_patches
    supporting_stems = set()
    for key in left.keys() & right.keys():
        shared = len(left[key] & right[key])
        overlap = shared/max(min(len(left[key]), len(right[key])), 1)
        if shared >= config.minimum_shared_patches and overlap >= config.image_overlap:
            supporting_stems.add(key[0])
    return len(supporting_stems) >= config.minimum_shared_views


def select_regions(regions, budget, config=None, *, representative_only=False):
    config = config or DeduplicationConfig()
    if type(budget) is not int or budget < 0:
        raise ValueError("budget cannot be negative")
    if not 0 < config.radius_fraction <= 1 or not 0 < config.minimum_patch_overlap <= 1:
        raise ValueError("invalid deduplication thresholds")
    if config.minimum_shared_patches < 1 or not 0 <= config.normal_cosine <= 1:
        raise ValueError("invalid support or normal threshold")
    if config.minimum_shared_views < 2 or not 0.6 <= config.image_overlap <= 1:
        raise ValueError("unverified geometry requires strong overlap in at least two distinct images")
    patches = []
    for region in regions:
        if not isinstance(region.get("alignment_verified", False), bool):
            raise ValueError("alignment_verified must be boolean")
        patches.append(_patches(region))
        if not np.isfinite([*region["xyz"], region["score"], region["radius"]]).all() or region["radius"] <= 0:
            raise ValueError("non-finite candidate or nonpositive radius")
        if len(region["xyz"]) != 3:
            raise ValueError("xyz must contain three coordinates")
        if "normal" in region:
            normal = np.asarray(region["normal"])
            if normal.shape != (3,) or not np.isfinite(normal).all():
                raise ValueError("normal must contain three finite coordinates")
    clusters = []
    order = sorted(range(len(regions)), key=lambda index: (-regions[index]["score"], str(regions[index]["chunk"]), index))
    for index in order:
        for cluster in clusters:
            compared_members = cluster[:1] if representative_only else cluster
            if all(shared_image_surface(regions[index], regions[member], config, patches[index], patches[member])
                   or same_surface(regions[index], regions[member], config) for member in compared_members):
                cluster.append(index)
                break
        else:
            clusters.append([index])
    merged = []
    for cluster in clusters:
        members = [regions[index] for index in cluster]
        scores = [member["score"] for member in members]
        aligned = all(member.get("alignment_verified", False) for member in members)
        location = np.median([member["xyz"] for member in members], axis=0).tolist() if aligned else list(members[0]["xyz"])
        merged.append(dict(xyz=location,
                           score=float(np.mean(scores)), score_range=[float(min(scores)), float(max(scores))],
                           members=cluster, chunks=[member["chunk"] for member in members],
                           scene=members[0].get("scene"), frame_id=members[0].get("frame_id"),
                           correspondence_verified=len(cluster) > 1,
                           alignment_verified=all(member.get("alignment_verified", False) for member in members)))
    merged.sort(key=lambda region: (-region["score"], region["members"][0]))
    return dict(input_regions=len(regions), deduplicated_candidates=len(merged), budget=budget,
                merged_regions=merged, selected=merged[:budget],
                unverified_alignment_candidates=sum(not row["alignment_verified"] for row in merged),
                boundary="Conservative candidate dedup, not SLRF or certified unique physical surfaces. Unknown correspondence stays separate; repeated observations do not increase scores.")


def allocate_retake_budget(regions, budget, config=None):
    result = select_regions(regions, budget, config, representative_only=True)
    candidates = []
    for cluster in result["merged_regions"]:
        indices = cluster["members"]
        representative = min(indices, key=lambda index: (-regions[index]["score"], str(regions[index]["chunk"]), index))
        original = regions[representative]
        candidate = {**original, **cluster}
        candidate.update(xyz=list(original["xyz"]), score=float(original["score"]), representative=representative,
                         query_count=len(indices), chunks=sorted(set(cluster["chunks"]), key=str),
                         location_hypotheses=[dict(query=index, chunk=regions[index]["chunk"], xyz=regions[index]["xyz"])
                                              for index in indices],
                         maximum_location_disagreement=float(max(np.linalg.norm(np.asarray(regions[index]["xyz"])-original["xyz"])
                                                                 for index in indices)))
        candidate["image_evidence"] = original.get("image_evidence", [])
        candidates.append(candidate)
    candidates.sort(key=lambda region: (-region["score"], region["representative"]))
    result.update(schema="retake_budget_v2", merged_regions=candidates, selected=candidates[:budget],
                  duplicates_removed=len(regions)-len(candidates),
                  score_rule="maximum original risk; no bonus for repeated observations",
                  grouping_rule="each member must directly match its fixed highest-risk representative; no transitive chaining",
                  location_rule="highest-risk original location; retain alternatives, never average uncertain coordinates",
                  boundary="Merge before allocating slots. Two shared image areas can identify repeats without proving global alignment. Unknown correspondences remain separate; not flight-ready coordinates.")
    return result
