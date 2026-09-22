#!/usr/bin/env python3
"""Generate geometry-only recapture candidates around defect surfaces.

The generator intentionally has no image, GS, probe, depth, or simulator-pose
input.  Coordinates are local ENU (x east, y north, z up).  Aircraft yaw is
DJI convention: degrees clockwise from north.
"""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import jsonschema


GENERATOR_NAME = "openfly_geometry_candidate_generator"
GENERATOR_VERSION = "0.3.0"
EPSILON = 1.0e-9


def add(a: list[float], b: list[float]) -> list[float]:
    return [a[index] + b[index] for index in range(3)]


def subtract(a: list[float], b: list[float]) -> list[float]:
    return [a[index] - b[index] for index in range(3)]


def scale(a: list[float], value: float) -> list[float]:
    return [component * value for component in a]


def dot(a: list[float], b: list[float]) -> float:
    return sum(a[index] * b[index] for index in range(3))


def norm(a: list[float]) -> float:
    return math.sqrt(dot(a, a))


def normalized(a: list[float], label: str) -> list[float]:
    length = norm(a)
    if length <= EPSILON:
        raise ValueError(f"{label} must be non-zero")
    return scale(a, 1.0 / length)


def cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def horizontal_unit(vector: list[float], fallback: list[float]) -> list[float]:
    projected = [vector[0], vector[1], 0.0]
    if norm(projected) <= EPSILON:
        projected = fallback
    return normalized(projected, "horizontal direction")


def rotate_xy(vector: list[float], angle_deg: float) -> list[float]:
    angle = math.radians(angle_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        cosine * vector[0] - sine * vector[1],
        sine * vector[0] + cosine * vector[1],
        0.0,
    ]


def angle_between_deg(a: list[float], b: list[float]) -> float:
    a_unit = normalized(a, "first direction")
    b_unit = normalized(b, "second direction")
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(a_unit, b_unit)))))


def route_id_for(payload: dict, region: dict, variant: str) -> str:
    """Return an order-independent, filesystem-safe ID scoped to scene and episode."""
    semantic = "|".join(
        (str(payload["scene_id"]), str(payload["episode_id"]), str(region["region_id"]), variant)
    )
    readable = "__".join(
        "".join(character if character.isalnum() or character in "-_" else "-" for character in part)
        for part in (str(payload["scene_id"]), str(payload["episode_id"]), str(region["region_id"]), variant)
    )
    return f"{readable}__{hashlib.sha256(semantic.encode('utf-8')).hexdigest()[:10]}"


def point_in_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    """Ray-casting test; boundary points count as inside for conservative gating."""
    x, y = point[:2]
    inside = False
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        x1, y1 = start
        x2, y2 = end
        cross_value = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross_value) <= 1.0e-8 and min(x1, x2) - EPSILON <= x <= max(x1, x2) + EPSILON \
                and min(y1, y2) - EPSILON <= y <= max(y1, y2) + EPSILON:
            return True
        intersects = (y1 > y) != (y2 > y)
        if intersects:
            edge_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x <= edge_x:
                inside = not inside
    return inside


def yaw_pitch_to_target(camera: list[float], target: list[float]) -> tuple[float, float]:
    east, north, up = subtract(target, camera)
    horizontal = math.hypot(east, north)
    yaw = math.degrees(math.atan2(east, north)) % 360.0 if horizontal > EPSILON else 0.0
    pitch = math.degrees(math.atan2(up, horizontal))
    return yaw, max(-90.0, min(30.0, pitch))


def path_length(points: list[list[float]]) -> float:
    return sum(norm(subtract(right, left)) for left, right in zip(points, points[1:]))


def sample_line(center: list[float], axis: list[float], length_m: float, spacing_m: float) -> list[list[float]]:
    segment_count = max(1, math.ceil(length_m / spacing_m))
    return [
        add(center, scale(axis, -0.5 * length_m + length_m * index / segment_count))
        for index in range(segment_count + 1)
    ]


def cadence_safe_speed(
    points: list[list[float]], proposed_speed_mps: float, camera: dict, constraints: dict
) -> float:
    distances = [norm(subtract(right, left)) for left, right in zip(points, points[1:])]
    if not distances:
        raise ValueError("candidate route needs at least two distinct capture points")
    interval_s = float(camera["minimum_capture_interval_s"])
    speed_mps = min(proposed_speed_mps, min(distances) / interval_s)
    if speed_mps + EPSILON < float(constraints["minimum_speed_mps"]):
        raise ValueError(
            "route is too short to satisfy camera cadence at the configured minimum speed"
        )
    return speed_mps


def capture_spacing_and_speed(
    range_m: float, camera: dict, constraints: dict
) -> tuple[float, float]:
    footprint_m = 2.0 * max(range_m, 1.0) * math.tan(
        math.radians(float(camera["horizontal_fov_deg"])) / 2.0
    )
    overlap_spacing_m = footprint_m * (1.0 - float(constraints["forward_overlap"]))
    maximum_segment_m = float(constraints.get("maximum_capture_segment_m", 30.0))
    spacing_m = min(overlap_spacing_m, maximum_segment_m)
    interval_s = float(camera["minimum_capture_interval_s"])
    nominal_speed_mps = float(constraints["nominal_speed_mps"])
    minimum_speed_mps = float(constraints["minimum_speed_mps"])
    speed_mps = min(nominal_speed_mps, spacing_m / interval_s)
    if speed_mps + EPSILON < minimum_speed_mps:
        required_spacing_m = minimum_speed_mps * interval_s
        if required_spacing_m > min(overlap_spacing_m, maximum_segment_m) + EPSILON:
            raise ValueError(
                "camera cadence, overlap, and minimum speed constraints are mutually incompatible"
            )
        speed_mps = minimum_speed_mps
        spacing_m = required_spacing_m
    return max(spacing_m, 0.25), speed_mps


def clamp_flight_z(surface_z: float, desired_z: float, constraints: dict) -> float:
    ground_z = float(constraints["takeoff_ground_z_enu_m"])
    minimum_z = ground_z + float(constraints["minimum_relative_altitude_m"])
    maximum_z = ground_z + float(constraints["maximum_relative_altitude_m"])
    clearance_z = surface_z + float(constraints["minimum_surface_clearance_m"])
    lower = max(minimum_z, clearance_z)
    if lower > maximum_z + EPSILON:
        raise ValueError("surface plus required clearance exceeds maximum flight altitude")
    return min(max(desired_z, lower), maximum_z)


def default_modes(region: dict) -> list[str]:
    surface_class = region["surface_class"]
    if surface_class == "ROOF":
        return ["NADIR_STRIP", "OBLIQUE_STRIP"]
    if surface_class == "FACADE":
        return ["FACADE_STRIP", "LOCAL_ORBIT"]
    return ["NADIR_STRIP", "LOCAL_ORBIT"]


def make_waypoints(
    camera_points: list[list[float]],
    target_points: list[list[float]],
    speed_mps: float,
    capture_view: str,
    target_surface_ids: list[str],
) -> list[dict]:
    waypoints = []
    for camera, target in zip(camera_points, target_points, strict=True):
        yaw, pitch = yaw_pitch_to_target(camera, target)
        waypoints.append({
            "position_enu_m": [round(value, 6) for value in camera],
            "aircraft_yaw_deg": round(yaw, 6),
            "gimbal_pitch_deg": round(pitch, 6),
            "speed_mps": round(speed_mps, 6),
            "capture": True,
            "capture_view": capture_view,
            "target_surface_ids": target_surface_ids,
        })
    return waypoints


def insert_non_capture_turn_transitions(
    waypoints: list[dict], max_segment_m: float, max_yaw_delta_deg: float = 45.0
) -> list[dict]:
    """Densify discontinuous bridge-to-strip joins without inventing extra photos."""
    if len(waypoints) < 2:
        return waypoints
    result = [waypoints[0]]
    for left, right in zip(waypoints, waypoints[1:], strict=False):
        leg = make_transit_leg(
            left["position_enu_m"], right["position_enu_m"],
            left["aircraft_yaw_deg"], right["aircraft_yaw_deg"],
            left["gimbal_pitch_deg"], right["gimbal_pitch_deg"],
            min(float(left["speed_mps"]), float(right["speed_mps"])),
            max_segment_m, max_yaw_delta_deg,
        )
        result.extend(leg[1:-1])
        result.append(right)
    return result


def interpolate_angle_deg(start: float, end: float, ratio: float) -> float:
    delta = (end - start + 540.0) % 360.0 - 180.0
    return (start + delta * ratio) % 360.0


def make_transit_leg(
    start: list[float], end: list[float], start_yaw: float, end_yaw: float,
    start_pitch: float, end_pitch: float, speed_mps: float, max_segment_m: float,
    max_yaw_delta_deg: float = 45.0,
) -> list[dict]:
    distance = norm(subtract(end, start))
    yaw_delta = abs((end_yaw - start_yaw + 180.0) % 360.0 - 180.0)
    count = max(
        1,
        math.ceil(distance / max_segment_m),
        math.ceil(yaw_delta / max_yaw_delta_deg),
    )
    points = []
    for index in range(count + 1):
        ratio = index / count
        position = [start[axis] + (end[axis] - start[axis]) * ratio for axis in range(3)]
        points.append({
            "position_enu_m": [round(value, 6) for value in position],
            "aircraft_yaw_deg": round(interpolate_angle_deg(start_yaw, end_yaw, ratio), 6),
            "gimbal_pitch_deg": round(start_pitch + (end_pitch - start_pitch) * ratio, 6),
            "speed_mps": round(speed_mps, 6),
            "capture": False,
            "is_safe_transit": True,
        })
    return points


def segment_allowed_xy(start: list[float], end: list[float], constraints: dict) -> bool:
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    count = max(1, math.ceil(distance / 1.0))
    boundary = constraints.get("operational_boundary_xy_m")
    exclusions = constraints.get("exclusion_zones", [])
    for index in range(count + 1):
        ratio = index / count
        point = [start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio, start[2]]
        if boundary and not point_in_polygon(point, boundary):
            return False
        if any(point_in_polygon(point, zone["polygon_xy_m"]) for zone in exclusions):
            return False
    return True


def safe_horizontal_path(start: list[float], end: list[float], constraints: dict) -> list[list[float]]:
    """Visibility-graph detour around declared 2-D exclusions at a safe altitude."""
    if segment_allowed_xy(start, end, constraints):
        return [start, end]
    nodes = [start, end]
    margin = 3.0
    for zone in constraints.get("exclusion_zones", []):
        polygon = zone["polygon_xy_m"]
        centroid = [sum(point[0] for point in polygon) / len(polygon), sum(point[1] for point in polygon) / len(polygon)]
        for x, y in polygon:
            direction = normalized([x - centroid[0], y - centroid[1], 0.0], "exclusion corner direction")
            nodes.append([x + margin * direction[0], y + margin * direction[1], start[2]])
    adjacency: list[list[tuple[float, int]]] = [[] for _ in nodes]
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if segment_allowed_xy(nodes[left], nodes[right], constraints):
                distance = norm(subtract(nodes[right], nodes[left]))
                adjacency[left].append((distance, right))
                adjacency[right].append((distance, left))
    queue = [(0.0, 0)]
    distances = {0: 0.0}
    previous: dict[int, int] = {}
    while queue:
        distance, node = heapq.heappop(queue)
        if node == 1:
            break
        if distance > distances.get(node, math.inf):
            continue
        for edge, neighbor in adjacency[node]:
            proposal = distance + edge
            if proposal < distances.get(neighbor, math.inf):
                distances[neighbor] = proposal
                previous[neighbor] = node
                heapq.heappush(queue, (proposal, neighbor))
    if 1 not in distances:
        raise ValueError("no safe 2D transit path around declared exclusions")
    order = [1]
    while order[-1] != 0:
        order.append(previous[order[-1]])
    return [nodes[index] for index in reversed(order)]


def transit_along_path(points: list[list[float]], yaw: float, pitch: float, speed: float, max_segment: float) -> list[dict]:
    result: list[dict] = []
    for start, end in zip(points, points[1:]):
        leg = make_transit_leg(start, end, yaw, yaw, pitch, pitch, speed, max_segment)
        result.extend(leg if not result else leg[1:])
    return result


def attach_home_transit(route: dict, constraints: dict) -> dict:
    home = constraints.get("home_enu_m")
    route["requires_external_safe_connection"] = home is None
    if home is None:
        route["model_features"]["safe_connection_semantics"] = "external_planner_required"
        return route

    capture_points = route["waypoints"]
    first, last = capture_points[0], capture_points[-1]
    maximum_z = float(constraints["takeoff_ground_z_enu_m"]) + float(
        constraints["maximum_relative_altitude_m"]
    )
    requested_safe_z = float(constraints.get(
        "safe_transit_altitude_m",
        max(first["position_enu_m"][2], last["position_enu_m"][2], float(home[2])),
    ))
    safe_z = min(requested_safe_z, maximum_z)
    if safe_z + EPSILON < max(first["position_enu_m"][2], last["position_enu_m"][2]):
        raise ValueError("safe transit altitude is below the capture route")
    speed = float(constraints["nominal_speed_mps"])
    max_segment = float(constraints.get("maximum_capture_segment_m", 30.0))
    home_safe = [float(home[0]), float(home[1]), safe_z]
    first_safe = [*first["position_enu_m"][:2], safe_z]
    last_safe = [*last["position_enu_m"][:2], safe_z]
    inbound_horizontal = transit_along_path(
        safe_horizontal_path(home_safe, first_safe, constraints), first["aircraft_yaw_deg"],
        first["gimbal_pitch_deg"], speed, max_segment,
    )
    inbound_vertical = make_transit_leg(
        first_safe, first["position_enu_m"], first["aircraft_yaw_deg"], first["aircraft_yaw_deg"],
        first["gimbal_pitch_deg"], first["gimbal_pitch_deg"], speed, max_segment,
    )
    outbound_vertical = make_transit_leg(
        last["position_enu_m"], last_safe, last["aircraft_yaw_deg"], last["aircraft_yaw_deg"],
        last["gimbal_pitch_deg"], last["gimbal_pitch_deg"], speed, max_segment,
    )
    outbound_horizontal = transit_along_path(
        safe_horizontal_path(last_safe, home_safe, constraints), last["aircraft_yaw_deg"],
        last["gimbal_pitch_deg"], speed, max_segment,
    )
    # Drop duplicated joins while retaining explicit vertical arrival/departure.
    route["waypoints"] = (
        inbound_horizontal[:-1] + inbound_vertical[:-1] + capture_points
        + outbound_vertical[1:] + outbound_horizontal[1:]
    )
    total_length = path_length([point["position_enu_m"] for point in route["waypoints"]])
    total_time = sum(
        norm(subtract(right["position_enu_m"], left["position_enu_m"]))
        / max(float(right["speed_mps"]), EPSILON)
        for left, right in zip(route["waypoints"], route["waypoints"][1:])
    )
    capture_length = float(route["estimated_cost"]["capture_path_length_m"])
    route["estimated_cost"]["path_length_m"] = round(total_length, 6)
    route["estimated_cost"]["flight_time_s"] = round(total_time, 6)
    route["estimated_cost"]["transit_path_length_m"] = round(
        max(0.0, total_length - capture_length), 6
    )
    route["same_budget_control"]["flight_budget"] = {
        "photo_count": route["estimated_cost"]["photo_count"],
        "path_length_m": route["estimated_cost"]["path_length_m"],
        "flight_time_s": route["estimated_cost"]["flight_time_s"],
    }
    route["model_features"]["safe_connection_semantics"] = (
        "home_to_route_at_declared_safe_altitude_checked_against_2d_boundary_and_exclusions"
    )
    return route


def visibility_summary(
    cameras: list[list[float]], targets: list[list[float]], normal: list[float], region: dict, camera: dict
) -> dict:
    ranges = [norm(subtract(camera_point, target)) for camera_point, target in zip(cameras, targets, strict=True)]
    incidences = []
    for camera_point, target in zip(cameras, targets, strict=True):
        outward_view = normalized(subtract(camera_point, target), "surface-to-camera direction")
        cosine = max(-1.0, min(1.0, dot(normal, outward_view)))
        incidences.append(math.degrees(math.acos(cosine)))
    median_range = sorted(ranges)[len(ranges) // 2]
    footprint_u = 2.0 * median_range * math.tan(math.radians(float(camera["horizontal_fov_deg"])) / 2.0)
    footprint_v = 2.0 * median_range * math.tan(math.radians(float(camera["vertical_fov_deg"])) / 2.0)
    coverage = min(1.0, footprint_u / float(region["extent_u_m"])) * min(
        1.0, footprint_v / float(region["extent_v_m"])
    )
    return {
        "visible_surface_fraction": round(coverage, 6),
        "incidence_angle_p50_deg": round(sorted(incidences)[len(incidences) // 2], 6),
        "range_p50_m": round(median_range, 6),
        "occlusion_tested": False,
        "visibility_semantics": "fov_and_surface_orientation_only_not_collision_or_occlusion",
    }


def attach_acquisition_overlap_bridge(
    cameras: list[list[float]],
    targets: list[list[float]],
    region: dict,
    constraints: dict,
) -> tuple[list[list[float]], list[list[float]], dict]:
    """Ensure at least one capture direction overlaps an acquisition observation.

    Directions point from the target surface toward the observing camera.  When
    a proposed route is too novel, one bridge capture is inserted at the nearer
    end of the strip using the closest acquisition direction.  This is a
    geometry-only overlap gate; it does not claim feature matching succeeded.
    """
    references = [
        normalized([float(value) for value in direction], "acquisition view direction")
        for direction in region.get("acquisition_view_directions_enu", [])
    ]
    if not references:
        raise ValueError("non-water region requires acquisition view directions")
    threshold = float(constraints.get("maximum_acquisition_bridge_angle_deg", 35.0))

    def best_match() -> tuple[float, int]:
        best = (math.inf, -1)
        for index, (camera, target) in enumerate(zip(cameras, targets, strict=True)):
            direction = subtract(camera, target)
            delta = min(angle_between_deg(direction, reference) for reference in references)
            if delta < best[0]:
                best = (delta, index)
        return best

    minimum_delta, bridge_index = best_match()
    if minimum_delta > threshold + EPSILON:
        center = [float(value) for value in region["center_enu_m"]]
        standoff = float(region.get("preferred_standoff_m", max(region["extent_u_m"], 20.0)))
        # Pick the acquisition direction nearest either end of the proposed route.
        candidates = []
        for reference in references:
            bridge = add(center, scale(reference, standoff))
            bridge[2] = clamp_flight_z(center[2], bridge[2], constraints)
            candidates.append((min(norm(subtract(bridge, cameras[0])), norm(subtract(bridge, cameras[-1]))), bridge))
        _, bridge = min(candidates, key=lambda item: item[0])
        if norm(subtract(bridge, cameras[0])) <= norm(subtract(bridge, cameras[-1])):
            cameras = [bridge] + cameras
            targets = [center] + targets
        else:
            cameras = cameras + [bridge]
            targets = targets + [center]
        minimum_delta, bridge_index = best_match()
    if minimum_delta > threshold + EPSILON:
        raise ValueError(
            f"cannot preserve acquisition overlap: {minimum_delta:.2f} deg exceeds {threshold:.2f} deg"
        )
    return cameras, targets, {
        "verified": True,
        "reference_direction_count": len(references),
        "bridge_capture_indices": [bridge_index],
        "minimum_direction_delta_deg": round(minimum_delta, 6),
        "maximum_allowed_direction_delta_deg": threshold,
        "semantics": "candidate_capture_direction_bridges_acquisition_observation_of_same_surface",
    }


def route_record(
    route_id: str,
    region: dict,
    mode: str,
    cameras: list[list[float]],
    targets: list[list[float]],
    speed_mps: float,
    capture_view: str,
    normal: list[float],
    camera_profile: dict,
    constraints: dict,
    standoff_scale: float,
) -> dict:
    cameras, targets, acquisition_overlap = attach_acquisition_overlap_bridge(
        cameras, targets, region, constraints
    )
    speed_mps = cadence_safe_speed(cameras, speed_mps, camera_profile, constraints)
    target_surface_ids = region.get("target_surface_ids") or [region["region_id"]]
    capture_waypoints = make_waypoints(cameras, targets, speed_mps, capture_view, target_surface_ids)
    maximum_capture_pitch = float(constraints.get("maximum_capture_gimbal_pitch_deg", -5.0))
    if any(point["gimbal_pitch_deg"] > maximum_capture_pitch + EPSILON for point in capture_waypoints):
        raise ValueError("candidate contains a sky-dominant capture pitch")
    waypoints = insert_non_capture_turn_transitions(
        capture_waypoints, float(constraints.get("maximum_capture_segment_m", 30.0))
    )
    length_m = path_length(cameras)
    capture_intervals_s = [
        norm(subtract(right, left)) / speed_mps for left, right in zip(cameras, cameras[1:])
    ]
    return {
        "route_id": route_id,
        "target_region_ids": [region["region_id"]],
        "generator_reason": {
            "defect_type": region["defect_type"],
            "observation_mode": mode,
            "explanation": (
                f"geometry-only {mode.lower()} around {region['surface_class'].lower()} surface; "
                "no candidate image or GS metric was read"
            ),
        },
        "estimated_cost": {
            "path_length_m": round(length_m, 6),
            "flight_time_s": round(length_m / speed_mps, 6),
            "capture_path_length_m": round(length_m, 6),
            "capture_time_s": round(length_m / speed_mps, 6),
            "transit_path_length_m": 0.0,
            "photo_count": len(capture_waypoints),
            "trigger_interval_s": round(min(capture_intervals_s), 6),
        },
        "visibility_summary": visibility_summary(
            cameras, targets, normal, region, camera_profile
        ),
        "model_features": {
            "surface_confidence": float(region["confidence"]),
            "surface_area_m2": round(float(region["extent_u_m"]) * float(region["extent_v_m"]), 6),
            "route_surface_range_m": round(
                sum(norm(subtract(camera, target)) for camera, target in zip(cameras, targets, strict=True))
                / len(cameras),
                6,
            ),
            "standoff_scale": round(standoff_scale, 6),
            "relative_height_m": round(
                sum(camera[2] - target[2] for camera, target in zip(cameras, targets, strict=True))
                / len(cameras),
                6,
            ),
            "minimum_capture_interval_s": float(camera_profile["minimum_capture_interval_s"]),
            "achieved_min_capture_interval_s": round(min(capture_intervals_s), 6),
            "maximum_capture_gimbal_pitch_deg": maximum_capture_pitch,
            "occlusion_verified": False,
            "collision_safe": False,
        },
        "source_chunk_ids": region.get("source_chunk_ids") or [f"unassigned:{region['region_id']}"],
        "acquisition_overlap": acquisition_overlap,
        "same_budget_control": {
            "protocol": "SAME_BASE_SAME_RNG_SAME_STEPS_NO_CANDIDATE_RGB",
            "incremental_optimization_steps": int(
                constraints.get("counterfactual_extra_optimization_steps", 500)
            ),
            "same_base_checkpoint": True,
            "same_rng_state": True,
            "same_probe_manifest": True,
            "candidate_rgb_added_to_control": False,
            "flight_budget": {
                "photo_count": len(capture_waypoints),
                "path_length_m": round(length_m, 6),
                "flight_time_s": round(length_m / speed_mps, 6),
            },
        },
        "legality_checks": {
            "altitude_within_declared_limits": True,
            "capture_pitch_not_sky_dominant": True,
            "capture_cadence_valid": True,
            "operational_boundary_2d_valid": True,
            "exclusion_zones_2d_valid": True,
            "acquisition_overlap_valid": True,
            "three_dimensional_collision_verified": False,
        },
        "waypoints": waypoints,
    }


def generate_region_routes(payload: dict, region: dict, camera: dict, constraints: dict) -> list[dict]:
    if region["surface_class"] == "WATER":
        return []
    center = [float(value) for value in region["center_enu_m"]]
    normal = normalized([float(value) for value in region["normal_enu"]], "normal_enu")
    provided_tangent = [float(value) for value in region.get("tangent_enu", [1.0, 0.0, 0.0])]
    tangent = horizontal_unit(provided_tangent, cross([0.0, 0.0, 1.0], normal))
    perpendicular = normalized([-tangent[1], tangent[0], 0.0], "perpendicular direction")
    preferred_standoff = float(region.get("preferred_standoff_m", max(region["extent_u_m"], 20.0)))
    clearance = max(
        float(constraints["minimum_surface_clearance_m"]),
        float(region.get("height_margin_m", 0.0)),
    )
    modes = region.get("suggested_observation_modes") or default_modes(region)
    standoff_scales = [
        float(value) for value in constraints.get("candidate_standoff_scales", [0.85, 1.15])
    ]
    routes: list[dict] = []

    for mode in modes:
        if mode == "NADIR_STRIP":
            for scale_index, standoff_scale in enumerate(standoff_scales):
                requested_range = max(preferred_standoff * standoff_scale, clearance)
                flight_z = clamp_flight_z(center[2], center[2] + requested_range, constraints)
                range_m = flight_z - center[2]
                spacing_m, speed_mps = capture_spacing_and_speed(range_m, camera, constraints)
                for axis_index, axis in enumerate((tangent, perpendicular)):
                    camera_center = [center[0], center[1], flight_z]
                    length_m = max(float(region["extent_u_m"]), float(region["extent_v_m"])) * 1.2
                    camera_points = sample_line(camera_center, axis, length_m, spacing_m)
                    target_points = [[point[0], point[1], center[2]] for point in camera_points]
                    variant = f"nadir_axis{axis_index}_range{scale_index}"
                    routes.append(route_record(
                        route_id_for(payload, region, variant), region, mode, camera_points,
                        target_points, speed_mps, "NADIR", [0.0, 0.0, 1.0], camera,
                        constraints, standoff_scale,
                    ))
        elif mode == "OBLIQUE_STRIP":
            variants = (
                ("forward", perpendicular, -1.0, tangent, "FORWARD_OBLIQUE"),
                ("backward", perpendicular, 1.0, tangent, "BACKWARD_OBLIQUE"),
                ("left", tangent, -1.0, perpendicular, "LEFT_OBLIQUE"),
                ("right", tangent, 1.0, perpendicular, "RIGHT_OBLIQUE"),
            )
            for scale_index, standoff_scale in enumerate(standoff_scales):
                scaled_standoff = preferred_standoff * standoff_scale
                for variant_name, offset_axis, side, scan_axis, capture_view in variants:
                    horizontal_offset = 0.55 * scaled_standoff
                    vertical_offset = max(clearance, math.sqrt(max(scaled_standoff**2 - horizontal_offset**2, 0.0)))
                    flight_z = clamp_flight_z(center[2], center[2] + vertical_offset, constraints)
                    camera_center = add([center[0], center[1], flight_z], scale(offset_axis, side * horizontal_offset))
                    actual_range = norm(subtract(camera_center, center))
                    spacing_m, speed_mps = capture_spacing_and_speed(actual_range, camera, constraints)
                    length_m = max(float(region["extent_u_m"]), 8.0) * 1.2
                    camera_points = sample_line(camera_center, scan_axis, length_m, spacing_m)
                    target_points = [add(center, scale(scan_axis, dot(subtract(point, camera_center), scan_axis))) for point in camera_points]
                    variant = f"oblique_{variant_name}_range{scale_index}"
                    routes.append(route_record(
                        route_id_for(payload, region, variant), region, mode, camera_points,
                        target_points, speed_mps, capture_view, [0.0, 0.0, 1.0], camera,
                        constraints, standoff_scale,
                    ))
        elif mode == "FACADE_STRIP":
            facade_normal = horizontal_unit(normal, perpendicular)
            facade_tangent = horizontal_unit(provided_tangent, cross([0.0, 0.0, 1.0], facade_normal))
            for scale_index, standoff_scale in enumerate(standoff_scales):
                scaled_standoff = preferred_standoff * standoff_scale
                for height_index, height_fraction in enumerate((-0.25, 0.25)):
                    target_z = center[2] + height_fraction * float(region["extent_v_m"])
                    desired_camera_z = max(target_z, center[2] + clearance)
                    flight_z = clamp_flight_z(center[2], desired_camera_z, constraints)
                    camera_center = add([center[0], center[1], flight_z], scale(facade_normal, scaled_standoff))
                    range_m = norm(subtract(camera_center, [center[0], center[1], target_z]))
                    spacing_m, speed_mps = capture_spacing_and_speed(range_m, camera, constraints)
                    length_m = max(float(region["extent_u_m"]), 8.0) * 1.2
                    camera_points = sample_line(camera_center, facade_tangent, length_m, spacing_m)
                    target_points = [
                        add([center[0], center[1], target_z], scale(facade_tangent, dot(subtract(point, camera_center), facade_tangent)))
                        for point in camera_points
                    ]
                    variant = f"facade_height{height_index}_range{scale_index}"
                    routes.append(route_record(
                        route_id_for(payload, region, variant), region, mode, camera_points,
                        target_points, speed_mps, "LOCAL_OBLIQUE", facade_normal, camera,
                        constraints, standoff_scale,
                    ))
        elif mode == "LOCAL_ORBIT":
            base_angle_deg = math.degrees(math.atan2(normal[1], normal[0])) if norm(normal[:2] + [0.0]) > EPSILON else 0.0
            for scale_index, standoff_scale in enumerate(standoff_scales):
                range_m = preferred_standoff * standoff_scale
                spacing_m, speed_mps = capture_spacing_and_speed(range_m, camera, constraints)
                point_count = max(4, math.ceil(0.5 * math.pi * range_m / spacing_m) + 1)
                for arc_index, start_offset in enumerate((-45.0, 45.0)):
                    angles = [base_angle_deg + start_offset + 90.0 * index / (point_count - 1) for index in range(point_count)]
                    flight_z = clamp_flight_z(center[2], center[2] + clearance, constraints)
                    camera_points = [
                        [center[0] + range_m * math.cos(math.radians(angle)),
                         center[1] + range_m * math.sin(math.radians(angle)), flight_z]
                        for angle in angles
                    ]
                    target_points = [center[:] for _ in camera_points]
                    variant = f"orbit_arc{arc_index}_range{scale_index}"
                    routes.append(route_record(
                        route_id_for(payload, region, variant), region, mode, camera_points,
                        target_points, speed_mps, "LOCAL_OBLIQUE", normal, camera,
                        constraints, standoff_scale,
                    ))
        else:
            raise ValueError(f"unsupported observation mode {mode!r}")
    return routes


def route_allowed(route: dict, constraints: dict) -> tuple[bool, str | None]:
    boundary = constraints.get("operational_boundary_xy_m")
    exclusions = constraints.get("exclusion_zones", [])
    positions = [waypoint["position_enu_m"] for waypoint in route["waypoints"]]
    sampled_points = positions[:1]
    for start, end in zip(positions, positions[1:]):
        segment_count = max(1, math.ceil(norm(subtract(end, start)) / 2.0))
        sampled_points.extend([
            [start[axis] + (end[axis] - start[axis]) * index / segment_count for axis in range(3)]
            for index in range(1, segment_count + 1)
        ])
    for point in sampled_points:
        if boundary and not point_in_polygon(point, boundary):
            return False, "outside_operational_boundary"
        for zone in exclusions:
            if point_in_polygon(point, zone["polygon_xy_m"]):
                return False, f"inside_exclusion_zone:{zone['zone_id']}"
    return True, None


def generate(payload: dict, input_sha256: str) -> tuple[dict, dict]:
    camera = payload["camera_profile"]
    constraints = payload["flight_constraints"]
    routes = []
    rejected = []
    for region in payload["defect_regions"]:
        if region["surface_class"] == "WATER":
            rejected.append({"region_id": region["region_id"], "reason": "surface_class_water"})
            continue
        try:
            region_routes = generate_region_routes(payload, region, camera, constraints)
        except ValueError as error:
            rejected.append({"region_id": region["region_id"], "reason": str(error)})
            continue
        for route in region_routes:
            route = attach_home_transit(route, constraints)
            allowed, reason = route_allowed(route, constraints)
            if allowed:
                routes.append(route)
            else:
                rejected.append({"route_id": route["route_id"], "reason": reason})
    if not routes:
        raise ValueError("no candidate route survives geometry and exclusion constraints")
    config_material = {
        "generator_name": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "camera_profile": camera,
        "flight_constraints": constraints,
    }
    config_sha256 = hashlib.sha256(
        json.dumps(config_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "schema_version": 2,
        "scene_id": payload["scene_id"],
        "episode_id": payload["episode_id"],
        "coordinate_frame": payload["coordinate_frame"],
        "camera_profile_id": camera["camera_profile_id"],
        "input_manifest_sha256": input_sha256,
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
            "config_sha256": config_sha256,
        },
        "routes": routes,
    }
    report = {
        "schema_version": 2,
        "scene_id": payload["scene_id"],
        "episode_id": payload["episode_id"],
        "input_manifest_sha256": input_sha256,
        "route_count": len(routes),
        "region_count": len(payload["defect_regions"]),
        "rejected": rejected,
        "safety_statement": (
            "Routes pass only altitude, boundary, exclusion-zone, cadence, and continuity checks; "
            "they are not collision-safe until an approved obstacle source verifies them."
        ),
        "online_input_policy": "geometry_only_no_future_rgb_no_probe_no_gs_render_no_ground_truth_depth",
    }
    return result, report


def validate_json(payload: dict, schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("defect_regions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--input-schema",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "schemas" / "defect_regions.schema.json",
    )
    parser.add_argument(
        "--output-schema",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "schemas" / "candidate_routes.schema.json",
    )
    args = parser.parse_args()
    raw = args.defect_regions.read_bytes()
    payload = json.loads(raw)
    validate_json(payload, args.input_schema)
    result, report = generate(payload, hashlib.sha256(raw).hexdigest())
    validate_json(result, args.output_schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path = args.report or args.output.with_name(args.output.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "output": str(args.output),
        "report": str(report_path),
        "route_count": len(result["routes"]),
        "rejected_count": len(report["rejected"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
