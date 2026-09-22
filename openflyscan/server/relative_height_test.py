"""Explicit reconstruction-only vertical frame; never an ASL or flight mode."""
import json
import math
from pathlib import Path

from PIL import Image

MODE = "relative_height_test"
ABSOLUTE_MODE = "absolute_asl"
FRAME = "local_east_north_relative_to_takeoff_m_TEST_ONLY"
WARNING = (
    "RECONSTRUCTION TEST ONLY: ASL is unknown; Z is relative to each takeoff. "
    "Battery-change/takeoff offsets are not verified. No flight mission export."
)
ALLOWED_ARTIFACTS = {
    "point_cloud.ply", "cloud.bin", "cloud_meta.json", "viewer.json",
    "coordinate_reference.json",
}


def is_relative_test(state):
    return state.get("config", {}).get("altitude_mode") == MODE


def mode_config(payload):
    mode = payload.get("altitude_mode", ABSOLUTE_MODE)
    if mode not in {MODE, ABSOLUTE_MODE}:
        raise ValueError("unsupported altitude_mode")
    if mode == MODE:
        if payload.get("acknowledge_no_flight") is not True:
            raise ValueError("relative_height_test requires acknowledge_no_flight=true")
        if payload.get("takeoff_absolute_altitude_m") is not None:
            raise ValueError("relative_height_test must not supply a takeoff ASL")
    return {"altitude_mode": mode, "test_only": mode == MODE,
            "flight_export_allowed": mode != MODE}


def finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_openfly_telemetry(path: Path):
    with Image.open(path) as image:
        try:
            comment = image.getexif().get_ifd(34665).get(37510)
        except (AttributeError, KeyError):
            return {}
    if isinstance(comment, bytes):
        if comment.startswith(b"ASCII\x00\x00\x00"):
            comment = comment[8:].decode("utf-8", "replace")
        else:
            return {}
    if not isinstance(comment, str):
        return {}
    try:
        data = json.loads(comment)
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict) or data.get("source") != "dji_video_downlink":
        return {}
    result = {"pose_metadata_source": "openfly_exif_user_comment"}
    for source, target in [
        ("relative_altitude_m", "RelativeAltitude"),
        ("gimbal_pitch_deg", "GimbalPitchDegree"),
        ("yaw_deg", "FlightYawDegree"),
    ]:
        value = finite(data.get(source))
        if value is not None:
            result[target] = value
    return result


def test_contract():
    return {
        "test_only": True, "flight_export_allowed": False,
        "safe_to_execute": False, "altitude_mode": MODE,
        "coordinate_frame": FRAME, "absolute_altitude_available": False,
        "warning": WARNING,
        "task_union": "disabled_reconstruction_only_no_flight",
        "localization": "horizontal GPS with relative-to-takeoff Z, not ASL",
        "geometry": "reconstruction-only Pi3X point cloud",
        "quality_predictor": "not_run_reconstruction_only",
    }
