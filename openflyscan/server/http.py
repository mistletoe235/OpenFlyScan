from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mimetypes
import os
import re
import secrets
import traceback
import time
from pathlib import Path

from aiohttp import web
from PIL import ExifTags, Image

from .pipeline import (
    PipelineCancelled, atomic_json, cancel_session_processes, dji_xmp_telemetry,
    load_state, mutate_state, process_session, update_state, workstation_config, interrupt_session_processes,
)
from .relative_height_test import (
    ALLOWED_ARTIFACTS, FRAME, finite, is_relative_test, mode_config,
    read_openfly_telemetry, test_contract,
)


from openflyscan.missions.export import mission_mode
from openflyscan.reconstruction.cameras import upload_metadata


SESSION_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{5,63}$")
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ROLLING_PREVIEW_MIN_IMAGES = 30


def safe_filename(value: str) -> str:
    name = Path(value or "image.jpg").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "image"
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        suffix = ".jpg"
    return stem[:100] + suffix


def rational(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(value[0]) / float(value[1])


def degrees(value) -> float:
    parts = list(value)
    return rational(parts[0]) + rational(parts[1]) / 60.0 + rational(parts[2]) / 3600.0


def exif_byte(value, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, (bytes, bytearray)):
        return int.from_bytes(value, byteorder="big", signed=False) if value else default
    return int(value)


def image_metadata(path: Path) -> dict:
    result = {"latitude": None, "longitude": None, "altitude_m": None, "timestamp": None}
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        exif = image.getexif()
        result["width"] = image.width
        result["height"] = image.height
        for key in (36867, 36868, 306):
            if exif.get(key):
                result["timestamp"] = str(exif.get(key))
                break
        if result["timestamp"] is None:
            try:
                exif_ifd = exif.get_ifd(34665)
            except (AttributeError, KeyError):
                exif_ifd = {}
            for key in (36867, 36868):
                if exif_ifd.get(key):
                    result["timestamp"] = str(exif_ifd.get(key))
                    break
        try:
            gps = exif.get_ifd(34853)
        except (AttributeError, KeyError):
            gps = {}
        if gps.get(2) and gps.get(4):
            latitude = degrees(gps[2])
            longitude = degrees(gps[4])
            if str(gps.get(1, "N")).upper().startswith("S"):
                latitude = -latitude
            if str(gps.get(3, "E")).upper().startswith("W"):
                longitude = -longitude
            result["latitude"] = latitude
            result["longitude"] = longitude
        if gps.get(6) is not None:
            altitude = rational(gps[6])
            if exif_byte(gps.get(5), default=0) == 1:
                altitude = -altitude
            result["altitude_m"] = altitude
    result.update(dji_xmp_telemetry(path))
    openfly = read_openfly_telemetry(path)
    if openfly:
        result.update(openfly)
        result["relative_altitude_m"] = openfly.get("RelativeAltitude")
    return result


def header_float(request: web.Request, name: str, default):
    value = request.headers.get(name)
    if value is None or not value.strip():
        return default
    number = float(value)
    if not (-1.0e7 < number < 1.0e7):
        raise web.HTTPBadRequest(text=f"invalid {name}")
    return number


@web.middleware
async def auth_middleware(request: web.Request, handler):
    if request.path in {"/", "/health"} or request.path.startswith("/static/"):
        return await handler(request)
    supplied = request.headers.get("Authorization", "")
    if supplied.startswith("Bearer "):
        supplied = supplied[7:]
    else:
        supplied = request.query.get("token", "")
    if not secrets.compare_digest(supplied, request.app["access_token"]):
        raise web.HTTPUnauthorized(text="access code required")
    return await handler(request)


def session_dir(request: web.Request) -> Path:
    session_id = request.match_info["session_id"]
    if not SESSION_ID.fullmatch(session_id):
        raise web.HTTPBadRequest(text="invalid session id")
    path = request.app["data_root"] / "sessions" / session_id
    if not (path / "state.json").is_file():
        raise web.HTTPNotFound(text="session not found")
    return path


async def health(request):
    worker = request.app.get("worker_task")
    running = worker is not None and not worker.done()
    queue = request.app.get("job_queue")
    return web.json_response(dict(ok=running, service="openflyscan-workstation", api_version=1,
        pipeline="pi3x_quality_directional", supported_mission_schemas=[13, 14],
        worker_running=running, worker_error=None if running else "worker unavailable",
        queued_jobs=queue.qsize() if queue else 0, running_jobs=len(request.app["running_jobs"]),
        time=int(time.time())))


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(request.app["static_root"] / "index.html")


async def viewer(request: web.Request) -> web.FileResponse:
    session_dir(request)
    return web.FileResponse(request.app["static_root"] / "viewer.html")


async def create_session(request: web.Request) -> web.Response:
    payload = await request.json()
    try:
        altitude_config = {**mode_config(payload), **mission_mode(payload)}
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error))
    name = str(payload.get("name") or "手机航拍任务").strip()[:80]
    session_id = time.strftime("s%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    path = request.app["data_root"] / "sessions" / session_id
    (path / "images").mkdir(parents=True)
    (path / "logs").mkdir()
    (path / "artifacts").mkdir()
    config = {
        **altitude_config,
        "horizontal_fov_deg": float(payload.get("horizontal_fov_deg", 73.7)),
        "takeoff_absolute_altitude_m": payload.get("takeoff_absolute_altitude_m"),
        "camera_model": str(payload.get("camera_model") or "DJI/phone RGB")[:80],
        "pipeline": "pi3x_quality_predictor",
        "max_relative_altitude_m": float(payload.get("max_relative_altitude_m", 80)),
        "minimum_capture_interval_s": float(payload.get("minimum_capture_interval_s", 2)),
        "speed_mps": float(payload.get("speed_mps", 4)),
        "auto_preview": bool(payload.get("auto_preview", True)),
        "maximum_tasks": max(1, min(30, int(payload.get("maximum_tasks", 10)))),
    }
    if config["takeoff_absolute_altitude_m"] is not None:
        config["takeoff_absolute_altitude_m"] = float(config["takeoff_absolute_altitude_m"])
    if not 10.0 <= config["horizontal_fov_deg"] <= 150.0:
        raise web.HTTPBadRequest(text="horizontal_fov_deg must be in [10, 150]")
    import math
    for key, lower, upper in (("max_relative_altitude_m", 5, 120), ("minimum_capture_interval_s", .5, 60), ("speed_mps", .1, 15)):
        if not math.isfinite(config[key]) or not lower <= config[key] <= upper:
            raise web.HTTPBadRequest(text="invalid " + key)
    if config["takeoff_absolute_altitude_m"] is not None and not math.isfinite(config["takeoff_absolute_altitude_m"]):
        raise web.HTTPBadRequest(text="takeoff altitude must be finite")
    now = int(time.time() * 1000)
    state = {
        "id": session_id, "name": name, "created_at_epoch_ms": now,
        "updated_at_epoch_ms": now, "phase": "receiving", "progress": 0.0,
        "message": "等待手机上传照片", "images": [], "image_count": 0,
        "sealed": False, "running": False, "completed": False,
        "cancelled": False,
        "preview_ready": False, "config": config, "artifacts": {}, "error": None,
        "fast_sfm_completed_images": 0, "fast_sfm_target_images": 0,
        "fast_sfm_resume_from": 0,
        "sfm_lane_phase": "idle", "sfm_lane_progress": 0.0,
        "sfm_lane_message": "Waiting for Pi3X observations",
        "scal3r_lane_phase": "idle", "scal3r_lane_progress": 0.0,
        "scal3r_lane_message": "等待首个冻结窗口",
        "contract": {
            "pipeline": "Pi3X + Quality Predictor + directional strip planner",
            "localization": "Pi3X local geometry aligned to uploaded GPS",
            "gs_or_gt_used": False,
            "mission_schema_version": config["mission_schema_version"],
            "recapture_flight_mode": config["recapture_flight_mode"],
            "safe_to_execute": False,
        },
    }
    if is_relative_test({"config": config}):
        state["contract"].update(test_contract())
        state["name"] = "[RELATIVE TEST / NO FLIGHT] " + name
    atomic_json(path / "state.json", state)
    return web.json_response(state, status=201)


async def list_sessions(request: web.Request) -> web.Response:
    rows = []
    for path in sorted((request.app["data_root"] / "sessions").glob("*/state.json"), reverse=True):
        try:
            state = json.loads(path.read_text())
            rows.append({key: state.get(key) for key in (
                "id", "name", "phase", "progress", "message", "image_count",
                "sealed", "running", "completed", "cancelled", "preview_ready", "artifacts", "error",
                "fast_sfm_completed_images", "fast_sfm_target_images", "fast_sfm_resume_from",
                "sfm_lane_phase", "sfm_lane_progress", "sfm_lane_message",
                "scal3r_lane_phase", "scal3r_lane_progress", "scal3r_lane_message",
                "scal3r_lane_snapshot_images", "scal3r_lane_target_windows",
                "scal3r_lane_completed_windows", "scal3r_lane_cache_hits",
                "scal3r_lane_cache_misses",
            )})
        except Exception:
            continue
    return web.json_response(rows[:100])


async def get_session(request: web.Request) -> web.Response:
    return web.json_response(load_state(session_dir(request)))


async def result(request):
    path = session_dir(request)
    state = load_state(path)
    artifacts = state.get("artifacts", {})
    relative_test = is_relative_test(state)
    viewer_path = path / "artifacts/viewer.json"
    viewer = json.loads(viewer_path.read_text()) if viewer_path.exists() else {}
    mission_name = Path(artifacts.get("mission") or "mission.json").name
    mission_path = path / "artifacts" / mission_name
    version = None
    if artifacts.get("mission") and mission_path.exists() and not relative_test:
        version = json.loads(mission_path.read_text())["schema_version"]
    return web.json_response(dict(session_id=state["id"], test_only=relative_test,
        flight_export_allowed=not relative_test, phase=state.get("phase"), completed=bool(state.get("completed")),
        message=state.get("message"), geometry_kind=viewer.get("geometry_kind"),
        point_cloud=dict(url=artifacts["point_cloud_ply"], format="ply", encoding="binary_little_endian",
                         coordinate_frame=FRAME if relative_test else "local_metric_xyz_aligned_to_uploaded_gps")
                    if artifacts.get("point_cloud_ply") else None,
        candidates=dict(url=artifacts.get("viewer_data"), count=len(viewer.get("candidates", [])),
                        semantics="Quality Predictor regional error"),
        quality_predictor=dict(url=artifacts.get("predictions")),
        surface_tasks=dict(url=artifacts.get("surface_tasks")),
        openfly_v5_mission=dict(url=artifacts["mission"], schema_version=version, safe_to_execute=False)
                          if version is not None else None,
        contract=state.get("contract", {})))


async def upload_image(request: web.Request) -> web.Response:
    path = session_dir(request)
    state = load_state(path)
    if state.get("cancelled"):
        raise web.HTTPConflict(text="session is cancelled")
    if state.get("sealed"):
        raise web.HTTPConflict(text="session is finalized")
    sequence = int(request.match_info["sequence"])
    if not 0 <= sequence <= 999_999:
        raise web.HTTPBadRequest(text="invalid sequence")
    original_name = safe_filename(request.headers.get("X-Filename", f"image_{sequence:06d}.jpg"))
    stored_name = f"{sequence:06d}_{original_name}"
    target = path / "images" / stored_name
    temporary = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    with temporary.open("wb") as handle:
        async for chunk in request.content.iter_chunked(1024 * 1024):
            size += len(chunk)
            if size > 48 * 1024 * 1024:
                temporary.unlink(missing_ok=True)
                raise web.HTTPRequestEntityTooLarge(max_size=48 * 1024 * 1024, actual_size=size)
            digest.update(chunk)
            handle.write(chunk)
    if size < 128:
        temporary.unlink(missing_ok=True)
        raise web.HTTPBadRequest(text="image body is empty")
    try:
        metadata = image_metadata(temporary)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise web.HTTPBadRequest(text=f"invalid image: {error}")
    metadata["latitude"] = header_float(request, "X-Latitude", metadata["latitude"])
    metadata["longitude"] = header_float(request, "X-Longitude", metadata["longitude"])
    metadata["altitude_m"] = header_float(request, "X-Altitude", metadata["altitude_m"])
    if is_relative_test(state):
        if request.headers.get("X-Altitude") is not None:
            temporary.unlink(missing_ok=True)
            raise web.HTTPBadRequest(text="relative_height_test forbids X-Altitude; ASL must remain unknown")
        if any(finite(metadata.get(key)) is None for key in ("latitude", "longitude", "relative_altitude_m")):
            temporary.unlink(missing_ok=True)
            raise web.HTTPBadRequest(text="relative_height_test requires real GPS and OpenFly EXIF relative altitude")
        metadata["altitude_m"] = None
        metadata["coordinate_frame"] = FRAME
    supplied_altitude_source = request.headers.get("X-Altitude-Source", "").strip().lower()
    if supplied_altitude_source not in {"", "aircraft_asl", "takeoff_asl_plus_relative"}:
        temporary.unlink(missing_ok=True)
        raise web.HTTPBadRequest(text="unsupported altitude source")
    metadata["timestamp"] = request.headers.get("X-Timestamp") or metadata["timestamp"]
    try:
        metadata = upload_metadata(request.headers, metadata)
    except (ValueError, TypeError) as error:
        temporary.unlink(missing_ok=True)
        raise web.HTTPBadRequest(text=str(error))
    supplied_capture_view = request.headers.get("X-Capture-View")
    capture_view = str(
        metadata.get("capture_view") or supplied_capture_view or "NADIR"
    ).upper()
    if capture_view not in {
        "NADIR", "NORTH_OBLIQUE", "EAST_OBLIQUE", "SOUTH_OBLIQUE", "WEST_OBLIQUE",
        "FORWARD_OBLIQUE", "BACKWARD_OBLIQUE", "LEFT_OBLIQUE", "RIGHT_OBLIQUE",
    }:
        temporary.unlink(missing_ok=True)
        raise web.HTTPBadRequest(text="unsupported capture view")
    sha256 = digest.hexdigest()
    existing = next((row for row in state["images"] if row["sha256"] == sha256), None)
    if existing:
        temporary.unlink(missing_ok=True)
        return web.json_response({"duplicate": True, "image": existing})
    temporary.replace(target)
    row = {
        "sequence": sequence, "filename": original_name, "stored_name": stored_name,
        "bytes": size, "sha256": sha256, "capture_view": capture_view,
        "capture_view_source": (
            "dji_xmp" if metadata.get("capture_view")
            else ("http_header" if supplied_capture_view else "default_nadir")
        ),
        "gps_source": ("http_camera_telemetry" if request.headers.get("X-Latitude") else "image_exif"),
        "altitude_source": ("openfly_relative_height_TEST_ONLY" if is_relative_test(state)
                            else supplied_altitude_source or "image_exif"),
        **metadata,
    }

    def add_image(current: dict) -> None:
        replaced = [item for item in current["images"] if int(item["sequence"]) == sequence]
        for item in replaced:
            # The new file has already atomically replaced the old file when the
            # sequence and sanitized filename are identical.  Do not delete it.
            if item["stored_name"] != stored_name:
                (path / "images" / item["stored_name"]).unlink(missing_ok=True)
        current["images"] = sorted(
            [item for item in current["images"] if int(item["sequence"]) != sequence] + [row],
            key=lambda item: int(item["sequence"]),
        )
        current["image_count"] = len(current["images"])
        current["completed"] = False
        if not current.get("running"):
            current["phase"] = "receiving"
            current["message"] = f"已接收 {current['image_count']} 张照片"
    state = mutate_state(path, add_image)
    # Start as soon as a minimally solvable overlapping prefix exists. While a
    # lane is running, schedule() coalesces subsequent uploads into one pass on
    # the newest prefix, so this remains incremental without launching one SfM
    # process per incoming frame.
    if (state["config"].get("auto_preview") and state["image_count"] >= ROLLING_PREVIEW_MIN_IMAGES
            and state["image_count"] - state.get("last_preview_images", 0) >= 26):
        await schedule(request.app, path.name, final=False)
    return web.json_response({"duplicate": False, "image": row, "image_count": state["image_count"]}, status=201)


async def finalize(request: web.Request) -> web.Response:
    path = session_dir(request)
    if load_state(path).get("cancelled"):
        raise web.HTTPConflict(text="session is cancelled")
    state = update_state(path, sealed=True, completed=False, phase="queued", message="已结束上传，等待完整处理")
    await schedule(request.app, path.name, final=True)
    return web.json_response(state, status=202)


async def retry(request: web.Request) -> web.Response:
    path = session_dir(request)
    state = load_state(path)
    if state.get("cancelled"):
        raise web.HTTPConflict(text="session is cancelled")
    final = bool(state.get("sealed"))
    update_state(path, completed=False, error=None, phase="queued", message="等待重新处理")
    await schedule(request.app, path.name, final=final)
    return web.json_response(load_state(path), status=202)


async def cancel(request: web.Request) -> web.Response:
    """Permanently stop remote work for a session while preserving its files."""
    path = session_dir(request)
    state = load_state(path)
    if state.get("cancelled"):
        return web.json_response(state)

    session_id = path.name
    request.app["desired_jobs"].pop(session_id, None)
    request.app["desired_scal3r_jobs"].pop(session_id, None)
    request.app["queued_jobs"].discard(session_id)
    request.app["queued_scal3r_jobs"].discard(session_id)
    terminated = cancel_session_processes(path)
    state = update_state(
        path,
        cancelled=True,
        sealed=True,
        running=False,
        completed=False,
        phase="cancelled",
        progress=float(state.get("progress", 0.0)),
        message="任务已由手机终止；远端处理已停止",
        error=None,
        cancelled_at_epoch_ms=int(time.time() * 1000),
        terminated_processes=terminated,
        sfm_lane_phase="cancelled",
        sfm_lane_message="已终止",
        scal3r_lane_phase="cancelled",
        scal3r_lane_message="已终止",
    )
    return web.json_response(state)


async def events(request: web.Request) -> web.StreamResponse:
    path = session_dir(request)
    response = web.StreamResponse(headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "Connection": "keep-alive", "X-Accel-Buffering": "no",
    })
    await response.prepare(request)
    previous = None
    try:
        while True:
            state = load_state(path)
            serialized = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
            if serialized != previous:
                await response.write(f"data: {serialized}\n\n".encode())
                previous = serialized
            else:
                await response.write(b": keepalive\n\n")
            await asyncio.sleep(1.0)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return response


async def artifact(request: web.Request) -> web.StreamResponse:
    path = session_dir(request)
    name = Path(request.match_info["name"]).name
    if name != request.match_info["name"]:
        raise web.HTTPBadRequest(text="invalid artifact path")
    if is_relative_test(load_state(path)) and name not in ALLOWED_ARTIFACTS:
        raise web.HTTPForbidden(text="relative-height reconstruction test: flight/task artifacts are disabled")
    target = path / "artifacts" / name
    if not target.is_file():
        raise web.HTTPNotFound(text="artifact not ready")
    content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    response = web.FileResponse(target, headers={"Cache-Control": "no-store"})
    response.content_type = content_type
    if name.endswith("mission-schema13.json"):
        response.headers["Content-Disposition"] = f'attachment; filename="{path.name}-openfly-v5-schema13.json"'
    return response


async def schedule(app: web.Application, session_id: str, final: bool) -> None:
    if load_state(app["data_root"] / "sessions" / session_id).get("cancelled"):
        return
    desired = app["desired_jobs"]
    desired[session_id] = bool(final or desired.get(session_id, False))
    if session_id not in app["queued_jobs"] and session_id not in app["running_jobs"]:
        app["queued_jobs"].add(session_id)
        await app["job_queue"].put(session_id)




def refresh_running_state(app, session_id):
    path = app["data_root"] / "sessions" / session_id
    active = session_id in app["running_jobs"] or session_id in app["queued_jobs"]
    state = load_state(path)
    if not state.get("cancelled"):
        update_state(path, running=active)


async def job_worker(app):
    while True:
        session_id = await app["job_queue"].get()
        app["queued_jobs"].discard(session_id)
        final = bool(app["desired_jobs"].pop(session_id, False))
        path = app["data_root"] / "sessions" / session_id
        try:
            if not load_state(path).get("cancelled"):
                app["running_jobs"].add(session_id)
                await asyncio.to_thread(process_session, path, final)
        except PipelineCancelled:
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            traceback.print_exc()
        finally:
            app["running_jobs"].discard(session_id)
            app["job_queue"].task_done()
        if session_id in app["desired_jobs"] and not load_state(path).get("cancelled"):
            app["queued_jobs"].add(session_id)
            await app["job_queue"].put(session_id)
        refresh_running_state(app, session_id)




async def start_background(app):
    app["job_queue"] = asyncio.Queue()
    app["worker_task"] = asyncio.create_task(job_worker(app))
    for state_path in (app["data_root"] / "sessions").glob("*/state.json"):
        try:
            state = json.loads(state_path.read_text())
            if state.get("running"):
                update_state(state_path.parent, running=False, phase="interrupted", message="Service restarted; retry processing")
        except (OSError, ValueError):
            traceback.print_exc()


async def stop_background(app):
    for session_id in list(app["running_jobs"]):
        interrupt_session_processes(app["data_root"] / "sessions" / session_id)
    app["worker_task"].cancel()
    await asyncio.gather(app["worker_task"], return_exceptions=True)


def build_app(data_root: Path, static_root: Path, access_token: str) -> web.Application:
    app = web.Application(middlewares=[auth_middleware], client_max_size=50 * 1024 * 1024)
    app.update({
        "data_root": data_root, "static_root": static_root, "access_token": access_token,
        "job_queue": None, "desired_jobs": {}, "queued_jobs": set(), "running_jobs": set(),
        "scal3r_job_queue": None, "desired_scal3r_jobs": {},
        "queued_scal3r_jobs": set(), "running_scal3r_jobs": set(),
    })
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_static("/static/", static_root)
    app.router.add_get("/s/{session_id}/viewer", viewer)
    app.router.add_post("/api/sessions", create_session)
    app.router.add_get("/api/sessions", list_sessions)
    app.router.add_get("/api/sessions/{session_id}", get_session)
    app.router.add_get("/api/sessions/{session_id}/result", result)
    app.router.add_put("/api/sessions/{session_id}/images/{sequence}", upload_image)
    app.router.add_post("/api/sessions/{session_id}/finalize", finalize)
    app.router.add_post("/api/sessions/{session_id}/retry", retry)
    app.router.add_post("/api/sessions/{session_id}/cancel", cancel)
    app.router.add_get("/api/sessions/{session_id}/events", events)
    app.router.add_get("/api/sessions/{session_id}/artifacts/{name}", artifact)
    app.on_startup.append(start_background)
    app.on_cleanup.append(stop_background)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=58080)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--static-root", type=Path, default=Path(__file__).parent / "static")
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    (args.data_root / "sessions").mkdir(exist_ok=True)
    workstation_config()
    token_file = args.data_root / "access_token"
    if token_file.exists():
        token = token_file.read_text().strip()
    else:
        token = secrets.token_urlsafe(18)
        token_file.write_text(token + "\n")
        token_file.chmod(0o600)
    web.run_app(
        build_app(args.data_root, args.static_root, token),
        host=args.host, port=args.port, access_log=None,
    )


if __name__ == "__main__":
    main()
