# Workstation service

The workstation service belongs in this repository alongside model training
and evaluation. It receives mobile images, reconstructs local geometry, runs
the Quality Predictor, and returns point clouds and reacquisition missions.
Its source is part of the `openflyscan` package, not a separate service repository.
The API retains existing mobile session paths while the worker uses Pi3X,
the shared Quality Predictor and directional whole-strip planning.

## Layout

| Module | Responsibility |
| --- | --- |
| `openflyscan/server/` | Authentication, sessions, uploads, job queue, progress and artifact delivery |
| `openflyscan/reconstruction/` | Camera preparation, Pi3X spatial groups, GPS-aligned local geometry and preview PLY |
| `openflyscan/quality_predictor/` | Shared model implementation for training and deployed inference; already imported |
| `openflyscan/planning/` | Target grouping, directional candidate strips, coverage selection and route ordering |
| `openflyscan/missions/` | Coordinates, takeoff-relative heights, mobile mission serialization and validation |

Runtime data, credentials, large backbone weights and scene caches remain outside
source Git; the released head is included at `weights/quality_predictor.pt`.
Initial five-direction coverage planning currently exists in the mobile
`SurveyPlanner`; do not duplicate it on the server merely to rename the system.
Server-generated reacquisition missions use the shared mobile mission contract.

## Run the service

Install the main repository with `python -m pip install -e '.[server]'`.
Use the Pi3X/GeoFF3D environment for inference; the server installation alone
does not install that external backbone. Copy `configs/workstation.example.json`
to a private runtime directory and set the backbone/source/checkpoint paths.
An optional `worker_python` selects a separate prepared inference interpreter.

```bash
export OPENFLYSCAN_CONFIG=/data/openflyscan/workstation.json
openflyscan-server --host 127.0.0.1 --port 55000 --data-root /data/openflyscan/service
```

The first launch creates a private `access_token` file under the data root.
Requests use `Authorization: Bearer <token>`. Keep that file outside Git and put
HTTPS in front of an externally accessible deployment. `/health` reports
`openflyscan-workstation`, pipeline identity, supported mission schemas and jobs.

The example binds to the workstation loopback interface. A phone must use a
reachable LAN/VPN address or HTTPS reverse proxy, not `127.0.0.1` or an SSH hostname
alone. Configure the listener/proxy and firewall accordingly without exposing an
unauthenticated service. Use the same root URL and token in the mobile cloud page.
iOS upload controls are documented in the client's `docs/CLOUD_UPLOAD.md`.

## Mobile API

- `POST /api/sessions`: create a session with camera FOV, optional takeoff
  absolute altitude, target count and mission capabilities.
- `PUT /api/sessions/{id}/images/{sequence}`: upload JPEG/PNG; existing GPS,
  timestamp, capture-view and altitude headers remain supported.
- `POST /api/sessions/{id}/finalize`: seal uploads and run quality/planning.
- `GET /api/sessions/{id}` and `/events`: state and progress updates.
- `GET /api/sessions/{id}/result` and `/artifacts/{name}`: PLY, regional output
  and mission files. `/retry` and `/cancel` retain their existing POST semantics.

Declare continuous export explicitly when creating a compatible Android V4/V5 or iOS session:

```json
{
  "name": "Aerial survey",
  "horizontal_fov_deg": 73.7,
  "takeoff_absolute_altitude_m": 25.0,
  "maximum_tasks": 12,
  "supported_mission_schemas": [13, 14],
  "recapture_flight_mode": "CONTINUOUS_EXPERIMENTAL"
}
```

Clients without capability fields retain schema 13 stop-and-capture behavior.
The takeoff altitude and FOV above are examples, not deployment defaults or camera
calibration. FOV must match uploaded image geometry, including video crop/resizing.
Without a supplied takeoff datum, the service returns quality/geometry but no
flight mission. Relative-height test sessions remain point-cloud-only.

### Camera metadata

Uploads additionally accept `X-Camera-Yaw`, `X-Camera-Pitch`, `X-Camera-Roll`
(degrees; DJI camera attitude in ENU), or `X-OpenFly-Metadata` containing:

```json
{
  "schema_version": 1,
  "camera_ypr_deg": [90.0, -45.0, 0.0],
  "intrinsics": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
  "intrinsics_image_size": [1920, 1080]
}
```

Intrinsics are scaled from the declared width/height to the uploaded image.
DJI XMP and OpenFly downlink EXIF telemetry are also read. When full camera
attitudes are unavailable, Pi3X pose conditioning is disabled for that batch;
predicted geometry is still aligned to uploaded GPS. A capture-view label alone
does not become a measured pose. Camera intrinsics fall back to the declared
FOV when a calibrated matrix is unavailable.

Uploads and job snapshots are isolated. Geometry previews update before final
quality scoring; final scoring uses the training feature assembly, including
source-patch recovery and the explicit geometric evidence features. Final
planning reuses the Expo West directional coverage implementation. The later
simulation-only surface-append experiment is not substituted for this planner.

## Deployment — September 21, 2026

The dsw1 service on port **55000** now runs `openflyscan.server.http` from
`/mnt/petrelfs/youzhongrui/mt/gs/OpenFlyScan`. The existing service address,
access token and four historical sessions are preserved. Runtime configuration,
dependencies and logs are under
`/mnt/petrelfs/youzhongrui/mt/gs/openflyscan-runtime`; session data stays at its
original location. The private pre-upgrade backup is under
`deployment_backups/openflyscan_20260921` on the same host.

Before switching, a 30-image Expo West replay exercised the actual upload,
Pi3X, Quality Predictor and mission-export path on a separate port. Native
Android V5 decoding and validation passed for schema 14 exports. After switching,
health, historical PLY delivery and schema 13/14 session creation/cancellation
passed. See [validation](validation.md) for the scope of these checks.

Continuous mode requires the explicit schema 14 capability fields shown above;
existing clients retain schema 13. No mobile application was changed by this
deployment. All server source is in this main repository.

## Pre-upgrade audit — September 21, 2026

The previous `streaming_service.server` on dsw1 listened on port 55000. Its health
response identifies `v86-streaming-active-mapping`, with reconstruction and
Scal3R workers active and no jobs queued or running at the time of inspection.
Its source used SfM + Scal3R, the older V50/V78 processing chain and a
joblib quality head, rather than the paper's Pi3X + Quality Predictor pipeline.
The old mission compiler and result descriptor both declared schema 13.

The upgraded service preserves session/upload paths, cancellation, progress,
PLY delivery and mission download. Algorithm caches live in a new versioned
namespace and cannot be mistaken for the old SfM/Scal3R results.

## Migration rationale (historical audit)

1. **Capture metadata.** The inspected Android upload interface explicitly
   sends image identity, GPS, altitude/source, timestamp and capture-view label.
   It does not explicitly transmit full aircraft/gimbal orientation or camera
   intrinsics. Define a versioned metadata record including their coordinate,
   time, unit and image-resize conventions. A view label is not a full pose.
   Existing JPEG metadata may be useful but must not be assumed present in
   decoded video frames. Mark missing priors explicitly.
2. **Pi3X geometry.** Port the camera preparation and spatial grouping used by
   the final Expo West pipeline. Preserve image/pixel associations and local-to-
   global transforms when assembling previews and predictor inputs. Separate
   progressive preview updates from finalized quality/planning jobs.
3. **Quality prediction.** Load the paper checkpoint through the shared
   `QualityPredictor`; reproduce its feature assembly, normalization, regional
   scores and target selection before enabling online planning.
4. **Reacquisition planning.** Port the Expo West directional whole-strip
   coverage planner and continuous trajectory construction. Keep the later
   surface-support append planner as a separately traceable algorithm stage;
   its September 13 evidence concerns the simulation experiments, not a new
   Expo West flight. Do not substitute an old V78 plan or a scene-specific
   frozen target list for the general planner.
5. **Mission export.** Export explicit execution mode and schema, preserving
   capture poses, transit semantics and takeoff-relative height conversion.
   Return the actual schema in the result descriptor instead of hardcoding 13.

## Mobile mission compatibility

| Client in the current source snapshot | Accepted schemas | Continuous reacquisition |
| --- | --- | --- |
| Android V5 | 1–14 | Schema 14 with `recapture_flight_mode=CONTINUOUS_EXPERIMENTAL` |
| Android V4 | 1–14 | Experimental app-side Virtual Stick with schema 14; updated 2026-09-22 build required |
| iOS | 1–14 | Experimental app-side Virtual Stick with schema 14; updated 2026-09-22 build required |

Schema 13 retains stop-and-capture semantics. Schema 14 is not obtained by
changing a number: the execution-mode field and client behavior are part of
the contract. The current V5 implementation permits continuous passage only
at eligible interior capture points; turns, height changes and segment
boundaries may still stop. Android V4 and iOS use app-side Virtual Stick, not KMZ: only
aligned interior capture points are eligible, photo acknowledgements gate progress,
and failed/missed captures pause rather than silently succeed. Both require foreground
connectivity; the new continuous path still needs device acceptance.
Capture actions remain meaningful in continuous mode.

Select a supported export mode from explicit client capabilities. Reject an
unsupported continuous request rather than silently downgrading it. Preserve
the existing approval metadata and import/preflight checks. A generated mission
is not automatically authorized for flight; the inspected result endpoint
currently returns `safe_to_execute=false`.

## Integration sequence (completed)

The integration first replayed an Expo West pre-reacquisition input set offline, checking camera
coordinates, features, selected targets and generated capture poses against
the frozen reference. Tests covered upload retries/cancellation and both schema
exports, including native mobile decode and execution-mode handling. The
new worker ran on a separate port/data root before production was switched.
Algorithm-specific cache identities prevent reuse across incompatible versions.

The migration reuses the trained quality model and connects the final
feature/geometry/planning implementations to the service, with explicit metadata
and mission-schema negotiation at the API boundary.

## Source map

- Previous service on dsw1 (retained for rollback):
  `/mnt/petrelfs/youzhongrui/mt/gs/v15_cross_scene_validation_20260823/streaming_service/`.
- Geometry audit reference: `scripts/run_sensor_slrf_geometry.py` in the
  active-mapping source. The deployed runner instead follows the final Expo West
  local-group GPS alignment path, rather than hierarchical SLRF.
- Expo West planning: `scripts/run_directional_surface_cover.py`,
  `scripts/directional_surface_cover.py`, `scripts/continuous_capture_path.py`;
  frozen reference under
  `artifacts/directional_surface_cover_20260910/two_buildings_continuous_final/`.
- Existing height/export reference: `scripts/export_continuous_schema13.py` and
  `docs/TWO_BUILDINGS_SCHEMA13_HEIGHT_HANDOFF_20260910.md`.
- Later planner stage: `scripts/recapture_nbv/append_surface_planner.py` and
  `docs/APPEND_PLANNER_METHOD_AND_PAPER_RESULTS_20260913.md`.
- Mobile contracts: each app's `SurveyMissionJson`/`SurveyMissionJSON`, plus
  Android V5 `CONTINUOUS_RECAPTURE_SWITCH_2026-09-12.md` and `CLOUD_ROUTE_WORKFLOW.md`.

Historical paths above are provenance, not proposed public module names.
