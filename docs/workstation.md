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
to a private runtime directory. Run from the OpenFlyScan repository:

```bash
mkdir -p /data/openflyscan
cp configs/workstation.example.json /data/openflyscan/workstation.json
```

`/data/openflyscan` is an example writable directory; replace it throughout if needed.
Edit these fields before starting the service:

| Field | Value |
| --- | --- |
| `geoff3d_root` | Absolute path to the patched GeoFF3D checkout from [dependencies](dependencies.md) |
| `pi3x_checkpoint` | Absolute path to the UAVFF3D Pi3X backbone checkpoint |
| `quality_checkpoint` | Absolute path to this repository's `weights/quality_predictor.pt` |
| `device` | GPU device, for example `cuda:0` |
| `worker_python` (optional) | Python executable in a separate inference environment with both OpenFlyScan and the backbone installed |

The default worker uses the server's Python interpreter. Then launch:

```bash
export OPENFLYSCAN_CONFIG=/data/openflyscan/workstation.json
openflyscan-server --host 127.0.0.1 --port 55000 --data-root /data/openflyscan/service
```

The first launch creates a private `access_token` file under the data root.
Requests use `Authorization: Bearer <token>`. Keep that file outside Git and put
HTTPS in front of an externally accessible deployment. `/health` reports
`openflyscan-workstation`, pipeline identity, supported mission schemas and jobs.
Check local availability with `curl http://127.0.0.1:55000/health`; a running
service reports `ok: true`. This checks the server, not a completed Pi3X inference.

The example binds to the workstation loopback interface. A phone must use a
reachable LAN/VPN address or HTTPS reverse proxy, not `127.0.0.1` or an SSH hostname
alone. Configure the listener/proxy and firewall accordingly without exposing an
unauthenticated service. Use an HTTPS endpoint with a certificate trusted by the phone; the Android V4
release does not allow cleartext HTTP. Enter the endpoint's root URL (without
`/api/sessions`) and the generated token in the mobile cloud page.
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

## Deployment history

The original private-service migration and source mapping are retained in the
[September 21 migration record](workstation_migration_20260921.md). Those paths
and retired services are not needed for a fresh installation.
