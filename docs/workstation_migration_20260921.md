# Workstation migration record — September 21, 2026

Historical deployment notes, retained for provenance. Hostnames, absolute paths
and migration tasks below refer to the original private deployment, not prerequisites
for installing OpenFlyScan. Some audit text predates the completed integration.
For a new installation, follow [the workstation guide](workstation.md).

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
