# Initial repository scope

This local repository is the main project entry and training/testing codebase,
not the complete private experiment archive. No remote has been created yet.

## Included

- Quality Predictor dependency closure and original reference configuration.
- Distributed training entry point, selected CPU tests and reference evaluator.
- Public checkpoint inference interface, component registry and App/UE guides.
- Source hashes and archive provenance in `source_import.json`.
- Integrated mobile-facing workstation API, Pi3X feature pipeline, shared Quality
  Predictor, Expo West directional planner and schema 13/14 mission export.

## Before publishing

1. Assign and verify the main, Android, iOS and UE source repository URLs.
2. Finalize source licensing and retain attribution for external dependencies.
3. Review the pinned Pi3X/GeoFF3D patch and COLMAP reader inventory before publication.
4. Export portable training/evaluation data manifests and matching checkpoints.
5. Verify direct training and checkpoint resume with the released data bundle.
6. Package portable deployment configuration and dependency instructions for the
   [workstation service](workstation.md), already replay-tested and deployed on
   dsw1. Keep credentials, intermediate training checkpoints and runtime sessions
   outside source Git; include the released head and small inference example.

Public modules and commands follow the paper terminology; checkpoint fields
and model behavior are preserved. Original source hashes and rename mappings
are recorded in `source_import.json`; see [naming](naming.md). One imported test
resolves its temporary path for macOS `/var` and `/private/var` aliases.
CPU tests and real-image workstation replay are recorded in
[validation](validation.md); full retraining remains separate. The released head and small inference example ship in this repository. Large
scene and simulator packages belong in the HF dataset.
