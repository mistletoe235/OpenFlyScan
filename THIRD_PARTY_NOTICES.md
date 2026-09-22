# Third-party source inventory

Original OpenFlyScan code, the released Quality Predictor head and the small
inference example use Apache-2.0, as provided in LICENSE. This inventory records
third-party provenance and does not relicense third-party code, models or data.

| Component | Use | Source record |
| --- | --- | --- |
| Quality Predictor and training/evaluation code | Learned regional quality prediction | `docs/source_import.json` identifies the archived source files and hashes |
| Workstation, geometry and directional planner | Mobile feedback and reacquisition | `docs/workstation_source_import.json` identifies imported implementation files |
| GeoFF3D | Pi3X runtime and scene/camera utilities | `configs/dependencies.lock.json`; local changes in `patches/geoff3d-runtime.patch` |
| Pi3X and DINO implementation inside the backbone | Geometry and appearance features | Retained in the external checkout; Pi3 source uses BSD-3-Clause, copied in `LICENSES/Pi3-BSD-3-Clause.txt`; retain other nested notices |
| UAVFF3D Pi3X checkpoint | Backbone weights | External asset; official Pi3X weights use CC BY-NC 4.0, and UAVFF3D's project-page license does not grant a blanket checkpoint license |
| COLMAP model reader | Source/teacher camera mapping during training | Original BSD-3-Clause terms retained in `vendor/colmap/read_write_model.py` and `LICENSES/COLMAP-BSD-3-Clause.txt`; exact archive revision and hash in the dependency lock |
| PyTorch, NumPy, SciPy, Pillow, OpenCV, aiohttp, jsonschema | Runtime libraries | Installed dependencies, not vendored source |

The GeoFF3D checkout's root Apache-2.0 license is preserved in
`LICENSES/GeoFF3D-Apache-2.0.txt`. That root file alone is not a statement that
every nested external model or dataset uses identical terms.

Android/iOS and UE maintain their own `LICENSE` and `THIRD_PARTY_NOTICES.md`.
Do not copy their blanket project license onto scene data, checkpoints or other
upstream assets. Scene attribution and publication status belong in the HF
manifest and data card.

The GeoFF3D patch records the modified runtime paths and hashes in
`configs/dependencies.lock.json`. Preserve upstream copyright and license notices
when applying or redistributing it. Model/checkpoint and SDK boundaries are
summarized in `docs/licensing.md`.
