# Data and external dependencies

## Predictor-only inference

The released `head_training_v3_checkpoint` is included at
`weights/quality_predictor.pt`, with a small real-feature sample under
`examples/quality_predictor/`. Other inputs must use the same feature pipeline. Pi3X/DINO are not run by `predict_quality.py`.
Normalization of the learned target must remain the one used during training;
the inference output is normalized, not PSNR in dB.

## Image-to-feature extraction and training

The trainer expects a prepared JSON plan with:

- Pi3X input policy, external model root and checkpoint identity.
- Per-scene source images/cameras, footprints and Base GS teacher grids.
- DINO cache manifest and compatible image/patch indexing.
- Training/validation scene assignments and Base/Sparse target policies.
- GeoFF3D dependency directory supplied through the plan's root configuration.

The exact validation contract lives in `openflyscan/quality_predictor/run_contract.py`; teacher
mapping and exposure metadata are checked in `openflyscan/quality_predictor/teacher_grid.py`.
Install `geoff3d` externally and link the bundled COLMAP `read_write_model`
helper as described in the dependency guide.

The archived run used Python 3.10 and PyTorch 2.5 with CUDA 12.1. The deployed
backbone revision, runtime patch and package inventory are recorded in
[dependencies](dependencies.md). Plan preparation and direct single-/multi-GPU
training commands are documented in [training data](training_data.md).
The remaining data release work is the teacher/cache bundle and its attribution.
