# Image-to-feature dependencies

The CPU predictor example only needs the main Python package. Image-to-feature
inference additionally uses the GeoFF3D/Pi3X source and the UAVFF3D Pi3X checkpoint.
The large backbone checkpoint is external; the small Quality Predictor checkpoint
is included at `weights/quality_predictor.pt`.

## Pinned backbone source

`configs/dependencies.lock.json` records the deployed GeoFF3D base revision and
the hashes of eight modified runtime files. The base revision alone is not the
deployed implementation: `patches/geoff3d-runtime.patch` also restores its prior
handling, scale normalization and related runtime changes.

In a fresh external checkout:

```bash
git clone https://github.com/yanxian-ll/GeoFF3D.git /opt/GeoFF3D
git -C /opt/GeoFF3D checkout --detach 8eda172f9f15d315ec8b033f76ea6032ee9f7c47
git -C /opt/GeoFF3D submodule update --init --recursive
git -C /opt/GeoFF3D apply --check "$PWD/patches/geoff3d-runtime.patch"
git -C /opt/GeoFF3D apply "$PWD/patches/geoff3d-runtime.patch"
python scripts/check_dependencies.py --geoff3d /opt/GeoFF3D
```

Run these commands from the OpenFlyScan repository. Do not apply the patch over
an existing modified deployment. The patch excludes private launch scripts,
local backups and the unrelated untracked full-map exporter.

## Python environment and weights

The deployed reference uses Python 3.10 and PyTorch 2.5.0 with CUDA 12.1. Observed
package versions are recorded in the lock file. Follow the pinned backbone's
installation instructions for its CUDA/optional packages, then install
OpenFlyScan with `python -m pip install -e '.[server]'` in that environment.
The version inventory is a tested-environment record, not a universal pip lock
for every OS or GPU.

Set `geoff3d_root`, `pi3x_checkpoint` and `quality_checkpoint` in your private
workstation config. Pi3X uses the UAVFF3D fine-tuned checkpoint; the Quality
Predictor uses the included `weights/quality_predictor.pt`; set its absolute path
in the workstation config.
Only load trusted Pi3X training checkpoints: the backbone loader reads their
full training-checkpoint format. The Quality Predictor uses weights-only loading.

## Training compatibility tree

The trainer resolves dependencies below its plan's `root`:

```text
DEPENDENCY_ROOT/
  UAVFF3D/GeoFF3D/                        patched source checkout
  UAVFF3D/pi3x_finetuning/checkpoint-best.pth
  open-lixel-h3dgs-color11-20260808/preprocess/read_write_model.py
```

Use private symlinks to existing sources/weights if desired. The last path is a
compatibility location for the COLMAP model reader; it does not require importing
the private GS training repository. The exact reader is included at
`vendor/colmap/read_write_model.py`, with its original notices and content hash
recorded in the dependency lock. Link that file to the compatibility location.
See [training data](training_data.md).

The patch and bundled code retain their upstream terms. The main project's Apache-2.0 license
does not replace model-weight or nested third-party terms; see
[licensing](licensing.md) and [release readiness](release_readiness.md).
