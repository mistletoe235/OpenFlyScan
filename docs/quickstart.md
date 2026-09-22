# Quick start

Choose the path you need; these are independent entry points, not required steps
in one installation:

| Goal | Start here | Additional requirements |
| --- | --- | --- |
| Install a capture app | [App downloads and setup](apps.md) | Compatible phone/aircraft; GitHub access |
| Open the included Expo East scene | [HIL simulator](simulator.md) | Linux GPU machine; separate HF access |
| Try the trained head | CPU example below | Python 3.10+; no backbone download |
| Process phone images | [Workstation setup](workstation.md) | GPU, patched GeoFF3D/Pi3X and backbone weights |
| Train the head | [Training guide](training_data.md) | CUDA and a prepared image/teacher/feature plan |

## 1. Run the included trained head on CPU

From the repository, using Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/predict_quality.py \
  --checkpoint weights/quality_predictor.pt \
  --inputs examples/quality_predictor/expo_west_sample_inputs.npz \
  --output outputs/scores.npy --device cpu
```

The 5.05 MB trained checkpoint, eight-region Expo West feature sample and
reference scores are included in the repository. No HF download or GPU is needed
for this example. Reference predictions are in
`examples/quality_predictor/expo_west_sample_scores.npy`; file hashes are in
`configs/quality_predictor.release.json`.

Use `--device cuda` for GPU inference, or replace `--inputs` with features from
your own scene. The feature contract is described in
[Quality Predictor](quality_predictor.md). Choose a fresh output path each time.

## 2. Optional synthetic interface check

```bash
python -m openflyscan.demo --output outputs/synthetic-demo
```

This generates random weights and synthetic inputs to check the interface;
use the trained checkpoint above for actual predictions.

## 3. Connect a phone to a workstation

Install the server extra with `python -m pip install -e '.[server]'`, then
prepare the backbone and private configuration using [workstation setup](workstation.md).
Set `quality_checkpoint` to the absolute path of this repository's
`weights/quality_predictor.pt`; only the Pi3X backbone weights need an external download.

```bash
export OPENFLYSCAN_CONFIG=/data/openflyscan/workstation.json
openflyscan-server --host 127.0.0.1 --port 55000 --data-root /data/openflyscan/sessions
```

Expose the service through your HTTPS endpoint, and configure that address and
the generated access token in the phone app. Keep the token private. See
[workstation API](workstation.md) for camera metadata and schema negotiation.
The CPU example does not connect to this service or send any aircraft commands.

## 4. Train the Quality Predictor

Training requires source images/cameras, GS error grids, footprints and DINO
caches, as described in [training data](training_data.md). With the prepared plan:

```bash
python scripts/train_quality_predictor.py \
  --plan /data/plan.json --config configs/quality_predictor.json \
  --output outputs/training-run --expected-world-size 1 --chunks-per-rank 2
```

This starts single-GPU training directly once those assets exist. The downloadable
teacher/cache bundle is not yet published, and the inference NPZ is not a training
dataset. The single-GPU example needs at least two training scenes and one
separate validation scene; see the training guide for plan construction.
