# OpenFlyScan

**Quality-guided aerial reconstruction with consumer drones.**

OpenFlyScan connects mobile aerial capture, early regional GS quality prediction,
targeted reacquisition, and reconstructed scenes for drone simulation. This is the
main project entry point and the home of the workstation service and Quality
Predictor training and evaluation code. Mobile clients and UE/HIL remain separate projects.

> Start here: [Mobile apps](docs/apps.md) · [Workstation setup](docs/workstation.md) ·
> [HIL simulator](docs/simulator.md) · [Quality Predictor](docs/quality_predictor.md)

## Project components

| Component | Purpose | Entry |
| --- | --- | --- |
| **Quality Predictor** | Feature inputs, network, training, checkpoint inference and evaluation | [Code guide](docs/quality_predictor.md) |
| **Workstation service** | Mobile uploads, Pi3X geometry, quality prediction and reacquisition missions | [Service guide](docs/workstation.md) |
| **Mobile apps** | Android V4/V5 and iOS capture applications | [App repositories](docs/apps.md) |
| **UE / HIL simulator** | GS scene rendering, PLY conversion and Android/DJI HIL | [Simulator](docs/simulator.md) |
| **GS scenes and simulator packages** | Large downloadable assets, separate from source Git | [HF dataset](https://huggingface.co/datasets/IPEC-COMMUNITY/openflyscan) |
| **Project website** | System overview and demonstrations | [OpenFlyScan](https://openflyscan.github.io/) |

The HF dataset currently contains the Expo East Linux HIL simulator and is
private. The trained Quality Predictor checkpoint (5.05 MB) and a small real-feature
example are included in this repository; standalone GS scenes remain separate assets.
All five source repositories are private and require collaborator access:
[main](https://github.com/mistletoe235/OpenFlyScan),
[Android V4](https://github.com/mistletoe235/OpenFlyGo-Android-V4),
[Android V5](https://github.com/mistletoe235/OpenFlyGo-Android-V5),
[iOS](https://github.com/mistletoe235/OpenFlyGo-iOS), and
[UE / HIL](https://github.com/mistletoe235/OpenFlyScan-UE).
GitHub source access and access to the private HF dataset are managed separately.

## Install

For a self-contained CPU check, start with the [quick start](docs/quickstart.md).
It runs the included trained head on an eight-region Expo West feature sample;
no additional model or scene download is needed.

For mobile capture → upload → point cloud → reviewed reacquisition, start with
the [workstation guide](docs/workstation.md) and [client guide](docs/apps.md).
An SSH connection from the phone is not required: clients use a reachable
HTTP/HTTPS service URL and bearer access code.

| Client | Pinned DJI SDK / reference aircraft | Mission schemas | Upload and cloud results |
| --- | --- | --- | --- |
| Android V4 | MSDK 4.16.4 / Mini 2 | 1–14 | Survey-trigger frames, historical photos, PLY and missions |
| Android V5 | MSDK 5.18.0 / Mini 4 Pro | 1–14 | Same; experimental continuous recapture requires schema 14 and DJI KMZ |
| iOS | MSDK 4.16.2 / Mini 2 | 1–14 | New source builds add session creation, trigger frames / historical photos and explicit finalize; PLY and missions supported |

Reference aircraft are not a guarantee for every SDK-supported product or every
build. See [SDK support links and platform differences](docs/apps.md). iOS live
uploads are aspect-preserving downlink frames, not SD-card originals; historical
photos require original GPS and ASL metadata. Existing distributed builds must be
updated to gain newly added features.

Create a session, collect/upload compatible images, explicitly finalize, then
inspect the cloud point cloud and proposed mission. Current workstation exports
carry `safe_to_execute=false` and `flight_authorized=false`: generation/download
is **not flight authorization**. Review and existing camera/altitude/preflight
checks remain required; no automatic takeoff or mission execution is implied.

Use Python 3.10 or newer. For GPU training, install a CUDA-compatible PyTorch
build for your system, then run:

```bash
python -m pip install -e '.[test,server]'
python -m unittest discover -s tests -p 'test_*.py'
```

This installs the Quality Predictor, service and CPU-test dependencies. Image-to-feature
extraction additionally requires the pinned Pi3X/GeoFF3D environment and model
weights; see [data and dependencies](docs/data.md).

## Predict and evaluate

Run the included trained checkpoint and real-feature sample:

```bash
python scripts/predict_quality.py \
  --checkpoint weights/quality_predictor.pt \
  --inputs examples/quality_predictor/expo_west_sample_inputs.npz \
  --output outputs/scores.npy --device cpu
```

The output preserves input region order. Larger scores indicate larger predicted
normalized regional GS reconstruction error. Feature fields, checkpoint format and
the shared-region evaluation command are documented in [the predictor guide](docs/quality_predictor.md).

## Training

The repository includes the paper configuration, Base/Sparse supervision,
source-patch recovery, view auxiliary loss, distributed trainer and selected
original tests. The public API is `openflyscan.QualityPredictor`.

```bash
python scripts/train_quality_predictor.py \
  --plan /data/plan.json --config configs/quality_predictor.json \
  --output outputs/training-run --expected-world-size 1 --chunks-per-rank 2
```

Training starts directly from the repository with a prepared data plan and CUDA
dependencies. The [training guide](docs/training_data.md) covers data preparation,
single-/multi-GPU execution and checkpoint resume. The downloadable teacher/cache
bundle is still pending.

## Layout

```text
openflyscan/                     Public API and checkpoint inference
openflyscan/quality_predictor/   Model, feature assembly, teachers and losses
openflyscan/server/              Sessions, uploads, jobs and artifact delivery
openflyscan/reconstruction/      Pi3X geometry and preview point clouds
openflyscan/planning/            Quality-guided reacquisition planning
openflyscan/missions/            Mobile mission export and validation
scripts/        Training, inference and shared-region evaluation
configs/        Paper reference configuration and asset manifests
weights/        Released Quality Predictor checkpoint
examples/       Small real-feature inference sample and reference scores
tests/          CPU regression tests and inference checks
docs/           App/UE entry points, data requirements and source provenance
```

Mobile and UE projects stay in independent repositories rather than being copied
here. The released head and small inference sample are included; full image sets,
training caches, backbone weights, GS PLYs and simulator archives stay in external
storage/HF. See [release scope](docs/release.md)
for the remaining preparation work and licensing review.

See [naming and compatibility](docs/naming.md) for archived checkpoint identifiers.

## License

Original OpenFlyScan code, the released Quality Predictor head and the small
inference example use [Apache-2.0](LICENSE). Third-party components retain their
own terms. Official Pi3X backbone weights use CC BY-NC 4.0; the source license
does not grant unrestricted commercial use of those weights. See
[licensing scope](docs/licensing.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
