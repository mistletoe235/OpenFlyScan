# Training data entry point

The trainer consumes a prepared plan, not a folder of photographs alone.
Keep these assets outside source Git:

| Field | Required content |
| --- | --- |
| `source` | Current images and camera metadata readable by GeoFF3D `scene_io` |
| `views` | Number of available images; each sampled scene needs at least 30 |
| `full_root` | Base reconstruction/calibration assets used by the teacher mapping |
| `evaluation` | `grid_errors.npz` and `metrics.json` for exposure-corrected Base GS error |
| `footprints` | NPZ containing `stems`, `centers`, `bbox_mins`, `bbox_maxs` |
| `dino_manifest` | Scene/image-indexed feature-cache paths compatible with the current patch layout |
| `split`, `physical_scene_id` | Explicit training/validation assignment without cross-split scene duplication |

The teacher grid must preserve image names, pixel validity, exposure metadata
and source-to-teacher camera correspondence. A rendered image folder without
these mappings cannot replace it. Each DINO cache contains the `feature` and
`rgb_stats` arrays expected by `pi3x_features.py`; cache paths inside its manifest
must resolve on the training machine.

## Resolve a portable manifest

Install the [backbone dependencies](dependencies.md) and OpenFlyScan in a
CUDA-enabled environment. Copy `configs/training_plan.example.json` and replace
its example scene entries with your dataset, keeping training and validation
scenes separate.

```bash
python scripts/resolve_training_plan.py \
  --template /data/training_plan.template.json \
  --data-root /data/openflyscan-training \
  --dependency-root /opt/openflyscan-dependencies \
  --output outputs/plan.json --expected-world-size 1 --chunks-per-rank 2
```

The resolver expands the root placeholders and checks the plan. Teacher grids
and DINO caches must already exist; paths inside cache manifests must also be
valid on the training machine. Use the same process count and per-process
sampling here as in the training command below.

## Single-GPU training

Run directly from the repository:

```bash
python scripts/train_quality_predictor.py \
  --plan outputs/plan.json --config configs/quality_predictor.json \
  --output outputs/training-run --expected-world-size 1 --chunks-per-rank 2
```

Training is the default mode. Each global step needs at least two Base/Sparse
pairs for cross-group ranking, so the single-GPU command uses two pairs per
process. This setting requires at least two training scenes and one separate
validation scene. It changes the global batch size relative to the paper setup.

Use `--max-steps 100` for a shorter run. Other model, optimizer and loss settings
are in the JSON configuration. Training records the effective configuration,
data metadata, logs and checkpoints automatically; choose an empty output
directory for a new run.

## Multi-GPU training

For four GPUs on one machine:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_quality_predictor.py \
  --plan outputs/plan.json --config configs/quality_predictor.json \
  --output outputs/training-run-4gpu --expected-world-size 4
```

`--expected-world-size` must match the total number of launched processes. The
unmodified reference configuration uses 48 processes and at least eight training
scenes. Keep it for that reference setup; set the process count and
`--chunks-per-rank` to match your resources for other runs.

## Resume or initialize from a checkpoint

To resume an interrupted run, repeat its command with `--resume`:

```bash
python scripts/train_quality_predictor.py \
  --plan outputs/plan.json --config configs/quality_predictor.json \
  --output outputs/training-run --expected-world-size 1 --chunks-per-rank 2 \
  --resume
```

Resume restores `head_last.pt` and optimizer progress, checking that the data,
configuration and source match. To start a new run from existing weights instead,
use a fresh output directory and `--init-checkpoint /data/head_last.pt`.
Only load training checkpoints from trusted sources.

An optional `--mode preflight` checks the data plan without starting GPU training;
it is not a required step.

The downloadable teacher/cache bundle has not been published yet.
The archived COLMAP reader is pinned in the dependency lock and included under
`vendor/colmap`; its compatibility-path setup is documented in the dependency guide.
