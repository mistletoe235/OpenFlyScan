# Quality Predictor

## Implementation

The GS quality model combines a pretrained Pi3X backbone with the trainable
`openflyscan.QualityPredictor`, configured by `configs/quality_predictor.json`.
The predictor has geometric and appearance branches, multiview aggregation and
a regional error output. The paper configuration includes source-patch recovery
and Base-only view supervision.

## Prepared inputs

`predict_quality.py` reads an NPZ with the following fields. `R` is the number of
regions and `V` the padded number of associated observations. Exact dimensions
come from the checkpoint's model configuration.

| Array | Shape | Source |
| --- | --- | --- |
| `point_features` | R × V × 1024 | Pi3X point decoder features at associated image patches |
| `confidence_features` | R × V × 1024 | Pi3X confidence decoder features at associated patches |
| `dino_features` | R × V × 1024 | Cached DINO image features |
| `pose_geometry_features` | R × V × 10 | Camera-relative region geometry |
| `quality_geometry_features` | R × V × configured dimension | Explicit point/projection evidence |
| `rgb_stats` | R × V × 6 | Local color statistics |
| `query_features` | R × configured dimension | Region geometry and observation summary |
| `view_mask` | R × V, boolean | Valid region-observation associations |

Feature construction is in `openflyscan/quality_predictor/`: `region_features.py`,
`geometry_evidence.py` and `view_supervision.py`. Inputs must preserve their
training-time coordinate and scaling conventions. All values must be finite,
including padding, and each scored region must have a valid observation.

## Checkpoint inference

```bash
python scripts/predict_quality.py --checkpoint weights/quality_predictor.pt \
  --inputs examples/quality_predictor/expo_west_sample_inputs.npz \
  --output outputs/scores.npy --device cpu
```

The 5.05 MB checkpoint and eight-region Expo West example are included in the
repository. Reference predictions are in
`examples/quality_predictor/expo_west_sample_scores.npy`; file hashes are recorded
in `configs/quality_predictor.release.json`. Use `--device cuda` for GPU inference
or replace the sample with your own prepared features.

The loader accepts the original training checkpoint schema, reconstructs the
model from `training_config`, and strictly loads `head`. Deserialization uses
PyTorch's weights-only loader. The Base-only auxiliary output is not exported
as the regional prediction. The original model's single regional score is saved
in input order.

## Shared-region evaluation

The imported evaluator computes Recall@20 and Spearman correlation using frozen
region indices and teacher targets. It also reports selection-boundary ties.
Supply the hashes belonging to the population being evaluated, not the archived
default hashes unless evaluating that exact archived population.

```bash
python scripts/evaluate_quality.py \
  --common-dir data/evaluation --scores outputs/scores.npy --name OpenFlyScan \
  --expected-selection-sha256 YOUR_SELECTION_SHA256 \
  --expected-teacher-sha256 YOUR_TEACHER_SHA256 --output outputs/evaluation.json
```

The common directory must contain `selection_frozen.json` and `region_values.npz`.
Scoring one NPZ does not establish correspondence with those frozen regions:
the input preparation must preserve their exact ordering and population.
This evaluator is an imported reference, not a claim that every final paper
table is reproduced by the current checkout.

## Training entry point

Run `scripts/train_quality_predictor.py` with the backbone dependencies and a
prepared training plan. For one GPU:

```bash
python scripts/train_quality_predictor.py \
  --plan data/plan.json --config configs/quality_predictor.json \
  --output outputs/run --expected-world-size 1 --chunks-per-rank 2
```

The reference configuration retains the paper's 48-process setup; command-line
options select the available GPU count and per-process sampling. See the
[training guide](training_data.md) for required data, multi-GPU commands and
checkpoint resume.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Selected original tests cover feature assembly, Base targets, weak Sparse
targets, ranking, distributed gathering and view auxiliary supervision. Public
inference tests use synthetic inputs and do not claim real-scene accuracy.
