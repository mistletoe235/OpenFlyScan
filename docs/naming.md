# Project naming

| Scope | Public name |
| --- | --- |
| Project | OpenFlyScan |
| Python package | `openflyscan` |
| GS quality model | Pretrained Pi3X backbone + Quality Predictor |
| Trainable module | `QualityPredictor`, in `openflyscan/quality_predictor/` |
| Reference configuration | `configs/quality_predictor.json` |
| Supervision | Base / Sparse |
| Regional output | Predicted regional error; larger means higher reacquisition priority |
| Simulator assets | `HIL-simulator/` |
| Planned weights | `checkpoints/quality_predictor/` |

## Compatibility and provenance

OpenFlySplat and V6 are historical project/experiment names, not public module
names. Full/Missing corresponds to Base/Sparse in archived training data.
Checkpoint schemas, configuration fields and parameter keys retain their
original spelling so existing weights and data remain loadable. In particular,
`head_training_v3_checkpoint`, `exposure_head_v3`, `head`, `full_*`, `missing_*`
and internal `risk` fields are serialization/implementation identifiers.
The evaluator retains `--lower-is-riskier` as an alias of `--lower-is-worse`.

`source_import.json` records original source locations and hashes, the initial
import paths, and their current paths. Those hashes identify the source archive,
not the renamed files in this checkout.

The scene names are Expo West (historically Two Buildings) and Expo East
(historical disk alias `expo_north`). Existing UE package internals, including
`Linux/OpenFlySplatUE`, remain unchanged. DJI MSDK V4/V5 and simulator release
`v0.1.0` are SDK/release versions, not experiment labels.
