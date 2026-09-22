# Quality Predictor checkpoint

`quality_predictor.pt` is the OpenFlyScan Quality Predictor checkpoint, released
under the Apache License 2.0 in [LICENSE](../LICENSE). It contains the head's
parameters and configuration, not Pi3X or DINO backbone weights.

The checkpoint has 1,252,725 parameters and occupies 5,050,090 bytes. Its SHA-256
and matching example files are recorded in
[the asset manifest](../configs/quality_predictor.release.json).

Run the [included CPU example](../docs/quickstart.md), or supply regional features
from the documented feature pipeline. The checkpoint license does not replace
licenses for backbone weights, input data or other dependencies. In particular,
the official Pi3X weights are distributed under CC BY-NC 4.0; the corresponding
full pipeline must respect those terms or a separately obtained authorization.
