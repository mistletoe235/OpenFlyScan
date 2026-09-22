"""Write staged v2 control configurations; never launch or authorize training."""

import argparse
import dataclasses
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openflyscan.quality_predictor.training_base import TrainingConfigV2


def control_configurations(base):
    cross_weights = dict(full_cross_ranking_weight=0.0, missing_total_cross_ranking_weight=0.0,
                         missing_structure_cross_ranking_weight=0.0)
    no_dropout = dataclasses.replace(base, modality="full", dino_dropout=0.0, confidence_dropout=0.0)
    configurations = {
        "01_v1_loss_control_new_inputs": dataclasses.replace(no_dropout, missing_total_local_ranking_weight=0.0, **cross_weights),
        "02_missing_total_only": dataclasses.replace(no_dropout, **cross_weights),
        "03_with_cross_chunk": no_dropout,
        "04_with_modality_dropout": dataclasses.replace(base, modality="full"),
    }
    for mode in ("dino_only", "confidence_only", "point_only", "explicit_geometry_only"):
        configurations[mode] = dataclasses.replace(no_dropout, modality=mode)
    return configurations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = TrainingConfigV2(**json.loads(args.base.read_text()))
    configurations = control_configurations(base)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, config in configurations.items():
        (args.output_dir/(name+".json")).write_text(json.dumps(dataclasses.asdict(config), indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(configurations=list(configurations), output=str(args.output_dir.resolve()),
                          boundary="Not launched. All controls use v2 input dimensions, not an exact v1 reproduction. DINO-only structure output has no DINO path; compare total outputs.")))


if __name__ == "__main__":
    main()
