"""Resolve portable path placeholders and validate a plan; never start training."""

import argparse
import dataclasses
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openflyscan import QualityPredictorTrainingConfig
from openflyscan.quality_predictor.run_contract import validate_plan
from openflyscan.quality_predictor.training_base import minimum_training_scenes


def resolve_paths(value, data_root, dependency_root):
    if isinstance(value, dict):
        return {key: resolve_paths(item, data_root, dependency_root) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_paths(item, data_root, dependency_root) for item in value]
    if not isinstance(value, str):
        return value
    for marker, root in (("${DATA_ROOT}", data_root), ("${DEPENDENCY_ROOT}", dependency_root)):
        if value == marker or value.startswith(marker + "/"):
            base = Path(root).resolve()
            resolved = (base / value[len(marker):].lstrip("/")).resolve()
            if not resolved.is_relative_to(base):
                raise ValueError("A plan path escapes its declared root")
            return str(resolved)
    if "${" in value:
        raise ValueError("Unknown or embedded path placeholder")
    return value


def prepare_plan(template, data_root, dependency_root, config):
    if "pi3x_checkpoint_identity" in template:
        raise ValueError("Do not relocate an old checkpoint stat identity; verify the new checkpoint in preflight")
    plan = resolve_paths(template, data_root, dependency_root)
    validate_plan(plan, minimum_training_scenes(config))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/quality_predictor.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--chunks-per-rank", type=int)
    args = parser.parse_args()
    config = QualityPredictorTrainingConfig(**json.loads(args.config.read_text()))
    overrides = {}
    for field in ("expected_world_size", "chunks_per_rank"):
        value = getattr(args, field)
        if value is not None:
            if value < 1:
                parser.error(f"--{field.replace('_', '-')} must be positive")
            overrides[field] = value
    config = dataclasses.replace(config, **overrides)
    plan = prepare_plan(json.loads(args.template.read_text()), args.data_root, args.dependency_root, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    print("Plan paths validated. Pass the generated plan to train_quality_predictor.py to start training.")


if __name__ == "__main__":
    main()
