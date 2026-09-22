"""Self-contained CPU wiring check with explicitly synthetic data and weights."""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from . import QualityPredictor, QualityPredictorTrainingConfig
from .inference import predict


def run_demo(output, seed=42):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory; existing results are never overwritten")
    config = QualityPredictorTrainingConfig()
    dimensions = config.model_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = QualityPredictor(config).eval()
    generator = np.random.default_rng(seed)
    regions, views = 4, 6
    sizes = {
        "point_features": dimensions.point_feature_dim,
        "confidence_features": dimensions.confidence_feature_dim,
        "dino_features": dimensions.dino_feature_dim,
        "pose_geometry_features": dimensions.pose_geometry_dim,
        "quality_geometry_features": dimensions.quality_geometry_dim,
        "rgb_stats": dimensions.rgb_stats_dim,
    }
    arrays = {name: generator.standard_normal((regions, views, width)).astype(np.float32)
              for name, width in sizes.items()}
    arrays["query_features"] = generator.standard_normal((regions, dimensions.query_dim)).astype(np.float32)
    arrays["view_mask"] = np.ones((regions, views), dtype=bool)
    arrays["view_mask"][-1, -2:] = False
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "synthetic_checkpoint.pt"
    inputs = output / "synthetic_inputs.npz"
    torch.save({"schema": "head_training_v3_checkpoint", "training_config": dataclasses.asdict(config),
                "head": model.state_dict(), "synthetic_demo": True}, checkpoint)
    np.savez_compressed(inputs, **arrays)
    scores = predict(checkpoint, inputs, device="cpu")
    if scores.shape != (regions,) or not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise RuntimeError("Unexpected predictor output")
    np.save(output / "scores.npy", scores)
    manifest = {
        "kind": "synthetic_wiring_check", "trained_weights": False, "seed": seed,
        "checkpoint": checkpoint.name, "inputs": inputs.name, "scores": "scores.npy",
        "regions": regions, "views": views, "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "feature_shapes": {name: list(values.shape) for name, values in arrays.items()},
        "scope": "CPU loading and prediction only; random weights and inputs, not scene-quality evaluation",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = run_demo(args.output, args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
