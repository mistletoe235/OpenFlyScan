"""Load a training checkpoint and score prepared current-observation features."""

import dataclasses
from pathlib import Path

import numpy as np
import torch

from . import QualityPredictor, QualityPredictorTrainingConfig, RegionInputs


def load_predictor(checkpoint_path, device="cpu"):
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "head_training_v3_checkpoint":
        raise ValueError("Expected a head_training_v3_checkpoint with head and training_config")
    config = QualityPredictorTrainingConfig(**checkpoint["training_config"])
    model = QualityPredictor(config)
    model.load_state_dict(checkpoint["head"], strict=True)
    model.training_config = config
    return model.to(device).eval()


def load_inputs(path, model, device="cpu"):
    config = model.head.config
    dimensions = {
        "point_features": config.point_feature_dim,
        "confidence_features": config.confidence_feature_dim,
        "dino_features": config.dino_feature_dim,
        "pose_geometry_features": config.pose_geometry_dim,
        "quality_geometry_features": config.quality_geometry_dim,
        "rgb_stats": config.rgb_stats_dim,
        "query_features": config.query_dim,
    }
    fields = {field.name for field in dataclasses.fields(RegionInputs)}
    with np.load(path, allow_pickle=False) as archive:
        missing = fields - set(archive.files)
        if missing:
            raise ValueError(f"Missing input arrays: {sorted(missing)}")
        arrays = {name: archive[name] for name in fields}
    mask = arrays["view_mask"]
    if mask.dtype != np.bool_ or mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError("view_mask must be a nonempty boolean [regions, views] array")
    if not mask.any(axis=1).all():
        raise ValueError("Each region must have at least one valid observation")
    for name, dimension in dimensions.items():
        expected = (mask.shape[0], dimension) if name == "query_features" else (*mask.shape, dimension)
        if arrays[name].shape != expected or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} must contain finite values with shape {expected}")
    return RegionInputs(**{
        name: torch.as_tensor(values, dtype=torch.bool if name == "view_mask" else torch.float32, device=device)
        for name, values in arrays.items()
    })


def predict(checkpoint_path, inputs_path, device="cpu"):
    model = load_predictor(checkpoint_path, device)
    inputs = load_inputs(inputs_path, model, device)
    with torch.inference_mode():
        scores = model(inputs)["total"]
    return scores.cpu().numpy()
