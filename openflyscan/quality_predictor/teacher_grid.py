"""Shared weighted teacher grids with explicit input-to-teacher pixel mapping."""

from pathlib import Path
import json

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mapping_models(record):
    contract = record.get("teacher_mapping_contract")
    full = Path(record["full_root"])
    if contract is None:
        return full / "rgb_sfm", full / "published_scene_source/sparse/0"
    require(contract.get("schema") == "expanded_teacher_mapping_v1"
            and contract.get("accepted") is True, "teacher mapping contract is not accepted")
    for key in ("scene", "source", "evaluation"):
        require(contract.get(key) == record.get(key), f"teacher mapping contract {key} mismatch")
    require(contract.get("mapping") == record.get("teacher_mapping"), "teacher mapping kind mismatch")
    evidence = json.loads(Path(contract["evidence_path"]).read_text())
    require(evidence.get("scene") == record["scene"]
            and evidence.get("mapping", {}).get("accepted") is True,
            "teacher mapping evidence is not accepted")
    for key in ("source", "evaluation", "source_model", "teacher_model", "mapping"):
        require(evidence["mapping"].get(key) == contract.get(key),
                f"teacher mapping evidence {key} mismatch")
    return Path(contract["source_model"]), Path(contract["teacher_model"])


def validate_exposure(record, metadata):
    require(metadata.get("trained_exposure") is True,
            f"{record['scene']}: teacher exposure contract needs explicit adaptation; "
            "missing saved exposure is not evidence of historically disabled exposure")


def validate_grid(names, mse, weight):
    stems = [Path(str(name)).stem for name in names]
    require(len(stems) == len(set(stems)), "duplicate teacher stem")
    require(mse.ndim == 3 and mse.shape[0] == len(stems), "teacher shape mismatch")
    require(mse.shape[1] == mse.shape[2], "teacher grid must be square")
    require(weight is not None, "valid pixel weights unavailable; refusing fabricated weights")
    require(weight.shape == mse.shape, "weight shape mismatch")
    require(np.isfinite(mse).all() and (mse >= 0).all(), "invalid teacher MSE")
    require(np.isfinite(weight).all() and (weight >= 0).all(), "invalid teacher weights")
    require((weight.sum(axis=(1, 2)) > 0).all(), "empty teacher view")
    return dict(names=np.asarray(names), mse=mse, valid_pixel_weight=weight,
                index={stem: index for index, stem in enumerate(stems)})


def load_standard(evaluation, grid=16):
    evaluation = Path(evaluation)
    metadata = json.loads((evaluation / "metrics.json").read_text())
    suffix = "" if metadata["grid"] == grid else f"_grid{grid}"
    with np.load(evaluation / "grid_errors.npz", allow_pickle=False) as archive:
        teacher = validate_grid(archive["names"], archive["mse" + suffix],
                                archive["valid_pixel_weight" + suffix])
    teacher["metadata"] = metadata
    return teacher


def load_two_buildings(evaluation, grid=16, recovered_weights=None):
    evaluation = Path(evaluation)
    with np.load(evaluation / f"grid_{grid}x{grid}.npz", allow_pickle=False) as archive:
        require(np.allclose(archive["psnr"], -10 * np.log10(np.maximum(archive["mse"], 1e-12)),
                            atol=1e-5, rtol=1e-6), "Two Buildings MSE/PSNR mismatch")
        names, mse = archive["image_names"], archive["mse"]
    require(recovered_weights is not None, "Two Buildings per-tile weights were not saved")
    require(np.array_equal(names, recovered_weights["names"]), "weight image order mismatch")
    return validate_grid(names, mse, recovered_weights["valid_pixel_weight"])


def weighted_psnr(teacher):
    weights = teacher["valid_pixel_weight"].astype(np.float64)
    mse = (teacher["mse"].astype(np.float64) * weights).sum((1, 2)) / weights.sum((1, 2))
    return -10 * np.log10(np.maximum(mse, 1e-12))


def lookup_pixels(teacher, stem, pixels, input_wh, teacher_wh, *, mapping, mapper=None, return_cells=False):
    pixels = np.asarray(pixels, dtype=np.float64)
    require(pixels.shape[-1] == 2, "pixels must end in xy")
    input_wh, teacher_wh = np.asarray(input_wh, float), np.asarray(teacher_wh, float)
    require(input_wh.shape == teacher_wh.shape == (2,), "image dimensions must be width,height")
    require(np.isfinite(input_wh).all() and np.isfinite(teacher_wh).all()
            and (input_wh > 0).all() and (teacher_wh > 0).all(), "invalid image dimensions")
    shape = pixels.shape[:-1]
    values = np.full(shape, np.nan, dtype=np.float32)
    valid = np.isfinite(pixels).all(-1)
    valid &= (pixels >= -.5).all(-1) & (pixels < input_wh - .5).all(-1)
    if mapping == "rectified_resize":
        mapped = (pixels + .5) * teacher_wh / input_wh - .5
    elif mapping == "raw_to_rectified":
        require(mapper is not None, "raw input requires its calibrated coordinate mapper")
        mapped, mapped_valid = mapper(pixels)
        mapped = np.asarray(mapped)
        require(mapped.shape == pixels.shape, "coordinate mapper shape mismatch")
        valid &= np.asarray(mapped_valid, bool)
    else:
        raise ValueError(f"unsupported or unverified pixel mapping: {mapping}")
    valid &= np.isfinite(mapped).all(-1)
    valid &= (mapped >= -.5).all(-1) & (mapped < teacher_wh - .5).all(-1)
    grid = teacher["mse"].shape[1]
    require((teacher_wh >= grid).all(), "grid exceeds teacher dimensions")
    require(np.equal(teacher_wh, np.floor(teacher_wh)).all(), "noninteger teacher dimensions")
    columns = np.arange(grid + 1) * int(teacher_wh[0]) // grid
    rows = np.arange(grid + 1) * int(teacher_wh[1]) // grid
    rounded = np.floor(np.where(np.isfinite(mapped), mapped, 0) + .5)
    column = np.clip(np.searchsorted(columns, rounded[..., 0], side="right") - 1, 0, grid - 1)
    row = np.clip(np.searchsorted(rows, rounded[..., 1], side="right") - 1, 0, grid - 1)
    if stem not in teacher["index"]:
        if return_cells:
            return values, np.zeros(shape, bool), np.full((*shape, 2), -1, dtype=np.int64)
        return values, np.zeros(shape, bool)
    index = teacher["index"][stem]
    valid &= teacher["valid_pixel_weight"][index, row, column] > 0
    labels = np.log10(np.maximum(teacher["mse"][index, row, column], 1e-10))
    values[valid] = labels[valid]
    if return_cells:
        cells = np.stack([row, column], -1)
        cells[~valid] = -1
        return values, valid, cells
    return values, valid
