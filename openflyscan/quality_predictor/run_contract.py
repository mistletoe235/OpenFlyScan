"""Read-only input contracts and lossless recovery for the v2 trainer."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def validate_plan(plan, minimum_train_scenes):
    from .teacher_grid import mapping_models, validate_exposure

    if "full_label_policy" in plan:
        from .base_supervision import FullLabelPolicy
        FullLabelPolicy(**plan["full_label_policy"])
    required = {"root", "records", "dino_manifest", "input_policy"}
    missing = required-set(plan)
    if missing:
        raise ValueError(f"not a training plan; missing {sorted(missing)}; adapt the candidate plan first")
    if plan.get("training_ready") is False:
        raise ValueError("plan explicitly declares training_ready=false")
    policy = plan["input_policy"]
    if not isinstance(policy, dict) or policy.get("model") != "pi3x":
        raise ValueError("training plan must declare a Pi3X input policy")
    if policy.get("depth") != "none" or policy.get("bootstrap_depth", False):
        raise ValueError("v2 native inputs do not accept depth or bootstrap depth priors")
    if not isinstance(plan["records"], list) or not plan["records"]:
        raise ValueError("training plan needs nonempty records")
    scenes, physical_scenes, train, validation = set(), {}, [], []
    for record in plan["records"]:
        required_record = {"scene", "split", "source", "views", "full_root", "evaluation", "footprints"}
        if not isinstance(record, dict) or required_record-set(record):
            raise ValueError("record is missing fields required by the frozen Source")
        scene = record["scene"]
        if not isinstance(scene, str) or not scene or scene in scenes:
            raise ValueError(f"invalid or duplicate scene: {scene}")
        scenes.add(scene)
        physical = record.get("physical_scene_id", scene)
        if not isinstance(physical, str) or not physical or (physical in physical_scenes and not plan.get("allow_same_split_physical_groups", False)):
            raise ValueError(f"duplicate or invalid physical_scene_id: {physical}; grouped sampling is not integrated")
        if physical in physical_scenes and physical_scenes[physical] != record["split"]:
            raise ValueError(f"cross-split physical_scene_id: {physical}")
        physical_scenes[physical] = record["split"]
        if record.get("input_policy", policy) != policy:
            raise ValueError(f"{scene}: per-scene input policy differs; current Source only supports one global policy")
        if "teacher_mapping" in record and record["teacher_mapping"] not in ("rectified_resize", "raw_to_rectified"):
            raise ValueError(f"{scene}: unsupported teacher pixel mapping")
        if "teacher_mapping_contract" in record:
            mapping_models(record)
        if type(record["views"]) is not int or record["views"] < 30:
            raise ValueError(f"{scene}: current sampler requires at least 30 views")
        if record["split"] == "train":
            train.append(scene)
        elif record["split"] == "validation":
            validation.append(scene)
        else:
            raise ValueError(f"{scene}: unsupported split {record['split']}")
        for name in ("source", "full_root", "evaluation"):
            if not Path(record[name]).is_dir():
                raise ValueError(f"{scene}: missing {name} directory: {record[name]}")
        for path in (Path(record["footprints"]), Path(record["evaluation"])/"grid_errors.npz",
                     Path(record["evaluation"])/"metrics.json"):
            if not path.is_file():
                raise ValueError(f"{scene}: missing Source-compatible asset: {path}; a teacher adapter may be required")
        metadata = json.loads((Path(record["evaluation"])/"metrics.json").read_text())
        validate_exposure(record, metadata)
    if len(train) < minimum_train_scenes or not validation:
        raise ValueError("insufficient distinct training scenes or no validation scenes")
    if not Path(plan["dino_manifest"]).is_file():
        raise ValueError("DINO manifest is missing")
    return train, validation


def dependency_inventory(geoff):
    geoff = Path(geoff)
    required = ("geoff3d/slrf/model_runner.py", "geoff3d/models/external/pi3x/__init__.py",
                "geoff3d/models/external/pi3/models/pi3x.py")
    for relative in required:
        if not (geoff/relative).is_file():
            raise ValueError(f"missing GeoFF3D dependency: {relative}")
    paths = set((geoff/"geoff3d").rglob("*.py"))
    for suffix in ("*.yaml", "*.yml"):
        paths.update((geoff/"configs").rglob(suffix))
    if not any(path.is_relative_to(geoff/"configs") for path in paths):
        raise ValueError("GeoFF3D Hydra configurations are missing")
    return {str(path.relative_to(geoff)): sha256(path) for path in sorted(paths)}


def build_data_contract(plan, quality_limits, minimum_train_scenes):
    train, validation = validate_plan(plan, minimum_train_scenes)
    root = Path(plan["root"])
    checkpoint = root/"UAVFF3D/pi3x_finetuning/checkpoint-best.pth"
    paths = [Path(plan["dino_manifest"])]
    if any("teacher_mapping" in record for record in plan["records"]):
        paths.append(root/"open-lixel-h3dgs-color11-20260808/preprocess/read_write_model.py")
    for record in plan["records"]:
        paths.extend((Path(record["footprints"]), Path(record["evaluation"])/"grid_errors.npz",
                      Path(record["evaluation"])/"metrics.json"))
        if "teacher_mapping" in record:
            from .teacher_grid import mapping_models
            source_model, teacher_model = mapping_models(record)
            for folder in dict.fromkeys((source_model, teacher_model)):
                paths.extend(folder/name for name in ("cameras.bin", "images.bin"))
            if "teacher_mapping_contract" in record:
                paths.append(Path(record["teacher_mapping_contract"]["evidence_path"]))
    dependencies = dependency_inventory(root/"UAVFF3D/GeoFF3D")
    identity = plan.get("pi3x_checkpoint_identity")
    if identity is None:
        checkpoint_contract = dict(pi3x_sha256=sha256(checkpoint))
    else:
        checkpoint_stat = checkpoint.stat()
        expected_identity = dict(path=str(checkpoint), size_bytes=checkpoint_stat.st_size,
                                 mtime_ns=checkpoint_stat.st_mtime_ns,
                                 verification="pinned_metadata_not_content_hash")
        if identity != expected_identity:
            raise ValueError("Pi3X checkpoint metadata changed or invalid pinned identity")
        checkpoint_contract = dict(pi3x_sha256=None, pi3x_checkpoint_identity=expected_identity)
    return dict(schema="head_training_v2_data_contract_2", train_scenes=train, validation_scenes=validation,
                train_only_quality_limits=list(quality_limits), pi3x_checkpoint=str(checkpoint),
                **checkpoint_contract, artifacts={str(path): sha256(path) for path in paths},
                dependency_files=dependencies, dependency_sha256=contract_digest(dependencies),
                boundary="Source-readable assets and dependency hashes; optional pinned checkpoint metadata is not a content hash. Not geometric QA, full RGB/cache rehash, or training authorization.")


def require_same_contract(old, current, keys, label):
    for key in keys:
        if key not in old or old[key] != current[key]:
            raise ValueError(f"{label} changed: {key}")


def recover_after_checkpoint(output, step, world):
    output = Path(output)
    rewrites, move_paths = {}, []
    for path in [output/f"rank{rank}_steps.jsonl" for rank in range(world)]+[output/"validation.jsonl"]:
        if not path.exists():
            continue
        lines = path.read_text().splitlines(keepends=True)
        kept = []
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
                record_step = record["step"]
                if type(record_step) is not int:
                    raise ValueError("log step must be an integer")
            except (ValueError, KeyError) as error:
                if index != len(lines)-1:
                    raise ValueError(f"corrupt non-tail log record: {path}:{index+1}") from error
                break
            if record_step <= step:
                kept.append(line if line.endswith("\n") else line+"\n")
        retained = "".join(kept)
        if retained != "".join(lines):
            rewrites[path] = retained
    for pattern in ("head_step_*.pt", "sample_audit_rank*_step*.json", "missing_gradient_rank*_step*.json"):
        for path in output.glob(pattern):
            saved_step = int(path.stem.rsplit("step", 1)[1].lstrip("_"))
            if saved_step > step:
                move_paths.append(path)
    move_paths.extend(output.glob("completion_rank*.json"))
    if (output/"latest_step.json").exists():
        move_paths.append(output/"latest_step.json")
    if not rewrites and not move_paths:
        return None
    archive = Path(tempfile.mkdtemp(prefix=f"resume_after_{step:06d}_", dir=output))
    for path in rewrites:
        shutil.copy2(path, archive/path.name)
    for path in move_paths:
        shutil.copy2(path, archive/path.name)
    atomic_json(archive/"recovery.json", dict(checkpoint_step=step, rewritten_logs=[path.name for path in rewrites],
                                            archived_files=[path.name for path in move_paths]))
    for path, content in rewrites.items():
        temporary = path.with_suffix(path.suffix+".tmp")
        temporary.write_text(content)
        temporary.replace(path)
    for path in move_paths:
        path.unlink()
    return str(archive)
