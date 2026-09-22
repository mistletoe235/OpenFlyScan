"""Train the Quality Predictor directly from a prepared data plan and configuration."""

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from openflyscan.quality_predictor.paired_observation_source import SourceV2, concatenate_pairs
from openflyscan.quality_predictor.geometry_evidence import REGION_EVIDENCE_NAMES, VIEW_EVIDENCE_NAMES
from openflyscan.quality_predictor.training_base import (
    TrainingConfigV2, TrainingHeadV2, initialize_head, labels_to_device,
    minimum_training_scenes, modality_inputs, ranking_loss, smoke_requirements, training_loss,
)
from openflyscan.quality_predictor.run_contract import (
    atomic_json, build_data_contract, contract_digest, dependency_inventory,
    recover_after_checkpoint, require_same_contract, sha256, validate_plan,
)


def code_inventory():
    paths = sorted((PROJECT/"openflyscan").rglob("*.py"))
    paths += [PROJECT/"scripts"/name for name in (
        "train_quality_predictor_legacy.py", "map_rgb_to_rectified_uv.py", "expanded_teacher_adapter.py")]
    hashes = {str(path.relative_to(PROJECT)): sha256(path) for path in paths}
    return hashes, hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def training_device(local_rank):
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def read_training_source(plan_path, plan, config):
    if config.expected_world_size*config.chunks_per_rank < 2:
        raise ValueError("v2 cross-chunk training requires at least two pairs per global step")
    validate_plan(plan, minimum_training_scenes(config))
    source = SourceV2(plan_path, None, max_regions=config.max_regions)
    fixed_validation_scenes(source)
    return source


def fixed_validation_scenes(source):
    configured = getattr(source, "plan", {}).get("fixed_validation_scenes")
    scenes = source.val if configured is None else configured
    if not isinstance(scenes, list) or not scenes or any(not isinstance(scene, str) for scene in scenes):
        raise ValueError("fixed_validation_scenes must be a nonempty list of scene names")
    if len(set(scenes)) != len(scenes) or not set(scenes) <= set(source.val):
        raise ValueError("fixed_validation_scenes must be distinct validation scenes, never training scenes")
    return scenes


def rank_zero_action(action, rank, world):
    message = [None]
    if rank == 0:
        try:
            action()
        except Exception as error:
            message[0] = f"{type(error).__name__}: {error}"
    if world > 1:
        distributed.broadcast_object_list(message, src=0)
    if message[0]:
        raise RuntimeError(message[0])


def compact_sample(summary):
    result = {key: value for key, value in summary.items() if key not in ("label_audit", "evidence_audit", "choice")}
    result["deleted_count"] = len(summary["choice"]["deleted"])
    result["label_audit"] = {state: {key: value for key, value in audit.items() if key != "rows"}
                             for state, audit in summary["label_audit"].items()}
    result["evidence_audit"] = {state: {
        "median_projectable_views": float(np.median([row["projectable"] for row in rows])),
        "median_consistent_views_2pct": float(np.median([row["consistent_2pct"] for row in rows])),
    } for state, rows in summary["evidence_audit"].items()}
    return result


def branch_gradients(model):
    result = {}
    for name in ("point", "confidence", "dino", "pose_geometry", "quality_geometry", "query", "structure_pool", "appearance_pool", "total_fuse"):
        gradients = [parameter.grad.detach().square().sum() for key, parameter in model.named_parameters()
                     if key.startswith("head."+name+".") and parameter.grad is not None]
        result[name] = float(torch.stack(gradients).sum().sqrt()) if gradients else 0.0
    return result


def isolated_missing_gradient(model, inputs, labels, config):
    probe = copy.deepcopy(model).eval()
    prediction = probe(inputs)["total"]
    mask = labels["structure_mask"].bool() & (labels["state"] == 1)
    loss, counts = ranking_loss(prediction, labels["structure"], mask, labels["groups"], config)
    loss.backward()
    result = dict(loss=float(loss.detach()), **counts, gradient_norms=branch_gradients(probe),
                  boundary="Real inputs, isolated Missing total-ranking gradient; not an accuracy claim.")
    del probe, prediction, loss
    return result


def fixed_validation(head, source, config, device, rank, world, step, output):
    from scipy.stats import spearmanr

    head.eval()
    validation_scenes = fixed_validation_scenes(source)
    scene = validation_scenes[rank % len(validation_scenes)]
    inputs, label, summary, clouds = source.pair(scene, np.random.default_rng(424242+rank), seed=config.seed)
    labels = labels_to_device(label, device)
    inputs, _ = modality_inputs(inputs, labels["groups"], config, training=False)
    with torch.no_grad():
        predictions = head(inputs)
    rows = []
    for state, channel, target_name, mask_name in [
        ("full", "total", "total", "total_mask"),
        ("missing", "total", "structure", "structure_mask"),
        ("missing", "structure", "structure", "structure_mask"),
    ]:
        mask = labels[mask_name].bool()
        if state == "full":
            mask &= labels.get("total_rank_mask", labels[mask_name]).bool()
        rows.append(dict(scene=scene, seed=424242+rank, state=state, output=channel,
                         prediction=predictions[channel][mask].cpu().tolist(), target=labels[target_name][mask].cpu().tolist()))
    gathered = [None for _ in range(world)]
    if world > 1:
        distributed.all_gather_object(gathered, rows)
    else:
        gathered[0] = rows
    if rank == 0:
        def metrics(prediction, target):
            prediction, target = np.asarray(prediction), np.asarray(target)
            if not len(target):
                return dict(regions=0, recall=None, rho=None)
            budget = int(np.ceil(0.2*len(target)))
            selected = np.argsort(prediction, kind="stable")[-budget:]
            worst = np.argsort(target, kind="stable")[-budget:]
            rho = float(spearmanr(prediction, target).statistic) if np.std(prediction) > 1e-8 and np.std(target) > 1e-8 else None
            return dict(regions=len(target), recall=len(set(selected)&set(worst))/budget, rho=rho)

        all_rows = [row for rank_rows in gathered for row in rank_rows]
        results = {}
        for state, channel in [("full", "total"), ("missing", "total"), ("missing", "structure")]:
            selected = [row for row in all_rows if row["state"] == state and row["output"] == channel]
            individual = [metrics(row["prediction"], row["target"]) for row in selected]
            recalls = [row["recall"] for row in individual if row["recall"] is not None]
            results[state+"_"+channel] = dict(
                macro_recall=float(np.mean(recalls)) if recalls else None,
                pooled=metrics([value for row in selected for value in row["prediction"]],
                               [value for row in selected for value in row["target"]]), per_chunk=individual)
        with (output/"validation.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=step, metrics=results,
                                         cases=[dict(scene=rank_rows[0]["scene"], seed=rank_rows[0]["seed"]) for rank_rows in gathered],
                                         fixed_validation_scenes=validation_scenes,
                                         excluded_additional_validation_scenes=[scene for scene in source.val if scene not in validation_scenes],
                                         boundary="New v2 periodic validation, not old final-checkpoint reevaluation; Full and Missing targets are separate."), allow_nan=False)+"\n")
    del inputs, labels, clouds, predictions
    head.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "train"), default="train")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--chunks-per-rank", type=int)
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    if args.init_checkpoint:
        args.init_checkpoint = args.init_checkpoint.resolve()
    config = TrainingConfigV2(**json.loads(args.config.read_text()))
    overrides = {}
    for argument, field in (("max_steps", "steps"), ("chunks_per_rank", "chunks_per_rank"),
                            ("expected_world_size", "expected_world_size")):
        value = getattr(args, argument)
        if value is not None:
            if value < 1:
                parser.error(f"--{argument.replace('_', '-')} must be positive")
            overrides[field] = value
    if args.mode == "smoke" and (args.max_steps is None or args.max_steps > 20):
        parser.error("smoke requires --max-steps between 1 and 20")
    config = dataclasses.replace(config, **overrides)
    if args.resume and args.init_checkpoint:
        parser.error("resume and warm initialization are mutually exclusive")
    if args.mode == "preflight" and args.resume:
        parser.error("preflight checks a proposed new run; use resume for an existing run")
    plan = json.loads(args.plan.read_text())
    initialization = dict(mode="random", seed=config.seed)
    if args.init_checkpoint:
        initialization = dict(mode="warm", path=str(args.init_checkpoint), sha256=sha256(args.init_checkpoint))
    if args.resume:
        initialization = json.loads((args.output/"run_manifest.json").read_text()).get("initialization")
    if args.mode == "preflight":
        source = read_training_source(args.plan, plan, config)
        payload = build_data_contract(plan, source.quality_limits.tolist(), minimum_training_scenes(config))
        payload.update(torch=torch.__version__, python=sys.version)
        print(json.dumps(dict(status="preflight_passed", plan_sha256=sha256(args.plan),
                              config_sha256=contract_digest(dataclasses.asdict(config)), code_sha256=code_inventory()[1],
                              data_contract_sha256=contract_digest(payload), initialization=initialization,
                              output=str(args.output), steps=config.steps, world_size=config.expected_world_size,
                              train_scenes=len(source.train), validation_scenes=len(source.val),
                              fixed_validation_scenes=fixed_validation_scenes(source),
                              dependency_files=len(payload["dependency_files"]),
                              boundary="CPU Source loading and hashes only; no Pi3X inference, training, output writes."), allow_nan=False))
        return
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world != config.expected_world_size:
        raise ValueError(f"world size {world} does not match {config.expected_world_size}")
    if world*config.chunks_per_rank < 2:
        raise ValueError("v2 cross-chunk training requires at least two pairs per global step")
    config_dict = dataclasses.asdict(config)
    config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()
    hashes, code_hash = code_inventory()
    plan_hash = sha256(args.plan)
    contract = dict(schema="head_training_v2_run", pipeline_version=2, initialization=initialization,
                    mode=args.mode, plan_sha256=plan_hash,
                    config_sha256=config_hash, code_sha256=code_hash, config=config_dict,
                    model_config=dataclasses.asdict(config.model_config()), code_files=hashes,
                    input_policy=plan["input_policy"], output=str(args.output),
                    per_view_evidence=list(VIEW_EVIDENCE_NAMES), region_evidence=list(REGION_EVIDENCE_NAMES),
                    label_policy="Frozen v1 labels; unavailable remains unknown; no automatic monotonic pair labels.",
                    scope="No SLRF edits, final-checkpoint reevaluation, or physical-region dedup inside this trainer.")

    def inspect_run():
        manifest = args.output/"run_manifest.json"
        if args.resume:
            old = json.loads(manifest.read_text())
            require_same_contract(old, contract,
                                  ("mode", "plan_sha256", "config_sha256", "code_sha256", "output", "pipeline_version", "initialization"),
                                  "resume contract")
            if not (args.output/"head_last.pt").is_file():
                raise ValueError("resume checkpoint missing")
            return
        if args.output.exists():
            existing = set(path.name for path in args.output.iterdir())
            if existing:
                raise ValueError(f"refusing to reuse nonempty output: {sorted(existing)}")

    rank_zero_action(inspect_run, 0, 1)
    source = read_training_source(args.plan, plan, config)
    device = training_device(local_rank)
    if world > 1:
        distributed.init_process_group("nccl")

    def prepare():
        if args.resume:
            return
        args.output.mkdir(parents=True, exist_ok=True)
        existing = set(path.name for path in args.output.iterdir())
        if existing:
            raise ValueError(f"refusing to reuse nonempty output: {sorted(existing)}")
        atomic_json(args.output/"run_manifest.json", contract)
        atomic_json(args.output/"effective_config.json", config_dict)
        atomic_json(args.output/"plan_snapshot.json", plan)

    rank_zero_action(prepare, rank, world)
    root = Path(plan["root"])
    geoff = root/"UAVFF3D/GeoFF3D"
    data_contract_hash = [None]

    def record_data_contract():
        payload = build_data_contract(plan, source.quality_limits.tolist(), minimum_training_scenes(config))
        payload.update(torch=torch.__version__, python=sys.version)
        data_contract_hash[0] = contract_digest(payload)
        destination = args.output/"data_contract.json"
        if args.resume:
            require_same_contract(json.loads(destination.read_text()), payload, tuple(payload), "resume data/dependency contract")
        else:
            atomic_json(destination, payload)

    rank_zero_action(record_data_contract, rank, world)
    if world > 1:
        distributed.broadcast_object_list(data_contract_hash, src=0)
    frozen_dependencies = json.loads((args.output/"data_contract.json").read_text())["dependency_files"]
    sys.path.insert(1, str(geoff))
    os.chdir(geoff)
    from geoff3d.slrf.model_runner import (
        init_model_from_hydra, load_checkpoint, apply_runtime_prior_policy, build_prior_overrides,
    )
    torch.manual_seed(config.seed)
    model, _ = init_model_from_hydra("pi3x", "aws", ["model.model_config.load_pretrained_weights=false"]+
                                    build_prior_overrides("pi3x", plan["input_policy"]), device)
    pi3x_checkpoint = root/"UAVFF3D/pi3x_finetuning/checkpoint-best.pth"
    load_checkpoint(model, str(pi3x_checkpoint))
    apply_runtime_prior_policy(model, plan["input_policy"])
    model.eval().requires_grad_(False)
    source.model = model
    if set(source.train) & set(source.val):
        raise ValueError("training and validation scenes overlap")
    if len(source.train) < minimum_training_scenes(config):
        raise ValueError("not enough distinct training scenes for configured sampling contract")
    scene_ids = {scene: index for index, scene in enumerate(source.records)}

    if dependency_inventory(geoff) != frozen_dependencies:
        raise RuntimeError("GeoFF3D source/configuration changed during model initialization")
    torch.manual_seed(config.seed)
    raw_head = TrainingHeadV2(config).to(device)
    optimizer = torch.optim.AdamW(raw_head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    start = 0
    missing_gradient_verified = False
    observed_pairs = {}
    if args.resume:
        checkpoint = torch.load(args.output/"head_last.pt", map_location=device, weights_only=False)
        for key, value in [("plan_sha256", plan_hash), ("config_sha256", config_hash), ("code_sha256", code_hash),
                           ("data_contract_sha256", data_contract_hash[0])]:
            if checkpoint[key] != value:
                raise ValueError(f"checkpoint contract changed: {key}")
        raw_head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["step"]
        if type(start) is not int or start < 1:
            raise ValueError("invalid checkpoint step")
        progress = checkpoint["rank_progress"]
        if len(progress) != world or any(row["rank"] != index for index, row in enumerate(progress)):
            raise ValueError("checkpoint rank progress does not match world size")
        observed_pairs = dict(progress[rank]["observed_pairs"])
        missing_gradient_verified = bool(progress[rank]["missing_gradient_verified"])
    elif args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        initialization = initialize_head(raw_head, checkpoint)
        initialization.update(path=str(args.init_checkpoint.resolve()), sha256=sha256(args.init_checkpoint))
        rank_zero_action(lambda: atomic_json(args.output/"initialization.json", initialization), rank, world)
        del checkpoint
    if start >= config.steps:
        raise ValueError("checkpoint already reached configured steps; do not silently restart")
    if args.resume:
        rank_zero_action(lambda: recover_after_checkpoint(args.output, start, world), rank, world)
    head = DistributedDataParallel(raw_head, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else raw_head
    prepared_pairs = None
    if hasattr(source, "iter_prepared_pairs"):
        schedule = ((source.train[(scheduled_step+ordinal*world+rank) % len(source.train)],
                     config.seed+rank*100003+scheduled_step*7919+ordinal*104729)
                    for scheduled_step in range(start, config.steps) for ordinal in range(config.chunks_per_rank))
        prepared_pairs = source.iter_prepared_pairs(schedule)
    for step in range(start, config.steps):
        began = time.monotonic()
        source.profiler.reset()
        torch.manual_seed(config.seed+step*997+rank)
        pairs = []
        for ordinal in range(config.chunks_per_rank):
            scene = source.train[(step+ordinal*world+rank) % len(source.train)]
            rng = np.random.default_rng(config.seed+rank*100003+step*7919+ordinal*104729)
            print(json.dumps(dict(event="sample_start", step=step+1, rank=rank, scene=scene)), flush=True)
            if prepared_pairs is None:
                inputs, labels, summary, clouds = source.pair(scene, rng, seed=config.seed)
            else:
                prepared_scene, sample_seed, choice = next(prepared_pairs)
                if prepared_scene != scene or sample_seed != config.seed+rank*100003+step*7919+ordinal*104729:
                    raise RuntimeError("prefetch changed the training sample sequence")
                inputs, labels, summary, clouds = source.pair(scene, rng, seed=config.seed, choice=choice)
            pairs.append((inputs, labels, summary))
            del clouds
        inputs, labels = concatenate_pairs(pairs, rank, world)
        labels["scene_ids"] = np.concatenate([np.full(len(pair[1]["groups"]), scene_ids[pair[2]["scene"]], dtype=np.int64) for pair in pairs])
        labels = labels_to_device(labels, device)
        if not labels["total_mask"].any() or not labels["structure_mask"].any():
            raise RuntimeError("global-step sample has no supervised Full or Missing regions")
        if args.mode == "smoke" and config.missing_total_local_ranking_weight > 0 and not missing_gradient_verified:
            probe_inputs, _ = modality_inputs(inputs, labels["pair_ids"], config, training=False)
            probe = isolated_missing_gradient(raw_head, probe_inputs, labels, config)
            missing_gradient_verified = probe["pairs"] > 0 and probe["gradient_norms"]["total_fuse"] > 0
            atomic_json(args.output/f"missing_gradient_rank{rank}_step{step+1:06d}.json", probe)
            del probe_inputs
        if step == start or args.mode == "smoke":
            atomic_json(args.output/f"sample_audit_rank{rank}_step{step+1:06d}.json", [pair[2] for pair in pairs])
        generator = torch.Generator().manual_seed(config.seed+step*65537+rank)
        model_inputs, dropped = modality_inputs(inputs, labels["pair_ids"], config, generator)
        warmup = min(1.0, (step+1)/max(config.warmup_steps, 1))
        learning_rate = config.learning_rate*warmup*(0.01+0.99*(1+math.cos(math.pi*step/config.steps))/2)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        predictions = head(model_inputs)
        loss, pieces, statistics = training_loss(predictions, labels, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        loss.backward()
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("frozen Pi3X received gradients")
        norm = torch.nn.utils.clip_grad_norm_(raw_head.parameters(), 1.0, error_if_nonfinite=True)
        gradients = branch_gradients(raw_head)
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in raw_head.parameters()):
            raise FloatingPointError("non-finite head weights")
        if not all(torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)):
            raise FloatingPointError("non-finite optimizer state")
        for name, value in statistics.items():
            observed_pairs[name] = observed_pairs.get(name, 0)+value["pairs"]
        record = dict(event="optimizer_step", rank=rank, step=step+1, loss=float(loss.detach()),
                      learning_rate=learning_rate, grad_norm=float(norm), branch_grad_norms=gradients,
                      losses={name: float(value.detach()) for name, value in pieces.items()},
                      ranking=statistics, dropout=dropped, samples=[compact_sample(pair[2]) for pair in pairs],
                      performance=source.performance_snapshot(),
                      seconds=time.monotonic()-began)
        with (args.output/f"rank{rank}_steps.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
        print(json.dumps({key: value for key, value in record.items() if key != "samples"}), flush=True)
        if rank == 0:
            atomic_json(args.output/"latest_step.json", record)
        del inputs, model_inputs, labels, pairs, predictions, loss, pieces
        if config.validation_every and (step+1) % config.validation_every == 0:
            fixed_validation(head, source, config, device, rank, world, step+1, args.output)
        if step == start or (step+1) % config.save_every == 0 or step+1 == config.steps:
            progress = dict(rank=rank, observed_pairs=observed_pairs, missing_gradient_verified=missing_gradient_verified)
            rank_progress = [None for _ in range(world)]
            if world > 1:
                distributed.all_gather_object(rank_progress, progress)
            else:
                rank_progress[0] = progress
            def save_checkpoint():
                if code_inventory()[1] != code_hash or dependency_inventory(geoff) != frozen_dependencies:
                    raise RuntimeError("training code or GeoFF3D dependency changed; refusing checkpoint publication")
                destination = args.output/f"head_step_{step+1:06d}.pt"
                if destination.exists():
                    raise ValueError(f"refusing to overwrite checkpoint: {destination}")
                temporary = destination.with_suffix(".pt.tmp")
                torch.save(dict(schema="head_training_v2_checkpoint", head=raw_head.state_dict(),
                                optimizer=optimizer.state_dict(), step=step+1,
                                config=dataclasses.asdict(config.model_config()), training_config=config_dict,
                                plan_sha256=plan_hash, code_sha256=code_hash, config_sha256=config_hash,
                                data_contract_sha256=data_contract_hash[0], rank_progress=rank_progress), temporary)
                temporary.replace(destination)
                latest = args.output/"head_last.pt.tmp"
                if latest.exists():
                    latest.unlink()
                os.link(destination, latest)
                latest.replace(args.output/"head_last.pt")
            rank_zero_action(save_checkpoint, rank, world)
    if prepared_pairs is not None:
        prepared_pairs.close()
        source.close()
    requirements = smoke_requirements(config, observed_pairs, missing_gradient_verified)
    passed = args.mode != "smoke" or requirements["passed"]
    flag = torch.tensor(int(passed), device=device)
    if world > 1:
        distributed.all_reduce(flag, op=distributed.ReduceOp.MIN)
    if code_inventory()[1] != code_hash:
        raise RuntimeError("source code changed while this run was executing")
    result = dict(status="passed" if bool(flag.item()) else "failed_smoke_gate", mode=args.mode,
                  rank=rank, step=config.steps, observed_ranking_pairs=observed_pairs,
                  missing_total_gradient_verified=missing_gradient_verified,
                  smoke_requirements=requirements if args.mode == "smoke" else None,
                  boundary="Execution/gradient evidence only; no accuracy, physical-defect, or multimodal-gain claim.")
    atomic_json(args.output/f"completion_rank{rank}.json", result)
    if world > 1:
        distributed.destroy_process_group()
    if not flag.item():
        raise RuntimeError("smoke did not exercise every required ranking/gradient path")


if __name__ == "__main__":
    main()
