import dataclasses
import importlib.util
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

from openflyscan.quality_predictor.region_features import pack_current_region_inputs
from openflyscan.quality_predictor.paired_observation_source import concatenate_pairs
from openflyscan.quality_predictor.geometry_evidence import (
    REGION_EVIDENCE_NAMES, VIEW_EVIDENCE_NAMES, augment_current_inputs,
    current_geometry_evidence, label_support_audit, self_projection_audit,
)
from openflyscan.quality_predictor.training_base import (
    TrainingConfigV2, TrainingHeadV2, initialize_head, labels_to_device,
    modality_inputs, ranking_loss, smoke_requirements, training_loss,
)
from openflyscan.quality_predictor.model import LocalQueryInputs, LocalQueryQualityConfig, LocalQueryQualityHead
from openflyscan.quality_predictor.region_selection import select_regions


def load_entrypoint(name):
    path = Path(__file__).resolve().parents[1]/"scripts"/(name+".py")
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


control_configurations = load_entrypoint("make_quality_controls").control_configurations
trainer_entrypoint = load_entrypoint("train_quality_predictor_legacy")
fixed_validation = trainer_entrypoint.fixed_validation


def planar_sample():
    rows, columns = np.indices((4, 6))
    intrinsic = np.array([[100.0, 0, 42], [0, 100.0, 28], [0, 0, 1.0]])
    rays = np.stack([(columns+0.5)*14, (rows+0.5)*14, np.ones_like(columns)], -1) @ np.linalg.inv(intrinsic).T
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 0, 3] = [-1, 0, 1]
    maps = np.stack([rays*10+pose[:3, 3] for pose in poses])
    cloud = dict(maps=maps, camera_poses=poses, intrinsics=np.tile(intrinsic, (3, 1, 1)), conf=np.ones((3, 4, 6)))
    proposal = dict(centers=np.array([[0.0, 0, 10], [1000.0, 0, 10]]), origin=np.array([0.0, 0, 10]),
                    cell_size=2.0, valid_points=maps.reshape(-1, 3))
    torch.manual_seed(7)
    inputs = pack_current_region_inputs(cloud, proposal, *[torch.randn(3, 4, 6, 1024) for _ in range(3)], torch.randn(3, 4, 6, 6))
    return cloud, proposal, inputs


def synthetic_inputs(regions=8):
    config = TrainingConfigV2().model_config()
    torch.manual_seed(4)
    return LocalQueryInputs(
        torch.randn(regions, 3, 1024), torch.randn(regions, 3, 1024), torch.randn(regions, 3, 1024),
        torch.randn(regions, 3, 10), torch.randn(regions, 3, config.quality_geometry_dim),
        torch.randn(regions, 3, 6), torch.randn(regions, config.query_dim), torch.ones(regions, 3, dtype=torch.bool))


def distributed_labels(rank):
    if rank == 0:
        features = [0.2, 0.7, 0.1]
        target = [0.1, 0.9, 0.3]
        state = torch.tensor([0, 0, 1])
    else:
        features = [0.3, 0.8, 0.4, 0.9]
        target = [0.7, 0.2, 0.9, 0.1]
        state = torch.tensor([0, 0, 1, 1])
    labels = dict(total=torch.tensor(target), structure=torch.tensor(target), total_mask=state == 0,
                  structure_mask=state == 1, state=state, groups=rank*2+state,
                  scene_ids=torch.full_like(state, rank), pair_ids=torch.full_like(state, rank))
    return torch.tensor(features)[:, None], labels


def cross_only_config():
    return TrainingConfigV2(full_regression_weight=0, missing_structure_regression_weight=0,
                            full_local_ranking_weight=0, missing_structure_local_ranking_weight=0,
                            missing_total_local_ranking_weight=0, full_cross_ranking_weight=1,
                            missing_total_cross_ranking_weight=1, missing_structure_cross_ranking_weight=0)


def distributed_worker(rank, init_file, output_dir):
    torch.set_num_threads(1)
    distributed.init_process_group("gloo", init_method="file://"+init_file, rank=rank, world_size=2)
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    wrapped = DistributedDataParallel(model)
    features, labels = distributed_labels(rank)
    prediction = wrapped(features).squeeze(-1)
    loss, _, counts = training_loss(dict(total=prediction, structure=prediction), labels, cross_only_config())
    loss.backward()
    Path(output_dir, f"rank{rank}.json").write_text(json.dumps(dict(gradient=float(model.weight.grad), counts=counts)))
    distributed.destroy_process_group()


def distributed_head_worker(rank, init_file, output_dir):
    torch.set_num_threads(1)
    distributed.init_process_group("gloo", init_method="file://"+init_file, rank=rank, world_size=2)
    torch.manual_seed(7)
    config = TrainingConfigV2()
    model = TrainingHeadV2(config)
    wrapped = DistributedDataParallel(model, find_unused_parameters=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    features, labels = distributed_labels(rank)
    inputs = synthetic_inputs(len(features))
    before = model.head.total_fuse[-1].weight.detach().clone()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _, counts = training_loss(wrapped(inputs), labels, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    weight = model.head.total_fuse[-1].weight.detach()
    Path(output_dir, f"rank{rank}.json").write_text(json.dumps(dict(
        finite=bool(torch.isfinite(loss)), changed=not bool(torch.equal(before, weight)),
        weight=weight.tolist(), cross_pairs=counts["missing_total_cross_ranking"]["pairs"])))
    distributed.destroy_process_group()


class GeometryEvidenceTests(unittest.TestCase):
    def test_consistency_and_empty_region(self):
        cloud, proposal, inputs = planar_sample()
        augmented, audit = augment_current_inputs(inputs, cloud, proposal)
        self.assertEqual(audit[0]["consistent_2pct"], 3)
        self.assertEqual(audit[1]["projectable"], 0)
        self.assertEqual(augmented.query_features.shape[-1], 10+len(REGION_EVIDENCE_NAMES))
        self.assertEqual(augmented.quality_geometry_features.shape[-1], 4+len(VIEW_EVIDENCE_NAMES))
        self.assertTrue(torch.isfinite(TrainingHeadV2(TrainingConfigV2()).eval()(augmented)["total"]).all())

    def test_conflict_is_retained_even_with_high_confidence(self):
        cloud, proposal, inputs = planar_sample()
        cloud["maps"][2, ..., 2] += 3
        cloud["conf"][:] = 8
        per_view, region, audit = current_geometry_evidence(cloud, proposal)
        self.assertEqual(audit[0]["projectable"], 3)
        self.assertEqual(audit[0]["consistent_2pct"], 2)
        self.assertGreater(per_view[0, 2, 0], 0.2)
        self.assertGreater(region[0, REGION_EVIDENCE_NAMES.index("behind_surface_conflict_fraction_5pct")], 0)
        self.assertTrue(inputs.view_mask[0, 2])

    def test_similarity_transform_does_not_manufacture_conflict(self):
        cloud, proposal, _ = planar_sample()
        original_view, original_region, _ = current_geometry_evidence(cloud, proposal)
        rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
        scale, translation = 3.0, np.array([120.0, -73, 19])
        moved = {key: np.array(value, copy=True) for key, value in cloud.items()}
        moved["maps"] = scale*cloud["maps"] @ rotation.T+translation
        moved["camera_poses"][:, :3, :3] = rotation @ cloud["camera_poses"][:, :3, :3]
        moved["camera_poses"][:, :3, 3] = scale*cloud["camera_poses"][:, :3, 3] @ rotation.T+translation
        moved_proposal = {**proposal, "centers": scale*proposal["centers"] @ rotation.T+translation}
        actual_view, actual_region, _ = current_geometry_evidence(moved, moved_proposal)
        np.testing.assert_allclose(actual_view, original_view, atol=1e-6)
        np.testing.assert_allclose(actual_region, original_region, atol=1e-6)
        self.assertLess(self_projection_audit(moved)["p90_pixels"], 1e-6)

    def test_unavailable_teacher_stays_unknown(self):
        cloud, proposal, inputs = planar_sample()
        values = np.full(cloud["maps"].shape[:-1], np.nan)
        audit = label_support_audit(cloud, proposal, values, inputs.view_mask.numpy())
        self.assertEqual(audit["valid"], 0)
        self.assertEqual(audit["unavailable_despite_two_projectable_views"], 1)
        self.assertIsNone(audit["rows"][0]["median"])

    def test_view_permutation_does_not_change_prediction(self):
        cloud, proposal, inputs = planar_sample()
        augmented, _ = augment_current_inputs(inputs, cloud, proposal)
        head = TrainingHeadV2(TrainingConfigV2()).eval()
        shuffled = dataclasses.replace(augmented, **{
            field.name: getattr(augmented, field.name)[:, [2, 0, 1]]
            for field in dataclasses.fields(augmented) if field.name != "query_features"
        })
        torch.testing.assert_close(head(augmented)["total"], head(shuffled)["total"], atol=1e-6, rtol=1e-5)


class TrainingV2Tests(unittest.TestCase):
    def test_direct_training_reaches_data_loading_without_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root/"not_started"
            plan, config = root/"plan.json", root/"config.json"
            plan.write_text(json.dumps(dict(input_policy={"pose": "input"})))
            config.write_text(json.dumps(dataclasses.asdict(TrainingConfigV2(chunks_per_rank=2, expected_world_size=1))))
            arguments = ["train_quality_predictor_legacy.py", "--plan", str(plan), "--config", str(config),
                         "--output", str(output), "--mode", "train"]
            with mock.patch.object(trainer_entrypoint, "code_inventory", return_value=({}, "test_hash")), \
                 mock.patch.object(trainer_entrypoint, "read_training_source", side_effect=ValueError("data fixture reached")) as source, \
                 mock.patch.object(torch.cuda, "set_device"), \
                 mock.patch("sys.argv", arguments), \
                 mock.patch.dict("os.environ", {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}):
                with self.assertRaisesRegex(ValueError, "data fixture reached"):
                    trainer_entrypoint.main()
            source.assert_called_once()
            self.assertFalse(output.exists())

    def test_resume_rejects_changed_contract_before_data_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/"old_run"
            output.mkdir()
            manifest = output/"run_manifest.json"
            old = json.dumps(dict(mode="smoke", plan_sha256="different_plan"))
            manifest.write_text(old)
            plan, config = root/"plan.json", root/"config.json"
            plan.write_text(json.dumps(dict(input_policy={"pose": "input"})))
            config.write_text(json.dumps(dataclasses.asdict(TrainingConfigV2(chunks_per_rank=2, expected_world_size=1))))
            arguments = ["train_quality_predictor_legacy.py", "--plan", str(plan), "--config", str(config),
                         "--output", str(output), "--mode", "smoke", "--max-steps", "2", "--resume"]
            with mock.patch.object(trainer_entrypoint, "code_inventory", return_value=({}, "test_hash")), \
                 mock.patch.object(torch.cuda, "set_device"), \
                 mock.patch("sys.argv", arguments), \
                 mock.patch.dict("os.environ", {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}):
                with self.assertRaisesRegex(RuntimeError, "resume contract changed: plan_sha256"):
                    trainer_entrypoint.main()
            self.assertEqual(manifest.read_text(), old)

    def test_nonfinite_configuration_rejected(self):
        for name in ("learning_rate", "missing_total_local_ranking_weight", "dino_dropout"):
            with self.assertRaises(ValueError):
                TrainingConfigV2(**{name: float("nan")})

    def test_control_stages_change_only_the_intended_settings(self):
        controls = control_configurations(TrainingConfigV2())
        stages = [dataclasses.asdict(controls[name]) for name in (
            "01_v1_loss_control_new_inputs", "02_missing_total_only", "03_with_cross_chunk", "04_with_modality_dropout")]
        differences = [{name for name in left if left[name] != right[name]} for left, right in zip(stages, stages[1:])]
        self.assertEqual(differences[0], {"missing_total_local_ranking_weight"})
        self.assertEqual(differences[1], {"full_cross_ranking_weight", "missing_total_cross_ranking_weight", "missing_structure_cross_ranking_weight"})
        self.assertEqual(differences[2], {"dino_dropout", "confidence_dropout"})
        self.assertEqual(controls["dino_only"].model_config(), controls["confidence_only"].model_config())

    def test_smoke_checks_only_enabled_paths_without_weakening_default(self):
        config = TrainingConfigV2()
        paths = smoke_requirements(config, {}, False)["required_ranking_paths"]
        counts = {name: 1 for name in paths}
        self.assertFalse(smoke_requirements(config, counts, False)["passed"])
        self.assertTrue(smoke_requirements(config, counts, True)["passed"])
        counts["missing_total_cross_ranking"] = 0
        self.assertFalse(smoke_requirements(config, counts, True)["passed"])
        control = control_configurations(config)["01_v1_loss_control_new_inputs"]
        counts = {"full_local_ranking": 1, "missing_structure_local_ranking": 1}
        self.assertTrue(smoke_requirements(control, counts, False)["passed"])

    def test_periodic_validation_keeps_three_output_contracts_separate(self):
        class Source:
            val = ["unseen"]

            def pair(self, scene, rng, seed):
                state = np.array([0, 0, 0, 0, 1, 1, 1, 1])
                labels = dict(total=np.array([0.1, 0.3, 0.6, 0.9, 0, 0, 0, 0]),
                              structure=np.array([0, 0, 0, 0, 0.9, 0.6, 0.3, 0.1]),
                              total_mask=state == 0, structure_mask=state == 1, groups=state)
                return synthetic_inputs(), labels, {}, None

        class Head(torch.nn.Module):
            def forward(self, inputs):
                return dict(total=torch.tensor([0.1, 0.3, 0.6, 0.9, 0.9, 0.6, 0.3, 0.1]),
                            structure=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.6, 0.9]))

        head = Head()
        with tempfile.TemporaryDirectory() as directory:
            fixed_validation(head, Source(), TrainingConfigV2(), "cpu", 0, 1, 200, Path(directory))
            record = json.loads((Path(directory)/"validation.jsonl").read_text())
        self.assertTrue(head.training)
        self.assertEqual(set(record["metrics"]), {"full_total", "missing_total", "missing_structure"})
        self.assertEqual(record["metrics"]["full_total"]["macro_recall"], 1.0)
        self.assertEqual(record["metrics"]["missing_total"]["macro_recall"], 1.0)
        self.assertEqual(record["metrics"]["missing_structure"]["macro_recall"], 0.0)
        self.assertEqual(record["metrics"]["missing_total"]["pooled"]["regions"], 4)

    def test_missing_ranking_reaches_total_fusion(self):
        config = TrainingConfigV2()
        head = TrainingHeadV2(config).eval()
        prediction = head(synthetic_inputs())["total"]
        loss, counts = ranking_loss(prediction, torch.linspace(0, 1, 8), torch.ones(8, dtype=torch.bool), torch.zeros(8, dtype=torch.long), config)
        loss.backward()
        self.assertGreater(counts["pairs"], 0)
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in head.head.total_fuse.parameters()), 0)

    def test_cross_ranking_compares_distinct_chunks_and_scenes(self):
        target = torch.tensor([0.1, 0.2, 0.8, 0.9])
        groups = torch.tensor([0, 0, 2, 2])
        mask = torch.ones(4, dtype=torch.bool)
        scenes = torch.tensor([0, 0, 1, 1])
        good, counts = ranking_loss(target, target, mask, groups, TrainingConfigV2(), cross=True, scene_ids=scenes)
        bad, _ = ranking_loss(1-target, target, mask, groups, TrainingConfigV2(), cross=True, scene_ids=scenes)
        self.assertEqual(counts["pairs"], 4)
        self.assertEqual(counts["cross_scene_pairs"], 4)
        self.assertLess(good, bad)

    def test_full_and_missing_never_cross_compare(self):
        state = torch.tensor([0, 0, 1, 1])
        labels = dict(total=torch.tensor([0.0, 1.0, 123.0, 999.0]), structure=torch.tensor([999.0, 888.0, 0.1, 0.9]),
                      total_mask=state == 0, structure_mask=state == 1, state=state, groups=state)
        prediction = torch.tensor([0.1, 0.8, 0.2, 0.7], requires_grad=True)
        loss, _, counts = training_loss(dict(total=prediction, structure=prediction), labels, TrainingConfigV2())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(counts["full_cross_ranking"]["pairs"], 0)
        self.assertEqual(counts["missing_total_cross_ranking"]["pairs"], 0)

    def test_empty_masks_have_finite_zero_loss(self):
        prediction = torch.randn(5, requires_grad=True)
        loss, counts = ranking_loss(prediction, torch.zeros(5), torch.zeros(5, dtype=torch.bool), torch.zeros(5, dtype=torch.long), TrainingConfigV2(), cross=True)
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertEqual(counts["pairs"], 0)

    def test_dropout_is_pair_coherent_and_exclusive(self):
        inputs = synthetic_inputs()
        pair_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        config = TrainingConfigV2(dino_dropout=0.5, confidence_dropout=0.5)
        modified, counts = modality_inputs(inputs, pair_ids, config, torch.Generator().manual_seed(3))
        self.assertEqual(counts["dino"]+counts["confidence"], 2)
        for pair in [0, 1]:
            selected = pair_ids == pair
            empty_dino = bool((modified.dino_features[selected] == 0).all())
            empty_confidence = bool((modified.confidence_features[selected] == 0).all())
            self.assertNotEqual(empty_dino, empty_confidence)
        torch.testing.assert_close(modified.view_mask, inputs.view_mask)

    def test_dino_control_has_no_geometry_content(self):
        inputs = synthetic_inputs()
        modified, _ = modality_inputs(inputs, torch.zeros(8, dtype=torch.long), TrainingConfigV2(modality="dino_only"))
        torch.testing.assert_close(modified.dino_features, inputs.dino_features)
        self.assertFalse(modified.query_features.any())
        self.assertFalse(modified.quality_geometry_features.any())
        self.assertFalse(modified.confidence_features.any())

    def test_v1_warm_start_only_resets_expanded_projectors(self):
        old = LocalQueryQualityHead(LocalQueryQualityConfig())
        current = TrainingHeadV2(TrainingConfigV2())
        report = initialize_head(current, dict(head=old.state_dict(), step=7500))
        self.assertTrue(report["reset_tensors"])
        self.assertFalse(report["optimizer_restored"])
        torch.testing.assert_close(current.head.total_fuse[0].weight, old.total_fuse[0].weight)

    def test_pair_group_ids_are_unique_across_ranks(self):
        inputs = synthetic_inputs(4)
        labels = dict(total=np.zeros(4), structure=np.zeros(4), total_mask=np.array([True, True, False, False]),
                      structure_mask=np.array([False, False, True, True]), groups=np.array([0, 0, 1, 1]))
        pairs = [(inputs, labels, {})]*2
        _, left = concatenate_pairs(pairs, rank=0, world_size=2)
        _, right = concatenate_pairs(pairs, rank=1, world_size=2)
        self.assertFalse(set(left["groups"]) & set(right["groups"]))
        self.assertEqual(len(np.unique(left["groups"])), 4)

    @unittest.skipUnless(distributed.is_available() and distributed.is_gloo_available(), "gloo required")
    def test_actual_head_runs_two_distributed_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            processes = [context.Process(target=distributed_head_worker, args=(rank, str(Path(directory)/"init"), directory)) for rank in [0, 1]]
            for process in processes:
                process.start()
            for process in processes:
                process.join(60)
                if process.is_alive():
                    process.kill()
                    process.join()
                self.assertEqual(process.exitcode, 0)
            records = [json.loads(Path(directory, f"rank{rank}.json").read_text()) for rank in [0, 1]]
            for record in records:
                self.assertTrue(record["finite"])
                self.assertTrue(record["changed"])
                self.assertGreater(record["cross_pairs"], 0)
            np.testing.assert_allclose(records[0]["weight"], records[1]["weight"], atol=1e-7)

    @unittest.skipUnless(distributed.is_available() and distributed.is_gloo_available(), "gloo required")
    def test_two_rank_ragged_gather_matches_centralized_gradient(self):
        torch.set_num_threads(1)
        features, labels = zip(*(distributed_labels(rank) for rank in [0, 1]))
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        prediction = model(torch.cat(features)).squeeze(-1)
        combined = {name: torch.cat([part[name] for part in labels]) for name in labels[0]}
        loss, _, _ = training_loss(dict(total=prediction, structure=prediction), combined, cross_only_config())
        loss.backward()
        expected = float(model.weight.grad)
        self.assertNotEqual(expected, 0)
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            processes = [context.Process(target=distributed_worker, args=(rank, str(Path(directory)/"init"), directory)) for rank in [0, 1]]
            for process in processes:
                process.start()
            for process in processes:
                process.join(60)
                if process.is_alive():
                    process.kill()
                    process.join()
                self.assertEqual(process.exitcode, 0)
            for rank in [0, 1]:
                actual = json.loads(Path(directory, f"rank{rank}.json").read_text())
                self.assertAlmostEqual(actual["gradient"], expected, places=6)
                self.assertGreater(actual["counts"]["missing_total_cross_ranking"]["pairs"], 0)


class RegionSelectionTests(unittest.TestCase):
    def candidate(self, chunk, location, score=0.9):
        return dict(scene="scene", frame_id="verified_frame", alignment_verified=True,
                    chunk=chunk, xyz=[location, 0, 10], radius=1.0, score=score,
                    image_evidence=[dict(stem="shared", patch_hw=[20, 37], patch_pixels_yx=[[3, 4], [3, 5], [4, 4]])])

    def test_repeated_surface_uses_one_budget_slot(self):
        candidates = [self.candidate(0, 0), self.candidate(1, 0.1, 0.8), self.candidate(2, 5, 0.7)]
        result = select_regions(candidates, 2)
        self.assertEqual(result["deduplicated_candidates"], 2)
        self.assertEqual(len(result["selected"]), 2)
        self.assertAlmostEqual(result["selected"][0]["score"], 0.85)

    def test_nearby_points_without_shared_evidence_do_not_merge(self):
        candidates = [self.candidate(0, 0), self.candidate(1, 0.1)]
        candidates[1]["image_evidence"][0]["stem"] = "different"
        self.assertEqual(select_regions(candidates, 2)["deduplicated_candidates"], 2)

    def test_unknown_alignment_stays_separate(self):
        candidates = [self.candidate(0, 0), self.candidate(1, 0.1)]
        candidates[1]["alignment_verified"] = False
        result = select_regions(candidates, 2)
        self.assertEqual(result["deduplicated_candidates"], 2)
        self.assertEqual(result["unverified_alignment_candidates"], 1)

    def test_invalid_normals_do_not_create_correspondence(self):
        candidates = [self.candidate(0, 0), self.candidate(1, 0.1)]
        candidates[0]["normal"] = [0.0, 0.0, 0.0]
        candidates[1]["normal"] = [0.0, 0.0, 1.0]
        self.assertEqual(select_regions(candidates, 2)["deduplicated_candidates"], 2)
        candidates[0]["normal"] = [float("nan"), 0.0, 1.0]
        with self.assertRaises(ValueError):
            select_regions(candidates, 2)

    def test_invalid_correspondence_metadata_is_rejected(self):
        candidate = self.candidate(0, 0)
        candidate["alignment_verified"] = "false"
        with self.assertRaises(ValueError):
            select_regions([candidate], 1)
        candidate["alignment_verified"] = True
        candidate["image_evidence"][0]["patch_pixels_yx"] = [[-1, 0]]
        with self.assertRaises(ValueError):
            select_regions([candidate], 1)

    def test_no_transitive_chain_merging_or_cross_scene_merge(self):
        candidates = [self.candidate(0, 0), self.candidate(1, 0.4), self.candidate(2, 0.8)]
        self.assertEqual(select_regions(candidates, 3)["deduplicated_candidates"], 2)
        candidates[1]["scene"] = "other_scene"
        self.assertEqual(select_regions(candidates, 3)["deduplicated_candidates"], 3)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
