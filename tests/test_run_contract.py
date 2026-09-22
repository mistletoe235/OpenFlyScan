import contextlib
import copy
import dataclasses
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from openflyscan.quality_predictor.training_base import TrainingConfigV2
from openflyscan.quality_predictor.model import LocalQueryInputs
from openflyscan.quality_predictor.training_performance import StageProfiler
from openflyscan.quality_predictor.run_contract import (
    build_data_contract, contract_digest, dependency_inventory,
    recover_after_checkpoint, require_same_contract, validate_plan,
)


ENTRYPOINT = Path(__file__).resolve().parents[1]/"scripts/train_quality_predictor_legacy.py"
SPECIFICATION = importlib.util.spec_from_file_location("pipeline_trainer", ENTRYPOINT)
trainer = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(trainer)


def fixture_plan(root):
    geoff = root/"UAVFF3D/GeoFF3D"
    for relative in ("geoff3d/slrf/model_runner.py", "geoff3d/models/external/pi3x/__init__.py",
                     "geoff3d/models/external/pi3/models/pi3x.py", "configs/model/pi3x.yaml"):
        path = geoff/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value: 1\n" if path.suffix == ".yaml" else "value = 1\n")
    checkpoint = root/"UAVFF3D/pi3x_finetuning/checkpoint-best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture; model loading is mocked")
    manifest = root/"dino_manifest.json"
    manifest.write_text(json.dumps(dict(entries=[])))
    records = []
    for index in range(4):
        scene = f"scene_{index}"
        directory = root/scene
        for name in ("source", "full", "teacher"):
            (directory/name).mkdir(parents=True)
        names = np.array([f"image_{view:03d}" for view in range(30)])
        np.savez(directory/"footprints.npz", stems=names, centers=np.zeros((30, 2)),
                 bbox_mins=np.zeros((30, 2)), bbox_maxs=np.ones((30, 2)))
        np.savez(directory/"teacher/grid_errors.npz", names=names,
                 mse=np.linspace(0.001, 0.1, 30*16*16).reshape(30, 16, 16),
                 valid_pixel_weight=np.ones((30, 16, 16)))
        (directory/"teacher/metrics.json").write_text(json.dumps(dict(trained_exposure=True, grid=16)))
        records.append(dict(scene=scene, split="train" if index < 3 else "validation", views=30,
                            source=str(directory/"source"), full_root=str(directory/"full"),
                            evaluation=str(directory/"teacher"), footprints=str(directory/"footprints.npz")))
    plan = dict(root=str(root), records=records, dino_manifest=str(manifest),
                input_policy=dict(model="pi3x", pose="input", ray="input", depth="none"))
    path = root/"plan.json"
    path.write_text(json.dumps(plan))
    return path, plan


class SyntheticSource:
    def __init__(self, path, model, max_regions):
        records = json.loads(Path(path).read_text())["records"]
        self.records = {record["scene"]: record for record in records}
        self.train = [record["scene"] for record in records if record["split"] == "train"]
        self.val = [record["scene"] for record in records if record["split"] == "validation"]
        self.quality_limits = np.array([0.1, 0.9])
        self.model = model
        self.profiler = StageProfiler()

    def performance_snapshot(self):
        return dict(stages=self.profiler.snapshot())

    def iter_prepared_pairs(self, sequence):
        for scene, sample_seed in sequence:
            yield scene, sample_seed, None

    def close(self):
        pass

    def pair(self, scene, rng, seed, choice=None):
        sample = int(rng.integers(0, 100000))
        generator = torch.Generator().manual_seed(seed+sample)
        def features(width):
            return torch.randn(8, 3, width, generator=generator)
        inputs = LocalQueryInputs(features(1024), features(1024), features(1024), features(10),
                                  features(8), features(6), torch.randn(8, 32, generator=generator),
                                  torch.ones(8, 3, dtype=torch.bool))
        state = np.array([0]*4+[1]*4)
        labels = dict(total=np.array([0.1, 0.3, 0.6, 0.9, 0, 0, 0, 0], dtype=np.float32),
                      structure=np.array([0, 0, 0, 0, 0.9, 0.6, 0.3, 0.1], dtype=np.float32),
                      total_mask=state == 0, structure_mask=state == 1, groups=state)
        summary = dict(scene=scene, pair_hash=str(sample), choice=dict(deleted=list(range(10))),
                       label_audit={name: dict(rows=[], valid=4) for name in ("full", "missing")},
                       evidence_audit={name: [dict(projectable=3, consistent_2pct=2)] for name in ("full", "missing")})
        return inputs, labels, summary, None


class TrainingRunTests(unittest.TestCase):
    def test_candidate_schema_is_rejected_with_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "adapt the candidate plan first"):
            validate_plan(dict(train=[], validation=[]), 2)

    def test_declared_policy_mismatch_and_physical_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan = fixture_plan(Path(directory))
            modified = copy.deepcopy(plan)
            modified["records"][-1]["input_policy"] = dict(plan["input_policy"], pose="none")
            with self.assertRaisesRegex(ValueError, "one global policy"):
                validate_plan(modified, 2)
            modified = copy.deepcopy(plan)
            for record in modified["records"][:2]:
                record["physical_scene_id"] = "same_building"
            with self.assertRaisesRegex(ValueError, "grouped sampling is not integrated"):
                validate_plan(modified, 2)

    def test_same_split_physical_groups_require_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan = fixture_plan(Path(directory))
            plan['allow_same_split_physical_groups'] = True
            for record in plan['records'][:2]:
                record['physical_scene_id'] = 'same_building'
            train, validation = validate_plan(plan, 2)
            self.assertEqual(train, [record['scene'] for record in plan['records'] if record['split'] == 'train'])
            self.assertEqual(validation, [record['scene'] for record in plan['records'] if record['split'] == 'validation'])
            plan['records'][-1]['physical_scene_id'] = 'same_building'
            with self.assertRaisesRegex(ValueError, 'cross-split physical_scene_id'):
                validate_plan(plan, 2)

    def test_external_model_and_hydra_changes_invalidate_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, plan = fixture_plan(root)
            original = build_data_contract(plan, [0.1, 0.9], 2)
            geoff = root/"UAVFF3D/GeoFF3D"
            for relative in ("geoff3d/models/external/pi3/models/pi3x.py", "configs/model/pi3x.yaml"):
                path = geoff/relative
                self.assertIn(relative, original["dependency_files"])
                before = path.read_text()
                path.write_text(before.replace("1", "2"))
                changed = build_data_contract(plan, [0.1, 0.9], 2)
                self.assertNotEqual(contract_digest(original), contract_digest(changed))
                with self.assertRaisesRegex(ValueError, "dependency_files"):
                    require_same_contract(original, changed, tuple(original), "resume data/dependency contract")
                path.write_text(before)
            self.assertEqual(dependency_inventory(geoff), original["dependency_files"])

    def test_sfm_correspondence_assets_and_reader_are_bound_to_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, plan = fixture_plan(root)
            record = plan["records"][0]
            record["teacher_mapping"] = "rectified_resize"
            reader = root/"open-lixel-h3dgs-color11-20260808/preprocess/read_write_model.py"
            reader.parent.mkdir(parents=True)
            reader.write_text("parser = 1\n")
            for folder in ("rgb_sfm", "published_scene_source/sparse/0"):
                for name in ("cameras.bin", "images.bin"):
                    path = Path(record["full_root"])/folder/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"original")
            original = build_data_contract(plan, [.1, .9], 2)
            self.assertIn(str(reader), original["artifacts"])
            path = Path(record["full_root"])/"published_scene_source/sparse/0/images.bin"
            path.write_bytes(b"changed correspondences")
            modified = build_data_contract(plan, [.1, .9], 2)
            self.assertNotEqual(contract_digest(original), contract_digest(modified))

    def test_cpu_preflight_uses_real_source_without_output_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, _ = fixture_plan(root)
            config = root/"config.json"
            config.write_text(json.dumps(dataclasses.asdict(TrainingConfigV2(expected_world_size=1, chunks_per_rank=2))))
            output = root/"not_created"
            arguments = [str(ENTRYPOINT), "--plan", str(plan), "--config", str(config),
                         "--output", str(output), "--mode", "preflight"]
            with mock.patch("sys.argv", arguments), mock.patch.object(trainer, "training_device") as device, \
                 contextlib.redirect_stdout(io.StringIO()) as stream:
                trainer.main()
            device.assert_not_called()
            self.assertFalse(output.exists())
            record = json.loads(stream.getvalue())
            self.assertEqual(record["status"], "preflight_passed")
            self.assertEqual(record["train_scenes"], 3)
            self.assertEqual(len(record["data_contract_sha256"]), 64)

    def test_resume_archives_uncommitted_logs_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            original = '{"step": 1}\n{"step": 2}\n{"step": '
            (output/"rank0_steps.jsonl").write_text(original)
            (output/"validation.jsonl").write_text('{"step": 2}\n')
            (output/"head_step_000001.pt").write_bytes(b"committed")
            (output/"head_step_000002.pt").write_bytes(b"unpublished")
            (output/"completion_rank0.json").write_text('{}\n')
            archive = Path(recover_after_checkpoint(output, 1, 1))
            self.assertEqual((archive/"rank0_steps.jsonl").read_text(), original)
            self.assertEqual((output/"rank0_steps.jsonl").read_text(), '{"step": 1}\n')
            self.assertEqual((output/"validation.jsonl").read_text(), "")
            self.assertEqual((output/"head_step_000001.pt").read_bytes(), b"committed")
            self.assertFalse((output/"head_step_000002.pt").exists())
            self.assertEqual((archive/"head_step_000002.pt").read_bytes(), b"unpublished")
            self.assertFalse((output/"completion_rank0.json").exists())

    def test_interior_log_corruption_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            original = '{broken\n{"step": 2}\n'
            path = output/"rank0_steps.jsonl"
            path.write_text(original)
            with self.assertRaisesRegex(ValueError, "corrupt non-tail"):
                recover_after_checkpoint(output, 1, 1)
            self.assertEqual(path.read_text(), original)
            self.assertFalse(list(output.glob("resume_after_*")))

    def run_synthetic_training(self, root, output, *, resume=False, interrupt=False):
        model_runner = types.ModuleType("geoff3d.slrf.model_runner")
        model_runner.init_model_from_hydra = mock.Mock(return_value=(torch.nn.Linear(1, 1), None))
        model_runner.load_checkpoint = mock.Mock()
        model_runner.apply_runtime_prior_policy = mock.Mock()
        model_runner.build_prior_overrides = mock.Mock(return_value=[])
        self.last_model_runner = model_runner
        arguments = [str(ENTRYPOINT), "--plan", str(root/"plan.json"), "--config", str(root/"config.json"),
                     "--output", str(output), "--max-steps", "4"]
        if resume:
            arguments.append("--resume")
        actual_validation = trainer.fixed_validation
        def validation(*args):
            actual_validation(*args)
            if interrupt:
                raise RuntimeError("simulated interruption after validation before checkpoint")
        cwd, search_paths = Path.cwd(), list(sys.path)
        previous_module = sys.modules.get("geoff3d.slrf.model_runner")
        sys.modules["geoff3d.slrf.model_runner"] = model_runner
        try:
            with mock.patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}), \
                 mock.patch("sys.argv", arguments), \
                 mock.patch.object(trainer, "training_device", return_value=torch.device("cpu")), \
                 mock.patch.object(trainer, "SourceV2", SyntheticSource), \
                 mock.patch.object(trainer, "fixed_validation", side_effect=validation), \
                 contextlib.redirect_stdout(io.StringIO()):
                trainer.main()
        finally:
            os.chdir(cwd)
            sys.path[:] = search_paths
            if previous_module is None:
                sys.modules.pop("geoff3d.slrf.model_runner", None)
            else:
                sys.modules["geoff3d.slrf.model_runner"] = previous_module
        return model_runner

    def test_interrupted_training_resumes_exactly_with_real_head_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_plan(root)
            config = TrainingConfigV2(steps=4, expected_world_size=1, chunks_per_rank=2, hidden_dim=8,
                                      save_every=2, validation_every=2, dino_dropout=0.3, confidence_dropout=0.3)
            (root/"config.json").write_text(json.dumps(dataclasses.asdict(config)))
            reference, resumed = root/"reference", root/"resumed"
            self.run_synthetic_training(root, reference)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.run_synthetic_training(root, resumed, interrupt=True)
            before = torch.load(resumed/"head_last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(before["step"], 1)
            self.run_synthetic_training(root, resumed, resume=True)
            expected = torch.load(reference/"head_last.pt", map_location="cpu", weights_only=False)
            actual = torch.load(resumed/"head_last.pt", map_location="cpu", weights_only=False)
            torch.testing.assert_close(actual["head"], expected["head"], atol=0, rtol=0)
            torch.testing.assert_close(actual["optimizer"], expected["optimizer"], atol=0, rtol=0)
            self.assertEqual(actual["rank_progress"], expected["rank_progress"])
            expected_rows = [json.loads(line) for line in (reference/"rank0_steps.jsonl").read_text().splitlines()]
            actual_rows = [json.loads(line) for line in (resumed/"rank0_steps.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in actual_rows], [1, 2, 3, 4])
            for expected_row, actual_row in zip(expected_rows, actual_rows):
                for name in ("loss", "learning_rate", "dropout", "samples", "ranking"):
                    self.assertEqual(actual_row[name], expected_row[name])
            self.assertEqual((reference/"validation.jsonl").read_text(), (resumed/"validation.jsonl").read_text())
            completion = json.loads((resumed/"completion_rank0.json").read_text())
            self.assertEqual(completion["status"], "passed")
            self.assertTrue(list(resumed.glob("resume_after_000001_*")))

    def test_resume_rejects_changed_model_before_loading_pi3x(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_plan(root)
            config = TrainingConfigV2(steps=4, expected_world_size=1, chunks_per_rank=2, hidden_dim=8,
                                      save_every=2, validation_every=2)
            (root/"config.json").write_text(json.dumps(dataclasses.asdict(config)))
            output = root/"interrupted"
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.run_synthetic_training(root, output, interrupt=True)
            path = root/"UAVFF3D/GeoFF3D/geoff3d/models/external/pi3/models/pi3x.py"
            path.write_text("value = 2\n")
            original_log = (output/"rank0_steps.jsonl").read_text()
            with self.assertRaisesRegex(RuntimeError, "resume data/dependency contract changed: dependency_files"):
                self.run_synthetic_training(root, output, resume=True)
            self.last_model_runner.init_model_from_hydra.assert_not_called()
            self.assertEqual((output/"rank0_steps.jsonl").read_text(), original_log)
            self.assertFalse(list(output.glob("resume_after_*")))


if __name__ == "__main__":
    unittest.main()
