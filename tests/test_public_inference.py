import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from openflyscan import QualityPredictor, QualityPredictorTrainingConfig
from openflyscan.inference import load_predictor, load_inputs, predict


class PublicInferenceTests(unittest.TestCase):
    def test_bundled_checkpoint_matches_reference(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "configs/quality_predictor.release.json").read_text())
        for metadata in manifest["files"].values():
            asset = root / metadata["repository_path"]
            self.assertEqual(asset.stat().st_size, metadata["bytes"])
            self.assertEqual(hashlib.sha256(asset.read_bytes()).hexdigest(), metadata["sha256"])
        actual = predict(root / "weights/quality_predictor.pt",
                         root / "examples/quality_predictor/expo_west_sample_inputs.npz")
        expected = np.load(root / "examples/quality_predictor/expo_west_sample_scores.npy", allow_pickle=False)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_public_names_and_command_entrypoints(self):
        self.assertEqual(QualityPredictor.__name__, "QualityPredictor")
        self.assertEqual(QualityPredictorTrainingConfig.__name__, "QualityPredictorTrainingConfig")
        root = Path(__file__).resolve().parents[1]
        for script in ("train_quality_predictor.py", "predict_quality.py", "evaluate_quality.py"):
            result = subprocess.run([sys.executable, str(root / "scripts" / script), "--help"],
                                    cwd=root, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            if script == "evaluate_quality.py":
                self.assertIn("--lower-is-worse", result.stdout)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        config_path = Path(__file__).resolve().parents[1] / "configs/quality_predictor.json"
        import json
        self.config = QualityPredictorTrainingConfig(**json.loads(config_path.read_text()))
        self.model = QualityPredictor(self.config).eval()
        self.checkpoint = self.root / "head.pt"
        torch.save({"schema": "head_training_v3_checkpoint",
                    "training_config": dataclasses.asdict(self.config),
                    "head": self.model.state_dict()}, self.checkpoint)
        dimensions = self.config.model_config()
        self.arrays = {
            "point_features": np.ones((2, 3, dimensions.point_feature_dim), np.float32),
            "confidence_features": np.ones((2, 3, dimensions.confidence_feature_dim), np.float32),
            "dino_features": np.ones((2, 3, dimensions.dino_feature_dim), np.float32),
            "pose_geometry_features": np.ones((2, 3, dimensions.pose_geometry_dim), np.float32),
            "quality_geometry_features": np.ones((2, 3, dimensions.quality_geometry_dim), np.float32),
            "rgb_stats": np.ones((2, 3, dimensions.rgb_stats_dim), np.float32),
            "query_features": np.ones((2, dimensions.query_dim), np.float32),
            "view_mask": np.ones((2, 3), bool),
        }
        self.features = self.root / "inputs.npz"

    def test_roundtrip_matches_original_head(self):
        np.savez(self.features, **self.arrays)
        inputs = load_inputs(self.features, self.model)
        with torch.inference_mode():
            expected = self.model(inputs)["total"].numpy()
        np.testing.assert_allclose(predict(self.checkpoint, self.features), expected, atol=1e-7)
        self.assertTrue(np.all((expected >= 0) & (expected <= 1)))

    def test_rejects_no_observations(self):
        self.arrays["view_mask"][0] = False
        np.savez(self.features, **self.arrays)
        with self.assertRaisesRegex(ValueError, "valid observation"):
            load_inputs(self.features, self.model)

    def test_rejects_bad_shape_and_nonfinite_values(self):
        for values in [np.zeros((2, 1), np.float32), np.full_like(self.arrays["query_features"], np.nan)]:
            np.savez(self.features, **{**self.arrays, "query_features": values})
            with self.assertRaisesRegex(ValueError, "query_features"):
                load_inputs(self.features, self.model)

    def test_rejects_non_boolean_mask(self):
        self.arrays["view_mask"] = self.arrays["view_mask"].astype(np.int32)
        np.savez(self.features, **self.arrays)
        with self.assertRaisesRegex(ValueError, "boolean"):
            load_inputs(self.features, self.model)

    def test_requires_checkpoint_schema_and_strict_weights(self):
        torch.save({"head": self.model.state_dict()}, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "schema|head_training_v3_checkpoint"):
            load_predictor(self.checkpoint)
        torch.save({"schema": "head_training_v3_checkpoint",
                    "training_config": dataclasses.asdict(self.config), "head": {}}, self.checkpoint)
        with self.assertRaises(RuntimeError):
            load_predictor(self.checkpoint)


if __name__ == "__main__":
    unittest.main()
