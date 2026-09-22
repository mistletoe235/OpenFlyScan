import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openflyscan.demo import run_demo
from openflyscan.inference import predict


class DemoTests(unittest.TestCase):
    def test_synthetic_demo_is_explicit_and_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = run_demo(root / "demo")
            self.assertFalse(report["trained_weights"])
            self.assertEqual(report["kind"], "synthetic_wiring_check")
            saved = json.loads((root / "demo/manifest.json").read_text())
            self.assertEqual(saved, report)
            expected = np.load(root / "demo/scores.npy")
            actual = predict(root / "demo/synthetic_checkpoint.pt", root / "demo/synthetic_inputs.npz")
            np.testing.assert_array_equal(expected, actual)
            with self.assertRaisesRegex(ValueError, "empty"):
                run_demo(root / "demo")


if __name__ == "__main__":
    unittest.main()
