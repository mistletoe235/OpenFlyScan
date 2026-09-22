import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest
import json


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class PublicSetupTests(unittest.TestCase):
    def test_release_scan_redacts_secret_values(self):
        inspect = load_script("check_release").inspect_bytes
        secret = b"hf_" + b"a" * 34
        findings = inspect(b"example\n" + secret + b"\n")
        self.assertEqual(findings, [{"kind": "hf_token", "line": 2}])
        self.assertNotIn(secret.decode(), str(findings))

    def test_dependency_check_detects_changed_content(self):
        check = load_script("check_dependencies").check_snapshot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("value = 1\n")
            expected = hashlib.sha256((root / "model.py").read_bytes()).hexdigest()
            lock = {"geoff3d": {"patched_files": {"model.py": expected}}}
            self.assertEqual(check(root, lock, verify_revision=False), [])
            (root / "model.py").write_text("value = 2\n")
            self.assertEqual(len(check(root, lock, verify_revision=False)), 1)

    def test_plan_resolver_keeps_nonpath_semantics_and_rejects_escape(self):
        resolve = load_script("resolve_training_plan").resolve_paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            template = {"root": "${DEPENDENCY_ROOT}", "records": [{"source": "${DATA_ROOT}/scene/source", "split": "train", "views": 30}]}
            result = resolve(template, root / "data", root / "deps")
            self.assertEqual(result["root"], str(root / "deps"))
            self.assertEqual(result["records"][0]["source"], str(root / "data/scene/source"))
            self.assertEqual(result["records"][0]["split"], "train")
            self.assertEqual(result["records"][0]["views"], 30)
            for invalid in ("${DATA_ROOT}/../escape", "${UNKNOWN}/scene", "prefix/${DATA_ROOT}"):
                with self.assertRaises(ValueError):
                    resolve(invalid, root / "data", root / "deps")

    def test_old_checkpoint_identity_cannot_be_relocated(self):
        from openflyscan import QualityPredictorTrainingConfig
        prepare = load_script("resolve_training_plan").prepare_plan
        with self.assertRaisesRegex(ValueError, "checkpoint stat identity"):
            prepare({"pi3x_checkpoint_identity": {}}, "/tmp/data", "/tmp/deps", QualityPredictorTrainingConfig())


if __name__ == "__main__":
    unittest.main()
