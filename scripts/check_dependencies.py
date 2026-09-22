"""Check the pinned GeoFF3D base revision and runtime patch without loading models."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def check_snapshot(root, lock, verify_revision=True):
    root = Path(root)
    failures = []
    specification = lock["geoff3d"]
    if verify_revision:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        if result.returncode or result.stdout.strip() != specification["revision"]:
            failures.append("GeoFF3D base revision does not match the lock")
    for relative, expected in specification["patched_files"].items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"Missing or changed runtime file: {relative}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geoff3d", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().parents[1] / "configs/dependencies.lock.json")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    failures = check_snapshot(args.geoff3d, lock)
    print(json.dumps({"passed": not failures, "failures": failures}, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
