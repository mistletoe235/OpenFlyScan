"""Report source-release hazards without printing credential values or changing files."""

import argparse
import json
from pathlib import Path
import re
import subprocess


SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "hf_token": re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b"),
    "github_token": re.compile(rb"\b(?:ghp_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{70,})\b"),
}


def inspect_bytes(content):
    return [{"kind": name, "line": content[:match.start()].count(b"\n") + 1}
            for name, pattern in SECRET_PATTERNS.items() for match in pattern.finditer(content)]


def inspect_repository(root, maximum_bytes):
    root = Path(root).resolve()
    result = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                            capture_output=True, check=True)
    paths = sorted(set(name.decode() for name in result.stdout.split(b"\0") if name))
    findings = []
    for relative in paths:
        path = root / relative
        if path.is_symlink():
            if not path.resolve().is_relative_to(root):
                findings.append({"path": relative, "kind": "external_symlink"})
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > maximum_bytes:
            findings.append({"path": relative, "kind": "large_file", "bytes": size})
        if path.suffix.lower() in (".jks", ".keystore", ".p12", ".pfx"):
            findings.append({"path": relative, "kind": "signing_material"})
        if size <= 8 * 1024 * 1024:
            findings.extend({"path": relative, **entry} for entry in inspect_bytes(path.read_bytes()))
    return {"files_checked": len(paths), "findings": findings,
            "license_present": any((root / name).is_file() for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")),
            "scope": "Current tracked and unignored files; bounded token-pattern scan, not a complete security or license review"}


def inspect_history(root, maximum_bytes):
    command = ["git", "-C", str(Path(root).resolve())]
    listing = subprocess.run([*command, "rev-list", "--objects", "--all"], capture_output=True, text=True, check=True)
    objects = [line.split(" ", 1) for line in listing.stdout.splitlines()]
    if not objects:
        return {"blobs_checked": 0, "findings": []}
    paths = {row[0]: row[1] if len(row) > 1 else "" for row in objects}
    metadata = subprocess.run([*command, "cat-file", "--batch-check"], input="\n".join(paths) + "\n",
                              capture_output=True, text=True, check=True)
    blobs = [line.split() for line in metadata.stdout.splitlines() if " blob " in line]
    findings = []
    process = subprocess.Popen([*command, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for identity, kind, size_text in blobs:
            size = int(size_text)
            if size > maximum_bytes:
                findings.append({"blob": identity, "path": paths[identity], "kind": "large_history_blob", "bytes": size})
            if size > 8 * 1024 * 1024:
                continue
            process.stdin.write((identity + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3 or header[1] != b"blob" or int(header[2]) != size:
                raise RuntimeError("Unexpected Git object response")
            content = process.stdout.read(size)
            if len(content) != size or process.stdout.read(1) != b"\n":
                raise RuntimeError("Truncated Git object response")
            findings.extend({"blob": identity, "path": paths[identity], **entry} for entry in inspect_bytes(content))
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait()
    return {"blobs_checked": len(blobs), "findings": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-file-mib", type=int, default=50)
    parser.add_argument("--history", action="store_true", help="Also inspect all reachable Git blobs")
    args = parser.parse_args()
    if args.max_file_mib < 1:
        parser.error("max-file-mib must be positive")
    report = inspect_repository(args.repo, args.max_file_mib * 1024 * 1024)
    if args.history:
        report["history"] = inspect_history(args.repo, args.max_file_mib * 1024 * 1024)
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report["findings"] or report.get("history", {}).get("findings") or not report["license_present"] else 0)


if __name__ == "__main__":
    main()
