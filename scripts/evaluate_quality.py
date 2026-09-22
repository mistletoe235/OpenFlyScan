"""Evaluate regional quality scores against the frozen Expo West GS teacher.

This script is intentionally method-agnostic. A baseline must export exactly one
score per frozen query; larger values must mean worse reconstruction quality.
The script never changes the valid-region mask, top-k budget, or teacher labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_SELECTION_SHA256 = "b6704eeb9d75b9ad7ecf14db383655264b949bb73cc72782d60809aa6043f0e4"
EXPECTED_TEACHER_SHA256 = "67e2df9256c7d7cd0d18a15dede9cca2c2c1befccdf39486c3491db1a337ddf9"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scores(path: Path, field: str | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if suffix == ".npz":
        data = np.load(path)
        key = field or (data.files[0] if len(data.files) == 1 else None)
        if key is None or key not in data.files:
            raise ValueError(f"--score-field is required; available fields: {data.files}")
        return np.asarray(data[key], dtype=np.float64).reshape(-1)
    if suffix == ".json":
        data = json.loads(path.read_text())
        if field:
            for key in field.split("."):
                data = data[key]
        return np.asarray(data, dtype=np.float64).reshape(-1)
    if suffix == ".csv":
        if not field:
            raise ValueError("--score-field is required for CSV input")
        with path.open(newline="") as handle:
            return np.asarray([float(row[field]) for row in csv.DictReader(handle)], dtype=np.float64)
    raise ValueError(f"Unsupported score format: {path}")


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks, including exact-value ties."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    rank_a = average_ranks(a)
    rank_b = average_ranks(b)
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denominator = float(np.linalg.norm(rank_a) * np.linalg.norm(rank_b))
    if denominator == 0.0:
        return None
    return float(np.dot(rank_a, rank_b) / denominator)


def metrics(scores: np.ndarray, target: np.ndarray, mask: np.ndarray, fraction: float) -> dict:
    ids = np.flatnonzero(mask & np.isfinite(scores))
    if len(ids) < 2:
        return {"valid_regions": int(len(ids)), "budget_regions": 0, "hits": 0,
                "recall": None, "random_expected_recall": None, "spearman": None}
    k = int(np.ceil(fraction * len(ids)))
    worst = ids[np.argsort(target[ids], kind="stable")[-k:]]
    selected = ids[np.argsort(scores[ids], kind="stable")[-k:]]
    hits = len(np.intersect1d(worst, selected, assume_unique=False))
    corr = spearman_correlation(scores[ids], target[ids])

    # A stable argsort silently turns input order into a decision rule when the
    # score at the selection boundary is tied.  Report the deterministic value
    # for reproducibility, but also the expected/min/max hit count over all
    # possible choices inside that boundary tie.
    truth = np.zeros(len(scores), dtype=bool)
    truth[worst] = True
    cutoff = float(np.partition(scores[ids], len(ids) - k)[len(ids) - k])
    above = ids[scores[ids] > cutoff]
    tied = ids[scores[ids] == cutoff]
    needed_from_tie = k - len(above)
    hits_above = int(truth[above].sum())
    positives_in_tie = int(truth[tied].sum())
    expected_hits = hits_above + needed_from_tie * positives_in_tie / max(len(tied), 1)
    minimum_hits = hits_above + max(0, needed_from_tie - (len(tied) - positives_in_tie))
    maximum_hits = hits_above + min(needed_from_tie, positives_in_tie)
    return {
        "valid_regions": int(len(ids)),
        "budget_regions": k,
        "hits": int(hits),
        "recall": float(hits / k),
        "random_expected_recall": float(k / len(ids)),
        "spearman": corr,
        "selection_cutoff": cutoff,
        "boundary_tie_regions": int(len(tied)),
        "selected_from_boundary_tie": int(needed_from_tie),
        "tie_aware_expected_hits": float(expected_hits),
        "tie_aware_expected_recall": float(expected_hits / k),
        "tie_aware_min_recall": float(minimum_hits / k),
        "tie_aware_max_recall": float(maximum_hits / k),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-dir", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-field")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--lower-is-worse", "--lower-is-riskier", dest="lower_is_worse", action="store_true")
    parser.add_argument(
        "--expected-selection-sha256",
        default=EXPECTED_SELECTION_SHA256,
        help="Frozen selection hash. Pass the test-set contract hash when evaluating a new population.",
    )
    parser.add_argument(
        "--expected-teacher-sha256",
        default=EXPECTED_TEACHER_SHA256,
        help="Frozen teacher hash. Pass the test-set contract hash when evaluating a new population.",
    )
    args = parser.parse_args()

    selection_path = args.common_dir / "selection_frozen.json"
    teacher_path = args.common_dir / "region_values.npz"
    selection_hash = sha256(selection_path)
    teacher_hash = sha256(teacher_path)
    if selection_hash != args.expected_selection_sha256:
        raise RuntimeError(f"Unexpected selection hash: {selection_hash}")
    if teacher_hash != args.expected_teacher_sha256:
        raise RuntimeError(f"Unexpected teacher hash: {teacher_hash}")

    selection = json.loads(selection_path.read_text())
    teacher = np.load(teacher_path)
    target = np.asarray(teacher["target_log_mse"], dtype=np.float64)
    if "valid" in teacher:
        valid = np.asarray(teacher["valid"], dtype=bool)
    elif "support" in teacher:
        # Some real-scene teachers store the same validity contract as a view
        # support count rather than a precomputed boolean mask.
        valid = np.asarray(teacher["support"], dtype=np.int64) >= 2
    else:
        valid = np.isfinite(target)
    # A support count alone does not guarantee that GS produced a finite target.
    # Without this guard NumPy sorts NaNs to the end, incorrectly treating them
    # as the worst reconstruction regions and severely depressing Recall@20.
    valid &= np.isfinite(target)
    groups = np.asarray([int(query["chunk"]) for query in selection["queries"]], dtype=np.int64)
    scores = load_scores(args.scores, args.score_field)
    if args.lower_is_worse:
        scores = -scores
    if len(scores) != len(target):
        raise ValueError(f"Expected {len(target)} scores, received {len(scores)}")

    global_result = metrics(scores, target, valid, args.top_fraction)
    per_chunk = []
    for group in np.unique(groups):
        group_mask = valid & (groups == group)
        if int(group_mask.sum()) < 2:
            continue
        per_chunk.append({"chunk": int(group), **metrics(scores, target, group_mask, args.top_fraction)})
    chunk_recalls = [row["recall"] for row in per_chunk if row["recall"] is not None]
    chunk_spearman = [row["spearman"] for row in per_chunk if row["spearman"] is not None]

    finite_coverage = int(np.sum(valid & np.isfinite(scores)))
    report = {
        "schema": "openfly-region-quality-baseline-v1",
        "baseline": args.name,
        "risk_direction": "higher_is_worse",
        "top_fraction": args.top_fraction,
        "queries": int(len(scores)),
        "teacher_valid_regions": int(valid.sum()),
        "finite_score_regions_with_teacher": finite_coverage,
        "score_coverage_of_teacher": float(finite_coverage / max(int(valid.sum()), 1)),
        "global": global_result,
        "chunk_macro_recall": float(np.mean(chunk_recalls)) if chunk_recalls else None,
        "chunk_macro_spearman": float(np.mean(chunk_spearman)) if chunk_spearman else None,
        "per_chunk": per_chunk,
        "selection_sha256": selection_hash,
        "teacher_sha256": teacher_hash,
        "score_file": str(args.scores),
        "score_sha256": sha256(args.scores),
        "boundary": (
            "Queries repeat across overlapping windows and are not unique physical surfaces. "
            "GS labels are offline evaluation-only. Unsupported/non-finite baseline scores are "
            "reported as missing coverage and never silently replaced by zero."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_chunk"}, indent=2))


if __name__ == "__main__":
    main()
