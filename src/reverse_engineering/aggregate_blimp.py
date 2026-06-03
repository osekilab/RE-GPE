#!/usr/bin/env python3
"""Aggregate BLiMP results across all folds for a given model size.

Scans a directory containing per-fold lm-evaluation-harness outputs, pulls the
overall BLiMP accuracy and every per-subtask accuracy, and writes a single
summary JSON with:

- baseline accuracies (if a "baseline/" subdir is present)
- per-fold accuracies
- mean / std / sem across folds

The overall baseline-minus-trained delta is printed to the console.

Usage:
    # Small
    uv run python aggregate_blimp.py \
        --blimp-dir blimp_evaluation \
        --output-path blimp_evaluation/summary.json

    # Medium
    uv run python aggregate_blimp.py \
        --blimp-dir blimp_evaluation_md \
        --output-path blimp_evaluation_md/summary.json

    # Large
    uv run python aggregate_blimp.py \
        --blimp-dir blimp_evaluation_lg \
        --output-path blimp_evaluation_lg/summary.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


FOLD_RE = re.compile(r"^fold_(\d+)$")


def find_result_json(run_dir: Path) -> Optional[Path]:
    """Return the newest results_*.json inside any subdirectory of run_dir."""
    candidates = sorted(run_dir.rglob("results_*.json"))
    if not candidates:
        return None
    return candidates[-1]


def extract_accuracies(result_json: Path) -> Dict[str, float]:
    """Pull overall + per-subtask BLiMP accuracies from a harness result file."""
    with result_json.open() as f:
        data = json.load(f)
    results = data.get("results", {})
    flat: Dict[str, float] = {}
    for task_name, metrics in results.items():
        if not task_name.startswith("blimp"):
            continue
        acc = metrics.get("acc,none")
        if acc is None:
            continue
        flat[task_name] = float(acc)
    return flat


def summarize(per_fold: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean/std/sem across folds, per BLiMP subtask."""
    if not per_fold:
        return {}
    task_names: set = set()
    for acc_map in per_fold.values():
        task_names.update(acc_map.keys())

    summary: Dict[str, Dict[str, float]] = {}
    for task in sorted(task_names):
        values = [
            per_fold[f][task] for f in per_fold if task in per_fold[f]
        ]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        summary[task] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "sem": float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
            "n_folds": int(arr.size),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blimp-dir",
        type=str,
        required=True,
        help="Directory containing fold_*/ and (optionally) baseline/ subdirs",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to write aggregated summary JSON",
    )
    args = parser.parse_args()

    blimp_dir = Path(args.blimp_dir)
    if not blimp_dir.exists():
        raise SystemExit(f"Directory not found: {blimp_dir}")

    # Baseline
    baseline_acc: Optional[Dict[str, float]] = None
    baseline_dir = blimp_dir / "baseline"
    if baseline_dir.exists():
        baseline_json = find_result_json(baseline_dir)
        if baseline_json is not None:
            baseline_acc = extract_accuracies(baseline_json)
            print(
                f"Baseline: {baseline_json.relative_to(blimp_dir)} "
                f"(blimp overall={baseline_acc.get('blimp', float('nan')):.4f})"
            )
        else:
            print(f"Warning: no results_*.json under {baseline_dir}")

    # Folds
    per_fold: Dict[int, Dict[str, float]] = {}
    for child in sorted(blimp_dir.iterdir()):
        if not child.is_dir():
            continue
        m = FOLD_RE.match(child.name)
        if not m:
            continue
        fold_num = int(m.group(1))
        result_json = find_result_json(child)
        if result_json is None:
            print(f"Warning: no results_*.json for fold {fold_num} under {child}")
            continue
        acc_map = extract_accuracies(result_json)
        per_fold[fold_num] = acc_map
        print(
            f"Fold {fold_num}: {result_json.relative_to(blimp_dir)} "
            f"(blimp overall={acc_map.get('blimp', float('nan')):.4f})"
        )

    summary = summarize(per_fold)

    output = {
        "blimp_dir": str(blimp_dir),
        "baseline": baseline_acc,
        "folds": {str(k): per_fold[k] for k in sorted(per_fold)},
        "summary": summary,
    }

    # Overall banner
    overall = summary.get("blimp")
    if overall is not None:
        baseline_overall = (
            baseline_acc.get("blimp") if baseline_acc is not None else None
        )
        print("\n" + "=" * 60)
        print(f"BLiMP overall accuracy over {overall['n_folds']} folds:")
        print(
            f"  trained mean = {overall['mean']:.4f} "
            f"± {overall['sem']:.4f} (std={overall['std']:.4f})"
        )
        if baseline_overall is not None:
            print(f"  baseline     = {baseline_overall:.4f}")
            print(f"  delta        = {overall['mean'] - baseline_overall:+.4f}")
        print("=" * 60)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()
