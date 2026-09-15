#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ner_framework.exact_span import SCORING_PROTOCOL
from ner_framework.timing import LATENCY_PROTOCOL
from ner_framework.utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap strict Wiki matrix results from per-example counts.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reference-model", default="dbmdz-bert-base-turkish-cased-wiki-tuned")
    parser.add_argument("--reference-device", default="gpu")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(ROOT_DIR)
    return resolved


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_counts(row: Dict[str, str]) -> Tuple[List[str], np.ndarray]:
    records = read_jsonl(Path(row["predictions_path"]))
    if any(
        record.get("scoring_protocol") != SCORING_PROTOCOL or record.get("latency_protocol") != LATENCY_PROTOCOL
        for record in records
    ):
        raise ValueError(f"Prediction protocol mismatch: {row['predictions_path']}")
    ids = [str(record["example_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate example IDs: {row['predictions_path']}")
    counts = np.asarray([[record["tp"], record["fp"], record["fn"]] for record in records], dtype=np.int64)
    return ids, counts


def f1_from_sums(sums: np.ndarray) -> np.ndarray:
    tp = sums[..., 0].astype(np.float64)
    fp = sums[..., 1].astype(np.float64)
    fn = sums[..., 2].astype(np.float64)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)


def bootstrap_f1(counts: np.ndarray, iterations: int, rng: np.random.Generator) -> np.ndarray:
    indices = rng.integers(0, len(counts), size=(iterations, len(counts)))
    return f1_from_sums(counts[indices].sum(axis=1))


def interval(values: np.ndarray) -> Tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def row_id(row: Dict[str, str]) -> str:
    return f"{row['model_slug']}::{row.get('device_profile', '')}::{row['prompt_strategy']}::{row['seed']}"


def main() -> None:
    args = parse_args()
    results_dir = inside(args.results_dir)
    rows = [row for row in read_csv(results_dir / "matrix_results.csv") if row["status"] == "completed"]
    if not rows:
        raise ValueError("No completed matrix rows are available for analysis")
    rng = np.random.default_rng(args.seed)
    loaded: Dict[str, Tuple[List[str], np.ndarray]] = {row_id(row): load_counts(row) for row in rows}

    intervals: List[Dict[str, object]] = []
    for row in rows:
        _, counts = loaded[row_id(row)]
        distribution = bootstrap_f1(counts, args.iterations, rng)
        low, high = interval(distribution)
        intervals.append(
            {
                "approach": row["approach"],
                "model_slug": row["model_slug"],
                "device_profile": row.get("device_profile", ""),
                "prompt_strategy": row["prompt_strategy"],
                "seed": row["seed"],
                "examples": len(counts),
                "f1": row["f1"],
                "f1_ci95_low": low,
                "f1_ci95_high": high,
                "bootstrap_iterations": args.iterations,
                "bootstrap_seed": args.seed,
            }
        )

    reference_rows = [
        row
        for row in rows
        if row["model_slug"] == args.reference_model
        and row.get("device_profile", "") == args.reference_device
    ]
    if len(reference_rows) != 1:
        raise ValueError(
            f"Expected exactly one completed reference row for {args.reference_model} "
            f"on device profile {args.reference_device}"
        )
    reference = reference_rows[0]
    reference_ids, reference_counts = loaded[row_id(reference)]
    paired: List[Dict[str, object]] = []
    for row in rows:
        if row is reference:
            continue
        comparison_ids, comparison_counts = loaded[row_id(row)]
        if comparison_ids != reference_ids:
            raise ValueError(f"Example order differs between reference and {row_id(row)}")
        indices = rng.integers(0, len(reference_counts), size=(args.iterations, len(reference_counts)))
        reference_f1 = f1_from_sums(reference_counts[indices].sum(axis=1))
        comparison_f1 = f1_from_sums(comparison_counts[indices].sum(axis=1))
        differences = comparison_f1 - reference_f1
        low, high = interval(differences)
        paired.append(
            {
                "reference_model": args.reference_model,
                "reference_device": args.reference_device,
                "comparison_model": row["model_slug"],
                "comparison_device": row.get("device_profile", ""),
                "prompt_strategy": row["prompt_strategy"],
                "seed": row["seed"],
                "observed_f1_difference": float(row["f1"]) - float(reference["f1"]),
                "difference_ci95_low": low,
                "difference_ci95_high": high,
                "bootstrap_probability_difference_le_zero": float(np.mean(differences <= 0)),
                "bootstrap_iterations": args.iterations,
                "bootstrap_seed": args.seed,
            }
        )

    write_csv(results_dir / "bootstrap_f1_intervals.csv", intervals)
    write_csv(results_dir / "paired_f1_differences.csv", paired)
    (results_dir / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "scoring_protocol": SCORING_PROTOCOL,
                "latency_protocol": LATENCY_PROTOCOL,
                "reference_model": args.reference_model,
                "reference_device": args.reference_device,
                "bootstrap_iterations": args.iterations,
                "bootstrap_seed": args.seed,
                "completed_rows": len(rows),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote bootstrap analysis under {results_dir}")


if __name__ == "__main__":
    main()
