#!/usr/bin/env python3
"""Validate the Wiki reporting cohort and regenerate aggregate table and figure inputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
METRICS = ("precision", "recall", "f1", "accuracy", "latency_median_ms", "latency_p95_ms")
GROUP_FIELDS = ("approach", "model_slug", "device_profile", "prompt_strategy")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_list = list(fields if fields is not None else (rows[0] if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def expected_runs(row: dict[str, str]) -> int:
    if row["approach"] == "transformer_http" or row["prompt_strategy"] == "zero-shot":
        return 1
    return 3


def numeric(values: Iterable[str]) -> list[float]:
    return [float(value) for value in values if value not in {"", None}]


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in GROUP_FIELDS)].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        record: dict[str, Any] = {
            "status": "completed" if all(row["status"] == "completed" for row in group) else "partial",
            **dict(zip(GROUP_FIELDS, key)),
            "runs_expected": expected_runs(group[0]),
            "runs_completed": len(group),
        }
        for metric in METRICS:
            values = numeric(row[metric] for row in group)
            record[f"{metric}_mean"] = mean(values) if values else ""
            record[f"{metric}_std"] = stdev(values) if len(values) > 1 else ""
        output.append(record)
    return output


def validate(rows: list[dict[str, str]], cohort: dict[str, Any]) -> None:
    expected_total = int(cohort["expected_rows"]["total"])
    if len(rows) != expected_total:
        raise ValueError(f"Expected {expected_total} reporting rows, found {len(rows)}")
    allowed = set(cohort["included_llm_slugs"]) | set(cohort["included_transformer_slugs"])
    if {row["model_slug"] for row in rows} != allowed:
        raise ValueError("The model set does not match configs/reporting_cohort.json")
    if any(row["status"] != "completed" for row in rows):
        raise ValueError("The reporting cohort contains incomplete rows")


def reliability_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row["approach"] != "llm_http":
            continue
        output.append(
            {
                "model_slug": row["model_slug"],
                "prompt_strategy": row["prompt_strategy"],
                "seed": row["seed"],
                "examples_evaluated": row["examples_evaluated"],
                "successful_requests": row["successful_requests"],
                "failed_requests": row["failed_requests"],
                "valid_json_requests": row["valid_json_requests"],
                "schema_valid_requests": row["schema_valid_requests"],
                "truncated_requests": row["truncated_requests"],
                "rejected_prediction_reasons": row["rejected_prediction_reasons"],
                "finish_reasons": row["finish_reasons"],
            }
        )
    return output


def primary_points(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in aggregates
        if row["approach"] == "llm_http"
        or (row["approach"] == "transformer_http" and row["device_profile"] == "gpu")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT_DIR / "reference-results" / "wiki" / "matrix_results_81.csv",
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=ROOT_DIR / "configs" / "reporting_cohort.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "reference-results" / "wiki",
    )
    args = parser.parse_args()
    rows = read_csv(args.matrix)
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    validate(rows, cohort)
    aggregates = aggregate_rows(rows)
    expected_aggregates = 3 * len(cohort["included_llm_slugs"]) + 2 * len(cohort["included_transformer_slugs"])
    if len(aggregates) != expected_aggregates:
        raise ValueError(
            f"Expected {expected_aggregates} model, device, and strategy aggregates, found {len(aggregates)}"
        )
    if any(row["runs_expected"] != row["runs_completed"] for row in aggregates):
        raise ValueError("An aggregate group has fewer runs than the protocol requires")
    write_csv(args.output_dir / "wiki_aggregates.csv", aggregates)
    write_csv(args.output_dir / "table_quality_latency.csv", primary_points(aggregates))
    write_csv(args.output_dir / "table_output_reliability.csv", reliability_rows(rows))
    write_csv(args.output_dir / "figure_f1_latency_points.csv", primary_points(aggregates))
    print(f"Regenerated {len(aggregates)} Wiki aggregates from {len(rows)} retained rows.")


if __name__ == "__main__":
    main()
