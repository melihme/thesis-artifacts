#!/usr/bin/env python3
"""Check the frozen public reference results and cohort boundaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    wiki = ROOT_DIR / "reference-results" / "wiki"
    matrix = csv_rows(wiki / "matrix_results_81.csv")
    if len(matrix) != 81:
        raise ValueError(f"Expected 81 Wiki result rows, found {len(matrix)}")
    if any(row["status"] != "completed" for row in matrix):
        raise ValueError("The Wiki reporting cohort contains an incomplete row")
    if len([row for row in matrix if "trendyol" in row["model_slug"].lower()]) != 7:
        raise ValueError("The complete seven-row Trendyol block is not present")
    if any(int(row["examples_evaluated"]) != 1000 for row in matrix):
        raise ValueError("A Wiki row did not evaluate the full test split")
    if any(int(row["failed_requests"] or 0) != 0 for row in matrix):
        raise ValueError("A Wiki row contains failed requests")
    expected_protocols = {"exact_token_span_v1", "localhost_http_batch1_concurrency1_v1"}
    actual_protocols = {row["scoring_protocol"] for row in matrix} | {row["latency_protocol"] for row in matrix}
    if actual_protocols != expected_protocols:
        raise ValueError(f"Unexpected Wiki protocol set: {sorted(actual_protocols)}")

    status = json.loads((wiki / "matrix_status_public.json").read_text(encoding="utf-8"))
    if status["source_matrix_at_freeze"] != {"completed": 81, "pending": 0, "expected_rows": 81}:
        raise ValueError("The Wiki source snapshot is not the complete 81-row matrix")
    if status["reported_cohort"] != {"completed": 81, "pending": 0, "expected_rows": 81}:
        raise ValueError("The Wiki reporting-cohort status does not match the 81-row contract")

    domain_path = ROOT_DIR / "reference-results" / "domain-shift" / "domain_shift_summary.json"
    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    if domain["completed_runs"] != 70 or len(domain["rows"]) != 70:
        raise ValueError("Domain-shift reference must contain 70 evaluations")
    if sorted({row["seed"] for row in domain["rows"]}) != [42, 43, 44, 45, 46]:
        raise ValueError("Domain-shift reference does not contain the five fixed seeds")
    if {row["family"] for row in domain["rows"]} != {"berturk", "distilberturk"}:
        raise ValueError("Unexpected domain-shift model families")

    print("Reference results verified: Wiki source and reported cohort 81/81, domain shift 70/70.")


if __name__ == "__main__":
    main()
