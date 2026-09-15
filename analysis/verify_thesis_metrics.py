#!/usr/bin/env python3
"""Recompute Wiki aggregates and compare them with the frozen public files."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.build_wiki_reporting import aggregate_rows, read_csv, validate, write_csv


def normalized(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    wiki = ROOT_DIR / "reference-results" / "wiki"
    cohort = json.loads((ROOT_DIR / "configs" / "reporting_cohort.json").read_text(encoding="utf-8"))
    matrix = read_csv(wiki / "matrix_results_81.csv")
    validate(matrix, cohort)
    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / "wiki_aggregates.csv"
        write_csv(candidate, aggregate_rows(matrix))
        if normalized(candidate) != normalized(wiki / "wiki_aggregates.csv"):
            raise ValueError("wiki_aggregates.csv does not match a fresh aggregation of matrix_results_81.csv")
    print("Wiki aggregate metrics match the complete 81-row matrix.")


if __name__ == "__main__":
    main()
