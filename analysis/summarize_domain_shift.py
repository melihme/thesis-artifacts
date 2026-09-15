#!/usr/bin/env python3
"""Aggregate the complete five-seed domain-shift evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "configs" / "reproduction.json"
PURPOSES = {
    "wiki_to_wiki": "source in-domain baseline",
    "wiki_to_twitter": "main domain-shift comparison",
    "twitter_to_twitter": "target in-domain control",
    "twitter_to_wiki": "reverse-transfer check",
}


def metric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def collect_rows(results_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(results_root.glob("seed-*/*/*/*.json")):
        relative = path.relative_to(results_root)
        seed_part, family, cell, profile_file = relative.parts
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        rows.append(
            {
                "seed": int(seed_part.removeprefix("seed-")),
                "family": family,
                "cell": cell,
                "profile": Path(profile_file).stem,
                "purpose": PURPOSES[cell],
                "examples": int(payload["examples"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "bootstrap_f1_ci": payload.get("bootstrap_f1_ci"),
            }
        )
    return sorted(rows, key=lambda row: (row["family"], row["profile"], row["cell"], row["seed"]))


def aggregate_cells(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["profile"], row["cell"])].append(row)
    summaries = []
    for (family, profile, cell), group in sorted(grouped.items()):
        summaries.append(
            {
                "family": family,
                "profile": profile,
                "cell": cell,
                "seeds": [row["seed"] for row in group],
                "examples": sorted({row["examples"] for row in group}),
                "precision": metric_summary([row["precision"] for row in group]),
                "recall": metric_summary([row["recall"] for row in group]),
                "f1": metric_summary([row["f1"] for row in group]),
            }
        )
    return summaries


def paired_gap_summaries(rows: Iterable[dict[str, Any]], primary_profile: str) -> list[dict[str, Any]]:
    lookup = {
        (row["seed"], row["family"], row["cell"]): row["f1"]
        for row in rows
        if row["profile"] == primary_profile
    }
    formulas = (
        ("domain_transfer_gap", "wiki_to_wiki", "wiki_to_twitter"),
        ("target_adaptation_gain", "twitter_to_twitter", "wiki_to_twitter"),
        ("reverse_transfer_gap", "twitter_to_twitter", "twitter_to_wiki"),
    )
    summaries = []
    for family in sorted({family for _, family, _ in lookup}):
        seeds = sorted({seed for seed, row_family, _ in lookup if row_family == family})
        for name, left_cell, right_cell in formulas:
            paired = [
                lookup[(seed, family, left_cell)] - lookup[(seed, family, right_cell)]
                for seed in seeds
                if (seed, family, left_cell) in lookup and (seed, family, right_cell) in lookup
            ]
            summaries.append(
                {
                    "family": family,
                    "profile": primary_profile,
                    "metric": name,
                    "formula": f"F1({left_cell}) - F1({right_cell})",
                    "seeds": seeds,
                    "f1_points": metric_summary(paired),
                }
            )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT_DIR / ".artifacts" / "results" / "cross-domain",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / ".artifacts" / "results" / "domain_shift_summary.json",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = collect_rows(args.results_root.resolve())
    expected = int(config["domain_shift"]["expected_total_runs"])
    if len(rows) != expected and not args.allow_partial:
        raise ValueError(f"Expected {expected} domain-shift evaluations, found {len(rows)}")
    primary = config["domain_shift"]["primary_profile"]
    summary = {
        "method": {
            "primary_profile": primary,
            "aggregation": "Arithmetic mean and sample standard deviation over independently trained seeds.",
            "gap_definitions": {
                "domain_transfer_gap": "F1(wiki_to_wiki) - F1(wiki_to_twitter)",
                "target_adaptation_gain": "F1(twitter_to_twitter) - F1(wiki_to_twitter)",
                "reverse_transfer_gap": "F1(twitter_to_twitter) - F1(twitter_to_wiki)",
            },
        },
        "completed_runs": len(rows),
        "rows": rows,
        "cell_summaries": aggregate_cells(rows),
        "paired_gap_summaries": paired_gap_summaries(rows, primary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} aggregate rows to {args.output}")


if __name__ == "__main__":
    main()
