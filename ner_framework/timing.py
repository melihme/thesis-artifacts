from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence


LATENCY_PROTOCOL = "localhost_http_batch1_concurrency1_v1"


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    if not values:
        return None
    if not 0 <= percentile_value <= 100:
        raise ValueError("Percentile must be between 0 and 100.")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarise_milliseconds(values: Iterable[float]) -> Dict[str, float | int | None]:
    cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not cleaned:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(cleaned),
        "mean_ms": sum(cleaned) / len(cleaned),
        "median_ms": percentile(cleaned, 50),
        "p90_ms": percentile(cleaned, 90),
        "p95_ms": percentile(cleaned, 95),
        "p99_ms": percentile(cleaned, 99),
        "min_ms": min(cleaned),
        "max_ms": max(cleaned),
    }


def collect_nested_numeric(records: Sequence[Mapping[str, object]], path: Sequence[str]) -> List[float]:
    values: List[float] = []
    for record in records:
        current: object = record
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            values.append(float(current))
    return values


def latency_summary(
    records: Sequence[Mapping[str, object]],
    component_paths: Mapping[str, Sequence[str]],
) -> Dict[str, object]:
    return {
        name: summarise_milliseconds(collect_nested_numeric(records, path))
        for name, path in component_paths.items()
    }
