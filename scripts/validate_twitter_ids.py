#!/usr/bin/env python3
"""Validate the public Twitter split IDs without reading any post text."""

from __future__ import annotations

import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPECTED = {"train": 1347, "validation": 292, "test": 286}


def main() -> None:
    root = ROOT_DIR / "datasets" / "split_ids" / "twitter_ner"
    seen: set[str] = set()
    for split, expected_count in EXPECTED.items():
        path = root / f"{split}.ids"
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(values) != expected_count:
            raise ValueError(f"{split}: expected {expected_count} IDs, found {len(values)}")
        if len(values) != len(set(values)):
            raise ValueError(f"{split}: duplicate IDs")
        if any(not re.fullmatch(r"[0-9]+", value) for value in values):
            raise ValueError(f"{split}: every public identifier must be numeric")
        if seen.intersection(values):
            raise ValueError(f"{split}: IDs overlap another split")
        seen.update(values)
    print("Twitter split IDs are valid: train=1347, validation=292, test=286.")


if __name__ == "__main__":
    main()
