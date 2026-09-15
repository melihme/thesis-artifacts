#!/usr/bin/env python3
"""Check Wiki counts by default and domain-shift source counts when requested."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=["wiki_ner"])
    args = parser.parse_args()
    config = json.loads((ROOT_DIR / "configs" / "reproduction.json").read_text(encoding="utf-8"))
    for dataset_name in args.datasets:
        expected = config["datasets"][dataset_name]["expected_counts"]
        manifest_path = ROOT_DIR / ".artifacts" / "datasets" / dataset_name / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = manifest["counts"]
        if actual != expected:
            raise ValueError(f"{dataset_name}: expected {expected}, found {actual}")
        print(f"{dataset_name}: counts verified {actual}")


if __name__ == "__main__":
    main()
