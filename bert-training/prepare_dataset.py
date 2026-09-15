#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ner_framework.data import ensure_materialized_dataset
from ner_framework.utils import read_json


DEFAULT_MATERIALIZED_ROOT = ROOT_DIR / ".artifacts" / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct train/eval/test splits under the ignored .artifacts directory.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--split-ids-root",
        type=Path,
        default=None,
        help="Directory containing train.ids, validation.ids, and test.ids.",
    )
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--materialized-data-root", type=Path, default=DEFAULT_MATERIALIZED_ROOT)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_ids_root = args.split_ids_root or ROOT_DIR / "datasets" / "split_ids" / args.dataset_name
    dataset = ensure_materialized_dataset(
        dataset_name=args.dataset_name,
        materialized_data_root=args.materialized_data_root,
        source_root=args.source_root,
        train_file=args.train_file,
        eval_file=args.eval_file,
        test_file=args.test_file,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        force=args.force,
        split_ids_root=split_ids_root,
    )

    manifest = read_json(dataset.manifest_path)
    print(f"Materialized dataset: {dataset.dataset_name}")
    print(f"Dataset root: {dataset.dataset_root}")
    print(f"Counts: train={manifest['counts']['train']}, eval={manifest['counts']['eval']}, test={manifest['counts']['test']}")
    print(f"Labels: {len(manifest['labels'])}")


if __name__ == "__main__":
    main()
