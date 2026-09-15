#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ner_framework.data import ensure_materialized_dataset
from ner_framework.utils import read_json, sanitize_name, set_seed, write_json


os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_MATERIALIZED_ROOT = ROOT_DIR / ".artifacts" / "datasets"
DEFAULT_MODEL_ROOT = ROOT_DIR / ".artifacts" / "models" / "bert"
DEFAULT_TRAIN_CONFIG = {
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.0,
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.0,
    "warmup_steps": 0,
    "lr_scheduler_type": "linear",
    "max_grad_norm": 1.0,
    "max_steps": -1,
    "max_length": 512,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_strategy": "epoch",
    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 100,
    "save_total_limit": 2,
    "seed": 42,
    "dataloader_workers": 0,
    "resume": "auto",
    "delete_checkpoints_after_training": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Turkish BERT-style NER model from a materialized dataset.")
    parser.add_argument("--source-model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--source-model-revision", required=True, help="Immutable Hugging Face commit SHA.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Licensed source or locally materialized dataset root.")
    parser.add_argument("--split-ids-root", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--materialized-data-root", type=Path, default=DEFAULT_MATERIALIZED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_CONFIG["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_TRAIN_CONFIG["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_TRAIN_CONFIG["weight_decay"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_TRAIN_CONFIG["gradient_accumulation_steps"])
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_TRAIN_CONFIG["warmup_ratio"])
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_TRAIN_CONFIG["warmup_steps"])
    parser.add_argument("--lr-scheduler-type", default=DEFAULT_TRAIN_CONFIG["lr_scheduler_type"])
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_TRAIN_CONFIG["max_grad_norm"])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_TRAIN_CONFIG["max_steps"])
    parser.add_argument("--max-length", type=int, default=DEFAULT_TRAIN_CONFIG["max_length"])
    parser.add_argument("--eval-strategy", choices=["no", "steps", "epoch"], default=DEFAULT_TRAIN_CONFIG["eval_strategy"])
    parser.add_argument("--save-strategy", choices=["no", "steps", "epoch"], default=DEFAULT_TRAIN_CONFIG["save_strategy"])
    parser.add_argument("--logging-strategy", choices=["no", "steps", "epoch"], default=DEFAULT_TRAIN_CONFIG["logging_strategy"])
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_TRAIN_CONFIG["eval_steps"])
    parser.add_argument("--save-steps", type=int, default=DEFAULT_TRAIN_CONFIG["save_steps"])
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_TRAIN_CONFIG["logging_steps"])
    parser.add_argument("--save-total-limit", type=int, default=DEFAULT_TRAIN_CONFIG["save_total_limit"])
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_CONFIG["seed"])
    parser.add_argument("--dataloader-workers", type=int, default=DEFAULT_TRAIN_CONFIG["dataloader_workers"])
    parser.add_argument("--resume", choices=["auto", "never"], default=DEFAULT_TRAIN_CONFIG["resume"])
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--force-materialize", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def default_output_dir(source_model: str) -> Path:
    return DEFAULT_MODEL_ROOT / sanitize_name(source_model)


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    split_ids_root = args.split_ids_root or ROOT_DIR / "datasets" / "split_ids" / args.dataset_name
    dataset = ensure_materialized_dataset(
        dataset_name=args.dataset_name,
        materialized_data_root=args.materialized_data_root,
        source_root=args.dataset_root,
        train_file=args.train_file,
        eval_file=args.eval_file,
        test_file=args.test_file,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        force=args.force_materialize,
        split_ids_root=split_ids_root,
    )
    manifest = read_json(dataset.manifest_path)

    output_dir = args.output_dir or default_output_dir(args.source_model)

    print(f"Source model: {args.source_model}")
    print(f"Materialized dataset: {dataset.dataset_root}")
    print(f"Output model dir: {output_dir}")
    print(
        "Split counts: "
        f"train={manifest['counts']['train']}, "
        f"eval={manifest['counts']['eval']}, "
        f"test={manifest['counts']['test']}"
    )

    if args.inspect_only:
        print("Inspect only mode enabled. No model weights were loaded.")
        return

    from ner_framework.bert import train_token_classifier

    summary = train_token_classifier(
        source_model=args.source_model,
        source_model_revision=args.source_model_revision,
        dataset_root=dataset.dataset_root,
        model_output_dir=output_dir,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": args.warmup_steps,
            "lr_scheduler_type": args.lr_scheduler_type,
            "max_grad_norm": args.max_grad_norm,
            "max_steps": args.max_steps,
            "max_length": args.max_length,
            "eval_strategy": args.eval_strategy,
            "save_strategy": args.save_strategy,
            "logging_strategy": args.logging_strategy,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "logging_steps": args.logging_steps,
            "save_total_limit": args.save_total_limit,
            "seed": args.seed,
            "dataloader_workers": args.dataloader_workers,
            "resume": args.resume,
            "deterministic": args.deterministic,
        },
    )
    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "training_config.json", {"cli_args": vars(args), "dataset_manifest": manifest})
    print(f"Saved training summary to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
