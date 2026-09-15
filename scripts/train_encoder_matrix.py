#!/usr/bin/env python3
"""Train the Wiki encoder matrix, with extra source support only for domain shift."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "configs" / "reproduction.json"


def run(command: list[str], inspect_only: bool = False) -> None:
    print("$ " + shlex.join(command), flush=True)
    if not inspect_only:
        subprocess.run(command, cwd=ROOT_DIR, check=True)


def materialize(dataset_name: str, source_root: Path, inspect_only: bool) -> None:
    command = [
        sys.executable,
        str(ROOT_DIR / "bert-training" / "prepare_dataset.py"),
        "--dataset-name",
        dataset_name,
        "--source-root",
        str(source_root),
        "--materialized-data-root",
        str(ROOT_DIR / ".artifacts" / "datasets"),
    ]
    if dataset_name == "twitter_ner":
        command.extend(
            ["--split-ids-root", str(ROOT_DIR / "datasets" / "domain_shift" / "twitter_ner" / "split_ids")]
        )
    run(command, inspect_only=inspect_only)


def training_command(config: dict, dataset_name: str, encoder: dict, seed: int) -> list[str]:
    settings = config["encoder_training"]
    dataset_root = ROOT_DIR / ".artifacts" / "datasets" / dataset_name
    output_dir = ROOT_DIR / ".artifacts" / "models" / "bert" / f"seed-{seed}" / dataset_name / encoder["slug"]
    command = [
        sys.executable,
        str(ROOT_DIR / "bert-training" / "train.py"),
        "--source-model",
        encoder["model"],
        "--source-model-revision",
        encoder["revision"],
        "--dataset-name",
        dataset_name,
        "--dataset-root",
        str(dataset_root),
        "--materialized-data-root",
        str(ROOT_DIR / ".artifacts" / "datasets"),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(settings["epochs"]),
        "--batch-size",
        str(settings["batch_size"]),
        "--learning-rate",
        str(settings["learning_rate"]),
        "--weight-decay",
        str(settings["weight_decay"]),
        "--gradient-accumulation-steps",
        str(settings["gradient_accumulation_steps"]),
        "--warmup-ratio",
        str(settings["warmup_ratio"]),
        "--warmup-steps",
        str(settings["warmup_steps"]),
        "--lr-scheduler-type",
        settings["lr_scheduler_type"],
        "--max-grad-norm",
        str(settings["max_grad_norm"]),
        "--max-steps",
        str(settings["max_steps"]),
        "--max-length",
        str(settings["max_length"]),
        "--eval-strategy",
        settings["eval_strategy"],
        "--save-strategy",
        settings["save_strategy"],
        "--logging-strategy",
        settings["logging_strategy"],
        "--save-total-limit",
        str(settings["save_total_limit"]),
        "--seed",
        str(seed),
        "--dataloader-workers",
        str(settings["dataloader_workers"]),
        "--resume",
        settings["resume"],
    ]
    if dataset_name == "twitter_ner":
        command.extend(
            ["--split-ids-root", str(ROOT_DIR / "datasets" / "domain_shift" / "twitter_ner" / "split_ids")]
        )
    command.append("--deterministic" if settings.get("deterministic", True) else "--no-deterministic")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--wiki-source", type=Path)
    parser.add_argument("--twitter-source", type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["wiki_ner", "twitter_ner"],
        default=["wiki_ner"],
    )
    parser.add_argument("--encoders", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--allow-different-snapshot", action="store_true")
    args = parser.parse_args()

    source_roots = {"wiki_ner": args.wiki_source, "twitter_ner": args.twitter_source}
    missing = [name for name in args.datasets if source_roots[name] is None]
    if missing:
        flags = {"wiki_ner": "--wiki-source", "twitter_ner": "--twitter-source"}
        raise SystemExit("Missing required source argument(s): " + ", ".join(flags[name] for name in missing))

    validation_command = [sys.executable, str(ROOT_DIR / "scripts" / "validate_source_data.py")]
    for dataset_name in args.datasets:
        flag = "--wiki-source" if dataset_name == "wiki_ner" else "--twitter-source"
        validation_command.extend([flag, str(source_roots[dataset_name].resolve())])
    if args.allow_different_snapshot:
        validation_command.append("--allow-different-snapshot")
    run(validation_command, inspect_only=args.inspect_only)

    for dataset_name in args.datasets:
        materialize(dataset_name, source_roots[dataset_name].resolve(), args.inspect_only)
    if args.prepare_only:
        return

    config = json.loads(args.config.read_text(encoding="utf-8"))
    encoders = [item for item in config["encoders"] if args.encoders is None or item["slug"] in args.encoders]
    if not encoders:
        raise ValueError("No configured encoders matched --encoders")
    seeds = args.seeds or config["encoder_training"]["seeds"]
    undeclared = sorted(set(seeds) - set(config["encoder_training"]["seeds"]))
    if undeclared:
        raise ValueError(f"Seeds not declared in the reproduction config: {undeclared}")

    for seed in seeds:
        for dataset_name in args.datasets:
            for encoder in encoders:
                run(training_command(config, dataset_name, encoder, seed), inspect_only=args.inspect_only)


if __name__ == "__main__":
    main()
