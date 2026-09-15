#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "configs" / "reproduction.json"
CELLS = {
    "wiki_to_wiki": ("wiki_ner", "wiki", "wiki_ner"),
    "wiki_to_twitter": ("wiki_ner", "wiki", "twitter_ner"),
    "twitter_to_twitter": ("twitter_ner", "twitter", "twitter_ner"),
    "twitter_to_wiki": ("twitter_ner", "twitter", "wiki_ner"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate locally trained encoders across the thesis domains.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--encoders", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--profiles", nargs="+", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    encoders = [item for item in config["encoders"] if args.encoders is None or item["slug"] in args.encoders]
    seeds = args.seeds or config["encoder_training"]["seeds"]
    primary = config["domain_shift"]["primary_profile"]
    profiles = args.profiles or [primary]

    for seed in seeds:
        for encoder in encoders:
            for cell_name in config["domain_shift"]["cells"]:
                training_dataset, model_domain, target_dataset = CELLS[cell_name]
                cell_profiles = profiles
                if args.profiles is None and cell_name == "wiki_to_twitter":
                    cell_profiles = [primary, *config["domain_shift"]["sensitivity_profiles"]]
                for profile in cell_profiles:
                    model_dir = ROOT_DIR / ".artifacts" / "models" / "bert" / f"seed-{seed}" / training_dataset / encoder["slug"]
                    output = ROOT_DIR / ".artifacts" / "results" / "cross-domain" / f"seed-{seed}" / encoder["slug"] / cell_name / f"{profile}.json"
                    command = [
                        sys.executable,
                        str(ROOT_DIR / "analysis" / "evaluate_cross_domain.py"),
                        "--model-dir",
                        str(model_dir),
                        "--model-domain",
                        model_domain,
                        "--target-dataset",
                        target_dataset,
                        "--target-data-dir",
                        str(ROOT_DIR / ".artifacts" / "datasets" / target_dataset),
                        "--split",
                        config["domain_shift"]["split"],
                        "--profile",
                        profile,
                        "--seed",
                        str(seed),
                        "--output-path",
                        str(output),
                    ]
                    if args.max_examples is not None:
                        command.extend(["--max-examples", str(args.max_examples)])
                    print("$ " + shlex.join(command), flush=True)
                    if not args.inspect_only:
                        subprocess.run(command, cwd=ROOT_DIR, check=True)


if __name__ == "__main__":
    main()
