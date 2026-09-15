#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.mapping_profiles import PROFILES, map_bio_sequence, require_profile
from analysis.ner_metrics import safe_f1, score_entity_sequences


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def infer_domain(dataset_name: str) -> str:
    if dataset_name.startswith("wiki"):
        return "wiki"
    if dataset_name.startswith("twitter"):
        return "twitter"
    raise ValueError(f"Cannot infer domain from dataset name: {dataset_name}")


def choose_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_id2label(model) -> Dict[int, str]:
    return {int(index): label for index, label in dict(model.config.id2label).items()}


def decode_batch_predictions(
    tokenizer,
    model,
    id2label: Dict[int, str],
    examples: Sequence[Dict],
    device,
    max_length: int,
) -> tuple[List[List[str]], List[List[str]], List[Dict[str, object]]]:
    import torch

    batch_tokens = [example["tokens"] for example in examples]
    encoded = tokenizer(
        batch_tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
    predicted_ids = logits.argmax(dim=-1).detach().cpu().tolist()

    decoded_predictions: List[List[str]] = []
    decoded_gold: List[List[str]] = []
    metadata: List[Dict[str, object]] = []

    for batch_index, example in enumerate(examples):
        word_ids = tokenizer(
            [example["tokens"]],
            is_split_into_words=True,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=max_length,
        ).word_ids(batch_index=0)

        seen_word_ids = set()
        prediction_sequence: List[str] = []
        gold_sequence: List[str] = []
        covered_word_ids: List[int] = []

        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_word_ids:
                continue
            if word_id >= len(example["labels"]):
                continue
            seen_word_ids.add(word_id)
            covered_word_ids.append(word_id)
            prediction_sequence.append(id2label[int(predicted_ids[batch_index][token_index])])
            gold_sequence.append(example["labels"][word_id])

        decoded_predictions.append(prediction_sequence)
        decoded_gold.append(gold_sequence)
        metadata.append(
            {
                "example_id": example.get("id"),
                "tokens": len(example.get("tokens", [])),
                "scored_tokens": len(gold_sequence),
                "truncated": len(gold_sequence) < len(example.get("tokens", [])),
                "covered_word_ids": covered_word_ids,
            }
        )

    return decoded_predictions, decoded_gold, metadata


def bootstrap_f1_ci(per_example: Sequence[Dict], samples: int, seed: int) -> Dict[str, float | int | None]:
    if not per_example or samples <= 0:
        return {"samples": 0, "lower": None, "upper": None}

    rng = random.Random(seed)
    values = []
    item_count = len(per_example)
    for _ in range(samples):
        tp = fp = fn = 0
        for _ in range(item_count):
            item = per_example[rng.randrange(item_count)]
            tp += int(item["tp"])
            fp += int(item["fp"])
            fn += int(item["fn"])
        values.append(safe_f1(tp, fp, fn))

    values.sort()
    lower_index = int(0.025 * (len(values) - 1))
    upper_index = int(0.975 * (len(values) - 1))
    return {
        "samples": samples,
        "lower": values[lower_index],
        "upper": values[upper_index],
    }


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    require_profile(args.profile)
    experiment_root = args.experiment_root.resolve()
    target_dataset_root = (
        args.target_data_dir.resolve()
        if args.target_data_dir is not None
        else experiment_root / ".artifacts" / "datasets" / args.target_dataset
    )
    if args.split == "all":
        examples = []
        for split_name in ("train", "eval", "test"):
            examples.extend(load_jsonl(target_dataset_root / f"{split_name}.jsonl"))
    else:
        examples = load_jsonl(target_dataset_root / f"{args.split}.jsonl")
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    target_domain = args.target_domain or infer_domain(args.target_dataset)
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir, local_files_only=True)
    model.to(device)
    model.eval()
    id2label = model_id2label(model)

    all_predicted_mapped: List[List[str]] = []
    all_gold_mapped: List[List[str]] = []
    all_metadata: List[Dict[str, object]] = []
    started_at = time.perf_counter()

    for offset in range(0, len(examples), args.batch_size):
        batch_examples = examples[offset : offset + args.batch_size]
        predicted_labels, gold_labels, metadata = decode_batch_predictions(
            tokenizer=tokenizer,
            model=model,
            id2label=id2label,
            examples=batch_examples,
            device=device,
            max_length=args.max_length,
        )
        for prediction_sequence, gold_sequence in zip(predicted_labels, gold_labels):
            all_predicted_mapped.append(
                map_bio_sequence(prediction_sequence, domain=args.model_domain, profile_name=args.profile)
            )
            all_gold_mapped.append(
                map_bio_sequence(gold_sequence, domain=target_domain, profile_name=args.profile)
            )
        all_metadata.extend(metadata)

    wall_time = time.perf_counter() - started_at
    metrics = score_entity_sequences(gold_sequences=all_gold_mapped, predicted_sequences=all_predicted_mapped)
    token_total = sum(len(sequence) for sequence in all_gold_mapped)
    token_correct = sum(
        int(gold_label == predicted_label)
        for gold_sequence, predicted_sequence in zip(all_gold_mapped, all_predicted_mapped)
        for gold_label, predicted_label in zip(gold_sequence, predicted_sequence)
    )
    metrics["accuracy"] = token_correct / token_total if token_total else 0.0
    ci = bootstrap_f1_ci(metrics["per_example"], samples=args.bootstrap_samples, seed=args.seed)

    summary = {
        "model_dir": str(args.model_dir),
        "model_domain": args.model_domain,
        "target_dataset": args.target_dataset,
        "target_dataset_dir": str(target_dataset_root),
        "target_domain": target_domain,
        "split": args.split,
        "profile": args.profile,
        "profile_description": PROFILES[args.profile].description,
        "active_labels": PROFILES[args.profile].active_labels(),
        "examples": len(examples),
        "item_count": len(examples),
        "device": str(device),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "wall_time_seconds": wall_time,
        "average_latency_seconds": wall_time / max(1, len(examples)),
        "metrics": {key: value for key, value in metrics.items() if key != "per_example"},
        "bootstrap_f1_ci": ci,
        "per_example": metrics["per_example"],
        "example_metadata": all_metadata,
        "torch_version": torch.__version__,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mapping-aware BERT cross-domain NER evaluation.")
    parser.add_argument("--experiment-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-domain", choices=["wiki", "twitter"], required=True)
    parser.add_argument("--target-dataset", choices=["wiki_ner", "twitter_ner"], required=True)
    parser.add_argument(
        "--target-data-dir",
        type=Path,
        default=None,
        help="Local materialized dataset directory. Defaults to .artifacts/datasets/<target-dataset>.",
    )
    parser.add_argument("--target-domain", choices=["wiki", "twitter"], default=None)
    parser.add_argument("--split", choices=["train", "eval", "test", "all"], default="test")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="conservative")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def default_output_path(args: argparse.Namespace) -> Path:
    model_slug = args.model_dir.name
    filename = f"{model_slug}__{args.model_domain}_to_{args.target_dataset}_{args.split}__{args.profile}.json"
    if args.max_examples is not None:
        filename = filename.replace(".json", f"__max{args.max_examples}.json")
    return args.experiment_root / ".artifacts" / "results" / "cross-domain" / filename


def main() -> None:
    args = parse_args()
    args.model_dir = args.model_dir.resolve()
    args.experiment_root = args.experiment_root.resolve()
    summary = evaluate(args)
    output_path = args.output_path or default_output_path(args)
    write_json(output_path, summary)
    metrics = summary["metrics"]
    print(f"Saved evaluation to {output_path}")
    print(
        "F1={f1:.4f} Precision={precision:.4f} Recall={recall:.4f} Examples={examples}".format(
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            examples=summary["examples"],
        )
    )


if __name__ == "__main__":
    main()
