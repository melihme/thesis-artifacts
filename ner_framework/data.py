"""Materialize WikiNER and the Twitter source used only for domain-shift analysis."""

from __future__ import annotations

import csv
import hashlib
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .utils import read_json, read_jsonl, sanitize_name, write_json, write_jsonl


EXPECTED_SPLITS = ("train", "eval", "test")
SPLIT_ID_FILENAMES = {
    "train": "train.ids",
    "eval": "validation.ids",
    "test": "test.ids",
}
OFFICIAL_EVAL_FILENAMES = ("eval.conll", "dev.conll", "validation.conll", "val.conll")
TWITTER_TEXT_SPLIT_FILENAMES = {
    "train": "train_with_tweet_text.tsv",
    "eval": "val_with_tweet_text.tsv",
    "test": "test_with_tweet_text.tsv",
}
TWITTER_ENTITY_TYPE_ALIASES = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "LOCATION": "LOCATION",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "ORGANIZATION": "ORGANIZATION",
    "ORGANISATION": "ORGANIZATION",
    "ORG": "ORGANIZATION",
    "MONEY": "MONEY",
    "TIME": "TIME",
    "DATE": "TIME",
    "PRODUCT": "PRODUCT",
    "TVSHOW": "TVSHOW",
    "TV_SHOW": "TVSHOW",
    "TV-SHOW": "TVSHOW",
}
TWITTER_TOKEN_PATTERN = re.compile(
    r"https?://\S+|[@#][\w_]+(?:['’][\w_]+)*|\w+(?:['’][\w]+)*|[^\w\s]",
    flags=re.UNICODE,
)


@dataclass
class MaterializedDataset:
    dataset_name: str
    dataset_root: Path
    manifest_path: Path


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split_ids(split_ids_root: Path) -> Dict[str, List[str]]:
    split_ids: Dict[str, List[str]] = {}
    all_ids: Dict[str, str] = {}

    for split_name, filename in SPLIT_ID_FILENAMES.items():
        path = split_ids_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing split-ID file: {path}")
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not values:
            raise ValueError(f"Split-ID file is empty: {path}")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate identifiers in {path}")
        for value in values:
            previous = all_ids.get(value)
            if previous is not None:
                raise ValueError(f"Identifier {value!r} occurs in both {previous} and {split_name}")
            all_ids[value] = split_name
        split_ids[split_name] = values

    return split_ids


def build_split_id_manifest(split_ids_root: Path, split_ids: Dict[str, List[str]]) -> Dict[str, object]:
    return {
        "root": str(split_ids_root),
        "files": {
            split_name: {
                "path": str(split_ids_root / SPLIT_ID_FILENAMES[split_name]),
                "sha256": sha256_path(split_ids_root / SPLIT_ID_FILENAMES[split_name]),
                "count": len(split_ids[split_name]),
            }
            for split_name in EXPECTED_SPLITS
        },
    }


def apply_split_ids(
    examples_by_split: Dict[str, List[Dict]],
    split_ids_root: Path,
) -> Tuple[Dict[str, List[Dict]], Dict[str, object]]:
    requested_by_split = read_split_ids(split_ids_root)
    selected_by_split: Dict[str, List[Dict]] = {}

    for split_name in EXPECTED_SPLITS:
        examples = examples_by_split.get(split_name, [])
        requested_ids = requested_by_split[split_name]
        by_identifier: Dict[str, Dict] = {}
        for example in examples:
            identifier = str(example.get("tweet_id") or example.get("id") or "")
            if not identifier:
                raise ValueError(f"Example in {split_name} has neither tweet_id nor id")
            if identifier in by_identifier:
                raise ValueError(f"Source contains duplicate identifier {identifier!r} in {split_name}")
            by_identifier[identifier] = example

        missing = [identifier for identifier in requested_ids if identifier not in by_identifier]
        if missing:
            preview = ", ".join(missing[:10])
            raise RuntimeError(
                f"Unable to reconstruct {split_name}: {len(missing)} required IDs are missing "
                f"from the licensed source (first IDs: {preview}). This is not an equivalent reproduction."
            )

        selected_by_split[split_name] = [by_identifier[identifier] for identifier in requested_ids]
    return selected_by_split, build_split_id_manifest(split_ids_root, requested_by_split)


def is_materialized_dataset(root: Path) -> bool:
    return root.is_dir() and (root / "manifest.json").exists() and all(
        (root / f"{split}.jsonl").exists() for split in EXPECTED_SPLITS
    )


def infer_dataset_name(dataset_name: Optional[str], source_root: Optional[Path]) -> str:
    if dataset_name:
        return dataset_name
    if source_root is None:
        return "dataset"
    if source_root.is_file():
        return source_root.stem
    return source_root.name


def _candidate_split_roots(source_root: Path) -> List[Path]:
    candidates: List[Path] = []
    if source_root.is_dir():
        candidates.append(source_root)
        for train_path in sorted(source_root.rglob("train.conll")):
            candidates.append(train_path.parent)
    seen = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(candidate)
    return unique_candidates


def discover_official_split_paths(source_root: Path) -> Optional[Dict[str, Path]]:
    for candidate_root in _candidate_split_roots(source_root):
        train_path = candidate_root / "train.conll"
        test_path = candidate_root / "test.conll"
        if not train_path.exists() or not test_path.exists():
            continue

        for eval_filename in OFFICIAL_EVAL_FILENAMES:
            eval_path = candidate_root / eval_filename
            if eval_path.exists():
                return {
                    "train": train_path,
                    "eval": eval_path,
                    "test": test_path,
                }

    return None


def _candidate_twitter_roots(source_root: Path) -> List[Path]:
    candidates: List[Path] = []
    if source_root.is_dir():
        candidates.append(source_root)
        for train_path in sorted(source_root.rglob(TWITTER_TEXT_SPLIT_FILENAMES["train"])):
            candidates.append(train_path.parent)

    seen = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(candidate)
    return unique_candidates


def discover_twitter_split_paths(source_root: Path) -> Optional[Tuple[Path, Dict[str, Path]]]:
    for candidate_root in _candidate_twitter_roots(source_root):
        split_paths = {
            split_name: candidate_root / filename
            for split_name, filename in TWITTER_TEXT_SPLIT_FILENAMES.items()
        }
        if all(path.exists() for path in split_paths.values()):
            return candidate_root, split_paths
    return None


def normalize_twitter_entity_type(raw_label: object) -> Optional[str]:
    if not isinstance(raw_label, str):
        return None

    normalized = raw_label.strip().upper().replace(" ", "_")
    alias = TWITTER_ENTITY_TYPE_ALIASES.get(normalized)
    if alias is not None:
        return alias

    compact = normalized.replace("_", "").replace("-", "")
    return TWITTER_ENTITY_TYPE_ALIASES.get(compact)


def tokenize_twitter_text(text: str) -> Tuple[List[str], List[List[int]]]:
    tokens: List[str] = []
    offsets: List[List[int]] = []

    for match in TWITTER_TOKEN_PATTERN.finditer(text):
        tokens.append(match.group(0))
        offsets.append([match.start(), match.end()])

    return tokens, offsets


def build_twitter_labels_and_entities(
    text: str,
    token_offsets: Sequence[Sequence[int]],
    rows: Sequence[Dict[str, str]],
) -> Tuple[Optional[List[str]], Optional[List[Dict]], Optional[str]]:
    if not token_offsets:
        return None, None, "no_tokens"

    labels = ["O"] * len(token_offsets)
    entities: List[Dict] = []
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            int(row["start_pos"]),
            int(row["end_pos"]),
            row["named_entity_type"],
        ),
    )

    for row in sorted_rows:
        entity_type = normalize_twitter_entity_type(row.get("named_entity_type"))
        if entity_type is None:
            return None, None, f"unsupported_label:{row.get('named_entity_type')}"

        start_char = int(row["start_pos"])
        end_char = int(row["end_pos"])
        end_exclusive = end_char + 1

        if start_char < 0 or end_exclusive > len(text) or start_char >= end_exclusive:
            return None, None, "invalid_char_span"

        token_indices = [
            index
            for index, (token_start, token_end) in enumerate(token_offsets)
            if token_end > start_char and token_start < end_exclusive
        ]
        if not token_indices:
            return None, None, "unaligned_span"
        if any(labels[index] != "O" for index in token_indices):
            return None, None, "overlapping_span"

        labels[token_indices[0]] = f"B-{entity_type}"
        for token_index in token_indices[1:]:
            labels[token_index] = f"I-{entity_type}"

        entities.append(
            {
                "label": entity_type,
                "text": text[start_char:end_exclusive],
                "start_token": token_indices[0],
                "end_token": token_indices[-1],
                "start_char": start_char,
                "end_char": end_char,
            }
        )

    return labels, entities, None


def materialize_twitter_split(split_name: str, split_path: Path) -> Tuple[List[Dict], Dict[str, object]]:
    rows_by_tweet: Dict[str, List[Dict[str, str]]] = {}
    ordered_tweet_ids: List[str] = []

    with split_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required_columns = {
            "tweet_id",
            "start_pos",
            "end_pos",
            "named_entity_type",
            "tweet_text",
        }
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{split_path} must contain the columns: {sorted(required_columns)}"
            )

        for row in reader:
            tweet_id = (row.get("tweet_id") or "").strip()
            if not tweet_id:
                continue
            if tweet_id not in rows_by_tweet:
                ordered_tweet_ids.append(tweet_id)
                rows_by_tweet[tweet_id] = []
            rows_by_tweet[tweet_id].append(row)

    examples: List[Dict] = []
    conversion_failures: Dict[str, int] = {}
    tweets_dropped_missing_text = 0
    entity_rows_with_text = 0

    for tweet_id in ordered_tweet_ids:
        tweet_rows = rows_by_tweet[tweet_id]
        tweet_text = tweet_rows[0].get("tweet_text") or ""

        if not tweet_text.strip():
            tweets_dropped_missing_text += 1
            continue

        entity_rows_with_text += len(tweet_rows)
        tokens, token_offsets = tokenize_twitter_text(tweet_text)
        labels, entities, failure_reason = build_twitter_labels_and_entities(
            text=tweet_text,
            token_offsets=token_offsets,
            rows=tweet_rows,
        )
        if failure_reason is not None or labels is None or entities is None:
            key = failure_reason or "unknown_failure"
            conversion_failures[key] = conversion_failures.get(key, 0) + 1
            continue

        examples.append(
            {
                "id": f"{split_name}-{len(examples):06d}",
                "tweet_id": tweet_id,
                "tokens": tokens,
                "labels": labels,
                "text": tweet_text,
                "token_offsets": token_offsets,
                "entities": entities,
            }
        )

    if not examples:
        raise RuntimeError(f"No usable domain-shift Twitter examples were materialized from {split_path}.")

    kept_entity_rows = sum(len(example["entities"]) for example in examples)
    stats: Dict[str, object] = {
        "source_file": str(split_path),
        "annotation_tweets_total": len(ordered_tweet_ids),
        "annotation_rows_total": sum(len(rows) for rows in rows_by_tweet.values()),
        "tweets_with_text": len(ordered_tweet_ids) - tweets_dropped_missing_text,
        "tweets_kept": len(examples),
        "tweets_dropped_missing_text": tweets_dropped_missing_text,
        "tweets_dropped_conversion": len(ordered_tweet_ids) - tweets_dropped_missing_text - len(examples),
        "entity_rows_with_text": entity_rows_with_text,
        "entity_rows_kept": kept_entity_rows,
        "tweet_text_coverage_ratio": (
            (len(ordered_tweet_ids) - tweets_dropped_missing_text) / len(ordered_tweet_ids)
            if ordered_tweet_ids
            else 0.0
        ),
        "tweet_kept_ratio": (
            len(examples) / len(ordered_tweet_ids)
            if ordered_tweet_ids
            else 0.0
        ),
        "conversion_failures": conversion_failures,
    }
    return examples, stats


def read_conll(path: Path) -> List[Tuple[List[str], List[str]]]:
    sequences: List[Tuple[List[str], List[str]]] = []
    tokens: List[str] = []
    labels: List[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                if tokens:
                    sequences.append((tokens, labels))
                    tokens, labels = [], []
                continue

            if line.startswith("#") or line.startswith("-DOCSTART-"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            tokens.append(parts[0])
            labels.append(parts[-1])

    if tokens:
        sequences.append((tokens, labels))

    return sequences


def build_text_and_offsets(tokens: Sequence[str]) -> Tuple[str, List[List[int]]]:
    text_parts: List[str] = []
    offsets: List[List[int]] = []
    cursor = 0

    for index, token in enumerate(tokens):
        if index > 0:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(token)
        cursor += len(token)
        offsets.append([start, cursor])

    return "".join(text_parts), offsets


def labels_to_entities(
    tokens: Sequence[str],
    labels: Sequence[str],
    token_offsets: Sequence[Sequence[int]],
    text: str,
) -> List[Dict]:
    entities: List[Dict] = []
    current_type: Optional[str] = None
    current_start_token: Optional[int] = None
    current_end_token: Optional[int] = None

    def flush() -> None:
        nonlocal current_type, current_start_token, current_end_token
        if current_type is None or current_start_token is None or current_end_token is None:
            current_type = None
            current_start_token = None
            current_end_token = None
            return

        start_char = int(token_offsets[current_start_token][0])
        end_char = int(token_offsets[current_end_token][1]) - 1
        entities.append(
            {
                "label": current_type,
                "text": text[start_char : end_char + 1],
                "start_token": current_start_token,
                "end_token": current_end_token,
                "start_char": start_char,
                "end_char": end_char,
            }
        )
        current_type = None
        current_start_token = None
        current_end_token = None

    for token_index, label in enumerate(labels):
        if label == "O" or "-" not in label:
            flush()
            continue

        prefix, entity_type = label.split("-", 1)
        prefix = prefix.upper()

        if prefix not in {"B", "I"}:
            flush()
            continue

        if prefix == "B" or current_type != entity_type or current_end_token is None:
            flush()
            current_type = entity_type
            current_start_token = token_index
            current_end_token = token_index
            continue

        current_end_token = token_index

    flush()
    return entities


def sequence_to_example(
    split_name: str,
    example_index: int,
    tokens: Sequence[str],
    labels: Sequence[str],
) -> Dict:
    if len(tokens) != len(labels):
        raise ValueError("Token and label sequence lengths do not match.")

    text, token_offsets = build_text_and_offsets(tokens)
    entities = labels_to_entities(tokens=tokens, labels=labels, token_offsets=token_offsets, text=text)

    return {
        "id": f"{split_name}-{example_index:06d}",
        "tokens": list(tokens),
        "labels": list(labels),
        "text": text,
        "token_offsets": token_offsets,
        "entities": entities,
    }


def sequences_to_examples(split_name: str, sequences: Sequence[Tuple[List[str], List[str]]]) -> List[Dict]:
    return [
        sequence_to_example(split_name=split_name, example_index=index, tokens=tokens, labels=labels)
        for index, (tokens, labels) in enumerate(sequences)
    ]


def write_conll(path: Path, examples: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            for token, label in zip(example["tokens"], example["labels"]):
                handle.write(f"{token}\t{label}\n")
            handle.write("\n")


def collect_labels(examples_by_split: Dict[str, Sequence[Dict]]) -> List[str]:
    labels = {
        label
        for examples in examples_by_split.values()
        for example in examples
        for label in example["labels"]
    }
    return sorted(labels, key=lambda label: (label != "O", label))


def collect_entity_types(examples_by_split: Dict[str, Sequence[Dict]]) -> List[str]:
    entity_types = {
        entity["label"]
        for examples in examples_by_split.values()
        for example in examples
        for entity in example["entities"]
    }
    return sorted(entity_types)


def split_examples_randomly(
    examples: Sequence[Dict],
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Dict]]:
    if len(examples) < 3:
        raise ValueError("At least 3 examples are required to materialize train/eval/test splits.")

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be less than 1.")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    test_count = max(1, round(total * test_ratio))
    eval_count = max(1, round(total * validation_ratio))

    while total - test_count - eval_count < 1:
        if test_count >= eval_count and test_count > 1:
            test_count -= 1
        elif eval_count > 1:
            eval_count -= 1
        else:
            raise ValueError("Unable to leave at least one training example after splitting.")

    train_count = total - test_count - eval_count
    train_examples = shuffled[:train_count]
    eval_examples = shuffled[train_count : train_count + eval_count]
    test_examples = shuffled[train_count + eval_count :]

    return {
        "train": train_examples,
        "eval": eval_examples,
        "test": test_examples,
    }


def load_materialized_dataset(dataset_root: Path) -> Dict[str, List[Dict]]:
    return {
        split_name: read_jsonl(dataset_root / f"{split_name}.jsonl")
        for split_name in EXPECTED_SPLITS
    }


def load_manifest(dataset_root: Path) -> Dict:
    return read_json(dataset_root / "manifest.json")


def _copy_materialized_dataset(source_root: Path, target_root: Path, force: bool) -> None:
    if target_root.exists() and force:
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    for split_name in EXPECTED_SPLITS:
        shutil.copy2(source_root / f"{split_name}.jsonl", target_root / f"{split_name}.jsonl")
        conll_path = source_root / f"{split_name}.conll"
        if conll_path.exists():
            shutil.copy2(conll_path, target_root / f"{split_name}.conll")
    shutil.copy2(source_root / "manifest.json", target_root / "manifest.json")


def ensure_materialized_dataset(
    dataset_name: Optional[str],
    materialized_data_root: Path,
    source_root: Optional[Path] = None,
    train_file: Optional[Path] = None,
    eval_file: Optional[Path] = None,
    test_file: Optional[Path] = None,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    force: bool = False,
    split_ids_root: Optional[Path] = None,
) -> MaterializedDataset:
    materialized_data_root = materialized_data_root.resolve()
    source_root = source_root.resolve() if source_root is not None else None
    split_ids_root = split_ids_root.resolve() if split_ids_root is not None else None
    dataset_name = infer_dataset_name(dataset_name=dataset_name, source_root=source_root)
    target_root = materialized_data_root / sanitize_name(dataset_name)
    manifest_path = target_root / "manifest.json"

    if target_root.exists() and is_materialized_dataset(target_root) and not force:
        manifest = load_manifest(target_root)
        if split_ids_root is not None and "split_ids" not in manifest:
            raise RuntimeError(
                f"Existing materialized dataset at {target_root} was not built from the public split IDs. "
                "Rerun with --force-materialize."
            )
        if split_ids_root is not None:
            current_split_ids = read_split_ids(split_ids_root)
            current_manifest = build_split_id_manifest(split_ids_root, current_split_ids)
            recorded_files = manifest["split_ids"].get("files", {})
            for split_name in EXPECTED_SPLITS:
                recorded = recorded_files.get(split_name, {})
                current = current_manifest["files"][split_name]
                if recorded.get("sha256") != current["sha256"] or recorded.get("count") != current["count"]:
                    raise RuntimeError(
                        f"Public split IDs changed for {split_name}. Rerun with --force-materialize."
                    )
        return MaterializedDataset(
            dataset_name=manifest.get("dataset_name", dataset_name),
            dataset_root=target_root,
            manifest_path=manifest_path,
        )

    if source_root is not None and is_materialized_dataset(source_root):
        _copy_materialized_dataset(source_root=source_root, target_root=target_root, force=force)
        manifest = load_manifest(target_root)
        return MaterializedDataset(
            dataset_name=manifest.get("dataset_name", dataset_name),
            dataset_root=target_root,
            manifest_path=target_root / "manifest.json",
        )

    examples_by_split: Dict[str, List[Dict]]
    source_description: Dict[str, Optional[str]]
    extra_manifest: Dict[str, object] = {}

    if train_file is not None:
        train_sequences = read_conll(train_file)
        if eval_file is not None and test_file is not None:
            examples_by_split = {
                "train": sequences_to_examples("train", train_sequences),
                "eval": sequences_to_examples("eval", read_conll(eval_file)),
                "test": sequences_to_examples("test", read_conll(test_file)),
            }
        else:
            train_examples = sequences_to_examples("source", train_sequences)
            examples_by_split = split_examples_randomly(
                examples=train_examples,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
            for split_name in EXPECTED_SPLITS:
                examples_by_split[split_name] = [
                    {
                        **example,
                        "id": f"{split_name}-{index:06d}",
                    }
                    for index, example in enumerate(examples_by_split[split_name])
                ]

        source_description = {
            "mode": "explicit_files",
            "train_file": str(train_file),
            "eval_file": str(eval_file) if eval_file is not None else None,
            "test_file": str(test_file) if test_file is not None else None,
            "source_root": str(source_root) if source_root is not None else None,
        }
    elif source_root is not None:
        official_splits = discover_official_split_paths(source_root)
        if official_splits is not None:
            examples_by_split = {
                split_name: sequences_to_examples(split_name, read_conll(split_path))
                for split_name, split_path in official_splits.items()
            }
            source_description = {
                "mode": "official_conll_splits",
                "source_root": str(source_root),
                "train_file": str(official_splits["train"]),
                "eval_file": str(official_splits["eval"]),
                "test_file": str(official_splits["test"]),
            }
        else:
            twitter_source = discover_twitter_split_paths(source_root) if dataset_name == "twitter_ner" else None
            if twitter_source is not None:
                twitter_dataset_dir, twitter_split_paths = twitter_source
                twitter_examples_by_split = {
                    split_name: materialize_twitter_split(split_name, split_path)
                    for split_name, split_path in twitter_split_paths.items()
                }
                examples_by_split = {
                    split_name: split_examples
                    for split_name, (split_examples, _) in twitter_examples_by_split.items()
                }
                source_description = {
                    "mode": "twitter_tsv_with_text_splits",
                    "source_root": str(source_root),
                    "dataset_dir": str(twitter_dataset_dir),
                    "train_file": str(twitter_split_paths["train"]),
                    "eval_file": str(twitter_split_paths["eval"]),
                    "test_file": str(twitter_split_paths["test"]),
                }
                extra_manifest = {
                    "source_counts": {
                        split_name: split_stats
                        for split_name, (_, split_stats) in twitter_examples_by_split.items()
                    }
                }
            elif source_root.is_file() and source_root.suffix.lower() == ".conll":
                source_examples = sequences_to_examples("source", read_conll(source_root))
                examples_by_split = split_examples_randomly(
                    examples=source_examples,
                    validation_ratio=validation_ratio,
                    test_ratio=test_ratio,
                    seed=seed,
                )
                for split_name in EXPECTED_SPLITS:
                    examples_by_split[split_name] = [
                        {
                            **example,
                            "id": f"{split_name}-{index:06d}",
                        }
                        for index, example in enumerate(examples_by_split[split_name])
                    ]
                source_description = {
                    "mode": "single_conll_file",
                    "source_root": str(source_root),
                    "train_file": str(source_root),
                    "eval_file": None,
                    "test_file": None,
                }
            else:
                raise FileNotFoundError(
                    f"Could not discover supported dataset splits under {source_root}. "
                    "Pass explicit --train-file/--eval-file/--test-file instead."
                )
    else:
        raise ValueError("A source_root or explicit train/eval/test files are required.")

    if split_ids_root is not None:
        examples_by_split, split_id_manifest = apply_split_ids(
            examples_by_split=examples_by_split,
            split_ids_root=split_ids_root,
        )
        extra_manifest["split_ids"] = split_id_manifest

    target_root.mkdir(parents=True, exist_ok=True)

    for split_name, examples in examples_by_split.items():
        write_jsonl(target_root / f"{split_name}.jsonl", examples)
        write_conll(target_root / f"{split_name}.conll", examples)

    manifest = {
        "dataset_name": dataset_name,
        "dataset_root": str(target_root),
        "source": source_description,
        "counts": {split_name: len(examples) for split_name, examples in examples_by_split.items()},
        "labels": collect_labels(examples_by_split),
        "entity_types": collect_entity_types(examples_by_split),
        "splits": {
            split_name: {
                "jsonl": str(target_root / f"{split_name}.jsonl"),
                "conll": str(target_root / f"{split_name}.conll"),
            }
            for split_name in EXPECTED_SPLITS
        },
    }
    manifest.update(extra_manifest)
    write_json(manifest_path, manifest)

    return MaterializedDataset(
        dataset_name=dataset_name,
        dataset_root=target_root,
        manifest_path=manifest_path,
    )
