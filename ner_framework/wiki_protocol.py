from __future__ import annotations

import hashlib
import json
import random
from typing import Dict, List, Sequence


DEFAULT_WIKI_SEEDS = (42, 1771, 2401)
EXPECTED_WIKI_TEST_COUNT = 1000
PROMPT_POSITIONS = ("A", "B", "C")


def example_fingerprint(example: Dict) -> str:
    """Return a stable content fingerprint that is independent of split-local IDs."""
    payload = {
        "tokens": list(example.get("tokens", [])),
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_split_structure(examples_by_split: Dict[str, Sequence[Dict]]) -> None:
    expected_splits = {"train", "eval", "test"}
    missing = sorted(expected_splits - set(examples_by_split))
    if missing:
        raise ValueError(f"Wiki dataset is missing required splits: {', '.join(missing)}")

    ids_by_split: Dict[str, set[str]] = {}
    for split_name in sorted(expected_splits):
        ids = [str(example.get("id", "")) for example in examples_by_split[split_name]]
        if any(not example_id for example_id in ids):
            raise ValueError(f"Wiki {split_name} contains an example without an ID.")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Wiki {split_name} contains duplicate example IDs.")
        ids_by_split[split_name] = set(ids)

    for left, right in (("train", "eval"), ("train", "test"), ("eval", "test")):
        overlap = ids_by_split[left] & ids_by_split[right]
        if overlap:
            example_ids = ", ".join(sorted(overlap)[:3])
            raise ValueError(f"Wiki {left}/{right} ID overlap detected: {example_ids}")


def eligible_wiki_train_examples(examples_by_split: Dict[str, Sequence[Dict]]) -> List[Dict]:
    """Return unique, labeled train examples with no content duplicate in eval/test."""
    _validate_split_structure(examples_by_split)
    held_out_fingerprints = {
        example_fingerprint(example)
        for split_name in ("eval", "test")
        for example in examples_by_split[split_name]
    }

    eligible: List[Dict] = []
    seen_fingerprints: set[str] = set()
    for example in examples_by_split["train"]:
        tokens = example.get("tokens")
        labels = example.get("labels")
        if not example.get("text") or not example.get("entities"):
            continue
        if not isinstance(tokens, list) or not isinstance(labels, list) or len(tokens) != len(labels):
            continue

        fingerprint = example_fingerprint(example)
        if fingerprint in held_out_fingerprints or fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        eligible.append(example)

    if len(eligible) < len(PROMPT_POSITIONS):
        raise ValueError("Wiki train does not contain three eligible, non-overlapping prompt examples.")
    return eligible


def select_wiki_prompt_examples(
    examples_by_split: Dict[str, Sequence[Dict]],
    seed: int,
    count: int,
) -> List[Dict]:
    """Shuffle Wiki train deterministically and return the A/B/C prefix."""
    if count < 0 or count > len(PROMPT_POSITIONS):
        raise ValueError(f"Wiki prompt count must be between 0 and {len(PROMPT_POSITIONS)}; received {count}.")
    if count == 0:
        return []

    shuffled = list(eligible_wiki_train_examples(examples_by_split))
    random.Random(seed).shuffle(shuffled)
    selected: List[Dict] = []
    for order, example in enumerate(shuffled[:count], start=1):
        selected.append(
            {
                **example,
                "selection": {
                    "seed": seed,
                    "source_split": "train",
                    "order": order,
                    "position": PROMPT_POSITIONS[order - 1],
                    "fingerprint": example_fingerprint(example),
                },
            }
        )
    return selected


def validate_wiki_protocol(
    examples_by_split: Dict[str, Sequence[Dict]],
    prompt_examples: Sequence[Dict],
    seed: int,
    evaluation_split: str = "test",
    expected_test_count: int = EXPECTED_WIKI_TEST_COUNT,
) -> Dict[str, object]:
    """Fail fast on split leakage, prompt drift, or an incomplete test split."""
    _validate_split_structure(examples_by_split)
    if evaluation_split != "test":
        raise ValueError(f"Wiki comparison must evaluate split 'test'; received '{evaluation_split}'.")
    if len(examples_by_split["test"]) != expected_test_count:
        raise ValueError(
            f"Wiki test must contain exactly {expected_test_count} examples; "
            f"found {len(examples_by_split['test'])}."
        )

    train_ids = {str(example["id"]) for example in examples_by_split["train"]}
    held_out_ids = {
        str(example["id"])
        for split_name in ("eval", "test")
        for example in examples_by_split[split_name]
    }
    held_out_fingerprints = {
        example_fingerprint(example)
        for split_name in ("eval", "test")
        for example in examples_by_split[split_name]
    }
    prompt_ids = [str(example.get("id", "")) for example in prompt_examples]
    prompt_fingerprints = [example_fingerprint(example) for example in prompt_examples]

    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("Duplicate Wiki prompt example IDs were selected.")
    if len(prompt_fingerprints) != len(set(prompt_fingerprints)):
        raise ValueError("Duplicate Wiki prompt example content was selected.")
    if any(example_id not in train_ids for example_id in prompt_ids):
        raise ValueError("Every prompt example must originate from Wiki train.")
    if set(prompt_ids) & held_out_ids:
        raise ValueError("A prompt example ID overlaps Wiki validation or test.")
    if set(prompt_fingerprints) & held_out_fingerprints:
        raise ValueError("Prompt example content overlaps Wiki validation or test.")

    for order, example in enumerate(prompt_examples, start=1):
        selection = example.get("selection", {})
        if selection.get("source_split") != "train":
            raise ValueError("Prompt selection metadata must identify Wiki train as its source.")
        if selection.get("seed") != seed or selection.get("order") != order:
            raise ValueError("Prompt selection seed/order metadata is inconsistent.")
        if selection.get("position") != PROMPT_POSITIONS[order - 1]:
            raise ValueError("Prompt selection position metadata is inconsistent.")

    expected_prefix = select_wiki_prompt_examples(examples_by_split, seed=seed, count=len(prompt_examples))
    if prompt_ids != [str(example["id"]) for example in expected_prefix]:
        raise ValueError("Prompt examples are not the deterministic Wiki-train A/B/C prefix.")

    return {
        "evaluation_split": evaluation_split,
        "expected_test_count": expected_test_count,
        "actual_test_count": len(examples_by_split["test"]),
        "train_count": len(examples_by_split["train"]),
        "eval_count": len(examples_by_split["eval"]),
        "eligible_train_count": len(eligible_wiki_train_examples(examples_by_split)),
        "prompt_example_ids": prompt_ids,
        "prompt_source_split": "train",
        "seed": seed,
        "status": "passed",
    }
