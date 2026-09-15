from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ner_framework.wiki_protocol import (
    eligible_wiki_train_examples,
    select_wiki_prompt_examples,
    validate_wiki_protocol,
)


def example(split: str, index: int, token: str, label: str = "B-PERSON") -> dict:
    entities = [] if label == "O" else [{"text": token, "label": label.split("-", 1)[1]}]
    return {
        "id": f"{split}-{index:06d}",
        "tokens": [token],
        "labels": [label],
        "text": token,
        "token_offsets": [[0, len(token)]],
        "entities": entities,
    }


def dataset() -> dict:
    return {
        "train": [example("train", index, f"train-{index}") for index in range(8)],
        "eval": [example("eval", 0, "validation")],
        "test": [example("test", 0, "held-out")],
    }


class WikiProtocolTests(unittest.TestCase):
    def test_one_shot_is_prefix_of_three_shot(self) -> None:
        examples = dataset()
        for seed in (42, 1771, 2401):
            one_shot = select_wiki_prompt_examples(examples, seed=seed, count=1)
            three_shot = select_wiki_prompt_examples(examples, seed=seed, count=3)
            self.assertEqual([item["id"] for item in one_shot], [three_shot[0]["id"]])
            self.assertEqual([item["selection"]["position"] for item in three_shot], ["A", "B", "C"])
            self.assertEqual([item["selection"]["source_split"] for item in three_shot], ["train"] * 3)

    def test_selection_is_deterministic(self) -> None:
        examples = dataset()
        first = select_wiki_prompt_examples(examples, seed=42, count=3)
        second = select_wiki_prompt_examples(examples, seed=42, count=3)
        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])

    def test_held_out_content_is_not_eligible(self) -> None:
        examples = dataset()
        duplicate = copy.deepcopy(examples["test"][0])
        duplicate["id"] = "train-999999"
        examples["train"].append(duplicate)
        eligible_ids = {item["id"] for item in eligible_wiki_train_examples(examples)}
        self.assertNotIn("train-999999", eligible_ids)

    def test_protocol_rejects_non_test_evaluation(self) -> None:
        examples = dataset()
        selected = select_wiki_prompt_examples(examples, seed=42, count=3)
        with self.assertRaisesRegex(ValueError, "must evaluate split 'test'"):
            validate_wiki_protocol(
                examples,
                selected,
                seed=42,
                evaluation_split="eval",
                expected_test_count=1,
            )

    def test_protocol_rejects_wrong_test_count(self) -> None:
        examples = dataset()
        selected = select_wiki_prompt_examples(examples, seed=42, count=3)
        with self.assertRaisesRegex(ValueError, "exactly 1000"):
            validate_wiki_protocol(examples, selected, seed=42)

    def test_protocol_accepts_train_only_deterministic_selection(self) -> None:
        examples = dataset()
        selected = select_wiki_prompt_examples(examples, seed=1771, count=3)
        result = validate_wiki_protocol(
            examples,
            selected,
            seed=1771,
            evaluation_split="test",
            expected_test_count=1,
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["prompt_source_split"], "train")


if __name__ == "__main__":
    unittest.main()
