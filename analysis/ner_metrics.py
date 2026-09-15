from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


Entity = Tuple[str, int, int]


def entities_from_bio(labels: Sequence[str]) -> List[Entity]:
    entities: List[Entity] = []
    current_label = None
    current_start = None
    current_end = None

    def flush() -> None:
        nonlocal current_label, current_start, current_end
        if current_label is not None and current_start is not None and current_end is not None:
            entities.append((current_label, current_start, current_end))
        current_label = None
        current_start = None
        current_end = None

    for index, label in enumerate(labels):
        if label == "O" or "-" not in label:
            flush()
            continue

        prefix, entity_type = label.split("-", 1)
        prefix = prefix.upper()
        if prefix not in {"B", "I"}:
            flush()
            continue

        if prefix == "B" or current_label != entity_type or current_end is None:
            flush()
            current_label = entity_type
            current_start = index
            current_end = index
            continue

        current_end = index

    flush()
    return entities


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def safe_f1(tp: int, fp: int, fn: int) -> float:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_entity_sequences(
    gold_sequences: Iterable[Sequence[str]],
    predicted_sequences: Iterable[Sequence[str]],
) -> Dict[str, object]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_label_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gold": 0, "predicted": 0})
    per_example = []

    for example_index, (gold_labels, predicted_labels) in enumerate(zip(gold_sequences, predicted_sequences)):
        gold_entities = set(entities_from_bio(gold_labels))
        predicted_entities = set(entities_from_bio(predicted_labels))

        true_positive_entities = gold_entities & predicted_entities
        false_positive_entities = predicted_entities - gold_entities
        false_negative_entities = gold_entities - predicted_entities

        tp = len(true_positive_entities)
        fp = len(false_positive_entities)
        fn = len(false_negative_entities)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        for label, _, _ in gold_entities:
            per_label_counts[label]["gold"] += 1
        for label, _, _ in predicted_entities:
            per_label_counts[label]["predicted"] += 1
        for label, _, _ in true_positive_entities:
            per_label_counts[label]["tp"] += 1
        for label, _, _ in false_positive_entities:
            per_label_counts[label]["fp"] += 1
        for label, _, _ in false_negative_entities:
            per_label_counts[label]["fn"] += 1

        per_example.append(
            {
                "example_index": example_index,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "gold": len(gold_entities),
                "predicted": len(predicted_entities),
            }
        )

    per_label = {}
    for label, counts in sorted(per_label_counts.items()):
        label_tp = counts["tp"]
        label_fp = counts["fp"]
        label_fn = counts["fn"]
        per_label[label] = {
            **counts,
            "precision": safe_divide(label_tp, label_tp + label_fp),
            "recall": safe_divide(label_tp, label_tp + label_fn),
            "f1": safe_f1(label_tp, label_fp, label_fn),
        }

    return {
        "precision": safe_divide(total_tp, total_tp + total_fp),
        "recall": safe_divide(total_tp, total_tp + total_fn),
        "f1": safe_f1(total_tp, total_fp, total_fn),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "gold_entities": total_tp + total_fn,
        "predicted_entities": total_tp + total_fp,
        "per_label": per_label,
        "per_example": per_example,
    }
