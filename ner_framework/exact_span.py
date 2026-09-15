from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


SCORING_PROTOCOL = "exact_token_span_v1"
CanonicalKey = Tuple[int, int, str]


def _entity_type(label: object) -> str:
    value = str(label)
    if value == "O":
        return ""
    return value.split("-", 1)[1] if "-" in value else value


def bio_to_spans(labels: Sequence[object]) -> List[Dict[str, object]]:
    """Convert BIO labels to inclusive token spans using seqeval-compatible I handling."""
    spans: List[Dict[str, object]] = []
    active_start: int | None = None
    active_label: str | None = None

    def close(end_token: int) -> None:
        nonlocal active_start, active_label
        if active_start is not None and active_label is not None:
            spans.append(
                {
                    "start_token": active_start,
                    "end_token": end_token,
                    "label": active_label,
                }
            )
        active_start = None
        active_label = None

    for token_index, raw_label in enumerate(labels):
        label = str(raw_label)
        if label == "O" or not label:
            close(token_index - 1)
            continue

        if "-" in label:
            prefix, entity_label = label.split("-", 1)
        else:
            prefix, entity_label = "B", label

        if prefix == "I" and active_start is not None and active_label == entity_label:
            continue

        close(token_index - 1)
        active_start = token_index
        active_label = entity_label

    close(len(labels) - 1)
    return spans


def span_text(example: Mapping[str, object], start_token: int, end_token: int) -> str:
    text = str(example.get("text", ""))
    offsets = example.get("token_offsets")
    if isinstance(offsets, list) and len(offsets) > end_token:
        start_offset = offsets[start_token]
        end_offset = offsets[end_token]
        if (
            isinstance(start_offset, (list, tuple))
            and isinstance(end_offset, (list, tuple))
            and len(start_offset) >= 2
            and len(end_offset) >= 2
        ):
            return text[int(start_offset[0]) : int(end_offset[1])]

    tokens = list(example.get("tokens", []))
    return " ".join(str(token) for token in tokens[start_token : end_token + 1])


def enrich_spans(example: Mapping[str, object], spans: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for span in spans:
        start_token = int(span["start_token"])
        end_token = int(span["end_token"])
        enriched.append(
            {
                "start_token": start_token,
                "end_token": end_token,
                "text": span_text(example, start_token, end_token),
                "label": str(span["label"]),
            }
        )
    return enriched


def spans_to_bio(spans: Sequence[Mapping[str, object]], token_count: int) -> List[str]:
    labels = ["O"] * token_count
    for span in sorted(
        spans,
        key=lambda item: (int(item["start_token"]), int(item["end_token"]), str(item["label"])),
    ):
        start_token = int(span["start_token"])
        end_token = int(span["end_token"])
        entity_label = str(span["label"])
        if start_token < 0 or end_token >= token_count or start_token > end_token:
            raise ValueError(f"Invalid canonical span: {(start_token, end_token, entity_label)}")
        if any(labels[index] != "O" for index in range(start_token, end_token + 1)):
            raise ValueError(f"Overlapping canonical span: {(start_token, end_token, entity_label)}")
        labels[start_token] = f"B-{entity_label}"
        for token_index in range(start_token + 1, end_token + 1):
            labels[token_index] = f"I-{entity_label}"
    return labels


def canonical_key(span: Mapping[str, object]) -> CanonicalKey:
    return int(span["start_token"]), int(span["end_token"]), str(span["label"])


def validate_prediction_entities(
    example: Mapping[str, object],
    raw_entities: Sequence[object],
    allowed_entity_types: Sequence[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Validate without repairing. Rejected objects are retained and count as false positives."""
    tokens = list(example.get("tokens", []))
    allowed = set(allowed_entity_types)
    candidates: List[Tuple[Tuple[object, ...], int, object]] = []
    rejected: List[Dict[str, object]] = []

    for raw_index, entity in enumerate(raw_entities):
        if not isinstance(entity, dict):
            rejected.append({"raw_index": raw_index, "entity": entity, "reason": "not_an_object"})
            continue

        missing = [field for field in ("start_token", "end_token", "text", "label") if field not in entity]
        if missing:
            rejected.append(
                {
                    "raw_index": raw_index,
                    "entity": entity,
                    "reason": "missing_fields",
                    "details": missing,
                }
            )
            continue

        start_token = entity.get("start_token")
        end_token = entity.get("end_token")
        if (
            isinstance(start_token, bool)
            or isinstance(end_token, bool)
            or not isinstance(start_token, int)
            or not isinstance(end_token, int)
        ):
            rejected.append({"raw_index": raw_index, "entity": entity, "reason": "non_integer_indices"})
            continue

        label = entity.get("label")
        text = entity.get("text")
        if not isinstance(label, str) or label not in allowed:
            rejected.append({"raw_index": raw_index, "entity": entity, "reason": "invalid_label"})
            continue
        if not isinstance(text, str):
            rejected.append({"raw_index": raw_index, "entity": entity, "reason": "non_string_text"})
            continue
        if start_token < 0 or start_token > end_token or end_token >= len(tokens):
            rejected.append({"raw_index": raw_index, "entity": entity, "reason": "out_of_range_indices"})
            continue

        expected_text = span_text(example, start_token, end_token)
        if text != expected_text:
            rejected.append(
                {
                    "raw_index": raw_index,
                    "entity": entity,
                    "reason": "text_mismatch",
                    "expected_text": expected_text,
                }
            )
            continue

        sort_key = (start_token, end_token, label, text, raw_index)
        candidates.append((sort_key, raw_index, entity))

    accepted: List[Dict[str, object]] = []
    accepted_keys: set[CanonicalKey] = set()
    occupied_tokens: set[int] = set()
    for _, raw_index, entity in sorted(candidates, key=lambda item: item[0]):
        start_token = int(entity["start_token"])
        end_token = int(entity["end_token"])
        key = (start_token, end_token, str(entity["label"]))
        if key in accepted_keys:
            rejected.append({"raw_index": raw_index, "entity": entity, "reason": "duplicate_prediction"})
            continue
        token_range = set(range(start_token, end_token + 1))
        if token_range & occupied_tokens:
            rejected.append({"raw_index": raw_index, "entity": entity, "reason": "overlapping_prediction"})
            continue
        accepted.append(
            {
                "start_token": start_token,
                "end_token": end_token,
                "text": str(entity["text"]),
                "label": str(entity["label"]),
            }
        )
        accepted_keys.add(key)
        occupied_tokens.update(token_range)

    rejected.sort(key=lambda item: int(item["raw_index"]))
    return accepted, rejected


def score_exact_spans(
    gold_spans: Sequence[Mapping[str, object]],
    predicted_spans: Sequence[Mapping[str, object]],
    rejected_prediction_count: int = 0,
) -> Dict[str, int]:
    gold_counts = Counter(canonical_key(span) for span in gold_spans)
    predicted_counts = Counter(canonical_key(span) for span in predicted_spans)
    true_positives = sum((gold_counts & predicted_counts).values())
    false_positives = sum(predicted_counts.values()) - true_positives + int(rejected_prediction_count)
    false_negatives = sum(gold_counts.values()) - true_positives
    return {"tp": true_positives, "fp": false_positives, "fn": false_negatives}


def metrics_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate_span_counts(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    tp = sum(int(record.get("tp", 0)) for record in records)
    fp = sum(int(record.get("fp", 0)) for record in records)
    fn = sum(int(record.get("fn", 0)) for record in records)
    return {"tp": tp, "fp": fp, "fn": fn, **metrics_from_counts(tp, fp, fn)}


def per_label_metrics(
    records: Sequence[Mapping[str, object]],
    entity_types: Sequence[str],
) -> Dict[str, Dict[str, float | int]]:
    result: Dict[str, Dict[str, float | int]] = {}
    for entity_type in entity_types:
        tp = fp = fn = 0
        for record in records:
            gold = [
                span for span in record.get("gold_spans", [])
                if isinstance(span, Mapping) and str(span.get("label")) == entity_type
            ]
            predicted = [
                span for span in record.get("predicted_spans", [])
                if isinstance(span, Mapping) and str(span.get("label")) == entity_type
            ]
            rejected_count = 0
            for rejected in record.get("rejected_predictions", []):
                if not isinstance(rejected, Mapping):
                    continue
                raw_entity = rejected.get("entity")
                if isinstance(raw_entity, Mapping) and raw_entity.get("label") == entity_type:
                    rejected_count += 1
            counts = score_exact_spans(gold, predicted, rejected_prediction_count=rejected_count)
            tp += counts["tp"]
            fp += counts["fp"]
            fn += counts["fn"]
        result[entity_type] = {"tp": tp, "fp": fp, "fn": fn, **metrics_from_counts(tp, fp, fn)}
    return result


def classify_span_errors(
    gold_spans: Sequence[Mapping[str, object]],
    predicted_spans: Sequence[Mapping[str, object]],
    rejected_prediction_count: int = 0,
) -> Dict[str, int]:
    """Return a deterministic diagnostic taxonomy; this never changes TP/FP/FN scoring."""
    gold_remaining = [canonical_key(span) for span in gold_spans]
    predicted_remaining = [canonical_key(span) for span in predicted_spans]

    for key in sorted(set(gold_remaining) & set(predicted_remaining)):
        matches = min(gold_remaining.count(key), predicted_remaining.count(key))
        for _ in range(matches):
            gold_remaining.remove(key)
            predicted_remaining.remove(key)

    counts: Counter[str] = Counter()
    for predicted in sorted(predicted_remaining):
        predicted_start, predicted_end, predicted_label = predicted
        match_index: int | None = None
        category: str | None = None
        for index, gold in enumerate(gold_remaining):
            gold_start, gold_end, gold_label = gold
            if (predicted_start, predicted_end) == (gold_start, gold_end):
                match_index, category = index, "label_error"
                break
        if match_index is None:
            for index, gold in enumerate(gold_remaining):
                gold_start, gold_end, gold_label = gold
                overlaps = predicted_end >= gold_start and predicted_start <= gold_end
                if overlaps and predicted_label == gold_label:
                    match_index, category = index, "boundary_error"
                    break
        if match_index is None:
            for index, gold in enumerate(gold_remaining):
                gold_start, gold_end, _ = gold
                if predicted_end >= gold_start and predicted_start <= gold_end:
                    match_index, category = index, "boundary_and_label_error"
                    break
        if match_index is None:
            counts["spurious_prediction"] += 1
        else:
            counts[str(category)] += 1
            gold_remaining.pop(match_index)

    counts["missing_entity"] += len(gold_remaining)
    counts["rejected_prediction"] += int(rejected_prediction_count)
    return dict(counts)


def score_bio_sequences(
    gold_sequences: Sequence[Sequence[object]],
    predicted_sequences: Sequence[Sequence[object]],
) -> Dict[str, object]:
    if len(gold_sequences) != len(predicted_sequences):
        raise ValueError("Gold and predicted sequence counts differ.")

    records: List[Dict[str, int]] = []
    correct_tokens = 0
    token_count = 0
    for gold_labels, predicted_labels in zip(gold_sequences, predicted_sequences):
        if len(gold_labels) != len(predicted_labels):
            raise ValueError("Gold and predicted token counts differ.")
        records.append(score_exact_spans(bio_to_spans(gold_labels), bio_to_spans(predicted_labels)))
        correct_tokens += sum(str(gold) == str(predicted) for gold, predicted in zip(gold_labels, predicted_labels))
        token_count += len(gold_labels)

    result = aggregate_span_counts(records)
    result["accuracy"] = correct_tokens / token_count if token_count else 0.0
    result["token_count"] = token_count
    result["scoring_protocol"] = SCORING_PROTOCOL
    return result
