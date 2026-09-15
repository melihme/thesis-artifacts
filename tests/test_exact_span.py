from __future__ import annotations

import unittest

from ner_framework.exact_span import (
    aggregate_span_counts,
    bio_to_spans,
    classify_span_errors,
    enrich_spans,
    metrics_from_counts,
    per_label_metrics,
    score_bio_sequences,
    score_exact_spans,
    spans_to_bio,
    validate_prediction_entities,
)


EXAMPLE = {
    "id": "example",
    "tokens": ["Microsoft", "ve", "Microsoft", "Türkiye"],
    "labels": ["B-ORG", "O", "B-ORG", "B-GPE"],
    "text": "Microsoft ve Microsoft Türkiye",
    "token_offsets": [[0, 9], [10, 12], [13, 22], [23, 30]],
}
ALLOWED = ["ORG", "GPE", "PERSON"]


class ExactSpanTests(unittest.TestCase):
    def gold(self):
        return enrich_spans(EXAMPLE, bio_to_spans(EXAMPLE["labels"]))

    def test_exact_match(self) -> None:
        raw = [{"start_token": 0, "end_token": 0, "text": "Microsoft", "label": "ORG"}]
        accepted, rejected = validate_prediction_entities(EXAMPLE, raw, ALLOWED)
        score = score_exact_spans(self.gold()[:1], accepted, len(rejected))
        self.assertEqual(score, {"tp": 1, "fp": 0, "fn": 0})

    def test_partial_text_receives_zero_credit(self) -> None:
        raw = [{"start_token": 0, "end_token": 0, "text": "soft", "label": "ORG"}]
        accepted, rejected = validate_prediction_entities(EXAMPLE, raw, ALLOWED)
        score = score_exact_spans(self.gold()[:1], accepted, len(rejected))
        self.assertEqual(score, {"tp": 0, "fp": 1, "fn": 1})
        self.assertEqual(metrics_from_counts(**score)["f1"], 0.0)

    def test_boundary_off_by_one_receives_zero_credit(self) -> None:
        raw = [{"start_token": 2, "end_token": 3, "text": "Microsoft Türkiye", "label": "ORG"}]
        accepted, rejected = validate_prediction_entities(EXAMPLE, raw, ALLOWED)
        score = score_exact_spans([self.gold()[1]], accepted, len(rejected))
        self.assertEqual(score, {"tp": 0, "fp": 1, "fn": 1})

    def test_wrong_label_receives_zero_credit(self) -> None:
        raw = [{"start_token": 0, "end_token": 0, "text": "Microsoft", "label": "PERSON"}]
        accepted, rejected = validate_prediction_entities(EXAMPLE, raw, ALLOWED)
        score = score_exact_spans(self.gold()[:1], accepted, len(rejected))
        self.assertEqual(score, {"tp": 0, "fp": 1, "fn": 1})

    def test_inconsistent_text_is_rejected(self) -> None:
        raw = [{"start_token": 2, "end_token": 2, "text": "microsoft", "label": "ORG"}]
        accepted, rejected = validate_prediction_entities(EXAMPLE, raw, ALLOWED)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["reason"], "text_mismatch")

    def test_invalid_indices_are_rejected(self) -> None:
        objects = [
            {"start_token": "0", "end_token": 0, "text": "Microsoft", "label": "ORG"},
            {"start_token": -1, "end_token": 0, "text": "Microsoft", "label": "ORG"},
            {"start_token": 0, "text": "Microsoft", "label": "ORG"},
        ]
        accepted, rejected = validate_prediction_entities(EXAMPLE, objects, ALLOWED)
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 3)

    def test_repeated_mentions_are_positionally_distinct(self) -> None:
        raw = [
            {"start_token": 0, "end_token": 0, "text": "Microsoft", "label": "ORG"},
            {"start_token": 2, "end_token": 2, "text": "Microsoft", "label": "ORG"},
        ]
        accepted, rejected = validate_prediction_entities(EXAMPLE, raw, ALLOWED)
        score = score_exact_spans(self.gold()[:2], accepted, len(rejected))
        self.assertEqual(score, {"tp": 2, "fp": 0, "fn": 0})

    def test_duplicate_prediction_is_false_positive(self) -> None:
        entity = {"start_token": 0, "end_token": 0, "text": "Microsoft", "label": "ORG"}
        accepted, rejected = validate_prediction_entities(EXAMPLE, [entity, dict(entity)], ALLOWED)
        score = score_exact_spans(self.gold()[:1], accepted, len(rejected))
        self.assertEqual(score, {"tp": 1, "fp": 1, "fn": 0})
        self.assertEqual(rejected[0]["reason"], "duplicate_prediction")

    def test_overlap_resolution_is_order_independent(self) -> None:
        candidates = [
            {"start_token": 2, "end_token": 3, "text": "Microsoft Türkiye", "label": "ORG"},
            {"start_token": 2, "end_token": 2, "text": "Microsoft", "label": "ORG"},
        ]
        first = validate_prediction_entities(EXAMPLE, candidates, ALLOWED)
        second = validate_prediction_entities(EXAMPLE, list(reversed(candidates)), ALLOWED)
        self.assertEqual(first[0], second[0])
        self.assertEqual([item["reason"] for item in first[1]], [item["reason"] for item in second[1]])

    def test_entity_order_does_not_change_score(self) -> None:
        candidates = [
            {"start_token": 3, "end_token": 3, "text": "Türkiye", "label": "GPE"},
            {"start_token": 0, "end_token": 0, "text": "Microsoft", "label": "ORG"},
        ]
        accepted_a, rejected_a = validate_prediction_entities(EXAMPLE, candidates, ALLOWED)
        accepted_b, rejected_b = validate_prediction_entities(EXAMPLE, list(reversed(candidates)), ALLOWED)
        self.assertEqual(
            score_exact_spans(self.gold(), accepted_a, len(rejected_a)),
            score_exact_spans(self.gold(), accepted_b, len(rejected_b)),
        )

    def test_bio_round_trip_and_invalid_i_handling(self) -> None:
        labels = ["I-ORG", "I-ORG", "O", "I-PERSON"]
        spans = bio_to_spans(labels)
        self.assertEqual(spans, [
            {"start_token": 0, "end_token": 1, "label": "ORG"},
            {"start_token": 3, "end_token": 3, "label": "PERSON"},
        ])
        self.assertEqual(spans_to_bio(spans, 4), ["B-ORG", "I-ORG", "O", "B-PERSON"])

    def test_bio_scoring_counts(self) -> None:
        scores = score_bio_sequences(
            [["B-ORG", "O", "B-GPE"]],
            [["B-ORG", "O", "B-ORG"]],
        )
        self.assertEqual((scores["tp"], scores["fp"], scores["fn"]), (1, 1, 1))
        self.assertEqual(aggregate_span_counts([scores])["f1"], 0.5)

    def test_per_label_metrics_keep_entity_types_separate(self) -> None:
        records = [
            {
                "gold_spans": self.gold(),
                "predicted_spans": self.gold()[:1],
                "rejected_predictions": [],
            }
        ]
        metrics = per_label_metrics(records, ["ORG", "GPE"])
        self.assertEqual((metrics["ORG"]["tp"], metrics["ORG"]["fn"]), (1, 1))
        self.assertEqual((metrics["GPE"]["tp"], metrics["GPE"]["fn"]), (0, 1))

    def test_error_taxonomy_distinguishes_boundary_and_label(self) -> None:
        gold = [
            {"start_token": 0, "end_token": 0, "label": "ORG"},
            {"start_token": 2, "end_token": 2, "label": "ORG"},
        ]
        predicted = [
            {"start_token": 0, "end_token": 0, "label": "PERSON"},
            {"start_token": 2, "end_token": 3, "label": "ORG"},
        ]
        taxonomy = classify_span_errors(gold, predicted, rejected_prediction_count=1)
        self.assertEqual(taxonomy["label_error"], 1)
        self.assertEqual(taxonomy["boundary_error"], 1)
        self.assertEqual(taxonomy["rejected_prediction"], 1)

    def test_shared_scorer_agrees_with_seqeval(self) -> None:
        try:
            from seqeval.metrics import f1_score, precision_score, recall_score
        except ImportError:
            self.skipTest("seqeval is installed by the Colab app requirements")
        gold = [
            ["B-ORG", "I-ORG", "O", "B-PERSON"],
            ["O", "B-GPE", "O"],
        ]
        predicted = [
            ["B-ORG", "I-ORG", "O", "B-PERSON"],
            ["O", "B-ORG", "O"],
        ]
        shared = score_bio_sequences(gold, predicted)
        self.assertAlmostEqual(shared["precision"], precision_score(gold, predicted))
        self.assertAlmostEqual(shared["recall"], recall_score(gold, predicted))
        self.assertAlmostEqual(shared["f1"], f1_score(gold, predicted))


if __name__ == "__main__":
    unittest.main()
