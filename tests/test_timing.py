from __future__ import annotations

import unittest

from ner_framework.timing import latency_summary, percentile, summarise_milliseconds


class TimingTests(unittest.TestCase):
    def test_interpolated_percentiles(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertAlmostEqual(percentile([0, 100], 95), 95)

    def test_empty_summary_uses_none_not_zero(self) -> None:
        summary = summarise_milliseconds([])
        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["median_ms"])

    def test_nested_component_collection(self) -> None:
        records = [
            {"request": {"client_e2e_ms": 10.0}},
            {"request": {"client_e2e_ms": 20.0}},
            {"request": {}},
        ]
        summary = latency_summary(records, {"client": ("request", "client_e2e_ms")})
        self.assertEqual(summary["client"]["count"], 2)
        self.assertEqual(summary["client"]["median_ms"], 15.0)


if __name__ == "__main__":
    unittest.main()
