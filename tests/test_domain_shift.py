from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


class DomainShiftContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT_DIR / "configs" / "reproduction.json").read_text(encoding="utf-8"))

    def test_five_fixed_seeds(self) -> None:
        self.assertEqual(self.config["encoder_training"]["seeds"], [42, 43, 44, 45, 46])

    def test_four_primary_cells(self) -> None:
        self.assertEqual(
            self.config["domain_shift"]["cells"],
            ["wiki_to_wiki", "wiki_to_twitter", "twitter_to_twitter", "twitter_to_wiki"],
        )

    def test_expected_evaluation_count(self) -> None:
        self.assertEqual(self.config["domain_shift"]["expected_primary_runs"], 40)
        self.assertEqual(self.config["domain_shift"]["expected_mapping_ablation_runs"], 30)
        self.assertEqual(self.config["domain_shift"]["expected_total_runs"], 70)


if __name__ == "__main__":
    unittest.main()
