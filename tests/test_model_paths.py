import unittest

from src.model_paths import parse_deck_model_path


class ModelPathTests(unittest.TestCase):
    def test_parse_v5b_stage_snapshot(self):
        parsed = parse_deck_model_path(
            "models/stage_snapshots/ppo_v5b_deck_bank_18_stage7_mixed_league.zip"
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["prefix"], "ppo_v5b_deck")
        self.assertEqual(parsed["deck_id"], "bank_18")
        self.assertEqual(parsed["variant"], "_stage7_mixed_league")


if __name__ == "__main__":
    unittest.main()
