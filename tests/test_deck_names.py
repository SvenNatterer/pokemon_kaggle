import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import utils


class DeckNameTests(unittest.TestCase):
    def test_deck_name_supports_numbered_and_bank_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "decks").mkdir()
            (root / "decks" / "deck_names.json").write_text(
                json.dumps({"7": "Hydrapple ex", "bank_18": "Mega Lucario ex"}),
                encoding="utf-8",
            )
            with patch.object(utils, "ROOT", root):
                self.assertEqual(utils.deck_name_for_path("decks/deck_7.csv"), "Ogerpon")
                self.assertEqual(
                    utils.deck_display_name_for_path("decks/deck_7.csv"),
                    "Ogerpon",
                )
                self.assertEqual(
                    utils.model_display_name_for_path("models/ppo_v5_deck_18.zip", "decks/deck_bank/bank_18.csv"),
                    "V5 Mega Lucario ex",
                )
                self.assertEqual(
                    utils.deck_name_for_path("decks/deck_bank/bank_18.csv"),
                    "Mega Lucario ex",
                )
                self.assertEqual(utils.deck_id_for_path("decks/deck_bank/bank_18.csv"), "bank_18")


if __name__ == "__main__":
    unittest.main()
