from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src.agents.bot_loader import _resolve_load_path


class BotLoaderCompatibilityTests(unittest.TestCase):
    def test_explicit_zip_wins_over_extracted_sibling_directory(self):
        with TemporaryDirectory() as directory:
            model_zip = Path(directory) / "opponent.zip"
            model_zip.write_bytes(b"checkpoint")
            model_zip.with_suffix("").mkdir()

            self.assertEqual(_resolve_load_path(str(model_zip)), str(model_zip))

    def test_missing_zip_keeps_suffixless_loader_compatibility(self):
        self.assertEqual(_resolve_load_path("models/opponent.zip"), "models/opponent")


if __name__ == "__main__":
    unittest.main()
