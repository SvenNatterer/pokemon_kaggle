from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src.bot_loader import _load_legacy_structured_model, _resolve_load_path
from src.legacy_policy import LegacyStructuredFeatureExtractor


class BotLoaderCompatibilityTests(unittest.TestCase):
    def test_explicit_zip_wins_over_extracted_sibling_directory(self):
        with TemporaryDirectory() as directory:
            model_zip = Path(directory) / "opponent.zip"
            model_zip.write_bytes(b"checkpoint")
            model_zip.with_suffix("").mkdir()

            self.assertEqual(_resolve_load_path(str(model_zip)), str(model_zip))

    def test_missing_zip_keeps_suffixless_loader_compatibility(self):
        self.assertEqual(_resolve_load_path("models/opponent.zip"), "models/opponent")

    def test_legacy_fallback_replaces_only_feature_extractor(self):
        loader = mock.Mock()
        loader.load.return_value = "model"
        saved = {"policy_kwargs": {"use_belief_actor": False}}
        with mock.patch(
            "stable_baselines3.common.save_util.load_from_zip_file",
            return_value=(saved, {}, {}),
        ):
            result = _load_legacy_structured_model(loader, "old-model")

        self.assertEqual(result, "model")
        custom = loader.load.call_args.kwargs["custom_objects"]["policy_kwargs"]
        self.assertIs(custom["features_extractor_class"], LegacyStructuredFeatureExtractor)
        self.assertFalse(custom["use_belief_actor"])


if __name__ == "__main__":
    unittest.main()
