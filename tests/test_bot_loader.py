from __future__ import annotations

import unittest
from unittest import mock

from src.bot_loader import _load_legacy_structured_model
from src.legacy_policy import LegacyStructuredFeatureExtractor


class BotLoaderCompatibilityTests(unittest.TestCase):
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
