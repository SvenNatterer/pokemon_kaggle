import unittest
import copy
from unittest.mock import patch

import numpy as np
import torch

from scripts.run_v6_architecture_ablation import (
    VARIANTS,
    evaluation_command,
    train_command,
    wait_for_base_b,
)
from src.custom_policy import PokemonTCGFeatureExtractor
from src.env_wrapper import V6_ACTION_SPACE_SIZE, PokemonTCGEnv


class V6ArchitectureAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PokemonTCGEnv(
            [6] * 60, [5] * 60, action_space_size=V6_ACTION_SPACE_SIZE
        )

    def test_compact_variants_reduce_feature_width(self):
        full = PokemonTCGFeatureExtractor(
            self.env.observation_space, feature_variant="full"
        )
        compact = PokemonTCGFeatureExtractor(
            self.env.observation_space, feature_variant="compact"
        )
        no_legacy = PokemonTCGFeatureExtractor(
            self.env.observation_space, feature_variant="compact_no_legacy"
        )

        self.assertEqual(20061, full.net[0].in_features)
        self.assertLess(compact.net[0].in_features, full.net[0].in_features // 4)
        self.assertEqual(256, compact.net[0].in_features - no_legacy.net[0].in_features)
        self.assertLess(
            sum(parameter.numel() for parameter in compact.parameters()),
            sum(parameter.numel() for parameter in full.parameters()) // 3,
        )

    def test_balanced_sits_between_full_and_compact(self):
        full = PokemonTCGFeatureExtractor(
            self.env.observation_space, feature_variant="full"
        )
        balanced = PokemonTCGFeatureExtractor(
            self.env.observation_space,
            feature_variant="balanced",
            use_card_table=True,
        )
        compact = PokemonTCGFeatureExtractor(
            self.env.observation_space, feature_variant="compact"
        )

        self.assertEqual(192, balanced.card_repr_dim)
        self.assertTrue(balanced.use_card_table)
        self.assertLess(compact.net[0].in_features, balanced.net[0].in_features)
        self.assertLess(balanced.net[0].in_features, full.net[0].in_features)
        self.assertLess(
            sum(parameter.numel() for parameter in compact.parameters()),
            sum(parameter.numel() for parameter in balanced.parameters()),
        )
        self.assertLess(
            sum(parameter.numel() for parameter in balanced.parameters()),
            sum(parameter.numel() for parameter in full.parameters()),
        )

    def test_all_variants_produce_finite_256_features(self):
        observation = self.env.observation_space.sample()
        observation["action_mask"][:] = 0
        observation["action_mask"][0] = 1
        tensors = {
            key: torch.as_tensor(np.asarray(value)).unsqueeze(0)
            for key, value in observation.items()
        }

        for variant in ("full", *VARIANTS):
            with self.subTest(variant=variant):
                extractor = PokemonTCGFeatureExtractor(
                    self.env.observation_space, feature_variant=variant
                )
                features = extractor(tensors)
                self.assertEqual((1, 256), tuple(features.shape))
                self.assertTrue(torch.isfinite(features).all())

    def test_card_table_preserves_outputs_and_gradients(self):
        observation = self.env.observation_space.sample()
        observation["action_mask"][:] = 0
        observation["action_mask"][:4] = 1
        tensors = {
            key: torch.as_tensor(np.asarray(value)).unsqueeze(0)
            for key, value in observation.items()
        }
        baseline = PokemonTCGFeatureExtractor(
            self.env.observation_space,
            feature_variant="compact",
            use_card_table=False,
        )
        card_table = copy.deepcopy(baseline)
        card_table.use_card_table = True

        baseline_features = baseline(tensors)
        table_features = card_table(tensors)
        torch.testing.assert_close(baseline_features, table_features, rtol=1e-5, atol=1e-6)

        baseline_features.square().mean().backward()
        table_features.square().mean().backward()
        table_parameters = dict(card_table.named_parameters())
        for name, parameter in baseline.named_parameters():
            with self.subTest(parameter=name):
                torch.testing.assert_close(
                    parameter.grad,
                    table_parameters[name].grad,
                    rtol=2e-4,
                    atol=2e-6,
                )

    def test_eval_card_table_is_reused_and_training_invalidates_it(self):
        observation = self.env.observation_space.sample()
        tensors = {
            key: torch.as_tensor(np.asarray(value)).unsqueeze(0)
            for key, value in observation.items()
        }
        extractor = PokemonTCGFeatureExtractor(
            self.env.observation_space,
            feature_variant="compact",
            use_card_table=True,
        )
        extractor.eval()

        with torch.no_grad():
            extractor(tensors)
            first_table = extractor._frozen_card_table
            extractor(tensors)

        self.assertIsNotNone(first_table)
        self.assertIs(first_table, extractor._frozen_card_table)
        extractor.train()
        self.assertIsNone(extractor._frozen_card_table)

    def test_training_arms_differ_only_by_feature_variant_and_output(self):
        commands = {variant: train_command(variant, 1234, "python") for variant in VARIANTS}
        for variant, command in commands.items():
            self.assertEqual(variant, command[command.index("--feature-variant") + 1])
            self.assertEqual("20260721", command[command.index("--seed") + 1])
            self.assertEqual("1234", command[command.index("--timesteps") + 1])
            self.assertEqual("v6", command[command.index("--policy-version") + 1])
        self.assertNotIn("--card-table", commands["compact"])
        self.assertNotIn("--card-table", commands["compact_no_legacy"])
        self.assertIn("--card-table", commands["balanced"])

    def test_evaluation_contains_exactly_four_candidates(self):
        command = evaluation_command(30, "python")

        self.assertEqual(4, command.count("--candidate"))

    def test_base_b_wait_requires_model_and_completion_marker(self):
        with patch(
            "scripts.run_v6_architecture_ablation.base_b_is_complete",
            side_effect=[False, True],
        ), patch("scripts.run_v6_architecture_ablation.time.sleep") as sleep:
            wait_for_base_b(1)

        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
