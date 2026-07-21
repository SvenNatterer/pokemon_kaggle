import argparse
from pathlib import Path
import unittest

from scripts.run_adaptive_stopping_ablation import (
    build_train_command,
    validate_extra_train_args,
)


class AdaptiveStoppingAblationTests(unittest.TestCase):
    def args(self):
        return argparse.Namespace(
            python="python",
            deck=Path("decks/deck_bank/bank_54.csv").resolve(),
            opp_pool=Path("decks/pool.json").resolve(),
            max_steps=1_000_000,
            seed=123,
            train_args=["--policy-version", "v6", "--n-epochs", "2"],
            kl_threshold=0.001,
            entropy_trend=0.002,
            min_steps=250_000,
            patience=8,
        )

    def test_only_adaptive_arm_gets_stopping_arguments(self):
        args = self.args()
        output = Path("models/ppo_v6_deck_bank_54_test.zip").resolve()
        fixed = build_train_command(args, output, adaptive=False)
        adaptive = build_train_command(args, output, adaptive=True)
        self.assertIn("--pfsp-lite", fixed)
        self.assertNotIn("--adaptive-stop", fixed)
        self.assertEqual(fixed, adaptive[: len(fixed)])
        self.assertIn("--adaptive-stop", adaptive)

    def test_runner_rejects_overrides_of_controlled_variables(self):
        for option in ("--seed", "--adaptive-stop", "--timesteps=12", "--opp-pool", "--no-pfsp-lite"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                validate_extra_train_args([option])


if __name__ == "__main__":
    unittest.main()
