import argparse
from pathlib import Path
import unittest

from scripts.run_training_pool_ablation import (
    COMPACT_V6_OPTIONS,
    PFSP_OPTIONS,
    build_train_command,
    validate_extra_train_args,
)


class TrainingPoolAblationTests(unittest.TestCase):
    def args(self):
        return argparse.Namespace(
            python="python",
            deck=Path("decks/deck_bank/bank_38.csv").resolve(),
            opp_pool=Path("experiments/pool.json").resolve(),
            steps=1_000_000,
            seed=20260718,
            train_args=["--n-epochs", "2"],
        )

    def test_only_pfsp_arm_gets_pfsp_arguments(self):
        args = self.args()
        output = Path("models/ppo_v6_deck_bank_38_test.zip").resolve()
        static = build_train_command(args, output, pfsp=False)
        pfsp = build_train_command(args, output, pfsp=True)
        self.assertIn("--no-pfsp-lite", static)
        self.assertNotIn("--no-pfsp-lite", pfsp)
        self.assertEqual(static[:-1], pfsp[: len(static) - 1])
        self.assertEqual(tuple(pfsp[len(static) - 1 :]), PFSP_OPTIONS)
        self.assertIn("1000000", static)
        self.assertEqual(
            tuple(static[static.index("--policy-version") : static.index("--policy-version") + len(COMPACT_V6_OPTIONS)]),
            COMPACT_V6_OPTIONS,
        )

    def test_runner_rejects_controlled_overrides(self):
        for option in (
            "--seed",
            "--pfsp-lite",
            "--timesteps=12",
            "--opp-pool",
            "--feature-variant",
            "--no-card-table",
        ):
            with self.subTest(option=option), self.assertRaises(ValueError):
                validate_extra_train_args([option])


if __name__ == "__main__":
    unittest.main()
