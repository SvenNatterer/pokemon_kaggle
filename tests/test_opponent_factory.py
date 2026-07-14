import copy
import json
from pathlib import Path
import unittest

from scripts.check_holdout_safe import deck_id_from_path
from scripts.run_opponent_factory import (
    DEFAULT_CONFIG,
    base_command,
    base_evaluation_command,
    finetune_command,
    read_json,
    target_assignments,
    target_candidate_path,
    validate_config,
    validate_static_inputs,
)


class OpponentFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json(DEFAULT_CONFIG)

    def test_checked_in_config_and_sources_are_valid(self):
        validate_config(self.config)
        validate_static_inputs(self.config)

    def test_splits_use_unique_exact_decks(self):
        split_decks = {}
        for target in self.config["targets"]:
            split_decks.setdefault(target["split"], set()).add(target["deck_id"])
        self.assertEqual(set(), split_decks["training"] & split_decks["validation"])
        self.assertEqual(set(), split_decks["training"] & split_decks["holdout"])
        self.assertEqual(set(), split_decks["validation"] & split_decks["holdout"])
        self.assertGreaterEqual(len(split_decks["validation"]), 6)
        self.assertGreaterEqual(len(split_decks["holdout"]), 6)

    def test_four_base_families_have_distinct_seeds(self):
        bases = self.config["bases"]
        all_bases = bases + self.config["fallback_bases"]
        self.assertGreaterEqual(len(bases), 4)
        self.assertEqual(2, len(self.config["fallback_bases"]))
        self.assertEqual(len(all_bases), len({item["seed"] for item in all_bases}))
        self.assertEqual(len(all_bases), len({item["id"] for item in all_bases}))
        command = base_command(self.config, bases[0], "python")
        self.assertIn("--belief-actor", command)
        self.assertIn("--card-table", command)
        self.assertIn("--rotate-perspective", command)
        self.assertEqual("v6", command[command.index("--policy-version") + 1])

    def test_base_evaluation_precedes_target_selection(self):
        command = base_evaluation_command(self.config, "python")
        self.assertIn("scripts/evaluate_submission.py", command)
        self.assertEqual(len(self.config["bases"]), command.count("--candidate"))
        assignments = target_assignments(self.config, dry_run=True)
        selected_count = self.config["base_evaluation"]["selected_base_count"]
        self.assertEqual(selected_count, len({base["id"] for _, base in assignments}))

    def test_fallback_evaluation_adds_exactly_two_bases(self):
        all_bases = self.config["bases"] + self.config["fallback_bases"]
        command = base_evaluation_command(self.config, "python", all_bases)
        self.assertEqual(len(self.config["bases"]) + 2, command.count("--candidate"))

        target_decks = {item["deck_id"] for item in self.config["targets"]}
        fallback_decks = {
            Path(item["deck_path"]).stem for item in self.config["fallback_bases"]
        }
        self.assertEqual(set(), target_decks & fallback_decks)

    def test_finetune_is_a_v6_seeded_copy_continuation(self):
        target = self.config["targets"][0]
        base = self.config["bases"][0]
        command = finetune_command(self.config, target, base, "python")
        self.assertIn("--continue-existing", command)
        self.assertIn("--card-table", command)
        self.assertIn("--seed", command)
        self.assertEqual("v6", command[command.index("--policy-version") + 1])
        self.assertTrue(target_candidate_path(target, base["id"]).name.startswith("ppo_v6_deck_"))

    def test_holdout_checker_recognizes_v6_model_ids(self):
        self.assertEqual(
            "bank_33",
            deck_id_from_path("models/ppo_v6_deck_bank_33_base_a.zip", model=True),
        )

    def test_rejects_duplicate_target_deck(self):
        invalid = copy.deepcopy(self.config)
        invalid["targets"][1]["deck_id"] = invalid["targets"][0]["deck_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_config(invalid)

    def test_config_is_plain_json_for_external_tooling(self):
        with Path(DEFAULT_CONFIG).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(3, payload["version"])
        self.assertEqual("v6", payload["policy_version"])


if __name__ == "__main__":
    unittest.main()
