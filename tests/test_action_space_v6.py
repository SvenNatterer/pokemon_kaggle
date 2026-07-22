from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.agents.rule_based_agent import RuleBasedPokemonAgent
from src.env.env_wrapper import (
    LEGACY_ACTION_SPACE_SIZE,
    LEGACY_STOP_ACTION,
    V6_ACTION_SPACE_SIZE,
    V6_STOP_ACTION,
    PokemonTCGEnv,
    advance_selection,
)
from src.training.train import validate_policy_action_space


def _selection(min_count=1, max_count=2, option_count=2):
    return SimpleNamespace(
        select=SimpleNamespace(
            option=[object() for _ in range(option_count)],
            minCount=min_count,
            maxCount=max_count,
        )
    )


class ActionSpaceV6Tests(unittest.TestCase):
    def test_v6_environment_has_exactly_66_actions(self):
        env = PokemonTCGEnv([6] * 60, [5] * 60, action_space_size=V6_ACTION_SPACE_SIZE)

        self.assertEqual(66, env.action_space.n)
        self.assertEqual((66,), env.observation_space["action_mask"].shape)
        self.assertEqual(65, env.stop_action)
        self.assertEqual("v6", env.policy_version)

    def test_legacy_environment_remains_1000_actions_by_default(self):
        env = PokemonTCGEnv([6] * 60, [5] * 60, action_space_size=LEGACY_ACTION_SPACE_SIZE)

        self.assertEqual(LEGACY_ACTION_SPACE_SIZE, env.action_space.n)
        self.assertEqual(LEGACY_STOP_ACTION, env.stop_action)

    def test_v6_stop_commits_an_autoregressive_selection(self):
        obs = _selection(min_count=1, max_count=2)

        pending, committed, invalid = advance_selection(obs, 0, [], stop_action=V6_STOP_ACTION)
        self.assertEqual(([0], False, False), (pending, committed, invalid))

        pending, committed, invalid = advance_selection(
            obs, V6_STOP_ACTION, pending, stop_action=V6_STOP_ACTION
        )
        self.assertEqual(([0], True, False), (pending, committed, invalid))

    def test_pending_selection_mask_follows_actor_ownership_in_both_perspectives(self):
        selection = _selection(min_count=2, max_count=3, option_count=3)
        selection.current = None
        selection.logs = []

        for learner_perspective in (0, 1):
            with self.subTest(learner_perspective=learner_perspective):
                env = PokemonTCGEnv(
                    [6] * 60,
                    [5] * 60,
                    learner_perspective=learner_perspective,
                    action_space_size=V6_ACTION_SPACE_SIZE,
                )
                env.pending_selection = [0, 2]
                env.opponent_pending_selection = [1]

                with patch("src.env.env_wrapper.to_observation_class", return_value=selection):
                    learner_obs = env._get_obs_python(
                        perspective=learner_perspective,
                        force_structured=False,
                    )
                    opponent_obs = env._get_obs_python(
                        perspective=1 - learner_perspective,
                        force_structured=False,
                    )

                self.assertEqual(0, learner_obs["action_mask"][0])
                self.assertEqual(1, learner_obs["action_mask"][1])
                self.assertEqual(0, learner_obs["action_mask"][2])
                self.assertEqual(1, learner_obs["action_mask"][V6_STOP_ACTION])
                self.assertEqual(1, opponent_obs["action_mask"][0])
                self.assertEqual(0, opponent_obs["action_mask"][1])
                self.assertEqual(1, opponent_obs["action_mask"][2])
                self.assertEqual(0, opponent_obs["action_mask"][V6_STOP_ACTION])

    def test_native_and_python_paths_share_pending_selection_ownership(self):
        for learner_perspective in (0, 1):
            with self.subTest(learner_perspective=learner_perspective):
                env = PokemonTCGEnv(
                    [6] * 60,
                    [5] * 60,
                    learner_perspective=learner_perspective,
                )
                env.pending_selection = [3]
                env.opponent_pending_selection = [7]

                self.assertIs(
                    env.pending_selection,
                    env._pending_selection_for_perspective(learner_perspective),
                )
                self.assertIs(
                    env.opponent_pending_selection,
                    env._pending_selection_for_perspective(1 - learner_perspective),
                )

    def test_rule_bot_returns_dynamic_v6_stop_index(self):
        observation = {
            "vector": np.zeros(1500, dtype=np.float32),
            "action_mask": np.zeros(V6_ACTION_SPACE_SIZE, dtype=np.int8),
        }
        observation["vector"][1491] = 1
        observation["action_mask"][V6_STOP_ACTION] = 1

        action = RuleBasedPokemonAgent().choose_action(observation)

        self.assertEqual(V6_STOP_ACTION, action)

    def test_environment_rejects_ambiguous_action_space_sizes(self):
        with self.assertRaisesRegex(ValueError, "Unsupported action space size"):
            PokemonTCGEnv([6] * 60, [5] * 60, action_space_size=67)

    def test_v5_checkpoint_cannot_continue_as_v6(self):
        legacy_model = SimpleNamespace(action_space=SimpleNamespace(n=LEGACY_ACTION_SPACE_SIZE))

        with self.assertRaisesRegex(RuntimeError, "intentionally incompatible"):
            validate_policy_action_space(legacy_model, V6_ACTION_SPACE_SIZE, "v6")


if __name__ == "__main__":
    unittest.main()
