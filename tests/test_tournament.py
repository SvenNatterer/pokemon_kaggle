from __future__ import annotations

import unittest
from unittest import mock

from src.tournament import build_evaluation_env


class TournamentPerspectiveTests(unittest.TestCase):
    def test_player_one_keeps_learner_and_opponent_decks(self):
        with mock.patch("src.tournament.PokemonTCGEnv", return_value="env") as env:
            result = build_evaluation_env([1], [2], "opponent.zip", learner_perspective=1)

        self.assertEqual(result, "env")
        env.assert_called_once_with(
            my_deck=[1],
            opponent_deck=[2],
            opponent_model_path="opponent.zip",
            learner_perspective=1,
        )


if __name__ == "__main__":
    unittest.main()
