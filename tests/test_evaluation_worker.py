from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.evaluate_submission import aggregate, parse_result
from src.evaluation_worker import create_result_file


class EvaluationWorkerTests(unittest.TestCase):
    def test_each_run_gets_a_unique_sanitized_result_file(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "src.evaluation_worker.ROOT", Path(temp)
        ):
            first = create_result_file("bot / one")
            second = create_result_file("bot / one")

        self.assertNotEqual(first, second)
        self.assertEqual(first.parent.name, "evaluation_results")
        self.assertTrue(first.name.startswith("bot_one_"))
        self.assertEqual(first.suffix, ".json")

    def test_evaluation_parser_keeps_game_diagnostics(self):
        wins, losses, draws, details = parse_result(
            'RESULT:2,1,0\nDETAIL:{"total_turns": 12, "perspective": {}}'
        )
        self.assertEqual((wins, losses, draws), (2, 1, 0))
        self.assertEqual(details["total_turns"], 12)

    def test_summary_reports_perspective_gap_and_failure_reasons(self):
        rows = [{
            "candidate": "candidate", "opponent": "opponent", "games": 4,
            "wins": 3, "losses": 1, "draws": 0, "score": 3.0,
            "score_rate": 0.75, "crashed": False, "details": {
                "total_turns": 20,
                "perspective": {
                    "player_0": {"games": 2, "wins": 2, "losses": 0, "draws": 0},
                    "player_1": {"games": 2, "wins": 1, "losses": 1, "draws": 0},
                },
                "candidate_win_reasons": {"prize": 3},
                "opponent_win_reasons": {"deckout": 1},
            },
        }]
        summary = aggregate(rows)[0]
        self.assertEqual(summary["mean_turns"], 5.0)
        self.assertEqual(summary["perspective_score_gap"], 0.5)
        self.assertEqual(summary["opponent_win_reasons"], {"deckout": 1})


if __name__ == "__main__":
    unittest.main()
