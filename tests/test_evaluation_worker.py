from pathlib import Path
import json
import sys
import tempfile
import threading
import unittest
from unittest import mock

from scripts.evaluate_submission import (
    DEFAULT_GAMES,
    aggregate,
    entry_from_spec,
    evaluate_pairs,
    pair_cache_signature,
    parse_result,
)
from src.arena.evaluation_worker import create_result_file


class EvaluationWorkerTests(unittest.TestCase):
    def test_evaluation_default_is_100_games_per_opponent(self):
        self.assertEqual(DEFAULT_GAMES, 100)

    def test_independent_matchups_run_in_parallel(self):
        candidates = [{"label": "candidate", "model_path": "candidate.zip", "deck_path": "candidate.csv"}]
        opponents = [
            {"label": "opponent-a", "model_path": "a.zip", "deck_path": "a.csv"},
            {"label": "opponent-b", "model_path": "b.zip", "deck_path": "b.csv"},
        ]
        rendezvous = threading.Barrier(2, timeout=2)
        thread_ids = set()

        def fake_evaluate_pair(candidate, opponent, games, timeout, worker_python):
            thread_ids.add(threading.get_ident())
            rendezvous.wait()
            return {"candidate": candidate["label"], "opponent": opponent["label"]}

        with mock.patch(
            "scripts.evaluate_submission.evaluate_pair",
            side_effect=fake_evaluate_pair,
        ):
            completed = list(
                evaluate_pairs(candidates, opponents, 100, 600, "python", workers=2)
            )

        self.assertEqual(len(completed), 2)
        self.assertEqual(len(thread_ids), 2)

    def test_rule_candidate_spec_keeps_profile_and_explicit_deck(self):
        with tempfile.TemporaryDirectory() as temp:
            deck = Path(temp) / "bank_10.csv"
            deck.write_text("card\n", encoding="utf-8")
            entry = entry_from_spec({
                "label": "rule_bot_balanced_bank10",
                "model_path": "rule_based:balanced",
                "deck_path": str(deck),
            })

        self.assertEqual(entry["label"], "rule_bot_balanced_bank10")
        self.assertEqual(entry["model_path"], "rule_based:balanced")
        self.assertEqual(entry["bot_type"], "rule_based")
        self.assertEqual(entry["deck_id"], "bank_10")

    def test_rule_candidate_can_build_persistent_cache_signature(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate_deck = root / "candidate.csv"
            opponent_deck = root / "opponent.csv"
            opponent_model = root / "opponent.zip"
            for path in (candidate_deck, opponent_deck, opponent_model):
                path.write_text(path.name, encoding="utf-8")
            signature = pair_cache_signature(
                {"label": "rule", "model_path": "rule_based:balanced", "deck_path": str(candidate_deck)},
                {"label": "ppo", "model_path": str(opponent_model), "deck_path": str(opponent_deck)},
                4,
            )

        self.assertEqual(signature["games"], 4)
        self.assertEqual(len(signature["candidate"]["model_sha256"]), 64)

    def test_successful_matchup_is_reused_from_persistent_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = {
                "label": "candidate",
                "model_path": str(root / "candidate.zip"),
                "deck_path": str(root / "candidate.csv"),
            }
            opponent = {
                "label": "opponent",
                "model_path": str(root / "opponent.zip"),
                "deck_path": str(root / "opponent.csv"),
            }
            for path, content in (
                (candidate["model_path"], b"candidate-model"),
                (candidate["deck_path"], b"candidate-deck"),
                (opponent["model_path"], b"opponent-model"),
                (opponent["deck_path"], b"opponent-deck"),
            ):
                Path(path).write_bytes(content)
            result = {
                "candidate": "candidate", "opponent": "opponent", "games": 4,
                "wins": 3, "losses": 1, "draws": 0, "score": 3.0,
                "score_rate": 0.75, "win_rate": 0.75, "crashed": False,
                "error": "", "seconds": 1.0, "details": {},
            }
            cache_dir = str(root / "cache")

            with mock.patch(
                "scripts.evaluate_submission.evaluate_pair", return_value=result
            ) as evaluate:
                first = list(evaluate_pairs(
                    [candidate], [opponent], 4, 600, "python", workers=1,
                    cache_dir=cache_dir,
                ))
            with mock.patch(
                "scripts.evaluate_submission.evaluate_pair",
                side_effect=AssertionError("cached matchup was replayed"),
            ) as evaluate_again:
                second = list(evaluate_pairs(
                    [candidate], [opponent], 4, 600, "python", workers=1,
                    cache_dir=cache_dir,
                ))

        evaluate.assert_called_once()
        evaluate_again.assert_not_called()
        self.assertFalse(first[0][2]["cached"])
        self.assertTrue(second[0][2]["cached"])
        self.assertEqual(second[0][2]["wins"], 3)

    def test_changed_model_invalidates_cached_matchup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = {
                "label": "candidate",
                "model_path": str(root / "candidate.zip"),
                "deck_path": str(root / "candidate.csv"),
            }
            opponent = {
                "label": "opponent",
                "model_path": str(root / "opponent.zip"),
                "deck_path": str(root / "opponent.csv"),
            }
            for path in (*candidate.values(), *opponent.values()):
                if path not in {"candidate", "opponent"}:
                    Path(path).write_text(path, encoding="utf-8")
            result = {
                "candidate": "candidate", "opponent": "opponent", "games": 2,
                "wins": 1, "losses": 1, "draws": 0, "score": 1.0,
                "score_rate": 0.5, "win_rate": 0.5, "crashed": False,
                "error": "", "seconds": 1.0, "details": {},
            }
            cache_dir = str(root / "cache")
            with mock.patch(
                "scripts.evaluate_submission.evaluate_pair", return_value=result
            ) as evaluate:
                list(evaluate_pairs(
                    [candidate], [opponent], 2, 600, "python", workers=1,
                    cache_dir=cache_dir,
                ))
                Path(candidate["model_path"]).write_bytes(b"changed-model-content")
                list(evaluate_pairs(
                    [candidate], [opponent], 2, 600, "python", workers=1,
                    cache_dir=cache_dir,
                ))

        self.assertEqual(evaluate.call_count, 2)

    def test_each_run_gets_a_unique_sanitized_result_file(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "src.arena.evaluation_worker.ROOT", Path(temp)
        ):
            first = create_result_file("bot / one")
            second = create_result_file("bot / one")

        self.assertNotEqual(first, second)
        self.assertEqual(first.parent.name, "evaluation_results")
        self.assertTrue(first.name.startswith("bot_one_"))
        self.assertEqual(first.suffix, ".json")

    def test_multi_candidate_result_filename_stays_below_filesystem_limit(self):
        bot_ids = ",".join(f"stage_snapshots__very_long_checkpoint_{index}_" + "x" * 60 for index in range(20))
        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "src.arena.evaluation_worker.ROOT", Path(temp)
        ):
            result_file = create_result_file(bot_ids)
            selection_file = result_file.with_name(result_file.stem + "_selection.json")

        self.assertTrue(result_file.name.startswith("batch_20_candidates_"))
        self.assertLess(len(selection_file.name.encode("utf-8")), 200)

    def test_evaluation_parser_keeps_game_diagnostics(self):
        wins, losses, draws, details = parse_result(
            'RESULT:2,1,0\nDETAIL:{"total_turns": 12, "perspective": {}}'
        )
        self.assertEqual((wins, losses, draws), (2, 1, 0))
        self.assertEqual(details["total_turns"], 12)

    def test_evaluation_parser_preserves_child_error(self):
        with self.assertRaisesRegex(RuntimeError, "missing model"):
            parse_result("CHILD ERROR: missing model")

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

    def test_health_gate_excludes_a_corrupted_candidate_from_selection_order(self):
        healthy = {
            "candidate": "healthy", "opponent": "opponent", "games": 4,
            "wins": 2, "losses": 2, "draws": 0, "score": 2.0,
            "score_rate": 0.5, "crashed": False, "details": {
                "health": {"engine_errors": 0, "invalid_learner_actions": 0, "option_overflows": 0},
            },
        }
        corrupted = {
            "candidate": "corrupted", "opponent": "opponent", "games": 4,
            "wins": 4, "losses": 0, "draws": 0, "score": 4.0,
            "score_rate": 1.0, "crashed": False, "details": {
                "health": {"engine_errors": 1, "invalid_learner_actions": 0, "option_overflows": 0},
            },
        }

        summaries = aggregate([corrupted, healthy])

        self.assertEqual(summaries[0]["candidate"], "healthy")
        self.assertTrue(summaries[0]["health_gate"]["passed"])
        self.assertFalse(summaries[1]["health_gate"]["passed"])
        self.assertIn("engine_errors=1", summaries[1]["health_gate"]["violations"])

    def test_promotion_rejects_a_selection_that_failed_the_health_gate(self):
        from scripts import promote_champion

        with tempfile.TemporaryDirectory() as temp:
            selection_path = Path(temp) / "selection.json"
            selection_path.write_text(json.dumps({
                "candidate": "corrupted",
                "summary": {
                    "wilson95_score_lb": 1.0,
                    "perspective_score_gap": 0.0,
                    "health_gate": {"passed": False, "violations": ["engine_errors=1"]},
                },
            }), encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = [
                    "promote_champion.py", "--selection", str(selection_path),
                    "--champion-file", str(Path(temp) / "champion.json"),
                ]
                with self.assertRaisesRegex(SystemExit, "health gate failed"):
                    promote_champion.main()
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
