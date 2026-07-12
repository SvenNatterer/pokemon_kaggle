from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
import zipfile

from src.arena_core import (
    ArenaStore,
    Participant,
    discover_participants,
    rank_participants,
    select_matchup,
    wilson_lower_bound,
)
from src.arena_match import load_holdout_results


def participant(bot_id: str, status: str = "loadable") -> Participant:
    return Participant(bot_id, bot_id.upper(), "rule_based", f"{bot_id}.csv", load_status=status)


class RankingTests(unittest.TestCase):
    def test_wilson_draw_is_half_success(self):
        self.assertAlmostEqual(wilson_lower_bound(0, 0, 2), wilson_lower_bound(1, 1, 0))
        self.assertGreater(wilson_lower_bound(10, 0, 0), wilson_lower_bound(1, 0, 0))

    def test_identical_elos_normalize_to_half(self):
        rows = rank_participants([participant("a"), participant("b")], [])
        self.assertEqual({row["normalized_elo"] for row in rows}, {0.5})

    def test_missing_holdout_is_visible_and_conservative(self):
        rows = rank_participants([participant("a"), participant("b")], [], {
            "a": {"games": 100, "score_rate": 0.9, "wilson95_score_lb": 0.8}
        })
        values = {row["bot_id"]: row for row in rows}
        self.assertFalse(values["a"]["holdout_missing"])
        self.assertTrue(values["b"]["holdout_missing"])
        self.assertEqual(values["b"]["ranking_components"]["holdout"], 0.35)
        self.assertGreater(values["a"]["ranking_score"], values["b"]["ranking_score"])

    def test_ranking_components_match_documented_formula(self):
        row = rank_participants([participant("a")], [])[0]
        components = row["ranking_components"]
        expected = 0.35 * components["arena_wilson"] + 0.25 * components["normalized_elo"]
        expected += 0.15 * components["arena_winrate"] + 0.25 * components["holdout"]
        self.assertAlmostEqual(row["ranking_score"], expected)


class MatchmakingTests(unittest.TestCase):
    def test_no_self_match_and_unloadable_bot_skipped(self):
        roster = [participant("a"), participant("b"), participant("broken", "unloadable")]
        first, second, _ = select_matchup(roster, [], random.Random(3))
        self.assertNotEqual(first.bot_id, second.bot_id)
        self.assertNotIn("broken", {first.bot_id, second.bot_id})

    def test_underrepresented_bot_is_selected_first(self):
        roster = [participant("a"), participant("b"), participant("c")]
        matches = [{"bot_a": "a", "bot_b": "b", "wins_a": 10, "wins_b": 0, "draws": 0,
                    "elo_a_after": 1210, "elo_b_after": 1190, "error_status": ""}]
        first, _, _ = select_matchup(roster, matches, random.Random(1))
        self.assertEqual(first.bot_id, "c")

    def test_pair_perspective_rotates(self):
        roster = [participant("a"), participant("b")]
        _, _, first_start = select_matchup(roster, [], random.Random(1))
        _, _, second_start = select_matchup(roster, [{"bot_a": "a", "bot_b": "b"}], random.Random(1))
        self.assertEqual((first_start, second_start), (0, 1))


class ParticipantDiscoveryTests(unittest.TestCase):
    def test_current_backup_legacy_and_rule_bot_are_discovered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "models" / "backup").mkdir(parents=True)
            (root / "models" / "curriculum_snapshots").mkdir(parents=True)
            (root / "decks").mkdir()
            (root / "decks" / "deck_1.csv").write_text("1\n", encoding="utf-8")
            (root / "decks" / "arena_agents.json").write_text(json.dumps({"agents": [{
                "id": "rules", "name": "Rules", "agent_type": "rule_based",
                "deck_path": "decks/deck_1.csv", "enabled": True,
            }]}), encoding="utf-8")
            for path in (root / "models" / "ppo_deck_1.zip", root / "models" / "backup" / "ppo_v4_deck_1_checkpoint_1.zip"):
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("data", "{}")
            with mock.patch("src.arena_core.ROOT", root), mock.patch(
                "src.arena_core.PARTICIPANT_MANIFEST", root / "decks" / "arena_agents.json"
            ):
                values = discover_participants()
        ids = {value.bot_id for value in values}
        self.assertIn("rules", ids)
        self.assertIn("ppo_deck_1", ids)
        legacy = next(value for value in values if "checkpoint" in value.bot_id)
        self.assertIn("historical_checkpoint", legacy.tags)
        self.assertIn("backup", legacy.tags)

    def test_missing_deck_is_diagnostic_not_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "models").mkdir()
            (root / "decks").mkdir()
            (root / "decks" / "arena_agents.json").write_text('{"agents": []}', encoding="utf-8")
            with zipfile.ZipFile(root / "models" / "ppo_deck_99.zip", "w") as archive:
                archive.writestr("data", "{}")
            with mock.patch("src.arena_core.ROOT", root), mock.patch(
                "src.arena_core.PARTICIPANT_MANIFEST", root / "decks" / "arena_agents.json"
            ):
                value = discover_participants()[0]
        self.assertEqual(value.load_status, "unloadable")
        self.assertIn("deck not found", value.load_error)


class PersistenceTests(unittest.TestCase):
    def test_holdout_history_preserves_latest_result_for_each_bot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "decks").mkdir()
            (root / "arena_data").mkdir()
            (root / "decks" / "submission_results.json").write_text(json.dumps({
                "summary": [{"candidate": "bot_a", "score_rate": 0.1}],
            }), encoding="utf-8")
            (root / "arena_data" / "evaluations.json").write_text(json.dumps([
                {"state": "completed", "results": {"summary": [
                    {"candidate": "bot_a", "score_rate": 0.6},
                ]}},
                {"state": "completed", "results": {"summary": [
                    {"candidate": "bot_b", "score_rate": 0.7},
                ]}},
                {"state": "completed", "results": {"summary": [
                    {"candidate": "bot_a", "score_rate": 0.8},
                ]}},
            ]), encoding="utf-8")
            with mock.patch("src.arena_match.ROOT", root):
                results = load_holdout_results()

        self.assertEqual(results["bot_a"]["score_rate"], 0.8)
        self.assertEqual(results["bot_b"]["score_rate"], 0.7)

    def test_reset_only_resets_arena_store(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.zip"
            model.write_bytes(b"model")
            store = ArenaStore(root / "arena")
            store.append_match({"match_id": "one"})
            store.reset()
            self.assertEqual(store.matches(), [])
            self.assertEqual(model.read_bytes(), b"model")


if __name__ == "__main__":
    unittest.main()
