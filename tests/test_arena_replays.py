from __future__ import annotations

import unittest
import subprocess
import tempfile
from unittest import mock

from src.arena.arena_core import ArenaStore, Participant
from src.arena.arena_match import execute_match, should_schedule_replay


def participant(bot_id: str) -> Participant:
    return Participant(
        bot_id, bot_id, "ppo", f"decks/{bot_id}.csv", f"models/{bot_id}.zip",
        load_status="loadable",
    )


class ArenaReplayTests(unittest.TestCase):
    def setUp(self):
        self.first = participant("first")
        self.second = participant("second")

    def matches_for_first(self, count: int, *, errors: int = 0):
        matches = [
            {"bot_a": "first", "bot_b": "other", "error_status": ""}
            for _ in range(count)
        ]
        matches.extend(
            {"bot_a": "first", "bot_b": "other", "error_status": "failed"}
            for _ in range(errors)
        )
        return matches

    def test_no_watched_bot_never_schedules_replay(self):
        self.assertFalse(should_schedule_replay(self.first, self.second, self.matches_for_first(14), set()))

    def test_watched_bot_schedules_its_fifteenth_successful_match(self):
        self.assertTrue(should_schedule_replay(self.first, self.second, self.matches_for_first(14), {"first"}))

    def test_errors_do_not_count_towards_interval(self):
        self.assertFalse(should_schedule_replay(self.first, self.second, self.matches_for_first(13, errors=1), {"first"}))

    def test_unwatched_opponent_count_does_not_trigger_replay(self):
        self.assertFalse(should_schedule_replay(self.first, self.second, self.matches_for_first(14), {"second"}))

    def test_match_subprocess_uses_devnull_for_stdin(self):
        completed = mock.Mock(returncode=0, stdout='RESULT:1,0,0\nDETAIL:{"total_turns": 1}', stderr="")
        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "src.arena.arena_match.discover_participants", return_value=[self.first, self.second]
        ), mock.patch("src.arena.arena_match.subprocess.run", return_value=completed) as run:
            execute_match(ArenaStore(temp), games=1)
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main()
