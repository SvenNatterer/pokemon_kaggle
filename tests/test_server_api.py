from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import src.server as server
from src.arena_core import ArenaStore, Participant


class FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


class ServerApiTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        self.previous_evaluation_process = server.evaluation_process
        server.evaluation_process = None

    def tearDown(self):
        server.evaluation_process = self.previous_evaluation_process

    def test_refresh_returns_backend_state(self):
        with mock.patch.object(server, "discover_participants", return_value=[]), mock.patch.object(
            server.arena_controller, "status", return_value={"state": "paused", "worker_alive": True}
        ), mock.patch.object(server.arena_store, "matches", return_value=[]):
            response = self.client.get("/api/refresh")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["state"], "paused")

    def test_factory_reset_requires_exact_confirmation(self):
        response = self.client.post("/api/reset", json={"confirmation": "yes"})
        self.assertEqual(response.status_code, 400)

    def test_factory_reset_calls_scoped_controller_reset(self):
        with mock.patch.object(server.arena_controller, "reset", return_value=(True, "reset")) as reset, mock.patch.object(
            server.arena_controller, "status", return_value={"state": "stopped"}
        ):
            response = self.client.post("/api/reset", json={"confirmation": "RESET ARENA"})
        self.assertEqual(response.status_code, 200)
        reset.assert_called_once_with()

    def test_rule_bot_is_not_sent_to_ppo_holdout_runner(self):
        rule = Participant("rule", "Rule", "rule_based", "deck.csv", load_status="loadable")
        with mock.patch.object(server, "discover_participants", return_value=[rule]):
            response = self.client.post("/api/evaluation/start", json={"bot_id": "rule", "games": 1})
        self.assertEqual(response.status_code, 400)

    def test_evaluation_start_does_not_write_arena_matches(self):
        ppo = Participant("ppo", "PPO", "ppo", "decks/deck_1.csv", "models/ppo_deck_1.zip", load_status="loadable")
        with tempfile.TemporaryDirectory() as temp:
            store = ArenaStore(temp)
            store.append_match({"match_id": "existing"})
            before = store.matches()
            with mock.patch.object(server, "discover_participants", return_value=[ppo]), mock.patch.object(
                server.arena_controller, "status", return_value={"state": "paused"}
            ), mock.patch.object(server.subprocess, "Popen", return_value=FakeProcess()):
                response = self.client.post("/api/evaluation/start", json={"bot_id": "ppo", "games": 1})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(store.matches(), before)

    def test_duplicate_evaluation_is_rejected(self):
        server.evaluation_process = FakeProcess()
        response = self.client.post("/api/evaluation/start", json={"bot_id": "anything"})
        self.assertEqual(response.status_code, 409)

    def test_champion_promotion_requires_completed_dashboard_evaluation(self):
        with mock.patch.object(server, "read_json", return_value={"state": "idle"}):
            response = self.client.post("/api/champion/promote", json={})
        self.assertEqual(response.status_code, 400)

    def test_replay_generation_requires_a_selected_bot(self):
        with mock.patch.object(server, "schedule_replay") as schedule:
            response = self.client.post("/api/replays/generate", json={"bot_ids": []})
        self.assertEqual(response.status_code, 400)
        schedule.assert_not_called()

    def test_replay_generation_uses_selected_bot_against_top_other_bot(self):
        top = Participant("top", "Top", "ppo", "decks/deck_1.csv", "models/ppo_deck_1.zip", load_status="loadable")
        other = Participant("other", "Other", "ppo", "decks/deck_2.csv", "models/ppo_deck_2.zip", load_status="loadable")
        with mock.patch.object(server, "discover_participants", return_value=[other, top]), mock.patch.object(
            server.arena_store, "matches", return_value=[]
        ), mock.patch.object(server, "load_holdout_results", return_value={}), mock.patch.object(
            server, "schedule_replay", return_value="replays/arena/test.json"
        ) as schedule:
            response = self.client.post("/api/replays/generate", json={"bot_ids": ["other"]})
        self.assertEqual(response.status_code, 200)
        schedule.assert_called_once()
        self.assertEqual(schedule.call_args.args[0].bot_id, "other")
        self.assertEqual(schedule.call_args.args[1].bot_id, "top")

    def test_replay_generation_creates_one_replay_for_each_selected_bot(self):
        top = Participant("top", "Top", "ppo", "decks/deck_1.csv", "models/ppo_deck_1.zip", load_status="loadable")
        selected = Participant("selected", "Selected", "ppo", "decks/deck_2.csv", "models/ppo_deck_2.zip", load_status="loadable")
        with mock.patch.object(server, "discover_participants", return_value=[selected, top]), mock.patch.object(
            server.arena_store, "matches", return_value=[]
        ), mock.patch.object(server, "load_holdout_results", return_value={}), mock.patch.object(
            server, "schedule_replay", return_value="replays/arena/test.json"
        ) as schedule:
            response = self.client.post("/api/replays/generate", json={"bot_ids": ["top", "selected"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(schedule.call_count, 2)
        self.assertEqual({call.args[0].bot_id for call in schedule.call_args_list}, {"top", "selected"})


if __name__ == "__main__":
    unittest.main()
