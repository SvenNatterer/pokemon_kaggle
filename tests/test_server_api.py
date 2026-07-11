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


if __name__ == "__main__":
    unittest.main()
