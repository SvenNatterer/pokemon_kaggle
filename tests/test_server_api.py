from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock

import src.arena.server as server
from src.arena.arena_core import ArenaStore, Participant


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

    def test_evaluation_payload_includes_results_and_selected_winner(self):
        state = {
            "state": "completed",
            "result_file": "arena_data/evaluation_results/run.json",
            "selection_file": "arena_data/evaluation_results/run_selection.json",
        }
        with mock.patch.object(server, "read_json", side_effect=[
            state, {"summary": [{"candidate": "a"}]}, {"candidate": "a"}
        ]):
            payload = server.evaluation_payload()
        self.assertEqual(payload["results"], [{"candidate": "a"}])
        self.assertEqual(payload["selection"], {"candidate": "a"})

    def test_evaluation_payload_marks_missing_worker_as_error(self):
        state = {"state": "running", "pid": 99999999}
        with mock.patch.object(server, "read_json", return_value=state), mock.patch.object(
            server, "atomic_write_json"
        ) as write:
            payload = server.evaluation_payload()
        self.assertEqual(payload["state"], "error")
        self.assertIn("stopped", payload["error"])
        write.assert_called_once_with(server.EVALUATION_FILE, payload)

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

    def test_rule_bot_is_sent_to_evaluation_with_profile_and_deck(self):
        rule = Participant(
            "rule", "Rule", "rule_based", "decks/deck_1.csv", "rule_based:balanced",
            load_status="loadable",
        )
        with mock.patch.object(server, "discover_participants", return_value=[rule]), mock.patch.object(
            server.arena_controller, "status", return_value={"state": "paused"}
        ), mock.patch.object(server.subprocess, "Popen", return_value=FakeProcess()) as popen:
            response = self.client.post(
                "/api/evaluation/start", json={"bot_id": "rule", "games": 1, "mode": "validation"}
            )
        self.assertEqual(response.status_code, 200)
        command = popen.call_args.args[0]
        spec = json.loads(command[command.index("--candidate-spec") + 1])
        self.assertEqual(spec["model_path"], "rule_based:balanced")
        self.assertEqual(spec["deck_path"], "decks/deck_1.csv")
        self.assertEqual(spec["bot_type"], "rule_based")

    def test_arena_cooldown_does_not_block_validation(self):
        ppo = Participant(
            "ppo", "PPO", "ppo", "decks/deck_1.csv", "models/ppo_deck_1.zip",
            load_status="cooldown",
        )
        with mock.patch.object(server, "discover_participants", return_value=[ppo]), mock.patch.object(
            server.arena_controller, "status", return_value={"state": "paused"}
        ), mock.patch.object(server.subprocess, "Popen", return_value=FakeProcess()) as popen:
            response = self.client.post(
                "/api/evaluation/start",
                json={"bot_id": "ppo", "games": 1, "mode": "validation"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(popen.call_args.kwargs.get("start_new_session", False))

    def test_evaluation_defaults_to_100_games_and_parallel_workers(self):
        ppo = Participant(
            "ppo", "PPO", "ppo", "decks/deck_1.csv", "models/ppo_deck_1.zip",
            load_status="loadable",
        )
        with mock.patch.object(server, "discover_participants", return_value=[ppo]), mock.patch.object(
            server.arena_controller, "status", return_value={"state": "paused"}
        ), mock.patch.object(server.subprocess, "Popen", return_value=FakeProcess()) as popen:
            response = self.client.post(
                "/api/evaluation/start",
                json={"bot_id": "ppo", "mode": "validation"},
            )

        self.assertEqual(response.status_code, 200)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--games") + 1], "100")
        self.assertGreater(int(command[command.index("--workers") + 1]), 0)

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

    def test_champion_promotion_runs_script_as_module(self):
        evaluation = {"state": "completed", "selection_file": "arena_data/selection.json"}
        completed = mock.Mock(returncode=0, stdout="Promoted champion", stderr="")
        with mock.patch.object(server, "read_json", side_effect=[evaluation, {}, {}]), mock.patch.object(
            server.subprocess, "run", return_value=completed
        ) as run:
            response = self.client.post("/api/champion/promote", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(run.call_args.args[0][:3], [server.sys.executable, "-m", "scripts.promote_champion"])

    def test_rule_bot_evaluation_winner_cannot_be_promoted(self):
        evaluation = {"state": "completed", "selection_file": "arena_data/selection.json"}
        selection = {"candidate": "rule", "candidate_spec": {"bot_type": "rule_based"}}
        with mock.patch.object(server, "read_json", side_effect=[evaluation, selection]), mock.patch.object(
            server.subprocess, "run"
        ) as run:
            response = self.client.post("/api/champion/promote", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Rule-based", response.get_json()["message"])
        run.assert_not_called()

    def test_watched_bots_are_deduplicated_and_saved_atomically(self):
        with mock.patch.object(server, "atomic_write_json") as write:
            response = self.client.post("/api/watched", json={"watched": ["b", "a", "b", ""]})
        self.assertEqual(response.status_code, 200)
        write.assert_called_once_with(server.WATCHED_FILE, {"watched": ["a", "b"]})

    def test_replay_location_organizes_known_sources(self):
        self.assertEqual(
            server.replay_location("replays/kaggle/54499398/episode-1-replay.json"),
            ("kaggle", "Kaggle", "Submission 54499398"),
        )
        self.assertEqual(
            server.replay_location("replays/test/generated_20260715/example.json"),
            ("test", "Tests", "generated_20260715"),
        )
        self.assertEqual(
            server.replay_location("replays/arena/example.json"),
            ("arena", "Arena", "Arena"),
        )

    def test_replay_details_extracts_kaggle_names_and_result(self):
        metadata, snapshots, result, status = server.replay_details({
            "info": {"EpisodeId": 42, "TeamNames": ["Alpha", "Beta"]},
            "rewards": [1, -1],
            "statuses": ["DONE", "DONE"],
            "steps": [[], []],
        })
        self.assertEqual(metadata, {"p0_name": "Alpha", "p1_name": "Beta", "episode_id": 42})
        self.assertEqual(snapshots, 2)
        self.assertEqual(result, "P0 win")
        self.assertEqual(status, "DONE / DONE")

    def test_kaggle_submission_descriptions_reads_metadata(self):
        payload = {"submissions": {"54499398": {"description": "Compact V6"}}}
        with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(payload))):
            self.assertEqual(
                server.kaggle_submission_descriptions(),
                {"54499398": "Compact V6"},
            )

    def test_bot_names_are_saved_per_checkpoint(self):
        participant = Participant("checkpoint-a", "Old", "ppo", "deck.csv", "model.zip")
        with mock.patch.object(server, "discover_participants", return_value=[participant]), mock.patch.object(
            server, "read_json", return_value={}
        ), mock.patch.object(server, "atomic_write_json") as write:
            response = self.client.post("/api/bot-names", json={"bot_id": "checkpoint-a", "name": "My checkpoint"})
        self.assertEqual(response.status_code, 200)
        write.assert_called_once_with(server.BOT_NAMES_FILE, {"checkpoint-a": "My checkpoint"})

    def test_bot_name_rejects_unknown_checkpoint(self):
        with mock.patch.object(server, "discover_participants", return_value=[]):
            response = self.client.post("/api/bot-names", json={"bot_id": "missing", "name": "Name"})
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
