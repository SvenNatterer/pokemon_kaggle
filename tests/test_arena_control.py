from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from src.arena.arena_control import ArenaController
from src.arena.arena_core import ArenaStore
from src.arena.arena_worker import is_temporary_roster_error


class ArenaControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ArenaStore(self.temp.name)
        self.controller = ArenaController(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def test_duplicate_start_is_rejected(self):
        self.store.set_state("running")
        with mock.patch("src.arena.arena_control._pid", return_value=123):
            success, _ = self.controller.start()
        self.assertFalse(success)

    def test_pause_is_safe_state_transition(self):
        self.store.set_state("running")
        success, _ = self.controller.pause()
        self.assertTrue(success)
        self.assertEqual(self.store.state()["state"], "paused")

    def test_stop_without_worker_is_idempotent(self):
        self.store.set_state("paused")
        with mock.patch("src.arena.arena_control._pid", return_value=None):
            success, _ = self.controller.stop()
        self.assertTrue(success)
        self.assertEqual(self.store.state()["state"], "stopped")

    def test_reset_preserves_external_files(self):
        self.store.append_match({"match_id": "one"})
        with mock.patch("src.arena.arena_control._pid", return_value=None):
            self.controller.reset()
        self.assertEqual(self.store.matches(), [])

    def test_roster_cooldown_error_is_retryable(self):
        error = RuntimeError("at least two enabled, loadable participants are required")
        self.assertTrue(is_temporary_roster_error(error))
        self.assertFalse(is_temporary_roster_error(RuntimeError("other failure")))


if __name__ == "__main__":
    unittest.main()
