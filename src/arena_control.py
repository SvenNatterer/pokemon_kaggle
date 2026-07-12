"""Process-safe control facade for the arena worker."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from src.arena_core import ARENA_DIR, ArenaStore, read_json


ROOT = Path(__file__).resolve().parents[1]
PID_FILE = ARENA_DIR / "worker.pid"


def _pid() -> int | None:
    value = read_json(PID_FILE, {})
    try:
        pid = int(value.get("pid"))
        os.kill(pid, 0)
        return pid
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return None


class ArenaController:
    def __init__(self, store: ArenaStore | None = None):
        self.store = store or ArenaStore()

    def status(self) -> dict[str, Any]:
        status = self.store.state()
        status["worker_pid"] = _pid()
        status["worker_alive"] = status["worker_pid"] is not None
        return status

    def start(self) -> tuple[bool, str]:
        state = self.store.state().get("state")
        pid = _pid()
        if pid is not None:
            if state == "running":
                return False, "Arena is already running."
            self.store.set_state("running", error="")
            return True, "Arena resumed."
        self.store.set_state("running", error="")
        subprocess.Popen(
            [sys.executable, "-m", "src.arena_worker"], cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=(ARENA_DIR / "worker.log").open("a", encoding="utf-8"),
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and _pid() is None:
            time.sleep(0.05)
        if _pid() is None:
            self.store.set_state("error", error="Arena worker did not start.")
            return False, "Arena worker did not start."
        return True, "Arena started."

    def pause(self) -> tuple[bool, str]:
        if self.store.state().get("state") != "running":
            return False, "Arena is not running."
        self.store.set_state("paused")
        return True, "Arena will pause after the current match."

    def stop(self, timeout: float = 10.0) -> tuple[bool, str]:
        pid = _pid()
        self.store.set_state("stopped", current_match=None)
        if pid is None:
            return True, "Arena is stopped."
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and _pid() is not None:
            time.sleep(0.1)
        if _pid() is not None:
            return True, "Arena stop requested; the worker will exit after the current match is persisted."
        return True, "Arena stopped."

    def reset(self) -> tuple[bool, str]:
        self.stop()
        self.store.reset()
        return True, "Arena match history and ranking were reset; models, decks, and evaluations were preserved."
