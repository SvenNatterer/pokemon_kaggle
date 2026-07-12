"""The single queue-free arena worker."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import time

from src.arena_core import ARENA_DIR, ArenaStore, atomic_write_json
from src.arena_match import execute_match


LOCK_FILE = ARENA_DIR / "worker.lock"
PID_FILE = ARENA_DIR / "worker.pid"
TEMPORARY_ROSTER_ERROR = "at least two enabled, loadable participants are required"


def is_temporary_roster_error(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and str(exc) == TEMPORARY_ROSTER_ERROR


def run_worker(games: int = 4, timeout: int = 300, poll_seconds: float = 0.5) -> int:
    ARENA_DIR.mkdir(parents=True, exist_ok=True)
    store = ArenaStore()
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with LOCK_FILE.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 2
        atomic_write_json(PID_FILE, {"pid": os.getpid()})
        try:
            while not stopping:
                state = store.state().get("state", "stopped")
                if state == "stopped":
                    break
                if state != "running":
                    time.sleep(poll_seconds)
                    continue
                try:
                    execute_match(store, games=games, timeout=timeout)
                except Exception as exc:
                    # Cooldowns are temporary. Keep the requested running state so
                    # the worker automatically retries as soon as two bots return.
                    if is_temporary_roster_error(exc):
                        store.set_state("running", error=str(exc), current_match=None)
                    else:
                        store.set_state("error", error=str(exc), current_match=None)
                    time.sleep(poll_seconds)
        finally:
            try:
                PID_FILE.unlink()
            except FileNotFoundError:
                pass
            if store.state().get("state") == "running":
                store.set_state("stopped", current_match=None)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4, help="Games per arena batch (default: 4).")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per batch in seconds.")
    args = parser.parse_args()
    return run_worker(args.games, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
