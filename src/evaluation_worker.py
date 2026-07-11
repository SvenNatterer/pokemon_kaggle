"""Thin state-reporting wrapper around scripts/evaluate_submission.py."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from src.arena_core import EVALUATION_FILE, atomic_write_json, read_json, utc_now


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--games", type=int, default=30)
    args = parser.parse_args()
    holdout = read_json(ROOT / "decks" / "holdout_opponents.json", {"opponents": []})
    opponent_labels = [entry.get("label", entry.get("deck_id")) for entry in holdout.get("opponents", [])]
    output_file = ROOT / "decks" / "submission_results.json"
    progress_file = EVALUATION_FILE
    atomic_write_json(progress_file, {
        "state": "running", "bot_id": args.bot_id, "model_path": args.model,
        "opponents": opponent_labels, "games_per_opponent": args.games,
        "planned_games": args.games * len(opponent_labels), "completed_games": 0,
        "wins": 0, "losses": 0, "draws": 0, "progress": 0.0,
        "started_at": utc_now(), "ended_at": None, "result_at": None,
        "error": "", "result_file": "decks/submission_results.json",
        "configuration": {"holdout_file": "decks/holdout_opponents.json", "games": args.games},
    })
    command = [
        sys.executable, "scripts/evaluate_submission.py", "--candidate", args.model,
        "--games", str(args.games), "--progress-file", str(progress_file),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    state = read_json(progress_file, {})
    if result.returncode:
        state.update({"state": "error", "error": (result.stderr or result.stdout).strip(), "ended_at": utc_now()})
        atomic_write_json(progress_file, state)
        return result.returncode
    state.update({"state": "completed", "ended_at": utc_now(), "result_at": utc_now(), "progress": 1.0})
    atomic_write_json(progress_file, state)
    history_file = ROOT / "arena_data" / "evaluations.json"
    history = read_json(history_file, [])
    history.append({**state, "results": read_json(output_file, {})})
    atomic_write_json(history_file, history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
