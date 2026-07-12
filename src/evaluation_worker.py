"""Thin state-reporting wrapper around scripts/evaluate_submission.py."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import uuid

from src.arena_core import EVALUATION_FILE, atomic_write_json, read_json, utc_now


ROOT = Path(__file__).resolve().parents[1]


def create_result_file(bot_id: str) -> Path:
    """Return a unique, stable destination for one evaluation run."""
    safe_bot_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", bot_id).strip("._") or "bot"
    run_id = f"{utc_now().replace(':', '').replace('+', '_')}_{uuid.uuid4().hex[:8]}"
    return ROOT / "arena_data" / "evaluation_results" / f"{safe_bot_id}_{run_id}.json"


def append_history(state: dict, results: dict) -> None:
    history_file = ROOT / "arena_data" / "evaluations.json"
    history = read_json(history_file, [])
    if not isinstance(history, list):
        history = []
    history.append({**state, "results": results})
    atomic_write_json(history_file, history)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot-id", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--holdout-file", default="decks/holdout_opponents.json")
    args = parser.parse_args()
    holdout = read_json(ROOT / args.holdout_file, {"opponents": []})
    opponent_labels = [entry.get("label", entry.get("deck_id")) for entry in holdout.get("opponents", [])]
    output_file = create_result_file(args.bot_id)
    relative_output_file = output_file.relative_to(ROOT).as_posix()
    progress_file = EVALUATION_FILE
    atomic_write_json(progress_file, {
        "state": "running", "bot_id": args.bot_id, "model_path": args.model,
        "opponents": opponent_labels, "games_per_opponent": args.games,
        "planned_games": args.games * len(opponent_labels) * len(args.model), "completed_games": 0,
        "wins": 0, "losses": 0, "draws": 0, "progress": 0.0,
        "started_at": utc_now(), "ended_at": None, "result_at": None,
        "error": "", "result_file": relative_output_file,
        "configuration": {"holdout_file": args.holdout_file, "games": args.games, "models": args.model},
    })
    command = [
        sys.executable, "scripts/evaluate_submission.py",
        "--games", str(args.games), "--progress-file", str(progress_file),
        "--results-file", relative_output_file,
        "--holdout-file", args.holdout_file,
        "--best-candidate-file", str(output_file.with_name(output_file.stem + "_selection.json").relative_to(ROOT)),
    ]
    for model in args.model:
        command.extend(["--candidate", model])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    state = read_json(progress_file, {})
    if result.returncode:
        state.update({"state": "error", "error": (result.stderr or result.stdout).strip(), "ended_at": utc_now()})
        atomic_write_json(progress_file, state)
        append_history(state, {})
        return result.returncode
    state.update({"state": "completed", "ended_at": utc_now(), "result_at": utc_now(), "progress": 1.0})
    state["selection_file"] = str(output_file.with_name(output_file.stem + "_selection.json").relative_to(ROOT))
    atomic_write_json(progress_file, state)
    append_history(state, read_json(output_file, {}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
