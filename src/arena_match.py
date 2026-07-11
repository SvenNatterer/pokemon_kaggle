"""Execute one arena batch and persist its aggregate result."""

from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from src.arena_core import (
    ArenaStore,
    DEFAULT_ELO,
    Participant,
    discover_participants,
    new_match_record,
    rank_participants,
    select_matchup,
    mark_bot_failure,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_result(stdout: str) -> tuple[int, int, int, str, dict[str, Any]]:
    parsed = None
    details: dict[str, Any] = {}
    for line in stdout.splitlines():
        if line.startswith("RESULT:"):
            parts = line.split(":", 1)[1].split(",")
            parsed = (int(parts[0]), int(parts[1]), int(parts[2]), "")
        if line.startswith("DETAIL:"):
            details = json.loads(line.split(":", 1)[1])
        if line.startswith("CHILD ERROR:"):
            return 0, 0, 0, line.split(":", 1)[1].strip(), details
    if parsed:
        return (*parsed, details)
    return 0, 0, 0, "evaluation produced no RESULT line", details


def _current_elos(participants: list[Participant], matches: list[dict[str, Any]]) -> dict[str, float]:
    return {row["bot_id"]: float(row["elo"]) for row in rank_participants(participants, matches)}


def _k_factor(games: int) -> int:
    if games < 50:
        return 32
    if games < 150:
        return 24
    return 16


def execute_match(store: ArenaStore, games: int = 4, timeout: int = 300) -> dict[str, Any]:
    participants = discover_participants()
    matches = store.matches()
    first, second, start_player = select_matchup(participants, matches)
    record = new_match_record(first, second, start_player)
    elos = _current_elos(participants, matches)
    record["elo_a_before"] = elos.get(first.bot_id, DEFAULT_ELO)
    record["elo_b_before"] = elos.get(second.bot_id, DEFAULT_ELO)
    store.set_state("running", current_match={"bot_a": first.bot_id, "bot_b": second.bot_id, "started_at": record["timestamp"]})

    # For odd batches, swapping arguments rotates which bot receives the extra
    # player-0 game. Results are mapped back to the stable A/B record fields.
    ordered = (first, second) if start_player == 0 else (second, first)
    command = [
        sys.executable, "src/evaluate_single.py",
        ordered[0].model_path or "rule_based", ordered[0].deck_path,
        ordered[1].model_path or "rule_based", ordered[1].deck_path,
        str(games),
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        wins_first, wins_second, draws, error, details = parse_result(result.stdout)
        if result.returncode != 0:
            error = (result.stderr or result.stdout).strip() or error
    except subprocess.TimeoutExpired:
        wins_first = wins_second = draws = 0
        error = f"match timed out after {timeout} seconds"
        details = {}

    if start_player == 0:
        wins_a, wins_b = wins_first, wins_second
    else:
        wins_a, wins_b = wins_second, wins_first
    record.update({"wins_a": wins_a, "wins_b": wins_b, "draws": draws, "error_status": error})
    record["turns"] = details.get("total_turns")
    reason_counts = details.get("reason_counts") or {}
    record["result_reason"] = max(reason_counts, key=reason_counts.get) if reason_counts else ""
    if error:
        record["result"] = "error"
        record["result_reason"] = error
        mark_bot_failure((first.bot_id, second.bot_id), error)
    else:
        record["winner"] = first.bot_id if wins_a > wins_b else second.bot_id if wins_b > wins_a else None
        record["result"] = "bot_a" if wins_a > wins_b else "bot_b" if wins_b > wins_a else "draw"
        total = wins_a + wins_b + draws
        expected_a = 1.0 / (1.0 + 10 ** ((record["elo_b_before"] - record["elo_a_before"]) / 400.0))
        board = {row["bot_id"]: row for row in rank_participants(participants, matches)}
        score_a = wins_a + 0.5 * draws
        delta_a = _k_factor(board[first.bot_id]["matches"]) * (score_a - expected_a * total)
        # Symmetric batch update keeps total rating mass stable.
        record["elo_a_after"] = record["elo_a_before"] + delta_a
        record["elo_b_after"] = record["elo_b_before"] - delta_a
    store.append_match(record)
    board = rank_participants(participants, store.matches(), load_holdout_results())
    store.save_leaderboard(board)
    requested_state = store.state().get("state", "running")
    final_state = requested_state if requested_state in {"paused", "stopped"} else "running"
    store.set_state(final_state, current_match=None, last_match=record["match_id"], error=error)
    return record


def load_holdout_results() -> dict[str, dict[str, Any]]:
    from src.arena_core import read_json

    candidates = [ROOT / "decks" / "submission_results.json", ROOT / "decks" / "deck18_holdout_results.json"]
    rows: dict[str, dict[str, Any]] = {}
    for path in candidates:
        data = read_json(path, {})
        for row in data.get("summary", []):
            candidate = str(row.get("candidate", ""))
            if candidate:
                rows[candidate] = row
    return rows
