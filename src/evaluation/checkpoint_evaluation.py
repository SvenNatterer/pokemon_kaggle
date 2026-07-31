"""Checkpoint evaluation against the versioned validation and holdout pools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.league.evaluation import evaluate_vs_opponent
from src.utils import atomic_write_json, read_json, wilson_lower_bound


def load_opponent_manifest(path: str | Path) -> list[dict[str, str]]:
    payload = read_json(path)
    entries = payload.get("opponents") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Opponent manifest must contain a non-empty list: {path}")
    normalized = []
    for entry in entries:
        deck = entry.get("deck_path", entry.get("deck"))
        model = entry.get("model_path", entry.get("model"))
        label = entry.get("label")
        if not all(isinstance(value, str) and value for value in (deck, model, label)):
            raise ValueError(f"Invalid opponent entry in {path}: {entry!r}")
        normalized.append({"label": label, "deck": deck, "model": model})
    return normalized


def evaluate_pool(
    candidate_model: str,
    candidate_deck: str,
    opponents: list[dict[str, str]],
    games_per_opponent: int,
) -> dict[str, Any]:
    if games_per_opponent <= 0:
        raise ValueError("games_per_opponent must be positive")
    results = []
    for opponent in opponents:
        result, details = evaluate_vs_opponent(
            candidate_model,
            candidate_deck,
            opponent["model"],
            opponent["deck"],
            games_per_opponent,
            return_details=True,
        )
        wins, losses, draws = (int(result[0]), int(result[1]), int(result[2]))
        score = (wins + 0.5 * draws) / games_per_opponent
        results.append({
            "label": opponent["label"],
            "games": games_per_opponent,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": score,
            "wilson_lower_bound_95": wilson_lower_bound(wins, losses, draws),
            "details": details,
        })

    total_games = sum(item["games"] for item in results)
    total_score = sum(item["wins"] + 0.5 * item["draws"] for item in results)
    worst_win_rate = min(results, key=lambda item: item["win_rate"])
    worst_wilson = min(results, key=lambda item: item["wilson_lower_bound_95"])
    return {
        "games_per_opponent": games_per_opponent,
        "macro_win_rate": sum(item["win_rate"] for item in results) / len(results),
        "micro_win_rate": total_score / total_games,
        "worst_win_rate": worst_win_rate["win_rate"],
        "worst_win_rate_opponent": worst_win_rate["label"],
        "worst_wilson_lower_bound_95": worst_wilson["wilson_lower_bound_95"],
        "worst_wilson_opponent": worst_wilson["label"],
        "opponents": results,
    }


def evaluate_checkpoint(
    *,
    candidate_model: str,
    candidate_deck: str,
    validation_manifest: str,
    holdout_manifest: str,
    games_per_opponent: int,
    output_path: str | Path,
    global_steps: int,
    training_pool: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "checkpoint_model": candidate_model,
        "candidate_deck": candidate_deck,
        "global_steps": int(global_steps),
        "training_pool": [
            {"label": entry["label"], "model": entry.get("model_path"), "weight": entry["weight"]}
            for entry in training_pool
        ],
        "validation": evaluate_pool(
            candidate_model, candidate_deck,
            load_opponent_manifest(validation_manifest), games_per_opponent,
        ),
        "holdout": evaluate_pool(
            candidate_model, candidate_deck,
            load_opponent_manifest(holdout_manifest), games_per_opponent,
        ),
    }
    atomic_write_json(output_path, report)
    return report
