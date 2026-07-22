#!/usr/bin/env python3
"""Compare one frozen Compact checkpoint with and without its card-table cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.arena.arena_core import atomic_write_json, read_json, utc_now
from src.agents.bot_loader import load_bot
from src.env.env_wrapper import _fit_observation_to_model_space
from src.league.tournament import build_evaluation_env, model_action_space_size, read_deck


DEFAULT_MODEL = "models/architecture_ablation/ppo_v6_deck_bank_54_compact.zip"
DEFAULT_DECK = "decks/deck_bank/bank_54.csv"
DEFAULT_MANIFEST = "decks/validation_opponents.json"
DEFAULT_RESULTS = "logs/v6_architecture_ablation/compact_card_cache_ab.json"


def state_difference(left, right) -> float:
    if left is None or right is None:
        return 0.0 if left is None and right is None else float("inf")
    if isinstance(left, (tuple, list)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return float("inf")
        return max((state_difference(a, b) for a, b in zip(left, right)), default=0.0)
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return float("inf")
    return float(np.max(np.abs(left_array - right_array), initial=0.0))


def configure_models(model_path: str):
    baseline = load_bot(model_path)
    cached = load_bot(model_path)
    baseline_extractor = baseline.policy.features_extractor
    cached_extractor = cached.policy.features_extractor
    baseline_extractor.use_card_table = False
    cached_extractor.use_card_table = True
    baseline.policy.set_training_mode(False)
    cached.policy.set_training_mode(False)
    return baseline, cached


def compare_direction(
    baseline,
    cached,
    learner_deck,
    opponent,
    games: int,
    learner_perspective: int,
    seed_offset: int,
) -> dict:
    env = build_evaluation_env(
        learner_deck,
        read_deck(opponent["deck_path"]),
        opponent["model_path"],
        learner_perspective,
        model_action_space_size(baseline),
    )
    model_space = baseline.observation_space
    result = {
        "games": games,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "decisions": 0,
        "action_mismatches": 0,
        "max_state_difference": 0.0,
        "baseline_seconds": 0.0,
        "cached_seconds": 0.0,
    }
    try:
        for game_index in range(games):
            obs, _ = env.reset(seed=seed_offset + game_index)
            done = False
            baseline_state = None
            cached_state = None
            episode_start = np.ones((1,), dtype=bool)
            while not done:
                fitted = _fit_observation_to_model_space(obs, model_space)
                started = time.perf_counter()
                baseline_action, baseline_state = baseline.predict(
                    fitted,
                    state=baseline_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                result["baseline_seconds"] += time.perf_counter() - started
                started = time.perf_counter()
                cached_action, cached_state = cached.predict(
                    fitted,
                    state=cached_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                result["cached_seconds"] += time.perf_counter() - started
                result["decisions"] += 1
                if not np.array_equal(baseline_action, cached_action):
                    result["action_mismatches"] += 1
                result["max_state_difference"] = max(
                    result["max_state_difference"],
                    state_difference(baseline_state, cached_state),
                )
                episode_start = np.zeros((1,), dtype=bool)
                obs, _, terminated, truncated, info = env.step(baseline_action)
                done = terminated or truncated

            winner = info.get("winner", -1)
            if winner == learner_perspective:
                result["wins"] += 1
            elif winner == 1 - learner_perspective:
                result["losses"] += 1
            else:
                result["draws"] += 1
    finally:
        env.close()
    return result


def merge_results(parts: list[dict]) -> dict:
    totals = {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "decisions": 0,
        "action_mismatches": 0,
        "max_state_difference": 0.0,
        "baseline_seconds": 0.0,
        "cached_seconds": 0.0,
    }
    for part in parts:
        for key in totals:
            if key == "max_state_difference":
                totals[key] = max(totals[key], part[key])
            else:
                totals[key] += part[key]
    totals["baseline_ms_per_decision"] = (
        totals["baseline_seconds"] * 1000.0 / totals["decisions"]
    )
    totals["cached_ms_per_decision"] = (
        totals["cached_seconds"] * 1000.0 / totals["decisions"]
    )
    totals["inference_speedup"] = totals["baseline_seconds"] / totals["cached_seconds"]
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--deck", default=DEFAULT_DECK)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--games", type=int, default=100, help="Games per validation opponent.")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.games <= 0:
        raise ValueError("--games must be positive")
    manifest = read_json(args.manifest, {})
    opponents = manifest.get("opponents", [])
    if not opponents:
        raise RuntimeError("Validation manifest has no opponents")
    learner_deck = read_deck(args.deck)
    baseline, cached = configure_models(args.model)
    rows = []
    for opponent_index, opponent in enumerate(opponents):
        label = opponent.get("label") or Path(opponent["model_path"]).stem
        print(f"[{opponent_index + 1}/{len(opponents)}] {label}", flush=True)
        games_as_player_0 = (args.games + 1) // 2
        games_as_player_1 = args.games // 2
        parts = [
            compare_direction(
                baseline,
                cached,
                learner_deck,
                opponent,
                games_as_player_0,
                0,
                args.seed + opponent_index * 100_000,
            )
        ]
        if games_as_player_1:
            parts.append(
                compare_direction(
                    baseline,
                    cached,
                    learner_deck,
                    opponent,
                    games_as_player_1,
                    1,
                    args.seed + opponent_index * 100_000 + 50_000,
                )
            )
        row = {"opponent": label, **merge_results(parts)}
        rows.append(row)
        print(
            f"  actions={row['decisions']} mismatches={row['action_mismatches']} "
            f"state_diff={row['max_state_difference']:.3g} "
            f"speedup={row['inference_speedup']:.2f}x",
            flush=True,
        )

    summary = merge_results(rows)
    payload = {
        "created_at": utc_now(),
        "model": args.model,
        "manifest": args.manifest,
        "games_per_opponent": args.games,
        "summary": summary,
        "by_opponent": rows,
    }
    atomic_write_json(args.results, payload)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved results to {args.results}", flush=True)
    if summary["action_mismatches"] or summary["max_state_difference"] > 1e-5:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
