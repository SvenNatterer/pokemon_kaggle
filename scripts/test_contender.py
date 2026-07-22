#!/usr/bin/env python3
"""Evaluate a contender model/deck against both the Arena and Validation pools."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

# Ensure workspace root is in path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.arena.arena_core import (
    discover_participants,
    enabled_participants,
    read_json,
    atomic_write_json,
    wilson_lower_bound,
)
from src.agents.rule_based_agent import is_rule_based_model_spec


DEFAULT_WORKER_PYTHON = os.path.join("venv", "bin", "python")
DEFAULT_LEADERBOARD_FILE = "arena_data/leaderboard.json"
DEFAULT_VALIDATION_FILE = "decks/validation_opponents.json"


def _k_factor(games: int) -> int:
    """K-factor used in Pokémon Kaggle Arena Elo updates."""
    if games < 50:
        return 32
    if games < 150:
        return 24
    return 16


def parse_result(stdout: str) -> tuple[int, int, int, str, dict[str, Any]]:
    """Parse output from evaluate_single.py."""
    parsed = None
    details: dict[str, Any] = {}
    for line in stdout.splitlines():
        if line.startswith("RESULT:"):
            parts = line.split(":", 1)[1].split(",")
            parsed = (int(parts[0]), int(parts[1]), int(parts[2]), "")
        elif line.startswith("DETAIL:"):
            try:
                details = json.loads(line.split(":", 1)[1])
            except json.JSONDecodeError:
                details = {"parse_error": "invalid DETAIL payload"}
        elif line.startswith("CHILD ERROR:"):
            return 0, 0, 0, line.split(":", 1)[1].strip(), details
    if parsed:
        return (*parsed, details)
    return 0, 0, 0, "evaluation produced no RESULT line", details


def distribute_games(total_games: int, opponents: list[dict[str, Any]]) -> list[tuple[dict[str, Any], int]]:
    """Distribute total games as evenly as possible across all opponents in a pool."""
    n = len(opponents)
    if n == 0:
        return []
    base_games = total_games // n
    remainder = total_games % n
    
    distribution = []
    for i, opp in enumerate(opponents):
        games = base_games + (1 if i < remainder else 0)
        if games > 0:
            distribution.append((opp, games))
    return distribution


def evaluate_pair(
    candidate: dict[str, str],
    opponent: dict[str, str],
    games: int,
    timeout: int,
    worker_python: str,
) -> dict[str, Any]:
    """Execute evaluation for a single candidate/opponent pair via evaluate_single.py."""
    command = [
        worker_python,
        "src/arena/evaluate_single.py",
        candidate["model_path"],
        candidate["deck_path"],
        opponent["model_path"],
        opponent["deck_path"],
        str(games),
    ]
    child_environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        child_environment[variable] = "1"
    
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=child_environment,
        )
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        wins, losses, draws, error, details = parse_result(result.stdout)
        crashed = False
    except Exception as exc:
        elapsed = time.monotonic() - started
        wins, draws = 0, 0
        losses = games
        crashed = True
        error = str(exc)
        details = {}

    score = wins + 0.5 * draws
    return {
        "candidate": candidate["label"],
        "opponent": opponent["label"],
        "opponent_deck": opponent["deck_path"],
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": score,
        "score_rate": score / games if games else 0.0,
        "win_rate": wins / games if games else 0.0,
        "crashed": crashed,
        "error": error,
        "seconds": elapsed,
        "details": details,
    }


def run_pool_test(
    candidate: dict[str, str],
    opponents: list[dict[str, Any]],
    total_games: int,
    workers: int,
    timeout: int,
    worker_python: str,
    pool_name: str,
) -> list[dict[str, Any]]:
    """Run all matchups in the given opponent pool and return the results."""
    distribution = distribute_games(total_games, opponents)
    if not distribution:
        print(f"No active opponents in pool '{pool_name}'. skipping.")
        return []
    
    print(f"\nEvaluating {candidate['label']} in {pool_name}: playing {total_games} games distributed across {len(distribution)} opponents...")
    
    results = []
    total_pairs = len(distribution)
    pair_index = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_pair,
                candidate,
                {
                    "label": opp["label"],
                    "model_path": opp["model_path"],
                    "deck_path": opp["deck_path"],
                },
                games,
                timeout,
                worker_python,
            ): opp for opp, games in distribution
        }
        
        for future in as_completed(futures):
            pair_index += 1
            row = future.result()
            results.append(row)
            marker = "CRASH" if row["crashed"] else "OK"
            print(
                f"[{pair_index}/{total_pairs} completed] {pool_name} -> vs {row['opponent']}: "
                f"{marker}: {row['wins']}-{row['losses']}-{row['draws']} "
                f"score={row['score_rate'] * 100:.1f}% "
                f"time={row['seconds']:.1f}s"
            )
            if row["error"]:
                print(f"  Error: {row['error']}")
                
    return results


def calculate_metrics(results: list[dict[str, Any]], opponent_elos: dict[str, float] | None = None) -> dict[str, Any]:
    """Calculate aggregated stats, dynamic Elo, Wilson score, and worst matchup."""
    if not results:
        return {}
    
    total_games = sum(row["games"] for row in results)
    total_wins = sum(row["wins"] for row in results)
    total_losses = sum(row["losses"] for row in results)
    total_draws = sum(row["draws"] for row in results)
    total_score = total_wins + 0.5 * total_draws
    
    win_rate = total_wins / total_games if total_games else 0.0
    score_rate = total_score / total_games if total_games else 0.0
    wilson_lb = wilson_lower_bound(total_wins, total_losses, total_draws)
    
    # Worst matchup calculation (by lowest score_rate)
    worst_row = min(results, key=lambda row: row["score_rate"])
    worst_matchup = {
        "opponent": worst_row["opponent"],
        "deck_path": worst_row["opponent_deck"],
        "wins": worst_row["wins"],
        "losses": worst_row["losses"],
        "draws": worst_row["draws"],
        "score_rate": worst_row["score_rate"]
    }
    
    # Calculate Elo if opponent Elos are provided
    contender_elo = 1200.0
    elo_progression = []
    if opponent_elos:
        # Sort results alphabetically by opponent to ensure deterministic Elo order
        sorted_results = sorted(results, key=lambda row: row["opponent"])
        games_played = 0
        for row in sorted_results:
            opp_label = row["opponent"]
            opp_elo = opponent_elos.get(opp_label, 1200.0)
            G = row["games"]
            if G == 0:
                continue
            # Expected score against this opponent
            expected = 1.0 / (1.0 + 10 ** ((opp_elo - contender_elo) / 400.0))
            actual = row["wins"] + 0.5 * row["draws"]
            # Elo update
            k = _k_factor(games_played)
            delta = k * (actual - expected * G)
            contender_elo += delta
            games_played += G
            elo_progression.append({
                "opponent": opp_label,
                "opponent_elo": opp_elo,
                "expected_score": expected * G,
                "actual_score": actual,
                "delta": delta,
                "new_contender_elo": contender_elo
            })
            
    return {
        "games": total_games,
        "wins": total_wins,
        "losses": total_losses,
        "draws": total_draws,
        "score": total_score,
        "win_rate": win_rate,
        "score_rate": score_rate,
        "wilson_lb": wilson_lb,
        "worst_matchup": worst_matchup,
        "elo": contender_elo if opponent_elos else None,
        "elo_progression": elo_progression if opponent_elos else []
    }


def print_report(title: str, metrics: dict[str, Any], results: list[dict[str, Any]]) -> None:
    """Print a clean formatted text report to console."""
    if not metrics:
        return
    
    print(f"\n==========================================")
    print(f" REPORT: {title}")
    print(f"==========================================")
    print(f"Total Games:    {metrics['games']}")
    print(f"Record (W-L-D): {metrics['wins']}-{metrics['losses']}-{metrics['draws']}")
    print(f"Total Score:    {metrics['score']:.1f} / {metrics['games']}.0")
    print(f"Win Rate:       {metrics['win_rate'] * 100:.2f}%")
    print(f"Score Rate:     {metrics['score_rate'] * 100:.2f}%")
    print(f"Wilson 95% LB:  {metrics['wilson_lb'] * 100:.2f}%")
    if metrics['elo'] is not None:
        print(f"Contender ELO:  {metrics['elo']:.1f}")
        
    worst = metrics["worst_matchup"]
    print(f"Worst Matchup:  vs {worst['opponent']} ({worst['wins']}-{worst['losses']}-{worst['draws']}, score={worst['score_rate'] * 100:.1f}%)")
    print(f"------------------------------------------")
    print(f"{'Opponent':30s} | {'Record':8s} | {'Score Rate':>10s}")
    print(f"-" * 55)
    for row in sorted(results, key=lambda r: r["score_rate"]):
        record_str = f"{row['wins']}-{row['losses']}-{row['draws']}"
        print(f"{row['opponent']:30s} | {record_str:8s} | {row['score_rate'] * 100:9.1f}%")
    print(f"==========================================\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", "-m", required=True, help="Path to contender model .zip file or rule_based:profile spec.")
    parser.add_argument("--deck-path", "-d", required=True, help="Path to contender deck CSV.")
    parser.add_argument("--label", "-l", default="", help="Label for the candidate model (defaults to filename stem).")
    parser.add_argument("--arena-games", type=int, default=200, help="Total games to run in the arena pool (default: 200).")
    parser.add_argument("--validation-games", type=int, default=100, help="Total games to run in the validation pool (default: 100).")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="Concurrent workers count.")
    parser.add_argument("--results-dir", default="arena_data", help="Directory to save JSON results.")
    parser.add_argument("--no-arena", action="store_true", help="Skip the arena pool test.")
    parser.add_argument("--no-validation", action="store_true", help="Skip the validation pool test.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds per match batch.")
    args = parser.parse_args()

    # Validate inputs
    if not is_rule_based_model_spec(args.model_path) and not os.path.isfile(args.model_path):
        print(f"Error: Model path '{args.model_path}' not found.", file=sys.stderr)
        return 1
    if not os.path.isfile(args.deck_path):
        print(f"Error: Deck path '{args.deck_path}' not found.", file=sys.stderr)
        return 1
    
    # Label
    label = args.label
    if not label:
        if is_rule_based_model_spec(args.model_path):
            label = args.model_path.replace(":", "_")
        else:
            label = Path(args.model_path).stem
            
    candidate = {
        "label": label,
        "model_path": args.model_path,
        "deck_path": args.deck_path
    }
    
    # Python executable
    worker_python = DEFAULT_WORKER_PYTHON if os.path.exists(DEFAULT_WORKER_PYTHON) else sys.executable
    print(f"Starting evaluations for candidate: {label}")
    print(f"Model: {args.model_path}")
    print(f"Deck:  {args.deck_path}")
    print(f"Python worker: {worker_python}")
    print(f"Parallel workers: {args.workers}")
    
    arena_results = []
    arena_metrics = {}
    validation_results = []
    validation_metrics = {}
    
    # 1. Run Arena pool if requested
    if not args.no_arena:
        # Load arena opponents and Elos
        arena_participants = enabled_participants(discover_participants())
        # Convert to dictionary opponent format expected by evaluator
        arena_opponents = []
        for p in arena_participants:
            # Exclude the candidate itself if it is already in the arena pool
            if p.model_path == args.model_path and p.deck_path == args.deck_path:
                continue
            arena_opponents.append({
                "label": p.bot_id,
                "model_path": p.model_path or "rule_based",
                "deck_path": p.deck_path,
            })
            
        # Get Elos from leaderboard
        leaderboard = read_json(DEFAULT_LEADERBOARD_FILE, {"rows": []})
        opponent_elos = {}
        for row in leaderboard.get("rows", []):
            bot_id = row.get("bot_id")
            elo = row.get("elo")
            if bot_id and elo is not None:
                opponent_elos[bot_id] = float(elo)
                
        arena_results = run_pool_test(
            candidate,
            arena_opponents,
            args.arena_games,
            args.workers,
            args.timeout,
            worker_python,
            "Arena Pool",
        )
        arena_metrics = calculate_metrics(arena_results, opponent_elos)
        print_report("ARENA LEAGUE RESULTS", arena_metrics, arena_results)
        
    # 2. Run Validation pool if requested
    if not args.no_validation:
        val_manifest = read_json(DEFAULT_VALIDATION_FILE, {"opponents": []})
        val_opponents = []
        for entry in val_manifest.get("opponents", []):
            val_opponents.append({
                "label": entry["label"],
                "model_path": entry["model_path"],
                "deck_path": entry["deck_path"],
            })
            
        validation_results = run_pool_test(
            candidate,
            val_opponents,
            args.validation_games,
            args.workers,
            args.timeout,
            worker_python,
            "Validation Pool",
        )
        validation_metrics = calculate_metrics(validation_results, opponent_elos=None)
        print_report("VALIDATION POOL RESULTS", validation_metrics, validation_results)
        
    # 3. Save combined results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"contender_test_{label}_{timestamp}.json"
    output_path = Path(args.results_dir) / output_filename
    
    output_payload = {
        "candidate": candidate,
        "timestamp": timestamp,
        "arena": {
            "metrics": arena_metrics,
            "matchups": arena_results
        } if not args.no_arena else None,
        "validation": {
            "metrics": validation_metrics,
            "matchups": validation_results
        } if not args.no_validation else None
    }
    
    atomic_write_json(output_path, output_payload)
    print(f"Successfully saved all contender evaluation results to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
