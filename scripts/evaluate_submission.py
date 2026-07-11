#!/usr/bin/env python3
"""Evaluate active models against a frozen opponent holdout."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from typing import Any


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.model_paths import discover_deck_models, parse_deck_model_path


DEFAULT_HOLDOUT_FILE = "decks/holdout_opponents.json"
DEFAULT_RESULTS_FILE = "decks/submission_results.json"
DEFAULT_WORKER_PYTHON = os.path.join("venv", "bin", "python")


def deck_path_for_id(deck_id: str) -> str:
    if str(deck_id).startswith("bank_"):
        return os.path.join("decks", "deck_bank", f"{deck_id}.csv")
    return os.path.join("decks", f"deck_{deck_id}.csv")


def normalize_path(path: str) -> str:
    return os.path.relpath(path, ROOT) if os.path.isabs(path) else path


def entry_from_model(model: dict[str, Any]) -> dict[str, str] | None:
    deck_path = deck_path_for_id(model["deck_id"])
    if not os.path.exists(deck_path):
        return None
    return {
        "label": model["name"],
        "deck_id": str(model["deck_id"]),
        "model_path": normalize_path(model["path"]),
        "deck_path": deck_path,
    }


def entry_from_path(model_path: str) -> dict[str, str] | None:
    model_path = normalize_path(model_path)
    if not model_path.endswith(".zip"):
        model_path = f"{model_path}.zip"
    parsed = parse_deck_model_path(model_path)
    if parsed is None:
        raise ValueError(f"Cannot parse deck id from model path: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return entry_from_model(parsed)


def discover_entries(model_dir: str, include_variants: bool = True) -> list[dict[str, str]]:
    entries = []
    for model in discover_deck_models(model_dir=model_dir, include_variants=include_variants):
        entry = entry_from_model(model)
        if entry is not None:
            entries.append(entry)
    return entries


def unique_by_deck(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    by_deck = {}
    for entry in entries:
        by_deck.setdefault(entry["deck_id"], entry)
    return list(by_deck.values())


def build_holdout(opponent_pool: str, num_opponents: int, seed: int) -> dict[str, Any]:
    entries = unique_by_deck(discover_entries(opponent_pool, include_variants=True))
    if not entries:
        raise RuntimeError(f"No usable opponent models found in {opponent_pool}")

    rng = random.Random(seed)
    shuffled = entries[:]
    rng.shuffle(shuffled)
    opponents = sorted(shuffled[:num_opponents], key=lambda item: item["label"])
    return {
        "version": 1,
        "seed": seed,
        "opponent_pool": opponent_pool,
        "note": "Frozen holdout. Do not train directly against these exact opponents.",
        "opponents": opponents,
    }


def load_or_create_holdout(args: argparse.Namespace) -> dict[str, Any]:
    should_create = args.refresh or not os.path.exists(args.holdout_file)
    if should_create:
        holdout = build_holdout(args.opponent_pool, args.num_opponents, args.seed)
        os.makedirs(os.path.dirname(args.holdout_file), exist_ok=True)
        with open(args.holdout_file, "w", encoding="utf-8") as handle:
            json.dump(holdout, handle, indent=2)
        return holdout

    with open(args.holdout_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_candidates(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.candidate:
        candidates = []
        for path in args.candidate:
            entry = entry_from_path(path)
            if entry is not None:
                candidates.append(entry)
            
            base_path = path
            if base_path.endswith(".zip"):
                base_path = base_path[:-4]
            import glob
            for ckpt_path in glob.glob(f"{base_path}_checkpoint_*.zip"):
                try:
                    ckpt_entry = entry_from_path(ckpt_path)
                    if ckpt_entry is not None:
                        candidates.append(ckpt_entry)
                except Exception as e:
                    print(f"Warning: could not load checkpoint {ckpt_path}: {e}")
        return candidates
    return discover_entries(args.candidate_pool, include_variants=args.include_variants)


def wilson_lower_bound(successes: float, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = successes / total
    denom = 1.0 + z * z / total
    center = phat + z * z / (2.0 * total)
    spread = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return max(0.0, (center - spread) / denom)


def parse_result(stdout: str) -> tuple[int, int, int]:
    for line in stdout.splitlines():
        if line.startswith("RESULT:"):
            parts = line.split(":", 1)[1].split(",")
            return int(parts[0]), int(parts[1]), int(parts[2])
    raise ValueError("No RESULT line in evaluation output")


def evaluate_pair(
    candidate: dict[str, str],
    opponent: dict[str, str],
    games: int,
    timeout: int,
    worker_python: str,
) -> dict[str, Any]:
    command = [
        worker_python,
        "src/evaluate_single.py",
        candidate["model_path"],
        candidate["deck_path"],
        opponent["model_path"],
        opponent["deck_path"],
        str(games),
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        wins, losses, draws = parse_result(result.stdout)
        crashed = False
        error = ""
    except Exception as exc:
        elapsed = time.monotonic() - started
        wins, draws = 0, 0
        losses = games
        crashed = True
        error = str(exc)

    score = wins + 0.5 * draws
    return {
        "candidate": candidate["label"],
        "opponent": opponent["label"],
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
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(row["candidate"], []).append(row)

    summaries = []
    for candidate, candidate_rows in by_candidate.items():
        games = sum(row["games"] for row in candidate_rows)
        wins = sum(row["wins"] for row in candidate_rows)
        losses = sum(row["losses"] for row in candidate_rows)
        draws = sum(row["draws"] for row in candidate_rows)
        score = sum(row["score"] for row in candidate_rows)
        worst_row = min(candidate_rows, key=lambda row: row["score_rate"])
        summaries.append(
            {
                "candidate": candidate,
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "score_rate": score / games if games else 0.0,
                "win_rate": wins / games if games else 0.0,
                "wilson95_score_lb": wilson_lower_bound(score, games),
                "worst_opponent": worst_row["opponent"],
                "worst_score_rate": worst_row["score_rate"],
                "crashes": sum(1 for row in candidate_rows if row["crashed"]),
            }
        )

    return sorted(
        summaries,
        key=lambda row: (
            row["wilson95_score_lb"],
            row["worst_score_rate"],
            row["score_rate"],
        ),
        reverse=True,
    )


def print_entries(title: str, entries: list[dict[str, str]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for entry in entries:
        print(f"{entry['label']:40s} {entry['deck_path']}")


def print_summary(summaries: list[dict[str, Any]]) -> None:
    print("\nHoldout leaderboard")
    print("-------------------")
    print(
        f"{'rank':>4s}  {'model':38s} {'score':>7s} {'win':>7s} "
        f"{'wilson':>7s} {'worst':>7s} {'crash':>5s}  worst opponent"
    )
    for rank, row in enumerate(summaries, 1):
        print(
            f"{rank:4d}  {row['candidate'][:38]:38s} "
            f"{row['score_rate'] * 100:6.1f}% "
            f"{row['win_rate'] * 100:6.1f}% "
            f"{row['wilson95_score_lb'] * 100:6.1f}% "
            f"{row['worst_score_rate'] * 100:6.1f}% "
            f"{row['crashes']:5d}  {row['worst_opponent']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-file", default=DEFAULT_HOLDOUT_FILE)
    parser.add_argument("--results-file", default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--opponent-pool", default="models/holdout")
    parser.add_argument("--candidate-pool", default="models")
    parser.add_argument("--num-opponents", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--worker-python", default=os.environ.get("POKEMON_PYTHON", ""))
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--max-opponents", type=int, default=0)
    parser.add_argument("--no-variants", dest="include_variants", action="store_false")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--init", action="store_true", help="Create or refresh the holdout file and exit.")
    parser.add_argument("--list", action="store_true", help="Print candidates and opponents without evaluating.")
    parser.add_argument("--no-save", action="store_true")
    parser.set_defaults(include_variants=True)
    args = parser.parse_args()

    holdout = load_or_create_holdout(args)
    opponents = holdout.get("opponents", [])
    candidates = discover_candidates(args)
    worker_python = args.worker_python
    if not worker_python:
        worker_python = DEFAULT_WORKER_PYTHON if os.path.exists(DEFAULT_WORKER_PYTHON) else sys.executable

    if args.max_opponents > 0:
        opponents = opponents[: args.max_opponents]
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    if not opponents:
        raise RuntimeError("Holdout has no opponents")
    if not candidates:
        raise RuntimeError("No candidate models found")

    print(f"Holdout file: {args.holdout_file}")
    print(f"Worker python: {worker_python}")
    print(f"Games per pair: {args.games}")
    print_entries("Opponents", opponents)
    print_entries("Candidates", candidates)

    if args.init or args.list:
        return 0

    rows = []
    total_pairs = len(candidates) * len(opponents)
    pair_index = 0
    for candidate in candidates:
        for opponent in opponents:
            pair_index += 1
            print(
                f"\n[{pair_index}/{total_pairs}] "
                f"{candidate['label']} vs {opponent['label']}"
            )
            row = evaluate_pair(candidate, opponent, args.games, args.timeout, worker_python)
            rows.append(row)
            marker = "CRASH" if row["crashed"] else "OK"
            print(
                f"{marker}: {row['wins']}-{row['losses']}-{row['draws']} "
                f"score={row['score_rate'] * 100:.1f}% "
                f"time={row['seconds']:.1f}s"
            )
            if row["error"]:
                print(f"error: {row['error']}")

    summaries = aggregate(rows)
    print_summary(summaries)

    if not args.no_save:
        payload = {
            "holdout_file": args.holdout_file,
            "games_per_pair": args.games,
            "created_at": int(time.time()),
            "summary": summaries,
            "matches": rows,
        }
        os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
        with open(args.results_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nSaved results to {args.results_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
