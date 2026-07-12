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
from src.arena_core import atomic_write_json, read_json, utc_now


DEFAULT_HOLDOUT_FILE = "decks/holdout_opponents.json"
DEFAULT_RESULTS_FILE = "decks/submission_results.json"
DEFAULT_WORKER_PYTHON = os.path.join("venv", "bin", "python")


def deck_path_for_id(deck_id: str) -> str:
    if str(deck_id).startswith("bank_"):
        return os.path.join("decks", "deck_bank", f"{deck_id}.csv")
    return os.path.join("decks", f"deck_{deck_id}.csv")


def normalize_path(path: str) -> str:
    return os.path.relpath(path, ROOT) if os.path.isabs(path) else path


def candidate_suggestions(model_path: str) -> list[str]:
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    suggestions = []
    for model in discover_deck_models("models", include_variants=True):
        name = model["name"]
        if base_name in name or name in base_name:
            suggestions.append(model["path"])
    return suggestions[:5]


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
        suggestions = candidate_suggestions(model_path)
        if suggestions:
            raise FileNotFoundError(
                f"Model not found: {model_path}. Similar models: {', '.join(suggestions)}"
            )
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


def parse_result(stdout: str) -> tuple[int, int, int, dict[str, Any]]:
    details: dict[str, Any] = {}
    result: tuple[int, int, int] | None = None
    for line in stdout.splitlines():
        if line.startswith("RESULT:"):
            parts = line.split(":", 1)[1].split(",")
            result = int(parts[0]), int(parts[1]), int(parts[2])
        elif line.startswith("DETAIL:"):
            try:
                details = json.loads(line.split(":", 1)[1])
            except json.JSONDecodeError:
                details = {"parse_error": "invalid DETAIL payload"}
    if result is not None:
        return *result, details
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
        wins, losses, draws, details = parse_result(result.stdout)
        crashed = False
        error = ""
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
        by_opponent = {
            row["opponent"]: {
                "games": row["games"], "score_rate": row["score_rate"],
                "wins": row["wins"], "losses": row["losses"], "draws": row["draws"],
            }
            for row in candidate_rows
        }
        perspective = {"player_0": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
                       "player_1": {"games": 0, "wins": 0, "losses": 0, "draws": 0}}
        candidate_win_reasons: dict[str, int] = {}
        opponent_win_reasons: dict[str, int] = {}
        total_turns = 0.0
        for row in candidate_rows:
            details = row.get("details") or {}
            total_turns += float(details.get("total_turns", 0) or 0)
            for side, values in (details.get("perspective") or {}).items():
                if side not in perspective:
                    continue
                for key in ("games", "wins", "losses", "draws"):
                    perspective[side][key] += int(values.get(key, 0) or 0)
            for target, source in ((candidate_win_reasons, details.get("candidate_win_reasons") or {}),
                                   (opponent_win_reasons, details.get("opponent_win_reasons") or {})):
                for reason, count in source.items():
                    target[reason] = target.get(reason, 0) + int(count)
        for values in perspective.values():
            values["score_rate"] = (
                (values["wins"] + 0.5 * values["draws"]) / values["games"]
                if values["games"] else 0.0
            )
        perspective_gap = abs(perspective["player_0"]["score_rate"] - perspective["player_1"]["score_rate"])
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
                "mean_turns": total_turns / games if games else 0.0,
                "perspective": perspective,
                "perspective_score_gap": perspective_gap,
                "candidate_win_reasons": candidate_win_reasons,
                "opponent_win_reasons": opponent_win_reasons,
                "by_opponent": by_opponent,
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
        f"{'rank':>4s}  {'model':32s} {'score':>7s} {'wilson':>7s} "
        f"{'worst':>7s} {'p-gap':>7s} {'turns':>7s}  worst opponent"
    )
    for rank, row in enumerate(summaries, 1):
        print(
            f"{rank:4d}  {row['candidate'][:32]:32s} "
            f"{row['score_rate'] * 100:6.1f}% "
            f"{row['wilson95_score_lb'] * 100:6.1f}% "
            f"{row['worst_score_rate'] * 100:6.1f}% "
            f"{row['perspective_score_gap'] * 100:6.1f}% "
            f"{row['mean_turns']:7.1f}  {row['worst_opponent']}"
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
    parser.add_argument(
        "--best-candidate-file", default="",
        help="Optional JSON destination for the top candidate selected by Wilson lower bound.",
    )
    parser.add_argument("--progress-file", default="", help="Optional JSON progress file for the arena dashboard.")
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
            if args.progress_file:
                completed_games = sum(item["games"] for item in rows)
                atomic_write_json(args.progress_file, {
                    "state": "running", "bot_id": read_json(args.progress_file, {}).get("bot_id", candidate["label"]),
                    "model_path": candidate["model_path"],
                    "opponents": [entry["label"] for entry in opponents], "games_per_opponent": args.games,
                    "planned_games": total_pairs * args.games, "completed_games": completed_games,
                    "wins": sum(item["wins"] for item in rows), "losses": sum(item["losses"] for item in rows),
                    "draws": sum(item["draws"] for item in rows),
                    "progress": completed_games / (total_pairs * args.games) if total_pairs else 0.0,
                    "started_at": read_json(args.progress_file, {}).get("started_at", utc_now()),
                    "ended_at": None, "result_at": None, "error": "",
                    "result_file": args.results_file,
                    "configuration": {"holdout_file": args.holdout_file, "games": args.games},
                })

    summaries = aggregate(rows)
    print_summary(summaries)

    if args.best_candidate_file:
        best = summaries[0]
        atomic_write_json(args.best_candidate_file, {
            "selected_at": utc_now(),
            "selection_metric": "wilson95_score_lb, worst_score_rate, score_rate",
            "candidate": best["candidate"],
            "summary": best,
            "holdout_file": args.holdout_file,
            "games_per_pair": args.games,
        })
        print(f"Selected best candidate: {best['candidate']} -> {args.best_candidate_file}")

    if not args.no_save:
        payload = {
            "holdout_file": args.holdout_file,
            "games_per_pair": args.games,
            "created_at": int(time.time()),
            "summary": summaries,
            "matches": rows,
        }
        atomic_write_json(args.results_file, payload)
        print(f"\nSaved results to {args.results_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
