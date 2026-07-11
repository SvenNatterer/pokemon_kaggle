#!/usr/bin/env python3
"""Evaluate newly trained models against fixed reference models."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from typing import Any


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.model_paths import discover_deck_models, parse_deck_model_path


DEFAULT_RESULTS_FILE = "decks/reference_eval_results.json"
DEFAULT_WORKER_PYTHON = os.path.join("venv", "bin", "python")
DEFAULT_REFERENCES = [
    "models/holdout/ppo_v4_deck_7.zip",
    "models/ppo_v4_deck_bank_47.zip",
    "models/ppo_deck_1.zip",
]
DEFAULT_CANDIDATE_POOL = "models"


def deck_path_for_id(deck_id: str) -> str:
    if str(deck_id).startswith("bank_"):
        return os.path.join("decks", "deck_bank", f"{deck_id}.csv")
    return os.path.join("decks", f"deck_{deck_id}.csv")


def normalize_path(path: str) -> str:
    return os.path.relpath(path, ROOT) if os.path.isabs(path) else path


def entry_from_path(model_path: str) -> dict[str, Any]:
    model_path = normalize_path(model_path)
    if not model_path.endswith(".zip"):
        model_path = f"{model_path}.zip"

    parsed = parse_deck_model_path(model_path)
    if parsed is None:
        raise ValueError(f"Cannot parse deck id from model path: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    deck_path = deck_path_for_id(parsed["deck_id"])
    if not os.path.exists(deck_path):
        raise FileNotFoundError(f"Deck not found for {model_path}: {deck_path}")

    return {
        "label": parsed["name"],
        "deck_id": str(parsed["deck_id"]),
        "variant": parsed["variant"],
        "model_path": model_path,
        "deck_path": deck_path,
        "mtime": os.path.getmtime(model_path),
    }


def add_pool_once(pools: list[str], path: str) -> None:
    if not os.path.isdir(path):
        return
    normalized = normalize_path(path)
    if normalized not in pools:
        pools.append(normalized)


def candidate_pools_for_args(args: argparse.Namespace) -> list[str]:
    pools: list[str] = []
    for pool in args.candidate_pool or [DEFAULT_CANDIDATE_POOL]:
        add_pool_once(pools, pool)

    auto_extra_pools = args.all and not args.candidate_pool
    if args.include_queue or auto_extra_pools:
        add_pool_once(pools, os.path.join("models", "queue"))
    if args.include_archive or auto_extra_pools:
        for path in sorted(glob.glob(os.path.join("models", "archive_*")), reverse=True):
            add_pool_once(pools, path)
    if args.include_backup:
        add_pool_once(pools, os.path.join("models", "backup"))

    return pools


def discover_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.candidate:
        candidates = []
        for path in args.candidate:
            candidates.append(entry_from_path(path))
            
            base_path = path
            if base_path.endswith(".zip"):
                base_path = base_path[:-4]
            import glob
            for ckpt_path in glob.glob(f"{base_path}_checkpoint_*.zip"):
                try:
                    candidates.append(entry_from_path(ckpt_path))
                except Exception as e:
                    print(f"Warning: could not load checkpoint {ckpt_path}: {e}")
        return candidates

    now = time.time()
    reference_paths = {os.path.abspath(path) for path in args.reference}
    candidates = []
    seen_paths = set()
    for pool in args.candidate_pools:
        for model in discover_deck_models(pool, include_variants=True):
            path = model["path"]
            abs_path = os.path.abspath(path)
            if abs_path in reference_paths or abs_path in seen_paths:
                continue
            seen_paths.add(abs_path)
            if args.since_hours > 0:
                age_hours = (now - os.path.getmtime(path)) / 3600.0
                if age_hours > args.since_hours:
                    continue
            try:
                candidates.append(entry_from_path(path))
            except FileNotFoundError:
                continue

    return sorted(candidates, key=lambda item: item["mtime"], reverse=True)


def discover_references(args: argparse.Namespace) -> list[dict[str, Any]]:
    references = []
    for path in args.reference:
        try:
            references.append(entry_from_path(path))
        except FileNotFoundError as exc:
            print(f"Skipping reference: {exc}", file=sys.stderr)
    return references


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
    candidate: dict[str, Any],
    reference: dict[str, Any],
    games: int,
    timeout: int,
    worker_python: str,
) -> dict[str, Any]:
    command = [
        worker_python,
        "src/evaluate_single.py",
        candidate["model_path"],
        candidate["deck_path"],
        reference["model_path"],
        reference["deck_path"],
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
        "candidate_deck_id": candidate["deck_id"],
        "candidate_model_path": candidate["model_path"],
        "reference": reference["label"],
        "reference_deck_id": reference["deck_id"],
        "reference_model_path": reference["model_path"],
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


def strength_band(score_rate: float) -> str:
    if score_rate < 0.25:
        return "too_weak"
    if score_rate < 0.40:
        return "weak_style"
    if score_rate <= 0.60:
        return "ideal"
    if score_rate <= 0.75:
        return "strong"
    return "dominant"


def holdout_fit_score(score_rate: float, worst_score_rate: float, crashes: int, target: float, wilson_score_lb: float = 0.0) -> float:
    if crashes:
        return -100.0 * crashes
    return wilson_score_lb + worst_score_rate * 0.5


def aggregate(rows: list[dict[str, Any]], target: float) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(row["candidate_model_path"], []).append(row)

    summaries = []
    for candidate_path, candidate_rows in by_candidate.items():
        games = sum(row["games"] for row in candidate_rows)
        wins = sum(row["wins"] for row in candidate_rows)
        losses = sum(row["losses"] for row in candidate_rows)
        draws = sum(row["draws"] for row in candidate_rows)
        score = sum(row["score"] for row in candidate_rows)
        score_rate = score / games if games else 0.0
        worst_row = min(candidate_rows, key=lambda row: row["score_rate"])
        best_row = max(candidate_rows, key=lambda row: row["score_rate"])
        crashes = sum(1 for row in candidate_rows if row["crashed"])
        summaries.append(
            {
                "candidate": candidate_rows[0]["candidate"],
                "candidate_model_path": candidate_path,
                "candidate_deck_id": candidate_rows[0]["candidate_deck_id"],
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "score_rate": score_rate,
                "win_rate": wins / games if games else 0.0,
                "wilson95_score_lb": wilson_lower_bound(score, games),
                "worst_reference": worst_row["reference"],
                "worst_score_rate": worst_row["score_rate"],
                "best_reference": best_row["reference"],
                "best_score_rate": best_row["score_rate"],
                "crashes": crashes,
                "band": strength_band(score_rate),
                "holdout_fit": holdout_fit_score(score_rate, worst_row["score_rate"], crashes, target, wilson_lower_bound(score, games)),
            }
        )
    return summaries


def best_per_deck(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in summaries:
        deck_id = row["candidate_deck_id"]
        current = best.get(deck_id)
        if current is None or (
            row["wilson95_score_lb"],
            row["worst_score_rate"],
            row["score_rate"],
        ) > (
            current["wilson95_score_lb"],
            current["worst_score_rate"],
            current["score_rate"],
        ):
            best[deck_id] = row
    return sorted(
        best.values(),
        key=lambda row: (row["wilson95_score_lb"], row["worst_score_rate"], row["score_rate"]),
        reverse=True,
    )


def print_entries(title: str, entries: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for entry in entries:
        age = (time.time() - entry["mtime"]) / 3600.0
        print(f"{entry['label']:42s} deck={entry['deck_id']:8s} age={age:5.1f}h")


def print_strength(summaries: list[dict[str, Any]]) -> None:
    rows = sorted(
        summaries,
        key=lambda row: (row["crashes"] == 0, row["score_rate"], row["worst_score_rate"]),
        reverse=True,
    )
    print("\nReference strength")
    print("------------------")
    print(
        f"{'rank':>4s}  {'model':40s} {'score':>7s} {'wilson':>7s} "
        f"{'worst':>7s} {'best':>7s} {'crash':>5s}  band"
    )
    for rank, row in enumerate(rows, 1):
        print(
            f"{rank:4d}  {row['candidate'][:40]:40s} "
            f"{row['score_rate'] * 100:6.1f}% "
            f"{row['wilson95_score_lb'] * 100:6.1f}% "
            f"{row['worst_score_rate'] * 100:6.1f}% "
            f"{row['best_score_rate'] * 100:6.1f}% "
            f"{row['crashes']:5d}  {row['band']}"
        )


def print_holdout_fit(summaries: list[dict[str, Any]]) -> None:
    rows = sorted(summaries, key=lambda row: row["holdout_fit"], reverse=True)
    print("\nHoldout fit")
    print("-----------")
    print(
        f"{'rank':>4s}  {'model':40s} {'fit':>7s} {'score':>7s} "
        f"{'worst':>7s} {'crash':>5s}  weakest ref"
    )
    for rank, row in enumerate(rows, 1):
        print(
            f"{rank:4d}  {row['candidate'][:40]:40s} "
            f"{row['holdout_fit'] * 100:6.1f}% "
            f"{row['score_rate'] * 100:6.1f}% "
            f"{row['worst_score_rate'] * 100:6.1f}% "
            f"{row['crashes']:5d}  {row['worst_reference']}"
        )


def print_best_per_deck(summaries: list[dict[str, Any]]) -> None:
    print("\nBest per deck")
    print("-------------")
    print(f"{'deck':8s} {'model':40s} {'score':>7s} {'worst':>7s} {'fit':>7s}  band")
    for row in best_per_deck(summaries):
        print(
            f"{row['candidate_deck_id']:8s} {row['candidate'][:40]:40s} "
            f"{row['score_rate'] * 100:6.1f}% "
            f"{row['worst_score_rate'] * 100:6.1f}% "
            f"{row['holdout_fit'] * 100:6.1f}%  {row['band']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-pool",
        action="append",
        default=[],
        help="Directory with candidate model .zip files. Can be passed more than once.",
    )
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--since-hours", type=float, default=36.0)
    parser.add_argument("--all", action="store_true", help="Evaluate all models in the candidate pool.")
    parser.add_argument("--include-archive", action="store_true", help="Also scan models/archive_* for candidates.")
    parser.add_argument("--include-backup", action="store_true", help="Also scan models/backup for candidates.")
    parser.add_argument("--include-queue", action="store_true", help="Also scan models/queue for candidates.")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--target-strength", type=float, default=0.50)
    parser.add_argument("--worker-python", default=os.environ.get("POKEMON_PYTHON", ""))
    parser.add_argument("--results-file", default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--max-references", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if not args.reference:
        args.reference = DEFAULT_REFERENCES[:]
    if args.all:
        args.since_hours = 0.0
    args.candidate_pools = candidate_pools_for_args(args)

    worker_python = args.worker_python
    if not worker_python:
        worker_python = DEFAULT_WORKER_PYTHON if os.path.exists(DEFAULT_WORKER_PYTHON) else sys.executable

    candidates = discover_candidates(args)
    references = discover_references(args)

    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    if args.max_references > 0:
        references = references[: args.max_references]

    if not candidates:
        if args.candidate:
            raise RuntimeError("No candidate models found from explicit --candidate paths")
        raise RuntimeError(
            "No candidate models found. "
            f"Searched pools: {', '.join(args.candidate_pools) or '(none)'}. "
            "Use --all to include archive/queue automatically, or pass --candidate-pool PATH."
        )
    if not references:
        raise RuntimeError("No reference models found")

    print(f"Worker python: {worker_python}")
    print(f"Games per pair: {args.games}")
    if args.candidate:
        print("Candidate mode: explicit paths")
    elif args.since_hours > 0:
        print(f"Candidate mode: models changed in the last {args.since_hours:g} hours")
    else:
        print("Candidate mode: all models in pool")
    if not args.candidate:
        print(f"Candidate pools: {', '.join(args.candidate_pools)}")
    print_entries("References", references)
    print_entries("Candidates", candidates)

    if args.list:
        return 0

    rows = []
    total_pairs = len(candidates) * len(references)
    pair_index = 0
    for candidate in candidates:
        for reference in references:
            pair_index += 1
            print(f"\n[{pair_index}/{total_pairs}] {candidate['label']} vs {reference['label']}")
            row = evaluate_pair(candidate, reference, args.games, args.timeout, worker_python)
            rows.append(row)
            marker = "CRASH" if row["crashed"] else "OK"
            print(
                f"{marker}: {row['wins']}-{row['losses']}-{row['draws']} "
                f"score={row['score_rate'] * 100:.1f}% time={row['seconds']:.1f}s"
            )
            if row["error"]:
                print(f"error: {row['error']}")

    summaries = aggregate(rows, args.target_strength)
    print_strength(summaries)
    print_holdout_fit(summaries)
    print_best_per_deck(summaries)

    if not args.no_save:
        payload = {
            "reference_models": [entry["model_path"] for entry in references],
            "games_per_pair": args.games,
            "target_strength": args.target_strength,
            "created_at": int(time.time()),
            "summary": sorted(
                summaries,
                key=lambda row: (row["wilson95_score_lb"], row["worst_score_rate"], row["score_rate"]),
                reverse=True,
            ),
            "matches": rows,
        }
        os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
        with open(args.results_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nSaved results to {args.results_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
