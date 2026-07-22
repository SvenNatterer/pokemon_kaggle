#!/usr/bin/env python3
"""Evaluate active models against a frozen opponent holdout."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.league.model_paths import discover_deck_models, parse_deck_model_path
from src.arena.arena_core import atomic_write_json, read_json, utc_now
from src.agents.rule_based_agent import is_rule_based_model_spec
from src.training.training_health import (
    health_gate,
    merge_option_count_histograms,
    summarize_health,
)


DEFAULT_HOLDOUT_FILE = "decks/holdout_opponents.json"
DEFAULT_RESULTS_FILE = "arena_data/submission_results.json"
DEFAULT_WORKER_PYTHON = os.path.join("venv", "bin", "python")
DEFAULT_GAMES = 100
DEFAULT_PARALLEL_WORKERS = max(1, os.cpu_count() or 1)
DEFAULT_CACHE_DIR = "arena_data/evaluation_cache"
# Increment when the evaluator or game semantics change incompatibly.
EVALUATION_CACHE_VERSION = 1
_FILE_DIGESTS: dict[tuple[str, int, int], str] = {}


def deck_path_for_id(deck_id: str) -> str:
    if str(deck_id).startswith("bank_"):
        return os.path.join("decks", "deck_bank", f"{deck_id}.csv")
    return os.path.join("decks", f"deck_{deck_id}.csv")


def normalize_path(path: str) -> str:
    return os.path.relpath(path, ROOT) if os.path.isabs(path) else path


def file_digest(path: str) -> str:
    """Return a memoized content digest for one evaluation input file."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(ROOT) / resolved
    resolved = resolved.resolve()
    stat = resolved.stat()
    cache_key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    digest = _FILE_DIGESTS.get(cache_key)
    if digest is not None:
        return digest

    hasher = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    _FILE_DIGESTS[cache_key] = digest
    return digest


def bot_digest(model_path: str) -> str:
    """Hash a PPO archive or the code and profile behind a rule-based bot."""
    if is_rule_based_model_spec(model_path):
        source_digest = file_digest(os.path.join(ROOT, "src", "agents", "rule_based_agent.py"))
        payload = f"{model_path.strip().lower()}:{source_digest}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return file_digest(model_path)


def pair_cache_signature(
    candidate: dict[str, str],
    opponent: dict[str, str],
    games: int,
) -> dict[str, Any]:
    """Identify every input that changes a candidate/opponent result."""
    return {
        "version": EVALUATION_CACHE_VERSION,
        "games": int(games),
        "candidate": {
            "model_sha256": bot_digest(candidate["model_path"]),
            "deck_sha256": file_digest(candidate["deck_path"]),
        },
        "opponent": {
            "model_sha256": bot_digest(opponent["model_path"]),
            "deck_sha256": file_digest(opponent["deck_path"]),
        },
    }


def pair_cache_path(cache_dir: str, signature: dict[str, Any]) -> Path:
    serialized = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{cache_key}.json"


def evaluation_health_gate(row: dict[str, Any]) -> dict[str, Any]:
    """Return the P0 gate for one candidate/opponent evaluation result."""
    details = row.get("details") or {}
    return health_gate(details.get("health"), crashes=int(bool(row.get("crashed"))))


def load_cached_pair(
    cache_dir: str,
    candidate: dict[str, str],
    opponent: dict[str, str],
    games: int,
) -> dict[str, Any] | None:
    signature = pair_cache_signature(candidate, opponent, games)
    payload = read_json(pair_cache_path(cache_dir, signature), {})
    if not isinstance(payload, dict) or payload.get("signature") != signature:
        return None
    row = payload.get("result")
    if not isinstance(row, dict) or row.get("crashed"):
        return None
    try:
        valid_game_count = int(row.get("games", -1)) == games
        completed_games = sum(
            int(row.get(key, 0)) for key in ("wins", "losses", "draws")
        )
    except (TypeError, ValueError):
        return None
    if not valid_game_count or completed_games != games:
        return None
    if not evaluation_health_gate(row)["passed"]:
        return None
    return {
        **row,
        "candidate": candidate["label"],
        "opponent": opponent["label"],
        "cached": True,
    }


def save_cached_pair(
    cache_dir: str,
    candidate: dict[str, str],
    opponent: dict[str, str],
    games: int,
    row: dict[str, Any],
) -> None:
    if row.get("crashed"):
        return
    if not evaluation_health_gate(row)["passed"]:
        return
    if sum(int(row.get(key, 0)) for key in ("wins", "losses", "draws")) != games:
        return
    signature = pair_cache_signature(candidate, opponent, games)
    stored_row = {**row, "cached": False}
    atomic_write_json(
        pair_cache_path(cache_dir, signature),
        {"signature": signature, "saved_at": utc_now(), "result": stored_row},
    )


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


def entry_from_spec(raw_spec: str | dict[str, Any]) -> dict[str, str]:
    """Build an evaluation candidate with an explicit model/profile and deck."""
    try:
        spec = json.loads(raw_spec) if isinstance(raw_spec, str) else dict(raw_spec)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid candidate spec: {exc}") from exc
    if not isinstance(spec, dict):
        raise ValueError("Candidate spec must be a JSON object")

    model_path = str(spec.get("model_path") or "").strip()
    deck_path = normalize_path(str(spec.get("deck_path") or "").strip())
    label = str(spec.get("label") or spec.get("bot_id") or model_path).strip()
    if not model_path or not deck_path or not label:
        raise ValueError("Candidate spec requires label, model_path, and deck_path")
    if not os.path.isfile(deck_path):
        raise FileNotFoundError(f"Candidate deck not found: {deck_path}")

    deck_stem = Path(deck_path).stem
    deck_id = deck_stem[5:] if deck_stem.startswith("deck_") else deck_stem
    if is_rule_based_model_spec(model_path):
        return {
            "label": label,
            "deck_id": deck_id,
            "model_path": model_path,
            "deck_path": deck_path,
            "bot_type": "rule_based",
        }

    entry = entry_from_path(model_path)
    if entry is None:
        raise ValueError(f"Cannot build candidate from model path: {model_path}")
    entry.update({
        "label": label,
        "deck_id": deck_id,
        "deck_path": deck_path,
        "bot_type": str(spec.get("bot_type") or "ppo"),
    })
    return entry


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
    if getattr(args, "candidate_spec", None):
        return [entry_from_spec(spec) for spec in args.candidate_spec]
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
        elif line.startswith("CHILD ERROR:"):
            raise RuntimeError(line.split(":", 1)[1].strip())
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


def evaluate_pairs(
    candidates: list[dict[str, str]],
    opponents: list[dict[str, str]],
    games: int,
    timeout: int,
    worker_python: str,
    workers: int,
    cache_dir: str | None = None,
    force: bool = False,
):
    """Yield cached or newly completed independent matchups."""
    pairs = []
    for candidate in candidates:
        for opponent in opponents:
            if cache_dir and not force:
                cached = load_cached_pair(cache_dir, candidate, opponent, games)
                if cached is not None:
                    yield candidate, opponent, cached
                    continue
            pairs.append((candidate, opponent))

    if not pairs:
        return
    active_workers = min(max(1, int(workers)), len(pairs))
    if active_workers == 1:
        for candidate, opponent in pairs:
            row = evaluate_pair(
                candidate, opponent, games, timeout, worker_python
            )
            row["cached"] = False
            if cache_dir:
                save_cached_pair(cache_dir, candidate, opponent, games, row)
            yield candidate, opponent, row
        return

    with ThreadPoolExecutor(max_workers=active_workers) as executor:
        futures = {
            executor.submit(
                evaluate_pair,
                candidate,
                opponent,
                games,
                timeout,
                worker_python,
            ): (candidate, opponent)
            for candidate, opponent in pairs
        }
        for future in as_completed(futures):
            candidate, opponent = futures[future]
            row = future.result()
            row["cached"] = False
            if cache_dir:
                save_cached_pair(cache_dir, candidate, opponent, games, row)
            yield candidate, opponent, row


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
        health_rows = []
        for row in candidate_rows:
            details = row.get("details") or {}
            total_turns += float(details.get("total_turns", 0) or 0)
            health_rows.append(details.get("health") or {})
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
        crashes = sum(1 for row in candidate_rows if row["crashed"])
        health = summarize_health(
            episodes=games,
            learner_decisions=sum(
                int(item.get("learner_decisions", 0) or 0) for item in health_rows
            ),
            invalid_learner_actions=sum(
                int(item.get("invalid_learner_actions", 0) or 0) for item in health_rows
            ),
            option_overflows=sum(
                int(item.get("option_overflows", 0) or 0) for item in health_rows
            ),
            engine_errors=sum(
                int(item.get("engine_errors", 0) or 0) for item in health_rows
            ),
            max_option_count_seen=max(
                (int(item.get("max_option_count_seen", 0) or 0) for item in health_rows),
                default=0,
            ),
            option_count_histogram=merge_option_count_histograms(
                item.get("option_count_histogram") for item in health_rows
            ),
        )
        health["gate"] = health_gate(health, crashes=crashes)
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
                "crashes": crashes,
                "mean_turns": total_turns / games if games else 0.0,
                "perspective": perspective,
                "perspective_score_gap": perspective_gap,
                "candidate_win_reasons": candidate_win_reasons,
                "opponent_win_reasons": opponent_win_reasons,
                "by_opponent": by_opponent,
                "health": health,
                "health_gate": health["gate"],
            }
        )

    return sorted(
        summaries,
        key=lambda row: (
            row["health_gate"]["passed"],
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
        f"{'rank':>4s}  {'model':32s} {'health':>7s} {'score':>7s} {'wilson':>7s} "
        f"{'worst':>7s} {'p-gap':>7s} {'turns':>7s}  worst opponent"
    )
    for rank, row in enumerate(summaries, 1):
        print(
            f"{rank:4d}  {row['candidate'][:32]:32s} "
            f"{'OK' if row['health_gate']['passed'] else 'BLOCK':>7s} "
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
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_PARALLEL_WORKERS,
        help=(
            "Parallel candidate/opponent processes. Defaults to the available "
            "CPU count and is capped by the number of matchups."
        ),
    )
    parser.add_argument("--worker-python", default=os.environ.get("POKEMON_PYTHON", ""))
    parser.add_argument("--candidate", action="append")
    parser.add_argument(
        "--candidate-spec",
        action="append",
        help="JSON candidate with label, model_path (or rule_based:profile), and deck_path.",
    )
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--max-opponents", type=int, default=0)
    parser.add_argument("--no-variants", dest="include_variants", action="store_false")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--init", action="store_true", help="Create or refresh the holdout file and exit.")
    parser.add_argument("--list", action="store_true", help="Print candidates and opponents without evaluating.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Persistent per-matchup result cache.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write persistent matchup results.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replay every matchup and replace its cached result.",
    )
    parser.add_argument(
        "--best-candidate-file", default="",
        help="Optional JSON destination for the top candidate selected by Wilson lower bound.",
    )
    parser.add_argument("--progress-file", default="", help="Optional JSON progress file for the arena dashboard.")
    parser.set_defaults(include_variants=True)
    args = parser.parse_args()
    if args.games <= 0 or args.timeout <= 0 or args.workers <= 0:
        parser.error("--games, --timeout, and --workers must be positive")

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
    active_workers = min(args.workers, total_pairs)
    print(f"Parallel workers: {active_workers}")
    cache_dir = None if args.no_cache else args.cache_dir
    print(f"Result cache: {cache_dir or 'disabled'}")
    pair_index = 0
    for candidate, opponent, row in evaluate_pairs(
        candidates,
        opponents,
        args.games,
        args.timeout,
        worker_python,
        args.workers,
        cache_dir=cache_dir,
        force=args.force,
    ):
        pair_index += 1
        print(
            f"\n[{pair_index}/{total_pairs} completed] "
            f"{candidate['label']} vs {opponent['label']}"
        )
        rows.append(row)
        marker = "CACHED" if row.get("cached") else "CRASH" if row["crashed"] else "OK"
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
                "configuration": {
                    "holdout_file": args.holdout_file,
                    "games": args.games,
                    "workers": active_workers,
                    "models": [entry["model_path"] for entry in candidates],
                    "candidates": candidates,
                    "cache_dir": cache_dir,
                    "cached_matches": sum(bool(item.get("cached")) for item in rows),
                },
            })

    candidate_order = {entry["label"]: index for index, entry in enumerate(candidates)}
    opponent_order = {entry["label"]: index for index, entry in enumerate(opponents)}
    rows.sort(
        key=lambda row: (
            candidate_order[row["candidate"]],
            opponent_order[row["opponent"]],
        )
    )

    summaries = aggregate(rows)
    print_summary(summaries)
    cached_matches = sum(bool(row.get("cached")) for row in rows)
    print(f"\nReused {cached_matches}/{total_pairs} matchups from the result cache.")

    if not args.no_save:
        payload = {
            "holdout_file": args.holdout_file,
            "games_per_pair": args.games,
            "parallel_workers": active_workers,
            "candidates": candidates,
            "cached_matches": cached_matches,
            "created_at": int(time.time()),
            "summary": summaries,
            "matches": rows,
        }
        atomic_write_json(args.results_file, payload)
        print(f"\nSaved results to {args.results_file}")

    eligible_summaries = [row for row in summaries if row["health_gate"]["passed"]]
    if not eligible_summaries:
        print("No candidate passed the evaluation health gate.", file=sys.stderr)
        return 2

    if args.best_candidate_file:
        best = eligible_summaries[0]
        best_spec = next(entry for entry in candidates if entry["label"] == best["candidate"])
        atomic_write_json(args.best_candidate_file, {
            "selected_at": utc_now(),
            "selection_metric": "health_gate, wilson95_score_lb, worst_score_rate, score_rate",
            "candidate": best["candidate"],
            "candidate_spec": best_spec,
            "summary": best,
            "holdout_file": args.holdout_file,
            "games_per_pair": args.games,
            "parallel_workers": active_workers,
            "cached_matches": cached_matches,
        })
        print(f"Selected best candidate: {best['candidate']} -> {args.best_candidate_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
