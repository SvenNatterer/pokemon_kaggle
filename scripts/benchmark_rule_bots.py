#!/usr/bin/env python3
"""Reproducible cross-play benchmark for the versioned rule-bot meta pool."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.league.tournament import evaluate_vs_opponent


DEFAULT_POOL = ROOT / "decks" / "rule_bot_meta_pool_v1.json"
DEFAULT_OPPONENTS = ROOT / "decks" / "validation_opponents.json"
DEFAULT_OUTPUT = ROOT / "reports" / "rule_bot_benchmark_v1.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--opponents", type=Path, default=DEFAULT_OPPONENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--candidate", action="append", default=[], help="Exact candidate label; repeatable")
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--max-opponents", type=int)
    parser.add_argument("--cross-play", action="store_true", help="Also use selected rule bots as opponents")
    parser.add_argument("--smoke", action="store_true", help="Run two games for two candidates and two opponents")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the matrix without games")
    parser.add_argument("--resume", action="store_true", help="Resume completed matchups from an existing output file")
    return parser.parse_args(argv)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pool(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_json(path)
    bots = list(payload.get("bots") or [])
    required = {"label", "archetype", "variant", "model", "deck", "meta_weight"}
    if not bots:
        raise ValueError("rule pool is empty")
    labels = set()
    for entry in bots:
        missing = required - set(entry)
        if missing:
            raise ValueError(f"pool entry is missing {sorted(missing)}")
        if entry["label"] in labels:
            raise ValueError(f"duplicate pool label: {entry['label']}")
        labels.add(entry["label"])
        deck = ROOT / entry["deck"]
        if not deck.is_file():
            raise ValueError(f"missing pool deck: {entry['deck']}")
        if sum(1 for line in deck.read_text(encoding="utf-8").splitlines() if line.strip()) != 60:
            raise ValueError(f"pool deck does not contain 60 cards: {entry['deck']}")
    total = sum(float(entry["meta_weight"]) for entry in bots)
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"meta weights must sum to 1.0, got {total:.8f}")
    return payload, bots


def load_opponents(path: Path) -> list[dict[str, str]]:
    payload = _read_json(path)
    raw_entries = payload.get("opponents") if isinstance(payload, dict) else payload
    opponents = []
    for entry in raw_entries or []:
        model = entry.get("model_path") or entry.get("model")
        deck = entry.get("deck_path") or entry.get("deck")
        label = entry.get("label")
        if model and deck and label:
            opponents.append({"label": str(label), "model": str(model), "deck": str(deck)})
    if not opponents:
        raise ValueError("opponent manifest contains no usable entries")
    return opponents


def wilson_lower(score_rate: float, games: int, z: float = 1.959963984540054) -> float:
    if games <= 0:
        return 0.0
    denominator = 1.0 + z * z / games
    center = score_rate + z * z / (2.0 * games)
    margin = z * math.sqrt(score_rate * (1.0 - score_rate) / games + z * z / (4.0 * games * games))
    return max(0.0, (center - margin) / denominator)


def training_probabilities(pool: dict[str, Any], bots: list[dict[str, Any]]) -> dict[str, float]:
    sampling = pool.get("sampling") or {}
    meta_fraction = float(sampling.get("meta_fraction", 0.60))
    uniform_fraction = float(sampling.get("uniform_fraction", 0.20))
    pfsp_fraction = float(sampling.get("pfsp_fraction", 0.20))
    total = meta_fraction + uniform_fraction + pfsp_fraction
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError("sampling fractions must sum to 1.0")
    uniform = 1.0 / len(bots)
    # Before results exist, PFSP uses its uniform prior. Later training may replace
    # this term with capped weakness probabilities without changing meta weights.
    return {
        entry["label"]: meta_fraction * float(entry["meta_weight"])
        + (uniform_fraction + pfsp_fraction) * uniform
        for entry in bots
    }


def summarize(candidate: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
    games = sum(item["games"] for item in matches)
    score = sum(item["score"] for item in matches)
    score_rate = score / games if games else 0.0
    perspective_games = {"player_0": 0, "player_1": 0}
    perspective_score = {"player_0": 0.0, "player_1": 0.0}
    decision_counts: dict[str, int] = {}
    for match in matches:
        details = match["details"]
        for perspective, values in details.get("perspective", {}).items():
            perspective_games[perspective] += int(values.get("games", 0))
            perspective_score[perspective] += float(values.get("wins", 0)) + 0.5 * float(values.get("draws", 0))
        for category, count in details.get("decision_counts", {}).items():
            decision_counts[category] = decision_counts.get(category, 0) + int(count)
    rates = {
        key: perspective_score[key] / perspective_games[key] if perspective_games[key] else 0.0
        for key in perspective_games
    }
    return {
        "candidate": candidate["label"],
        "model": candidate["model"],
        "archetype": candidate["archetype"],
        "variant": candidate["variant"],
        "games": games,
        "score": score,
        "score_rate": score_rate,
        "wilson95_score_lb": wilson_lower(score_rate, games),
        "worst_matchup_score_rate": min((item["score_rate"] for item in matches), default=0.0),
        "perspective_score_rates": rates,
        "perspective_gap": abs(rates["player_0"] - rates["player_1"]),
        "decision_counts": decision_counts,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.games <= 0:
        raise SystemExit("--games must be positive")
    pool_path = args.pool.expanduser().resolve()
    opponent_path = args.opponents.expanduser().resolve()
    pool, bots = load_pool(pool_path)
    opponents = load_opponents(opponent_path)

    if args.candidate:
        requested = set(args.candidate)
        bots = [entry for entry in bots if entry["label"] in requested]
        missing = requested - {entry["label"] for entry in bots}
        if missing:
            raise SystemExit(f"unknown candidate label(s): {', '.join(sorted(missing))}")
    if args.smoke:
        args.games = 2
        args.max_candidates = args.max_candidates or 2
        args.max_opponents = args.max_opponents or 2
    if args.max_candidates is not None:
        bots = bots[: args.max_candidates]
    if args.max_opponents is not None:
        opponents = opponents[: args.max_opponents]
    if args.cross_play:
        opponents.extend(
            {"label": entry["label"], "model": entry["model"], "deck": entry["deck"]}
            for entry in bots
        )

    matrix = [(candidate, opponent) for candidate in bots for opponent in opponents]
    print(
        f"Rule benchmark: {len(bots)} candidates × {len(opponents)} opponents × {args.games} games",
        flush=True,
    )
    if args.dry_run:
        for candidate, opponent in matrix:
            print(f"  {candidate['label']} vs {opponent['label']}")
        return 0

    output = args.output.expanduser().resolve()
    matches = []
    created_at = datetime.now(timezone.utc).isoformat()
    if args.resume and output.is_file():
        previous = _read_json(output)
        if int(previous.get("games_per_matchup", -1)) != args.games:
            raise SystemExit("cannot resume: games_per_matchup differs from the existing report")
        matches = list(previous.get("matches") or [])
        created_at = str(previous.get("created_at") or created_at)
    completed_keys = {(item["candidate"], item["opponent"]) for item in matches}
    pending_matrix = [
        (candidate, opponent)
        for candidate, opponent in matrix
        if (candidate["label"], opponent["label"]) not in completed_keys
    ]
    started = time.time()
    for index, (candidate, opponent) in enumerate(pending_matrix, start=1):
        finished_before = len(matrix) - len(pending_matrix)
        print(
            f"[{finished_before + index}/{len(matrix)}] {candidate['label']} vs {opponent['label']}",
            flush=True,
        )
        result, details = evaluate_vs_opponent(
            candidate["model"],
            str(ROOT / candidate["deck"]),
            opponent["model"],
            str(ROOT / opponent["deck"]),
            num_games=args.games,
            return_details=True,
        )
        wins, losses, draws, *_ = result
        score = wins + 0.5 * draws
        matches.append(
            {
                "candidate": candidate["label"],
                "opponent": opponent["label"],
                "games": args.games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "score": score,
                "score_rate": score / args.games,
                "details": details,
            }
        )
        checkpoint = {
            "schema_version": 1,
            "created_at": created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "complete": False,
            "pool": str(pool_path.relative_to(ROOT)),
            "opponents": str(opponent_path.relative_to(ROOT)),
            "games_per_matchup": args.games,
            "balanced_perspectives": True,
            "planned_matchups": len(matrix),
            "completed_matchups": len(matches),
            "training_probabilities": training_probabilities(pool, load_pool(pool_path)[1]),
            "matches": matches,
        }
        write_report(output, checkpoint)

    summaries = [
        summarize(candidate, [item for item in matches if item["candidate"] == candidate["label"]])
        for candidate in bots
    ]
    summaries.sort(key=lambda item: (item["wilson95_score_lb"], item["worst_matchup_score_rate"]), reverse=True)
    payload = {
        "schema_version": 1,
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "pool": str(pool_path.relative_to(ROOT)),
        "opponents": str(opponent_path.relative_to(ROOT)),
        "games_per_matchup": args.games,
        "balanced_perspectives": True,
        "elapsed_seconds": time.time() - started,
        "planned_matchups": len(matrix),
        "completed_matchups": len(matches),
        "training_probabilities": training_probabilities(pool, load_pool(pool_path)[1]),
        "summaries": summaries,
        "matches": matches,
    }
    write_report(output, payload)
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
