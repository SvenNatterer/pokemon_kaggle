#!/usr/bin/env python3
"""Tune interpretable rule-bot coefficients through capped league co-evolution."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.rule_based_policy import RuleParameters, normalize_archetype
from src.league.tournament import evaluate_vs_opponent


DEFAULT_POOL = ROOT / "decks" / "rule_bot_meta_pool_v1.json"
DEFAULT_PPO_POOL = ROOT / "decks" / "opponent_factory_v6_development_pool.json"

TUNABLE_BOUNDS = {
    "attack_damage_scale": (0.06, 0.16),
    "attack_knockout": (20.0, 55.0),
    "attack_prize": (8.0, 24.0),
    "attack_win_game": (60.0, 140.0),
    "attack_board_empty_penalty": (12.0, 45.0),
    "basic_empty_bench_bonus": (18.0, 55.0),
    "basic_thin_bench_bonus": (5.0, 24.0),
    "attach_one_away": (20.0, 50.0),
    "retreat_damage_weight": (12.0, 40.0),
    "deckout_penalty": (30.0, 90.0),
    "damage_counter_knockout": (50.0, 130.0),
    "damage_counter_prize": (10.0, 40.0),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archetype", required=True)
    parser.add_argument("--variant", choices=("balanced", "tempo", "engine", "control"), default="balanced")
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--rule-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--ppo-pool", type=Path, default=DEFAULT_PPO_POOL)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--rule-opponents", type=int, default=4)
    parser.add_argument("--ppo-opponents", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _load_entries(path: Path, key: str | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get(key) if key and isinstance(payload, dict) else payload
    if key == "bots":
        return [
            {"label": item["label"], "model": item["model"], "deck": item["deck"], "kind": "rule"}
            for item in entries or []
        ]
    return [
        {
            "label": item["label"],
            "model": item.get("model") or item.get("model_path"),
            "deck": item.get("deck") or item.get("deck_path"),
            "kind": "rule" if str(item.get("model") or item.get("model_path", "")).startswith("rule") else "ppo",
        }
        for item in entries or []
        if item.get("model") or item.get("model_path")
    ]


def _candidate_spec(archetype: str, variant: str, parameters: dict[str, float]) -> str:
    query = urlencode(sorted((key, f"{value:.8g}") for key, value in parameters.items()))
    return f"rule_based:v4:{archetype}:{variant}?{query}"


def _initial_distribution() -> tuple[dict[str, float], dict[str, float]]:
    defaults = asdict(RuleParameters())
    mean = {key: float(defaults[key]) for key in TUNABLE_BOUNDS}
    std = {key: (high - low) * 0.18 for key, (low, high) in TUNABLE_BOUNDS.items()}
    return mean, std


def _sample(rng: random.Random, mean: dict[str, float], std: dict[str, float]) -> dict[str, float]:
    result = {}
    for key, (low, high) in TUNABLE_BOUNDS.items():
        result[key] = min(high, max(low, rng.gauss(mean[key], std[key])))
    return result


def _score_candidate(
    spec: str,
    deck: Path,
    opponents: list[dict[str, Any]],
    games: int,
) -> tuple[float, list[dict[str, Any]]]:
    matches = []
    rule_scores = []
    ppo_scores = []
    perspective_gaps = []
    avoidable_losses = 0
    for opponent in opponents:
        result, details = evaluate_vs_opponent(
            spec,
            str(deck),
            opponent["model"],
            str(ROOT / opponent["deck"]),
            num_games=games,
            return_details=True,
        )
        wins, losses, draws, *_ = result
        score_rate = (wins + 0.5 * draws) / games
        target = rule_scores if opponent["kind"] == "rule" else ppo_scores
        target.append(score_rate)
        perspectives = details.get("perspective", {})
        rates = []
        for values in perspectives.values():
            count = int(values.get("games", 0))
            if count:
                rates.append((float(values.get("wins", 0)) + 0.5 * float(values.get("draws", 0))) / count)
        if len(rates) == 2:
            perspective_gaps.append(abs(rates[0] - rates[1]))
        reasons = details.get("opponent_win_reasons", {})
        avoidable_losses += int(reasons.get("deckout", 0)) + int(reasons.get("benchout", 0))
        matches.append(
            {
                "opponent": opponent["label"],
                "kind": opponent["kind"],
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "score_rate": score_rate,
                "perspective_gap": abs(rates[0] - rates[1]) if len(rates) == 2 else None,
                "loss_reasons": reasons,
            }
        )
    all_scores = rule_scores + ppo_scores
    rule_mean = sum(rule_scores) / len(rule_scores) if rule_scores else 0.0
    ppo_mean = sum(ppo_scores) / len(ppo_scores) if ppo_scores else rule_mean
    worst = min(all_scores, default=0.0)
    gap = sum(perspective_gaps) / len(perspective_gaps) if perspective_gaps else 0.0
    loss_rate = avoidable_losses / max(1, games * len(opponents))
    # External PPO strength prevents a closed rule-only meta. Worst-matchup and
    # safety terms prevent cyclic specialists from winning the average only.
    fitness = 0.40 * ppo_mean + 0.30 * rule_mean + 0.20 * worst - 0.06 * gap - 0.04 * loss_rate
    return fitness, matches


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if min(args.generations, args.population, args.games) <= 0:
        raise SystemExit("generations, population and games must be positive")
    archetype = normalize_archetype(args.archetype)
    deck = args.deck.expanduser().resolve()
    if not deck.is_file():
        raise SystemExit(f"deck does not exist: {deck}")
    if args.smoke:
        args.generations = 1
        args.population = 2
        args.games = 2
        args.rule_opponents = 1
        args.ppo_opponents = 1

    rule_entries = _load_entries(args.rule_pool.expanduser().resolve(), "bots")
    ppo_entries = _load_entries(args.ppo_pool.expanduser().resolve(), None)
    rng = random.Random(args.seed)
    rng.shuffle(rule_entries)
    rng.shuffle(ppo_entries)
    opponents = rule_entries[: args.rule_opponents] + [
        item for item in ppo_entries if item["kind"] == "ppo"
    ][: args.ppo_opponents]
    if not opponents:
        raise SystemExit("no tuning opponents selected")

    output = args.output or ROOT / "reports" / f"rule_tuning_{archetype}_{args.variant}.json"
    print(
        f"Rule tuning: {archetype}/{args.variant}, {args.population} candidates × "
        f"{args.generations} generations, {len(opponents)} opponents × {args.games} games"
    )
    if args.dry_run:
        print(json.dumps({"deck": str(deck), "opponents": opponents, "bounds": TUNABLE_BOUNDS}, indent=2))
        return 0

    mean, std = _initial_distribution()
    generations = []
    hall_of_fame = []
    for generation in range(args.generations):
        candidates = []
        sampled = [dict(mean)] + [
            _sample(rng, mean, std) for _ in range(max(0, args.population - 1))
        ]
        active_opponents = list(opponents)
        if hall_of_fame:
            active_opponents.append(hall_of_fame[-1])
        for index, parameters in enumerate(sampled, start=1):
            spec = _candidate_spec(archetype, args.variant, parameters)
            print(f"generation {generation + 1}/{args.generations}, candidate {index}/{len(sampled)}")
            fitness, matches = _score_candidate(spec, deck, active_opponents, args.games)
            candidates.append({"fitness": fitness, "spec": spec, "parameters": parameters, "matches": matches})
        candidates.sort(key=lambda item: item["fitness"], reverse=True)
        elite_count = max(2, math.ceil(len(candidates) * 0.25))
        elite = candidates[:elite_count]
        for key, (low, high) in TUNABLE_BOUNDS.items():
            values = [item["parameters"][key] for item in elite]
            mean[key] = sum(values) / len(values)
            variance = sum((value - mean[key]) ** 2 for value in values) / len(values)
            std[key] = max((high - low) * 0.025, math.sqrt(variance) * 0.90)
        best = candidates[0]
        hall_of_fame.append(
            {
                "label": f"hall_of_fame_generation_{generation + 1}",
                "model": best["spec"],
                "deck": str(deck.relative_to(ROOT)),
                "kind": "rule",
            }
        )
        generations.append({"generation": generation + 1, "mean": dict(mean), "std": dict(std), "candidates": candidates})

    best = max(
        (candidate for generation in generations for candidate in generation["candidates"]),
        key=lambda item: item["fitness"],
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archetype": archetype,
        "variant": args.variant,
        "deck": str(deck.relative_to(ROOT)),
        "seed": args.seed,
        "games_per_matchup": args.games,
        "opponents": opponents,
        "fitness": "40% PPO mean + 30% rule mean + 20% worst matchup - perspective/deckout/benchout penalties",
        "best": best,
        "hall_of_fame": hall_of_fame,
        "generations": generations,
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Best spec: {best['spec']}")
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
