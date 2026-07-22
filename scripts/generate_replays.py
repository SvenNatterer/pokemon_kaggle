#!/usr/bin/env python3
"""Unified CLI script to generate match replays across pools, single opponents, or random decks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.generate_replay import generate_replay_batch
from src.utils import resolve_deck_path, resolve_pool_path

DEFAULT_DECK = "decks/deck_bank/bank_38.csv"
DEFAULT_POOL = "kaggle_rule_bots_dev_pool.json"


def find_latest_model() -> Path | None:
    models_dir = ROOT / "models"
    if not models_dir.exists():
        return None

    # Search for all zip files in models/ recursively
    zips = list(models_dir.glob("**/*.zip"))
    if not zips:
        return None

    # Return the most recently modified zip file
    zips.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified replay generation CLI for Pokémon TCG AI experiments."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="",
        help="Path to candidate model .zip file or 'latest' (defaults to latest trained model).",
    )
    parser.add_argument(
        "--deck",
        "-d",
        type=str,
        default=DEFAULT_DECK,
        help="Path to candidate deck CSV (default: decks/deck_bank/bank_38.csv).",
    )
    parser.add_argument(
        "--pool",
        "-p",
        type=str,
        default="",
        help="Path or name of opponent pool JSON (e.g., kaggle_rule_bots_dev_pool.json).",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default="",
        help="Path or specification for single opponent model.",
    )
    parser.add_argument(
        "--deck-b",
        type=str,
        default="",
        help="Path to single opponent deck CSV.",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default="",
        help="Output directory for generated replays (default: replays/<model_name>).",
    )
    parser.add_argument(
        "--num-games",
        "-n",
        type=int,
        default=5,
        help="Number of games to generate if no pool or opponent model is specified.",
    )

    args = parser.parse_args()

    # 1. Resolve candidate model
    model_path: Path | None = None
    if args.model and args.model != "latest":
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
    else:
        latest = find_latest_model()
        if latest:
            model_path = latest
            print(f"Auto-selected latest model: {model_path.relative_to(ROOT)}")
        else:
            print("Error: No candidate model found.", file=sys.stderr)
            return 1

    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}", file=sys.stderr)
        return 1

    # 2. Resolve candidate deck
    resolved_deck = resolve_deck_path(args.deck)
    deck_str = str(resolved_deck) if resolved_deck and resolved_deck.exists() else args.deck

    # 3. Resolve pool path if provided or defaulted
    pool_path_str = ""
    if args.pool:
        resolved_pool = resolve_pool_path(args.pool)
        if resolved_pool and resolved_pool.exists():
            pool_path_str = str(resolved_pool)
        elif Path(args.pool).exists():
            pool_path_str = str(Path(args.pool).resolve())
        else:
            print(f"Warning: Pool file not found at '{args.pool}'.", file=sys.stderr)

    # 4. Resolve output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        out_dir = ROOT / "replays" / model_path.stem

    out_dir.mkdir(parents=True, exist_ok=True)

    # 5. Generate batch replays
    try:
        results = generate_replay_batch(
            model_a_path=str(model_path),
            deck_a_path=deck_str,
            pool_path=pool_path_str if pool_path_str else None,
            model_b_path=args.model_b if args.model_b else None,
            deck_b_path=args.deck_b if args.deck_b else None,
            out_dir=str(out_dir),
            num_games=args.num_games,
        )
        print(f"\nSuccessfully generated {len(results)} replay(s) in {out_dir}")
        return 0
    except Exception as e:
        print(f"Error generating replays: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
