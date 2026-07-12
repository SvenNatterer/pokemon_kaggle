#!/usr/bin/env python3
"""Build a validation manifest from models that are not training or final-holdout opponents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_submission import discover_entries, unique_by_deck


ROOT = Path(__file__).resolve().parents[1]


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {str(item["deck_id"]) for item in json.load(handle).get("opponents", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="decks/validation_opponents.json")
    parser.add_argument("--model-pool", default="models")
    parser.add_argument("--final-holdout", default="decks/holdout_opponents.json")
    parser.add_argument("--exclude-deck", action="append", default=[])
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    excluded = load_ids(ROOT / args.final_holdout) | {str(value) for value in args.exclude_deck}
    entries = [entry for entry in unique_by_deck(discover_entries(args.model_pool)) if entry["deck_id"] not in excluded]
    if len(entries) < args.count:
        raise RuntimeError(f"Need {args.count} eligible models, found {len(entries)}. Add models or reduce --count.")
    selected = sorted(entries, key=lambda item: item["label"])[:args.count]
    payload = {
        "version": 1,
        "purpose": "Repeatable model selection only; never train against these exact opponents.",
        "opponents": selected,
    }
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote validation manifest with {len(selected)} opponents: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
