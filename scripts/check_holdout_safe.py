#!/usr/bin/env python3
"""Fail closed when a training opponent overlaps the frozen holdout set."""

import argparse
import json
import os
import re
import sys


DECK_FILE_RE = re.compile(r"^(?:deck_)?(?P<deck_id>(?:bank_)?\d+)$")
MODEL_FILE_RE = re.compile(
    r"^ppo(?:_belief|_v4|_v5b?|_v6)?_deck_(?P<deck_id>(?:bank_)?\d+)(?:_.*)?$"
)


def normalized_path(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def deck_id_from_path(path, model=False):
    stem = os.path.splitext(os.path.basename(path))[0]
    match = (MODEL_FILE_RE if model else DECK_FILE_RE).match(stem)
    return match.group("deck_id") if match else None


def load_holdout(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    opponents = payload.get("opponents", [])
    if not isinstance(opponents, list) or not opponents:
        raise ValueError(f"Holdout file has no opponents: {path}")
    return opponents


def check_paths(holdout_file, decks, models, pools):
    opponents = load_holdout(holdout_file)
    holdout_ids = {str(entry["deck_id"]) for entry in opponents}
    holdout_decks = {
        normalized_path(entry["deck_path"])
        for entry in opponents
        if entry.get("deck_path")
    }
    holdout_models = {
        normalized_path(entry["model_path"])
        for entry in opponents
        if entry.get("model_path")
    }

    pool_entries = []
    for pool_path in pools:
        with open(pool_path, "r", encoding="utf-8") as handle:
            pool = json.load(handle)
        if not isinstance(pool, list):
            raise ValueError(f"Opponent pool must be a JSON list: {pool_path}")
        for index, entry in enumerate(pool):
            if not isinstance(entry, dict) or not entry.get("deck"):
                raise ValueError(f"Invalid entry {index} in opponent pool {pool_path}")
            pool_entries.append((entry["deck"], entry.get("model")))

    for deck_path, model_path in pool_entries:
        decks.append(deck_path)
        if model_path:
            models.append(model_path)

    violations = []
    for deck_path in decks:
        deck_id = deck_id_from_path(deck_path)
        if normalized_path(deck_path) in holdout_decks or deck_id in holdout_ids:
            violations.append(f"holdout deck: {deck_path} (id={deck_id})")

    for model_path in models:
        model_id = deck_id_from_path(model_path, model=True)
        normalized = normalized_path(model_path)
        path_parts = set(os.path.normpath(model_path).split(os.sep))
        if (
            normalized in holdout_models
            or model_id in holdout_ids
            or "holdout" in path_parts
        ):
            violations.append(f"holdout model: {model_path} (id={model_id})")

    if violations:
        raise ValueError("Unsafe training opponents:\n  - " + "\n  - ".join(violations))

    return holdout_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-file", default="decks/holdout_opponents.json")
    parser.add_argument("--deck", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--pool", action="append", default=[])
    args = parser.parse_args()

    try:
        holdout_ids = check_paths(
            args.holdout_file,
            list(args.deck),
            list(args.model),
            list(args.pool),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"HOLDOUT CHECK FAILED: {error}", file=sys.stderr)
        return 2

    print(
        "Holdout-safe. Reserved opponent IDs: "
        + ", ".join(sorted(holdout_ids))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
