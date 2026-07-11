#!/usr/bin/env python3
"""Create a manual model cleanup plan and optionally move stale models."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import sys
from typing import Any


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.model_paths import parse_deck_model_path


REFERENCE_MODELS = {
    "ppo_v4_deck_7",
    "ppo_v4_deck_bank_47",
    "ppo_deck_1",
}
REPORT_FILE = "decks/manual_filter_report.json"
REFERENCE_RESULTS_FILE = "decks/reference_eval_results.json"
HOLDOUT_FILE = "decks/holdout_opponents.json"
STATS_FILES = {
    "elo": "decks/elo_ratings.json",
    "games": "decks/games_played.json",
}


def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def safe_move(src: str, dst_dir: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(src)
    dst = os.path.join(dst_dir, base)
    if not os.path.exists(dst):
        shutil.move(src, dst)
        return dst

    stem, ext = os.path.splitext(base)
    counter = 1
    while True:
        candidate = os.path.join(dst_dir, f"{stem}_{counter}{ext}")
        if not os.path.exists(candidate):
            shutil.move(src, candidate)
            return candidate
        counter += 1


def model_entry(path: str, elos: dict[str, float], games: dict[str, int]) -> dict[str, Any] | None:
    parsed = parse_deck_model_path(path)
    if parsed is None:
        return None

    name = parsed["name"]
    try:
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
    except OSError:
        return None

    return {
        "name": name,
        "path": path,
        "deck_id": parsed["deck_id"],
        "variant": parsed["variant"],
        "mtime": mtime,
        "size": size,
        "elo": elos.get(name),
        "games": games.get(name, 0),
    }


def load_active_models(model_dir: str, elos: dict[str, float], games: dict[str, int]) -> list[dict[str, Any]]:
    entries = []
    for path in glob.glob(os.path.join(model_dir, "*.zip")):
        entry = model_entry(path, elos, games)
        if entry is not None:
            entries.append(entry)
    return sorted(entries, key=lambda item: item["mtime"], reverse=True)


def holdout_model_names(path: str) -> set[str]:
    data = load_json(path, {})
    names = set()
    for opponent in data.get("opponents", []):
        model_path = opponent.get("model_path", "")
        if model_path:
            names.add(os.path.splitext(os.path.basename(model_path))[0])
    return names


def reference_eval_by_model(path: str) -> dict[str, dict[str, Any]]:
    data = load_json(path, {})
    rows = {}
    for row in data.get("summary", []):
        candidate = row.get("candidate")
        if candidate:
            rows[candidate] = row
    return rows


def best_reference_eval_per_deck(rows: dict[str, dict[str, Any]]) -> set[str]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows.values():
        deck_id = row.get("candidate_deck_id")
        if not deck_id:
            continue
        current = best.get(deck_id)
        key = (
            float(row.get("holdout_fit", -999.0)),
            float(row.get("worst_score_rate", -1.0)),
            float(row.get("score_rate", -1.0)),
        )
        current_key = (
            float(current.get("holdout_fit", -999.0)),
            float(current.get("worst_score_rate", -1.0)),
            float(current.get("score_rate", -1.0)),
        ) if current else None
        if current is None or key > current_key:
            best[deck_id] = row
    return {row["candidate"] for row in best.values() if row.get("candidate")}


def model_rank(entry: dict[str, Any], ref_rows: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    ref = ref_rows.get(entry["name"])
    if ref:
        return (
            4,
            float(ref.get("holdout_fit", -999.0)),
            float(ref.get("worst_score_rate", -1.0)),
            float(ref.get("score_rate", -1.0)),
            entry["mtime"],
        )
    if not entry["variant"]:
        return (3, entry["elo"] if entry["elo"] is not None else 0.0, entry["games"], entry["mtime"])
    if entry["variant"].startswith("_checkpoint_"):
        return (2, entry["mtime"])
    return (1, entry["mtime"])


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    elos = load_json(STATS_FILES["elo"], {})
    games = load_json(STATS_FILES["games"], {})
    ref_rows = reference_eval_by_model(args.reference_results)
    move_reference_bands = set(args.move_reference_band)
    holdout_names = holdout_model_names(args.holdout_file)
    ref_best_names = {
        name for name in best_reference_eval_per_deck(ref_rows)
        if ref_rows.get(name, {}).get("band") not in move_reference_bands
    }
    explicit_protect = {os.path.splitext(os.path.basename(path))[0] for path in args.protect}
    protected_names = set(REFERENCE_MODELS) | holdout_names | ref_best_names | explicit_protect

    entries = load_active_models(args.model_dir, elos, games)
    by_deck: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_deck.setdefault(entry["deck_id"], []).append(entry)

    keep_names = set(protected_names)
    for deck_entries in by_deck.values():
        eligible = [
            entry for entry in deck_entries
            if ref_rows.get(entry["name"], {}).get("band") not in move_reference_bands
        ]
        ranked = sorted(eligible, key=lambda item: model_rank(item, ref_rows), reverse=True)
        keep_names.update(entry["name"] for entry in ranked[: args.keep_per_deck])

    actions = []
    for entry in entries:
        ref = ref_rows.get(entry["name"], {})
        reason = ""
        action = "keep"
        protected = entry["name"] in protected_names

        if protected:
            reason = "protected"
        elif ref.get("band") in move_reference_bands:
            action = "move"
            reason = f"reference_band:{ref.get('band')}"
        elif entry["name"] in keep_names:
            reason = "best_per_deck"
        elif entry["games"] < args.min_games and entry["name"] not in ref_rows and not args.archive_untested:
            reason = "untested"
        else:
            action = "move"
            reason = "extra_variant_or_lower_rank"

        actions.append(
            {
                "action": action,
                "reason": reason,
                "name": entry["name"],
                "path": entry["path"],
                "deck_id": entry["deck_id"],
                "variant": entry["variant"],
                "games": entry["games"],
                "elo": entry["elo"],
                "reference_score_rate": ref.get("score_rate"),
                "reference_worst_score_rate": ref.get("worst_score_rate"),
                "reference_holdout_fit": ref.get("holdout_fit"),
            }
        )

    return {
        "created_at": int(dt.datetime.now().timestamp()),
        "model_dir": args.model_dir,
        "destination": args.destination,
        "protected_names": sorted(protected_names),
        "actions": actions,
    }


def print_plan(plan: dict[str, Any]) -> None:
    actions = plan["actions"]
    keep = [row for row in actions if row["action"] == "keep"]
    move = [row for row in actions if row["action"] == "move"]

    print(f"Models scanned: {len(actions)}")
    print(f"Keep: {len(keep)}")
    print(f"Move: {len(move)}")
    print(f"Destination: {plan['destination']}")

    print("\nKeep")
    print("----")
    for row in keep:
        print(f"{row['name']:42s} deck={row['deck_id']:8s} reason={row['reason']}")

    print("\nMove candidates")
    print("---------------")
    for row in move:
        print(
            f"{row['name']:42s} deck={row['deck_id']:8s} "
            f"games={row['games']:4d} reason={row['reason']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reference-results", default=REFERENCE_RESULTS_FILE)
    parser.add_argument("--holdout-file", default=HOLDOUT_FILE)
    parser.add_argument("--report-file", default=REPORT_FILE)
    parser.add_argument("--keep-per-deck", type=int, default=1)
    parser.add_argument("--min-games", type=int, default=100)
    parser.add_argument("--archive-untested", action="store_true")
    parser.add_argument(
        "--move-reference-band",
        action="append",
        default=[],
        help="Move models whose reference-eval band matches this value, e.g. too_weak or weak_style.",
    )
    parser.add_argument("--protect", action="append", default=[])
    parser.add_argument("--destination", default="")
    parser.add_argument("--to-backup", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not args.destination:
        if args.to_backup:
            args.destination = "models/backup"
        else:
            today = dt.datetime.now().strftime("%Y%m%d")
            args.destination = f"models/archive_{today}"

    plan = build_plan(args)
    print_plan(plan)

    os.makedirs(os.path.dirname(args.report_file), exist_ok=True)
    with open(args.report_file, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
    print(f"\nWrote report to {args.report_file}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to move the listed candidates.")
        return 0

    moved = []
    for row in plan["actions"]:
        if row["action"] != "move":
            continue
        if not os.path.exists(row["path"]):
            continue
        dst = safe_move(row["path"], args.destination)
        moved.append((row["path"], dst))

    print(f"\nMoved {len(moved)} models:")
    for src, dst in moved:
        print(f"  {src} -> {dst}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
