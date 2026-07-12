#!/usr/bin/env python3
"""Promote a validation winner only when it clears explicit regression gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.arena_core import atomic_write_json, utc_now


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, help="JSON written by --best-candidate-file")
    parser.add_argument("--champion-file", default="models/champion.json")
    parser.add_argument("--min-wilson-improvement", type=float, default=0.0)
    parser.add_argument("--max-perspective-gap", type=float, default=0.10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    candidate = selection["summary"]
    if candidate.get("perspective_score_gap", 0.0) > args.max_perspective_gap:
        raise SystemExit("Rejected: perspective gap exceeds gate")
    champion_path = Path(args.champion_file)
    existing = json.loads(champion_path.read_text()) if champion_path.exists() else {}
    old = existing.get("summary", {})
    if candidate["wilson95_score_lb"] < old.get("wilson95_score_lb", 0.0) + args.min_wilson_improvement:
        raise SystemExit("Rejected: Wilson lower bound does not clear champion gate")
    record = {"promoted_at": utc_now(), "candidate": selection["candidate"], "summary": candidate,
              "selection_file": args.selection, "previous_champion": existing.get("candidate")}
    if args.dry_run:
        print(json.dumps(record, indent=2))
    else:
        atomic_write_json(champion_path, record)
        print(f"Promoted champion: {record['candidate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
