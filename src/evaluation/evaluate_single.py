#!/usr/bin/env python3
"""Single matchup evaluation worker process for parallel evaluation runs."""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.league.evaluation import evaluate_vs_baseline, evaluate_vs_opponent

if __name__ == "__main__":
    if len(sys.argv) == 6:
        model_a = sys.argv[1]
        deck_a = sys.argv[2]
        model_b = sys.argv[3]
        deck_b = sys.argv[4]
        num_games = int(sys.argv[5])

        try:
            result, details = evaluate_vs_opponent(
                model_a, deck_a, model_b, deck_b, num_games, return_details=True
            )
            wins, losses, draws, pw1, dw1, bw1, pw2, dw2, bw2 = result
            print(f"RESULT:{wins},{losses},{draws},{pw1},{dw1},{bw1},{pw2},{dw2},{bw2}")
            print(f"DETAIL:{json.dumps(details, sort_keys=True)}")
        except Exception as e:
            print(f"CHILD ERROR: {e}")
    elif len(sys.argv) == 4:
        model_path = sys.argv[1]
        deck_path = sys.argv[2]
        num_games = int(sys.argv[3])

        try:
            wins = evaluate_vs_baseline(model_path, deck_path, num_games)
            print(f"WINS:{wins}")
        except Exception as e:
            print(f"CHILD ERROR: {e}")
    else:
        print("Usage: evaluate_single.py <model_a> <deck_a> <model_b> <deck_b> <games> OR evaluate_single.py <model> <deck> <games>")
        sys.exit(1)
