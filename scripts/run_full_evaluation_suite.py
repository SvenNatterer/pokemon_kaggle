#!/usr/bin/env python3
"""Run evaluation suite across all checkpoints on validation and holdout sets, both with and without lookahead."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = str(ROOT / "venv" / "bin" / "python3")

CHECKPOINTS = [
    ("models/ppo_v6_exp016_7m_pfsp_league.zip", "decks/deck_bank/bank_38.csv"),
    ("models/ppo_v6_exp016_4m_pfsp_league.zip", "decks/deck_bank/bank_38.csv"),
    ("models/ppo_v6_exp016_2m_pfsp_league.zip", "decks/deck_bank/bank_38.csv"),
    ("models/ppo_v6_exp016_1m_pfsp_league.zip", "decks/deck_bank/bank_38.csv"),
    ("models/ppo_v6_exp014_2m_inference_gameplan.zip", "decks/deck_bank/bank_38.csv"),
    ("models/ppo_v6_exp014_1m_inference_gameplan.zip", "decks/deck_bank/bank_38.csv"),
    ("models/ppo_v6_exp012_scaled_resnet_mlp.zip", "decks/deck_bank/bank_38.csv"),
    ("models/ppo_v6_exp007_prize_archetype.zip", "decks/deck_bank/bank_38.csv"),
]

POOLS = [
    ("validation", "decks/pools/validation_opponents.json"),
    ("holdout", "decks/pools/holdout_opponents.json"),
]


def main():
    os.makedirs(ROOT / "evaluation_results", exist_ok=True)
    os.makedirs(ROOT / "replays", exist_ok=True)

    print("=== Starting Full Evaluation Suite ===")
    print(f"Total checkpoints: {len(CHECKPOINTS)}")
    print(f"Pools: {[p[0] for p in POOLS]}")

    for pool_name, pool_file in POOLS:
        for lookahead in [False, True]:
            mode_str = "with_lookahead" if lookahead else "no_lookahead"
            out_json = ROOT / "evaluation_results" / f"{pool_name}_{mode_str}.json"
            print(f"\n========================================================")
            print(f"Running Evaluation: Pool={pool_name.upper()}, Lookahead={lookahead}")
            print(f"Output File: {out_json}")
            print(f"========================================================")

            cmd = [
                PYTHON_BIN,
                "scripts/evaluate_submission.py",
                "--holdout-file", pool_file,
                "--results-file", str(out_json),
                "--games", "10",
                "--workers", str(min(8, os.cpu_count() or 4)),
            ]
            if lookahead:
                cmd.append("--lookahead")

            for model_path, deck_path in CHECKPOINTS:
                if os.path.exists(ROOT / model_path):
                    cmd.extend(["--candidate", str(model_path), "--candidate-deck", str(deck_path)])

            result = subprocess.run(cmd, cwd=str(ROOT))
            if result.returncode != 0:
                print(f"[ERROR] Evaluation failed for pool {pool_name} (lookahead={lookahead})")

    # Generate Replays for selected key checkpoints
    print("\n========================================================")
    print("Generating Replays for Checkpoints (with & without lookahead)")
    print("========================================================")

    replay_checkpoints = [
        "models/ppo_v6_exp016_7m_pfsp_league.zip",
        "models/ppo_v6_exp016_4m_pfsp_league.zip",
        "models/ppo_v6_exp014_2m_inference_gameplan.zip",
    ]

    for model_path in replay_checkpoints:
        if not os.path.exists(ROOT / model_path):
            continue
        for lookahead in [False, True]:
            cmd = [
                PYTHON_BIN,
                "scripts/generate_replays.py",
                "--model", str(model_path),
                "--pool", "kaggle_rule_bots_dev_pool.json",
            ]
            if lookahead:
                cmd.append("--lookahead")
            print(f"\nGenerating replays for {model_path} (lookahead={lookahead})...")
            subprocess.run(cmd, cwd=str(ROOT))

    print("\n=== Full Evaluation & Replay Generation Suite Finished! ===")


if __name__ == "__main__":
    main()
