#!/usr/bin/env python3
"""Measure simulation FPS (steps per second) against a rule-based opponent."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure workspace root is in path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.tournament import evaluate_vs_opponent


def measure_matchup(
    label: str,
    model1_path: str,
    deck1_path: str,
    model2_path: str,
    deck2_path: str,
    num_games: int = 10,
):
    print(f"\n==================================================")
    print(f"Matchup: {label}")
    print(f"Player 1 Model: {model1_path}")
    print(f"Player 1 Deck:  {deck1_path}")
    print(f"Player 2 Model: {model2_path}")
    print(f"Player 2 Deck:  {deck2_path}")
    print(f"Games to run:   {num_games}")
    print(f"==================================================")
    print("Running games...")

    start_time = time.perf_counter()
    result, details = evaluate_vs_opponent(
        model1_path,
        deck1_path,
        model2_path,
        deck2_path,
        num_games=num_games,
        return_details=True,
    )
    elapsed = time.perf_counter() - start_time

    wins, losses, draws, *_ = result
    total_turns = details["total_turns"]
    fps = total_turns / elapsed if elapsed > 0 else 0
    gps = num_games / elapsed if elapsed > 0 else 0

    print(f"Finished in {elapsed:.3f} seconds.")
    print(f"Wins: {wins} | Losses: {losses} | Draws: {draws}")
    print(f"Total steps (turns): {total_turns}")
    print(f"Average steps per game: {total_turns / num_games:.1f}")
    print(f"Performance:")
    print(f"  - FPS (Steps per second): {fps:.2f}")
    print(f"  - Games per second:       {gps:.3f}")
    
    return {
        "label": label,
        "elapsed": elapsed,
        "total_turns": total_turns,
        "fps": fps,
        "gps": gps,
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }


def main():
    # 1. PPO Champion vs Rule-Based Opponent
    ppo_model = "models/training_v6/ppo_v5b_deck_bank_18_compute_ft_aux0.zip"
    ppo_deck = "decks/deck_bank/bank_18.csv"
    
    rule_model = "rule_based:aggressive"
    rule_deck = "decks/deck_bank/bank_100.csv"
    
    # 2. Rule-Based Bot vs Rule-Based Opponent
    rule_cand_model = "rule_based:v4:alakazam:engine"
    rule_cand_deck = "decks/deck_bank/bank_28.csv"

    # Verify paths exist
    if not os.path.exists(ppo_model):
        print(f"Error: Champion model not found at {ppo_model}")
        return 1
    if not os.path.exists(ppo_deck):
        print(f"Error: Deck not found at {ppo_deck}")
        return 1
    if not os.path.exists(rule_deck):
        print(f"Error: Deck not found at {rule_deck}")
        return 1
    if not os.path.exists(rule_cand_deck):
        print(f"Error: Deck not found at {rule_cand_deck}")
        return 1

    # Measure PPO vs Rule-Based
    res_ppo = measure_matchup(
        label="PPO Champion vs Rule-Based (Aggressive)",
        model1_path=ppo_model,
        deck1_path=ppo_deck,
        model2_path=rule_model,
        deck2_path=rule_deck,
        num_games=10,
    )

    # Measure Rule-Based vs Rule-Based
    res_rule = measure_matchup(
        label="Rule-Based (Alakazam Engine) vs Rule-Based (Aggressive)",
        model1_path=rule_cand_model,
        deck1_path=rule_cand_deck,
        model2_path=rule_model,
        deck2_path=rule_deck,
        num_games=10,
    )

    print("\n================== SUMMARY ==================")
    print(f"{'Matchup':<45} | {'FPS':<10} | {'Games/sec':<10}")
    print("-" * 73)
    for res in [res_ppo, res_rule]:
        print(f"{res['label']:<45} | {res['fps']:<10.2f} | {res['gps']:<10.3f}")
    print("=============================================")


if __name__ == "__main__":
    sys.exit(main())
