import sys
import os
import json
import subprocess

# Ensure the root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena_utils import (
    run_continuous_match,
    get_elo_ratings
)
import random

def main():
    print("=====================================")
    print("AUTOMATED ARENA BATTLES - ENDLESS ELO")
    print("=====================================")
    
    matches_played = 0
    while True:
        # Run a single random match and update elo
        run_continuous_match()
        matches_played += 1
        
        # Save a replay occasionally (every 15 matches)
        if matches_played % 15 == 0:
            elos = get_elo_ratings()
            if elos:
                best_deck_id = max(elos, key=elos.get)
                # Find an opponent for replay
                active = list(elos.keys())
                opp_id = random.choice([d for d in active if d != best_deck_id] or [best_deck_id])
                
                print(f"\n--- Generating Replay for League Leader (Deck {best_deck_id}) ---")
                replay_dir = "PTCG_ABCS_Visualizer/replays"
                os.makedirs(replay_dir, exist_ok=True)
                out_file = f"{replay_dir}/leader_deck_{best_deck_id}_latest.json"
                
                cmd = [
                    "venv/bin/python", "src/generate_replay.py",
                    "--model-a", f"models/ppo_deck_{best_deck_id}.zip",
                    "--deck-a", f"decks/deck_{best_deck_id}.csv",
                    "--model-b", f"models/ppo_deck_{opp_id}.zip",
                    "--deck-b", f"decks/deck_{opp_id}.csv",
                    "--out", out_file
                ]
                subprocess.run(cmd)

if __name__ == "__main__":
    main()
