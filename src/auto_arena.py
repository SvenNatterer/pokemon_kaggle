import sys
import os
import json
import subprocess
import datetime

# Ensure the root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena_utils import (
    run_continuous_match,
    get_elo_ratings
)
from src.model_paths import parse_deck_model_path
import random

WATCHED_FILE = "decks/watched_models.json"

def get_watched_models():
    """Load the list of models the user wants replays for."""
    try:
        if os.path.exists(WATCHED_FILE):
            with open(WATCHED_FILE, "r") as f:
                return json.load(f).get("watched", [])
    except Exception:
        pass
    return []

def generate_replay_for(model_id, active_elos):
    """Generate a replay for a given model_id against a random active opponent."""
    model_a = f"models/{model_id}.zip"
    if not os.path.exists(model_a):
        print(f"Skipping replay for {model_id} – model file not found.")
        return

    parsed_a = parse_deck_model_path(model_a)
    if not parsed_a:
        print(f"Skipping replay for {model_id} – cannot parse deck id.")
        return

    active = list(active_elos.keys())
    opp_id = random.choice([d for d in active if d != model_id] or [model_id])
    model_b = f"models/{opp_id}.zip"
    parsed_b = parse_deck_model_path(model_b)

    if not parsed_b or not os.path.exists(model_b):
        print(f"Skipping replay for {model_id} – opponent {opp_id} not found.")
        return

    deck_num_a = parsed_a["deck_id"]
    deck_num_b = parsed_b["deck_id"]
    csv_a = f"decks/deck_bank/{deck_num_a}.csv" if str(deck_num_a).startswith("bank_") else f"decks/deck_{deck_num_a}.csv"
    csv_b = f"decks/deck_bank/{deck_num_b}.csv" if str(deck_num_b).startswith("bank_") else f"decks/deck_{deck_num_b}.csv"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    replay_dir = f"replays/{model_id}"
    os.makedirs(replay_dir, exist_ok=True)
    out_file = f"{replay_dir}/watched_{model_id}_vs_{opp_id}_{timestamp}.json"

    print(f"\n--- Generating Watched Replay: {model_id} vs {opp_id} ---")
    cmd = [
        sys.executable, "src/generate_replay.py",
        "--model-a", model_a, "--deck-a", csv_a,
        "--model-b", model_b, "--deck-b", csv_b,
        "--out", out_file
    ]
    subprocess.run(cmd)

def main():
    print("=====================================")
    print("AUTOMATED ARENA BATTLES - ENDLESS ELO")
    print("=====================================")

    matches_played = 0
    watched_cycle_index = 0  # Round-robin index through watched models

    while True:
        run_continuous_match()
        matches_played += 1

        # Generate a replay every 15 matches
        if matches_played % 15 == 0:
            elos = get_elo_ratings()
            if not elos:
                continue

            watched = get_watched_models()

            if watched:
                # Round-robin through watched models
                watched_cycle_index = watched_cycle_index % len(watched)
                target = watched[watched_cycle_index]
                watched_cycle_index += 1
                generate_replay_for(target, elos)
            else:
                # Fallback: use best model (original behavior)
                best_deck_id = max(elos, key=elos.get)
                generate_replay_for(best_deck_id, elos)

if __name__ == "__main__":
    main()
