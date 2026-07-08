import sys
import os

# Ensure the root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena_utils import (
    train_active_decks,
    champion_bonus_training
)

def main():
    iteration = 1
    while True:
        print("=====================================")
        print(f"AUTOMATED TRAINING LOOP - ITERATION {iteration}")
        print("=====================================")
        
        # Train all active decks
        # We pass an empty dict for scores since the leaderboard update inside
        # train_active_decks primarily relies on Elo ratings now.
        train_active_decks({})
        
        # Champion Bonus Training
        champion_bonus_training()
        
        print(f"\nTraining Iteration {iteration} Complete! Starting next iteration...\n")
        iteration += 1

if __name__ == "__main__":
    main()
