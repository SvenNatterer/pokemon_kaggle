import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from src.env_wrapper import PokemonTCGEnv, read_sample_deck
from src.agents.baseline_agent import RandomAgent

def evaluate():
    deck = read_sample_deck()
    env = PokemonTCGEnv(deck, deck)
    
    # Check if model exists
    model_path = "models/ppo_pokemon_final.zip"
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        model = PPO.load(model_path)
    else:
        print("Trained model not found. Using random agent for testing.")
        model = RandomAgent(env)
        
    num_episodes = 5
    wins = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        while not done:
            try:
                action, lstm_state = model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start = np.zeros((1,), dtype=bool)
            except TypeError:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
        winner = info.get("winner", -1)
        if winner == 0:
            print(f"Episode {ep+1}: Win!")
            wins += 1
        elif winner == 1:
            print(f"Episode {ep+1}: Loss")
        else:
            print(f"Episode {ep+1}: Draw")
            
    print(f"\nEvaluation finished! Win rate: {wins/num_episodes*100:.2f}%")

if __name__ == "__main__":
    evaluate()
