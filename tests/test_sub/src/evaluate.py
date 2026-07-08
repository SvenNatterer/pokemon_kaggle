import os
import sys

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
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
        if reward > 0:
            print(f"Episode {ep+1}: Win!")
            wins += 1
        elif reward < 0:
            print(f"Episode {ep+1}: Loss")
        else:
            print(f"Episode {ep+1}: Draw")
            
    print(f"\nEvaluation finished! Win rate: {wins/num_episodes*100:.2f}%")

if __name__ == "__main__":
    evaluate()
