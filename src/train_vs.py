import argparse
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Add src to pythonpath so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env_wrapper import PokemonTCGEnv

def read_deck(deck_path):
    import pandas as pd
    df = pd.read_csv(deck_path, header=None)
    return df[0].tolist()

def make_env(learning_deck_path, opponent_deck_path, opponent_model_path):
    def _init():
        learning_deck = read_deck(learning_deck_path)
        opponent_deck = read_deck(opponent_deck_path)
        return PokemonTCGEnv(my_deck=learning_deck, opponent_deck=opponent_deck, opponent_model_path=opponent_model_path)
    return _init

def main():
    parser = argparse.ArgumentParser(description='Train PPO agent against another trained agent.')
    parser.add_argument('--learning-deck', type=str, required=True, help='Path to deck.csv for the learning agent')
    parser.add_argument('--learning-model', type=str, required=True, help='Path to save/load the learning model')
    parser.add_argument('--opponent-deck', type=str, required=True, help='Path to deck.csv for the opponent')
    parser.add_argument('--opponent-model', type=str, required=True, help='Path to load the opponent model')
    parser.add_argument('--timesteps', type=int, default=25000, help='Number of timesteps to train')
    parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel environments')
    args = parser.parse_args()

    print(f"Initializing Multi-Agent environment with {args.num_envs} workers...")
    
    # Vectorized environment - using SubprocVecEnv for parallel execution
    env = SubprocVecEnv([make_env(args.learning_deck, args.opponent_deck, args.opponent_model) for _ in range(args.num_envs)])

    # Load or create the learning model
    model_path = args.learning_model
    if os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing learning model from {model_path}.zip...")
        model = PPO.load(model_path, env=env)
    else:
        print("Creating new PPO model for learning agent...")
        model = PPO('MultiInputPolicy', env, verbose=1)

    print(f"Starting Multi-Agent training for {args.timesteps} timesteps...")
    print(f"Learning Deck: {args.learning_deck} vs Opponent Deck: {args.opponent_deck}")
    
    model.learn(total_timesteps=args.timesteps)

    print("Training finished! Saving learning model...")
    model.save(model_path)

if __name__ == "__main__":
    main()
