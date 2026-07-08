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

from stable_baselines3.common.monitor import Monitor

def make_env(learning_deck_path, opponent_deck_path, opponent_model_path, reward_config=None):
    def _init():
        learning_deck = read_deck(learning_deck_path)
        opponent_deck = read_deck(opponent_deck_path)
        env = PokemonTCGEnv(my_deck=learning_deck, opponent_deck=opponent_deck, opponent_model_path=opponent_model_path, reward_config=reward_config)
        return Monitor(env)
    return _init

def main():
    parser = argparse.ArgumentParser(description='Train PPO agent against another trained agent.')
    parser.add_argument('--learning-deck', type=str, required=True, help='Path to deck.csv for the learning agent')
    parser.add_argument('--learning-model', type=str, required=True, help='Path to save/load the learning model')
    parser.add_argument('--opponent-deck', type=str, required=True, help='Path to deck.csv for the opponent')
    parser.add_argument('--opponent-model', type=str, required=True, help='Path to load the opponent model')
    parser.add_argument('--timesteps', type=int, default=25000, help='Number of timesteps to train')
    parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--algo', type=str, default='PPO', help='Algorithm to use: PPO or RecurrentPPO')
    parser.add_argument('--ent-start', type=float, default=0.1, help='Starting entropy coefficient (creativity)')
    parser.add_argument('--ent-end', type=float, default=0.01, help='Ending entropy coefficient (creativity)')
    parser.add_argument('--reward-config', type=str, default='{}', help='JSON string containing reward configuration overrides')
    args = parser.parse_args()

    import json
    try:
        reward_config = json.loads(args.reward_config)
    except Exception as e:
        print(f"Error parsing reward config: {e}")
        reward_config = {}

    print(f"Initializing Multi-Agent environment with {args.num_envs} workers...")
    
    # Vectorized environment - using SubprocVecEnv for parallel execution
    env = SubprocVecEnv([make_env(args.learning_deck, args.opponent_deck, args.opponent_model, reward_config) for _ in range(args.num_envs)])

    # Load or create the learning model
    model_path = args.learning_model
    
    from src.custom_ppo import CustomPPO, PokemonTCGRecurrentPolicy
    
    if args.algo == "RecurrentPPO":
        print("Using RecurrentPPO (CustomPPO) with LSTM memory...")
        algo_class = CustomPPO
        policy = PokemonTCGRecurrentPolicy
    else:
        print("Using standard PPO (Feedforward)...")
        algo_class = PPO
        policy = 'MultiInputPolicy'
        
    if os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing learning model from {model_path}.zip...")
        model = algo_class.load(model_path, env=env, tensorboard_log="logs/")
        # Update ent_coef on loaded model
        model.ent_coef = args.ent_start
    else:
        print(f"Creating new {args.algo} model for learning agent...")
        kwargs = {
            "policy": policy,
            "env": env,
            "verbose": 1,
            "ent_coef": args.ent_start,
            "tensorboard_log": "logs/",
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 1024,
            "n_epochs": 3
        }
        if args.algo == "RecurrentPPO":
            kwargs["c_aux"] = 0.5
            
        model = algo_class(**kwargs)

    from stable_baselines3.common.callbacks import BaseCallback
    import json
    
    class ProgressCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.progress_file = "decks/training_progress.json"
            
        def _on_step(self) -> bool:
            if hasattr(self.model, 'ent_coef'):
                # Calculate progress and apply linear decay to ent_coef
                total = self.locals.get("total_timesteps", 1)
                current = self.num_timesteps
                progress_made = min(1.0, current / total)
                new_ent_coef = args.ent_start - progress_made * (args.ent_start - args.ent_end)
                self.model.ent_coef = new_ent_coef

            if self.n_calls % 10 == 0:
                current_steps = self.n_calls * self.training_env.num_envs
                with open(self.progress_file, "w") as f:
                    json.dump({
                        "current": current_steps,
                        "total": self.locals.get("total_timesteps", 0),
                        "status": "running"
                    }, f)
            return True
            
        def _on_training_end(self) -> None:
            with open(self.progress_file, "w") as f:
                json.dump({
                    "current": self.locals.get("total_timesteps", 0),
                    "total": self.locals.get("total_timesteps", 0),
                    "status": "finished"
                }, f)

    # Initialize progress file at start
    with open("decks/training_progress.json", "w") as f:
        json.dump({"current": 0, "total": args.timesteps, "status": "running"}, f)

    print(f"Starting Multi-Agent training for {args.timesteps} timesteps...")
    print(f"Learning Deck: {args.learning_deck} vs Opponent Deck: {args.opponent_deck}")
    
    model.learn(total_timesteps=args.timesteps, callback=ProgressCallback())

    print("Training finished! Saving learning model...")
    model.save(model_path)

if __name__ == "__main__":
    main()
