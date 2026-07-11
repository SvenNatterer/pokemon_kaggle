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

def resolve_model_path(model_name):
    model_path = model_name if os.path.dirname(model_name) else os.path.join("models", model_name)
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    return model_path

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
    parser.add_argument('--algo', type=str, default='RecurrentPPO', choices=['RecurrentPPO', 'PPO'], help='Algorithm to use. RecurrentPPO applies action masks; PPO does not.')
    parser.add_argument('--ent-start', type=float, default=0.02, help='Starting entropy coefficient (creativity)')
    parser.add_argument('--ent-end', type=float, default=0.005, help='Ending entropy coefficient (creativity)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n-epochs', type=int, default=2, help='PPO epochs per rollout')
    parser.add_argument('--clip-range', type=float, default=0.1, help='PPO clipping range')
    parser.add_argument('--target-kl', type=float, default=0.05, help='Stop PPO update early above this KL')
    parser.add_argument('--reward-config', type=str, default='{}', help='JSON string containing reward configuration overrides')
    parser.add_argument('--aux-coef', type=float, default=0.5, help='Weight for hidden-card auxiliary loss')
    parser.add_argument('--belief-actor', action='store_true', help='Feed the learned hidden-card belief embedding into the actor')
    parser.add_argument('--belief-dim', type=int, default=64, help='Size of the learned belief embedding used by --belief-actor')
    parser.add_argument('--no-belief-detach', dest='belief_detach', action='store_false', help='Allow PPO loss gradients into the belief encoder')
    parser.set_defaults(belief_detach=True)
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
    from src.custom_policy import PokemonTCGFeatureExtractor
    policy_kwargs = dict(features_extractor_class=PokemonTCGFeatureExtractor)
    
    if args.algo == "RecurrentPPO":
        print("Using RecurrentPPO (CustomPPO) with LSTM memory...")
        algo_class = CustomPPO
        policy = PokemonTCGRecurrentPolicy
        policy_kwargs.update({
            "use_belief_actor": args.belief_actor,
            "belief_dim": args.belief_dim,
            "detach_belief_actor": args.belief_detach,
        })
    else:
        print("Using standard PPO (Feedforward, no action masking). RecurrentPPO is recommended for this env.")
        algo_class = PPO
        policy = 'MultiInputPolicy'
        
    model_path = resolve_model_path(model_path)
        
    model = None
    if os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing learning model from {model_path}.zip...")
        try:
            model = algo_class.load(model_path, env=env, tensorboard_log="logs/")
        except Exception as e:
            print(f"Could not load {model_path}.zip as {args.algo}: {e}")
            print("Creating a fresh model with the selected algorithm instead.")

        if model is not None:
            loaded_belief_actor = bool(getattr(getattr(model, "policy", None), "use_belief_actor", False))
            if args.belief_actor and not loaded_belief_actor:
                raise RuntimeError(
                    "--belief-actor was requested, but the existing checkpoint uses the legacy actor. "
                    "Use a fresh --learning-model for the belief-actor experiment."
                )
            if loaded_belief_actor and not args.belief_actor:
                print("Loaded a belief-actor checkpoint; continuing with its saved architecture.")
            # Update parameters on loaded model
            if hasattr(model, "c_aux"):
                model.c_aux = args.aux_coef
            model.ent_coef = args.ent_start
            model.learning_rate = args.lr
            from stable_baselines3.common.utils import get_schedule_fn
            model.lr_schedule = get_schedule_fn(args.lr)
            model.clip_range = get_schedule_fn(args.clip_range)
            model.target_kl = args.target_kl
            model.n_epochs = args.n_epochs
            if hasattr(model, 'policy') and hasattr(model.policy, 'optimizer'):
                for param_group in model.policy.optimizer.param_groups:
                    param_group['lr'] = args.lr

    if model is None:
        print(f"Creating new {args.algo} model for learning agent...")
        kwargs = {
            "policy": policy,
            "env": env,
            "verbose": 1,
            "ent_coef": args.ent_start,
            "tensorboard_log": "logs/",
            "learning_rate": args.lr,
            "n_steps": 1024,
            "batch_size": 1024,
            "n_epochs": args.n_epochs,
            "clip_range": args.clip_range,
            "target_kl": args.target_kl,
            "policy_kwargs": policy_kwargs
        }
        if args.algo == "RecurrentPPO":
            kwargs["c_aux"] = args.aux_coef
            
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
