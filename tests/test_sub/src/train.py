import os
import sys
import argparse
import pandas as pd

# Add src to pythonpath so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
import json
import os
os.environ["WANDB_MODE"] = "offline"
import wandb
from wandb.integration.sb3 import WandbCallback

class LiveStatusCallback(BaseCallback):
    def __init__(self, action_text, total_timesteps, verbose=0):
        super(LiveStatusCallback, self).__init__(verbose)
        self.action_text = action_text
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps % max(1, self.total_timesteps // 100) == 0 or self.num_timesteps == self.total_timesteps:
            status_data = {
                "action": self.action_text,
                "completed": self.num_timesteps,
                "total": self.total_timesteps
            }
            try:
                with open("decks/status.json", "w") as f:
                    json.dump(status_data, f)
            except Exception:
                pass
        return True

from stable_baselines3.common.monitor import Monitor
from src.env_wrapper import PokemonTCGEnv
from src.custom_ppo import CustomPPO, PokemonTCGRecurrentPolicy

def read_deck(deck_path):
    df = pd.read_csv(deck_path, header=None)
    return df[0].tolist()

def make_env(deck_path, opp_deck_path, opp_model_path):
    def _init():
        deck = read_deck(deck_path)
        opp_deck = read_deck(opp_deck_path)
        env = PokemonTCGEnv(deck, opp_deck, opponent_model_path=opp_model_path)
        return Monitor(env)
    return _init

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=str, required=True, help="Path to deck.csv")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model to save")
    parser.add_argument("--timesteps", type=int, default=25000, help="Number of training timesteps")
    parser.add_argument("--opp-deck", type=str, help="Path to opponent deck.csv", default=None)
    parser.add_argument("--opp-model", type=str, help="Path to opponent model .zip", default=None)
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments (default: 8)")
    args = parser.parse_args()

    opp_deck_path = args.opp_deck if args.opp_deck else args.deck
    
    print(f"Initializing environment with {args.num_envs} workers for deck {args.deck} against {opp_deck_path}...")
    # Vectorized environment - must use SubprocVecEnv because cg library uses a global singleton
    env = SubprocVecEnv([make_env(args.deck, opp_deck_path, args.opp_model) for _ in range(args.num_envs)])
    
    # Removed CheckpointCallback to stop spamming models/ folder
    
    model_path = f"models/{args.model_name}"
    
    if os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing model from {model_path}.zip...")
        model = CustomPPO.load(model_path, env=env)
    else:
        print("Creating new Custom PPO model...")
        model = CustomPPO(
            PokemonTCGRecurrentPolicy, 
            env, 
            verbose=1, 
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=1024, # MAXIMUM SPEED
            n_epochs=3,
            c_aux=0.5,
            device="cpu",
            tensorboard_log="logs/"
        )
    
    print(f"Starting training for {args.timesteps} timesteps...")
    deck_id = args.deck.split('_')[-1].split('.')[0]
    opp_id = opp_deck_path.split('_')[-1].split('.')[0]
    
    deck_name = "Unknown"
    opp_name = "Unknown"
    if os.path.exists("decks/deck_names.json"):
        try:
            with open("decks/deck_names.json", "r") as f:
                names = json.load(f)
                deck_name = names.get(str(deck_id), "Unknown")
                opp_name = names.get(str(opp_id), "Unknown")
        except: pass
        
    action_text = f"🧠 Training: {deck_name} (D{deck_id}) vs {opp_name} (D{opp_id})"
    
    # Initialize wandb
    run = wandb.init(
        project="pokemon_kaggle",
        name=f"D{deck_id}_vs_D{opp_id}_{args.timesteps}",
        group=f"deck_{deck_id}",
        config=vars(args),
        sync_tensorboard=True, # auto-upload sb3's tensorboard metrics
        monitor_gym=True,
        save_code=True,
    )
    
    live_status_callback = LiveStatusCallback(action_text=action_text, total_timesteps=args.timesteps)
    wandb_callback = WandbCallback(
        gradient_save_freq=0, # disable saving gradients to save space
        model_save_path=f"models/wandb/{run.id}",
        verbose=2,
    )
    callbacks = CallbackList([live_status_callback, wandb_callback])
    
    model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name=f"Deck_{deck_id}")
    
    run.finish()
    
    print("Training finished! Saving model...")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    train()
