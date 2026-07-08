import os
import sys
import pandas as pd
import json

# Add src to pythonpath so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

os.environ["WANDB_MODE"] = "offline"
import wandb
from wandb.integration.sb3 import WandbCallback

from src.env_wrapper import PokemonTCGEnv
from src.custom_ppo import CustomPPO, PokemonTCGRecurrentPolicy

def read_deck(deck_path):
    df = pd.read_csv(deck_path, header=None)
    return df[0].tolist()

def make_env(deck_path, opp_model_path):
    def _init():
        deck = read_deck(deck_path)
        # Self-play: opponent uses same deck
        env = PokemonTCGEnv(deck, deck, opponent_model_path=opp_model_path)
        return Monitor(env)
    return _init

def pretrain():
    deck_path = "decks/base_deck.csv"
    model_name = "ppo_base_brain"
    model_path = f"models/{model_name}"
    timesteps = 5_000_000  # 5 Million steps for the Grundschule!
    num_envs = 8
    
    print(f"==========================================")
    print(f"🚀 STARTING PRE-TRAINING (BASE BRAIN) 🚀")
    print(f"==========================================")
    print(f"Training {model_name} on {deck_path} against itself.")
    
    # Check if a model already exists to use as opponent (Self-Play evolution)
    opp_model_arg = model_name if os.path.exists(f"{model_path}.zip") else None
    
    print(f"Initializing {num_envs} environments...")
    env = SubprocVecEnv([make_env(deck_path, opp_model_arg) for _ in range(num_envs)])
    
    if os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing model from {model_path}.zip...")
        model = CustomPPO.load(model_path, env=env)
    else:
        print("Creating new Custom PPO model...")
        
        from src.custom_policy import PokemonTCGFeatureExtractor
        policy_kwargs = dict(
            features_extractor_class=PokemonTCGFeatureExtractor,
        )
        
        model = CustomPPO(
            PokemonTCGRecurrentPolicy, 
            env, 
            verbose=1, 
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=5,
            gamma=0.999,
            ent_coef=0.02,
            c_aux=0.5,
            device="cpu",
            tensorboard_log="logs/",
            policy_kwargs=policy_kwargs
        )
    
    # Initialize wandb
    run = wandb.init(
        project="pokemon_kaggle",
        name="Base_Brain_Pretraining",
        group="Pretrain",
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    
    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=f"models/wandb/{run.id}",
        model_save_freq=100_000,
        verbose=2,
    )
    
    print(f"Training for {timesteps} timesteps. Press Ctrl+C to stop early when W&B looks good.")
    
    try:
        model.learn(total_timesteps=timesteps, callback=wandb_callback)
    except KeyboardInterrupt:
        print("\nPre-Training interrupted by user. Saving progress...")
    
    run.finish()
    
    print("Saving Base Brain model...")
    model.save(model_path)
    print(f"✅ Foundation Model saved to {model_path}.zip!")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    pretrain()
