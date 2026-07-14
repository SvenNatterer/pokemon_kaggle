import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys

# Add current directory to path so src can be imported
try:
    AGENT_DIR = os.path.dirname(__file__)
except NameError:
    AGENT_DIR = "/kaggle_simulations/agent"

# Insert at 0 so our bundled packages are prioritized over Kaggle's broken system packages
sys.path.insert(0, AGENT_DIR)

# Kaggle's container currently has a numpy 2.x incompatibility breaking tensorboard & matplotlib
# We mock these before importing PyTorch/SB3 so they don't crash during initialization.
from unittest.mock import MagicMock
sys.modules['tensorboard'] = MagicMock()
sys.modules['tensorboard.compat'] = MagicMock()
sys.modules['torch.utils.tensorboard'] = MagicMock()
sys.modules['torch.utils.tensorboard.writer'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

import torch
import numpy as np
torch.set_num_threads(1)

from cg.api import Observation, to_observation_class
from src.custom_ppo import CustomPPO
from src.env_wrapper import LEGACY_ACTION_SPACE_SIZE, PokemonTCGEnv, advance_selection

model = None
dummy_env = None
lstm_state = None
episode_start = True

def read_deck_csv() -> list[int]:
    """Read deck.csv."""
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        if csv[i].strip():
            deck.append(int(csv[i].strip()))
    return deck

def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent."""
    global model, dummy_env, lstm_state, episode_start
    
    if obs_dict.get("select") is None:
        # Initial deck selection
        lstm_state = None
        episode_start = True
        return read_deck_csv()
        
    obs: Observation = to_observation_class(obs_dict)
        
    valid_options = len(obs.select.option) if obs.select and obs.select.option else 0
    
    if valid_options == 0:
        return []
        
    if model is None:
        model_path = "ppo_pokemon_final.zip"
        if not os.path.exists(model_path):
            model_path = "/kaggle_simulations/agent/" + model_path
        
        try:
            model = CustomPPO.load(model_path, device='cpu')
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            raise e
            
        my_deck = read_deck_csv()
        action_space_size = int(
            getattr(getattr(model, "action_space", None), "n", LEGACY_ACTION_SPACE_SIZE)
        )
        dummy_env = PokemonTCGEnv(my_deck, my_deck, action_space_size=action_space_size)
        
    dummy_env.current_obs_dict = obs_dict
    perspective = obs.current.yourIndex if obs.current else 0
    pending = []
    max_count = min(valid_options, max(0, int(obs.select.maxCount or 0)))

    for _ in range(max_count + 1):
        formatted_obs = dummy_env._get_obs(
            perspective=perspective,
            pending_selection=pending,
        )
        try:
            action, lstm_state = model.predict(
                formatted_obs,
                state=lstm_state,
                episode_start=np.array([episode_start], dtype=bool),
                deterministic=True,
            )
            episode_start = False
            action = int(np.asarray(action).item())
        except Exception as e:
            print(f"Error predicting action: {e}", file=sys.stderr)
            action = dummy_env.stop_action if len(pending) >= int(obs.select.minCount or 0) else 0

        pending, committed, invalid = advance_selection(
            obs, action, pending, stop_action=dummy_env.stop_action
        )
        if invalid:
            legal = [
                index
                for index in range(min(valid_options, dummy_env.stop_action))
                if index not in pending
            ]
            fallback_action = legal[0] if legal else dummy_env.stop_action
            pending, committed, _ = advance_selection(
                obs, fallback_action, pending, stop_action=dummy_env.stop_action
            )
        if committed:
            return [int(index) for index in pending]

    # Defensive fallback for malformed selection metadata.
    min_count = min(valid_options, max(0, int(obs.select.minCount or 0)))
    for index in range(valid_options):
        if len(pending) >= min_count:
            break
        if index not in pending:
            pending.append(index)
    return [int(index) for index in pending]
