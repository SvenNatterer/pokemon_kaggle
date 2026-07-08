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
torch.set_num_threads(1)

from cg.api import Observation, to_observation_class
from src.custom_ppo import CustomPPO
from src.env_wrapper import PokemonTCGEnv, select_action_indices

model = None
dummy_env = None

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
    global model, dummy_env
    
    obs: Observation = to_observation_class(obs_dict)
    if obs.select == None:
        # Initial deck selection
        return read_deck_csv()
        
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
        dummy_env = PokemonTCGEnv(my_deck, my_deck) # dummy opponent deck
        
    dummy_env.current_obs_dict = obs_dict
    perspective = obs.current.yourIndex if obs.current else 0
    formatted_obs = dummy_env._get_obs(perspective=perspective)
    
    try:
        action, _ = model.predict(formatted_obs, deterministic=True)
        action = int(action.item())
    except Exception as e:
        print(f"Error predicting action: {e}", file=sys.stderr)
        action = None # Deterministic heuristic fallback

    action_list = select_action_indices(
        obs,
        action,
        perspective=perspective,
        allow_policy_override=True,
    )
        
    return [int(x) for x in action_list]
