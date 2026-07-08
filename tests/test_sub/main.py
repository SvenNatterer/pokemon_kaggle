import os
import sys
import random

# Add current directory to path so src can be imported
try:
    AGENT_DIR = os.path.dirname(__file__)
except NameError:
    AGENT_DIR = "/kaggle_simulations/agent"

sys.path.append(AGENT_DIR)

from cg.api import Observation, to_observation_class
from src.custom_ppo import CustomPPO
from src.env_wrapper import PokemonTCGEnv

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
    max_c = obs.select.maxCount if obs.select else 1
    min_c = obs.select.minCount if obs.select else 1
    
    if valid_options == 0:
        return []
        
    if model is None:
        model_path = "ppo_pokemon_final.zip"
        if not os.path.exists(model_path):
            model_path = "/kaggle_simulations/agent/" + model_path
        model = CustomPPO.load(model_path)
        my_deck = read_deck_csv()
        dummy_env = PokemonTCGEnv(my_deck, my_deck) # dummy opponent deck
        
    dummy_env.current_obs_dict = obs_dict
    perspective = obs.current.yourIndex if obs.current else 0
    formatted_obs = dummy_env._get_obs(perspective=perspective)
    
    action, _ = model.predict(formatted_obs, deterministic=True)
    action = int(action.item())
    
    if action >= valid_options:
        sample_size = min(max_c, valid_options)
        action_list = random.sample(list(range(valid_options)), sample_size)
    else:
        action_list = [action]
        if len(action_list) < min_c:
            available = [i for i in range(valid_options) if i != action]
            remaining = min(min_c - len(action_list), len(available))
            if remaining > 0:
                action_list += random.sample(available, remaining)
        action_list = action_list[:max_c]
        
    return [int(x) for x in action_list]
