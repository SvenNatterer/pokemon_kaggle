import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys

# Add current directory and src directory to path so imports work both locally and in Kaggle
try:
    AGENT_DIR = os.path.dirname(__file__)
except NameError:
    AGENT_DIR = "/kaggle_simulations/agent"

sys.path.insert(0, AGENT_DIR)
src_dir = os.path.join(AGENT_DIR, "src")
if os.path.exists(src_dir):
    sys.path.insert(0, src_dir)
else:
    parent_src = os.path.abspath(os.path.join(AGENT_DIR, "..", "src"))
    if os.path.exists(parent_src) and parent_src not in sys.path:
        sys.path.insert(0, parent_src)




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

try:
    import src.cg.api as cg_api
    from src.cg.api import Observation, to_observation_class
    if "cg" not in sys.modules and "src.cg" in sys.modules:
        sys.modules["cg"] = sys.modules["src.cg"]
        sys.modules["cg.api"] = cg_api
except ImportError:
    from cg.api import Observation, to_observation_class

from src.custom_ppo import CustomPPO


from src.env_wrapper import (
    LEGACY_ACTION_SPACE_SIZE,
    PokemonTCGEnv,
    advance_selection,
)

import time

model = None
dummy_env = None
lstm_state = None
episode_start = True
match_accumulated_time = 0.0
step_counter = 0
fallback_mode = False

MAX_MATCH_TIME_SECONDS = 540.0
MIN_OVERAGE_TIME_SAFETY_SECONDS = 30.0

def read_deck_csv() -> list[int]:
    """Read deck.csv with local testing fallback."""
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/deck.csv"
    if not os.path.exists(file_path):
        file_path = os.path.join(AGENT_DIR, "deck.csv")
    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(AGENT_DIR), "decks", "deck_bank", "bank_18.csv")
    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(AGENT_DIR), "decks", "deck_0.csv")
        
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(len(csv)):
        if csv[i].strip() and len(deck) < 60:
            deck.append(int(csv[i].strip()))
    return deck


def _fast_legal_fallback(obs: Observation, valid_options: int, stop_action: int, pending: list[int]) -> list[int]:
    """Pure Python greedy legal action selection for error/timeout fallback."""
    min_count = max(0, int(obs.select.minCount or 0)) if obs and obs.select else 0
    max_count = min(valid_options, max(0, int(obs.select.maxCount or 0))) if obs and obs.select else 0
    
    # Try advancing selection with legal options
    for _ in range(max_count + 1):
        if len(pending) >= min_count and len(pending) > 0:
            action = stop_action
        else:
            legal = [i for i in range(min(valid_options, stop_action)) if i not in pending]
            action = legal[0] if legal else stop_action
        
        pending, committed, invalid = advance_selection(obs, action, pending, stop_action=stop_action)
        if invalid:
            legal = [i for i in range(min(valid_options, stop_action)) if i not in pending]
            fallback_action = legal[0] if legal else stop_action
            pending, committed, _ = advance_selection(obs, fallback_action, pending, stop_action=stop_action)
        if committed:
            return [int(x) for x in pending]
            
    # Final defensive filling if loop exhausted
    for index in range(valid_options):
        if len(pending) >= min_count:
            break
        if index not in pending:
            pending.append(index)
    return [int(x) for x in pending]

def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent with match time limit guardrails."""
    global model, dummy_env, lstm_state, episode_start, match_accumulated_time, step_counter, fallback_mode
    
    step_start_time = time.monotonic()
    step_counter += 1
    
    if obs_dict.get("select") is None:
        # Initial deck selection - reset match timer and states
        lstm_state = None
        episode_start = True
        match_accumulated_time = 0.0
        step_counter = 0
        fallback_mode = False
        if dummy_env is not None:
            try:
                dummy_env.reset_inference_guardrails()
            except Exception as e:
                print(f"[WARNING] Guardrail reset failed: {e}", file=sys.stderr)
        return read_deck_csv()
        
    # Check Kaggle remainingOverageTime if provided in obs_dict
    remaining_overage = obs_dict.get("remainingOverageTime")
    if remaining_overage is not None:
        try:
            rem_val = float(remaining_overage)
            if rem_val < MIN_OVERAGE_TIME_SAFETY_SECONDS and not fallback_mode:
                print(f"[SAFETY FALLBACK] Low remainingOverageTime ({rem_val:.2f}s). Switching to fast fallback.", file=sys.stderr)
                fallback_mode = True
        except (ValueError, TypeError):
            pass
            
    if match_accumulated_time > MAX_MATCH_TIME_SECONDS and not fallback_mode:
        print(f"[SAFETY FALLBACK] Accumulated match time ({match_accumulated_time:.2f}s) exceeded limit ({MAX_MATCH_TIME_SECONDS}s). Switching to fast fallback.", file=sys.stderr)
        fallback_mode = True

    try:
        obs: Observation = to_observation_class(obs_dict)
    except Exception as e:
        print(f"[ERROR] Failed to parse observation dict: {e}", file=sys.stderr)
        match_accumulated_time += time.monotonic() - step_start_time
        return []

    valid_options = len(obs.select.option) if obs.select and obs.select.option else 0
    if valid_options == 0:
        match_accumulated_time += time.monotonic() - step_start_time
        return []

    stop_action = valid_options

    if fallback_mode:
        res = _fast_legal_fallback(obs, valid_options, stop_action, [])
        match_accumulated_time += time.monotonic() - step_start_time
        return res

    if model is None:
        model_path = "ppo_pokemon_final.zip"
        if not os.path.exists(model_path):
            model_path = "/kaggle_simulations/agent/" + model_path
        
        try:
            model = CustomPPO.load(model_path, device='cpu')
            my_deck = read_deck_csv()
            action_space_size = int(
                getattr(getattr(model, "action_space", None), "n", LEGACY_ACTION_SPACE_SIZE)
            )
            dummy_env = PokemonTCGEnv(
                my_deck,
                my_deck,
                action_space_size=action_space_size,
                inference_guardrails=True,
                zone_aux_targets=bool(getattr(model.policy, "use_zone_aux", False)),
            )
        except Exception as e:
            print(f"[ERROR] Model initialization failed: {e}. Switching to fallback mode.", file=sys.stderr)
            fallback_mode = True
            res = _fast_legal_fallback(obs, valid_options, stop_action, [])
            match_accumulated_time += time.monotonic() - step_start_time
            return res
            
    if dummy_env is None:
        print(f"[ERROR] dummy_env is None. Switching to fallback mode.", file=sys.stderr)
        fallback_mode = True
        res = _fast_legal_fallback(obs, valid_options, stop_action, [])
        match_accumulated_time += time.monotonic() - step_start_time
        return res

    dummy_env.current_obs_dict = obs_dict
    perspective = obs.current.yourIndex if obs.current else 0
    pending = []
    max_count = min(valid_options, max(0, int(obs.select.maxCount or 0)))

    for _ in range(max_count + 1):
        if fallback_mode:
            res = _fast_legal_fallback(obs, valid_options, stop_action, pending)
            match_accumulated_time += time.monotonic() - step_start_time
            return res

        try:
            formatted_obs = dummy_env._get_obs_python(
                perspective=perspective,
                pending_selection=pending,
            )

            action, lstm_state = model.predict(
                formatted_obs,
                state=lstm_state,
                episode_start=np.array([episode_start], dtype=bool),
                deterministic=True,
            )
            episode_start = False
            action = int(np.asarray(action).item())
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}. Switching to fallback mode.", file=sys.stderr)
            fallback_mode = True
            res = _fast_legal_fallback(obs, valid_options, stop_action, pending)
            match_accumulated_time += time.monotonic() - step_start_time
            return res

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
            match_accumulated_time += time.monotonic() - step_start_time
            return [int(index) for index in pending]

    min_count = min(valid_options, max(0, int(obs.select.minCount or 0)))
    for index in range(valid_options):
        if len(pending) >= min_count:
            break
        if index not in pending:
            pending.append(index)
            
    match_accumulated_time += time.monotonic() - step_start_time
    return [int(index) for index in pending]

