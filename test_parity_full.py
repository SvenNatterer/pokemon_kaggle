import os
import sys
import numpy as np
import ctypes

# Add src to path
sys.path.insert(0, os.path.abspath('src'))
from env_wrapper import PokemonTCGEnv
from src.cg.api import to_observation_class
from src.cg.sim import lib, Battle, V6ObservationBuffer

def test():
    print("Testing parity...")
    env_py = PokemonTCGEnv(
        my_deck=[1]*60,
        opponent_deck=[1]*60,
        action_space_size=66
    )
    obs_raw, _ = env_py.reset(seed=42)
    # env_py._structured_observation was called inside reset, but let's call it manually
    raw_obs = env_py.cg_obs # It might be None if not stored.
    # Actually env_wrapper doesn't store cg_obs?
    # Let's get it:
    import src.cg.game as cg_game
    battle_obs_dict = cg_game._get_battle_data()
    obs_obj = to_observation_class(battle_obs_dict)
    
    # Python features
    py_features = env_py._structured_observation(obs_obj, env_py.learner_perspective, [])
    
    # C++ features
    buf = V6ObservationBuffer()
    lib.GetV6Observation(cg_game.Battle.battle_ptr, env_py.learner_perspective, None, 0, ctypes.byref(buf))
    
    # Map buf back to numpy dictionary
    cpp_features = {
        "aux_target": np.array(buf.aux_target, dtype=np.float32),
        "action_mask": np.array(buf.action_mask, dtype=np.float32),
        "entity_features": np.array(buf.entity_features, dtype=np.float32).reshape(12, 36),
        "entity_ids": np.array(buf.entity_ids, dtype=np.int32),
        "entity_tool_ids": np.array(buf.entity_tool_ids, dtype=np.int32),
        "entity_pre_evolution_ids": np.array(buf.entity_pre_evolution_ids, dtype=np.int32),
        "entity_energy_card_ids": np.array(buf.entity_energy_card_ids, dtype=np.int32),
        "hand_ids": np.array(buf.hand_ids, dtype=np.int32),
        "discard_ids": np.array(buf.discard_ids, dtype=np.int32),
        "revealed_ids": np.array(buf.revealed_ids, dtype=np.int32),
        "prize_ids": np.array(buf.prize_ids, dtype=np.int32),
        "search_ids": np.array(buf.search_ids, dtype=np.int32),
        "looking_ids": np.array(buf.looking_ids, dtype=np.int32),
        "own_deck_ids": np.array(buf.own_deck_ids, dtype=np.int32),
        "context_card_ids": np.array(buf.context_card_ids, dtype=np.int32),
        "log_card_ids": np.array(buf.log_card_ids, dtype=np.int32),
        "option_features": np.array(buf.option_features, dtype=np.float32).reshape(65, 21),
        "option_card_ids": np.array(buf.option_card_ids, dtype=np.int32),
        "option_attack_ids": np.array(buf.option_attack_ids, dtype=np.int32),
        "option_types": np.array(buf.option_types, dtype=np.int32),
        "option_areas": np.array(buf.option_areas, dtype=np.int32),
    }

    mismatch = False
    for key in py_features.keys():
        py_val = py_features[key]
        cpp_val = cpp_features[key]
        if py_val.shape != cpp_val.shape:
            print(f"Shape mismatch in {key}: {py_val.shape} vs {cpp_val.shape}")
            mismatch = True
        elif not np.allclose(py_val, cpp_val, rtol=1e-3, atol=1e-3):
            print(f"Mismatch in {key}!")
            # print("Python:", py_val)
            # print("C++   :", cpp_val)
            diff = py_val - cpp_val
            print("Max Diff:", np.max(np.abs(diff)))
            # find first mismatched index
            idx = np.where(np.abs(diff) > 1e-3)
            print("Mismatched indices:", idx)
            print("Python:", py_val[idx][:10])
            print("C++   :", cpp_val[idx][:10])
            mismatch = True
            
    if mismatch:
        print("Mismatches found!")
        return False
    print("All features match perfectly!")
    return True

if __name__ == '__main__':
    test()
