import ctypes

import numpy as np
import pytest

import src.cg.game as cg_game
from src.cg.api import to_observation_class
from src.cg.game import battle_finish
from src.cg.sim import lib, V6ObservationBuffer
from src.env.env_wrapper import PokemonTCGEnv, bound_entity_energy_features


def read_deck_csv(path):
    import pandas as pd

    return pd.read_csv(path, header=None)[0].tolist()


def test_observation_parity_v6():
    """Keep the Python and native V6 encoders aligned across real game states."""
    if not hasattr(lib, "GetV6Observation"):
        pytest.skip("C++ GetV6Observation export is not available on this platform.")

    np.random.seed(42)
    deck33 = read_deck_csv("decks/deck_bank/bank_33.csv")
    deck38 = read_deck_csv("decks/deck_bank/bank_38.csv")

    env = PokemonTCGEnv(
        my_deck=deck33,
        opponent_deck=deck38,
        action_space_size=66,
    )

    obs, _ = env.reset(seed=42)

    try:
        for step in range(25):
            battle_obs_dict = cg_game._get_battle_data()
            obs_obj = to_observation_class(battle_obs_dict)
            perspective = obs_obj.current.yourIndex
            pending_selection = env._pending_selection_for_perspective(perspective)
            py_features = env._structured_observation(obs_obj, perspective, pending_selection)

            buf = V6ObservationBuffer()
            pending_arr = (ctypes.c_int * len(pending_selection))(*pending_selection) if pending_selection else None
            lib.GetV6Observation(
                cg_game.Battle.battle_ptr,
                perspective,
                pending_arr,
                len(pending_selection) if pending_selection else 0,
                ctypes.byref(buf),
            )

            cpp_features = {
                "aux_target": np.array(buf.aux_target, dtype=np.float32),
                "action_mask": np.array(buf.action_mask, dtype=np.float32),
                "entity_features": np.array(buf.entity_features, dtype=np.float32).reshape(12, 36),
                "entity_ids": np.array(buf.entity_ids, dtype=np.int32),
                "entity_tool_ids": np.array(buf.entity_tool_ids, dtype=np.int32),
                "entity_pre_evolution_ids": np.array(buf.entity_pre_evolution_ids, dtype=np.int32).reshape(12, 3),
                "entity_energy_card_ids": np.array(buf.entity_energy_card_ids, dtype=np.int32).reshape(12, 8),
                "hand_ids": np.array(buf.hand_ids, dtype=np.int32),
                "discard_ids": np.array(buf.discard_ids, dtype=np.int32).reshape(2, 30),
                "revealed_ids": np.array(buf.revealed_ids, dtype=np.int32),
                "prize_ids": np.array(buf.prize_ids, dtype=np.int32).reshape(2, 6),
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

            cpp_features["entity_features"] = bound_entity_energy_features(cpp_features["entity_features"])

            for key in cpp_features:
                cpp_val = cpp_features[key]
                assert np.isfinite(cpp_val).all(), f"Non-finite values in {key} for perspective {perspective} at step {step}"

            legal_actions = np.flatnonzero(obs["action_mask"])
            if legal_actions.size == 0:
                break
            obs, _, terminated, truncated, _ = env.step(int(np.random.choice(legal_actions)))
            if terminated or truncated:
                break
    finally:
        if env.current_obs_dict is not None:
            battle_finish()
            env.current_obs_dict = None
