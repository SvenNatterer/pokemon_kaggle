import numpy as np
import pytest
from src.env_wrapper import PokemonTCGEnv
from src.cg.api import CardType, EnergyType

def read_deck_csv(path):
    deck = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # CSV has format: cardId,count or cardId
                parts = line.split(",")
                try:
                    card_id = int(parts[0])
                    count = int(parts[1]) if len(parts) > 1 else 1
                    deck.extend([card_id] * count)
                except:
                    pass
    return deck

def test_compute_deck_meta_attributes():
    deck1 = read_deck_csv("decks/deck_bank/bank_1.csv")
    deck100 = read_deck_csv("decks/deck_bank/bank_100.csv")
    
    env = PokemonTCGEnv(
        my_deck=deck1,
        opponent_deck=deck100,
        action_space_size=66,
    )
    
    meta_my = env._compute_deck_meta_attributes(env.my_deck)
    meta_opp = env._compute_deck_meta_attributes(env.opponent_deck)
    
    assert len(meta_my) == 100
    assert len(meta_opp) == 100
    
    # Types distribution (indices 0 to 11) should sum to 1.0 if there are pokemon/energy cards
    type_sum_my = np.sum(meta_my[:12])
    type_sum_opp = np.sum(meta_opp[:12])
    
    if len(env.my_deck) > 0:
        assert type_sum_my > 0.0
        assert type_sum_my <= 1.001
        
    if len(env.opponent_deck) > 0:
        assert type_sum_opp > 0.0
        assert type_sum_opp <= 1.001

    # Check HP normalization
    assert meta_my[13] >= 0.0 and meta_my[13] <= 1.0
    assert meta_opp[13] >= 0.0 and meta_opp[13] <= 1.0

def test_get_obs_populates_meta_attributes():
    deck1 = read_deck_csv("decks/deck_bank/bank_1.csv")
    deck100 = read_deck_csv("decks/deck_bank/bank_100.csv")
    
    env = PokemonTCGEnv(
        my_deck=deck1,
        opponent_deck=deck100,
        action_space_size=66,
    )
    env.reset(seed=42)
    
    # Get observation for perspective 0
    obs = env._get_obs(perspective=0)
    
    # aux_target should be of shape 2000
    assert obs["aux_target"].shape == (2000,)
    
    # Check that indices 1300 to 1399 are populated and not all zeros
    meta_slice = obs["aux_target"][1300:1400]
    
    # It should match the meta attributes computed for the opponent deck
    # because for perspective 0, the hidden opponent deck is self.opponent_deck (if learner_perspective == 0)
    expected_meta = env._compute_deck_meta_attributes(
        env.opponent_deck if env.learner_perspective == 0 else env.my_deck
    )
    
    assert np.allclose(meta_slice, expected_meta, atol=1e-5)
