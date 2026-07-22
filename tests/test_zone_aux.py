import numpy as np
import pandas as pd
import pytest
import torch
from gymnasium import spaces

from src.cg.sim import HAS_NATIVE_V6_OBSERVATION
from src.models.custom_policy import PokemonTCGFeatureExtractor
from src.training.custom_ppo import CustomPPO
from src.env.env_wrapper import (
    PokemonTCGEnv,
    V6_ACTION_SPACE_SIZE,
    _fit_observation_to_model_space,
)


def _deck(path):
    return pd.read_csv(path, header=None)[0].tolist()


def _zone_keys():
    return (
        "aux_own_deck_ids",
        "aux_own_prize_ids",
        "aux_opponent_hand_ids",
        "aux_opponent_deck_ids",
        "aux_opponent_prize_ids",
    )


def test_zone_aux_is_opt_in_for_checkpoint_compatibility():
    deck = _deck("decks/deck_bank/bank_38.csv")
    baseline = PokemonTCGEnv(deck, deck, action_space_size=V6_ACTION_SPACE_SIZE)
    extended = PokemonTCGEnv(
        deck,
        deck,
        action_space_size=V6_ACTION_SPACE_SIZE,
        zone_aux_targets=True,
    )
    try:
        for key in _zone_keys():
            assert key not in baseline.observation_space.spaces
            assert key in extended.observation_space.spaces
    finally:
        baseline.close()
        extended.close()


def test_public_python_encoder_zero_fills_privileged_targets():
    deck = _deck("decks/deck_bank/bank_38.csv")
    env = PokemonTCGEnv(
        deck,
        deck,
        action_space_size=V6_ACTION_SPACE_SIZE,
        zone_aux_targets=True,
    )
    try:
        env.reset(seed=3)
        observation = env._get_obs_python(perspective=env.learner_perspective)
        for key in _zone_keys():
            assert np.count_nonzero(observation[key]) == 0
    finally:
        env.close()


@pytest.mark.skipif(
    not HAS_NATIVE_V6_OBSERVATION,
    reason="native engine is required for privileged simulator targets",
)
def test_native_encoder_exports_exact_hidden_zone_ids():
    env = PokemonTCGEnv(
        _deck("decks/deck_bank/bank_38.csv"),
        _deck("decks/deck_bank/bank_33.csv"),
        action_space_size=V6_ACTION_SPACE_SIZE,
        zone_aux_targets=True,
    )
    try:
        observation, _ = env.reset(seed=7)
        for _ in range(20):
            if np.count_nonzero(observation["aux_own_prize_ids"]) == 6:
                break
            legal = np.flatnonzero(observation["action_mask"])
            observation, _, terminated, truncated, _ = env.step(int(legal[0]))
            if terminated or truncated:
                break

        assert np.count_nonzero(observation["aux_own_prize_ids"]) == 6
        assert np.count_nonzero(observation["aux_opponent_prize_ids"]) == 6
        assert np.count_nonzero(observation["aux_opponent_hand_ids"]) > 0
        assert np.count_nonzero(observation["aux_own_deck_ids"]) > 0
        assert np.count_nonzero(observation["aux_opponent_deck_ids"]) > 0
    finally:
        env.close()


def test_privileged_targets_do_not_change_actor_features():
    deck = _deck("decks/deck_bank/bank_38.csv")
    env = PokemonTCGEnv(
        deck,
        deck,
        action_space_size=V6_ACTION_SPACE_SIZE,
        zone_aux_targets=True,
    )
    extractor = PokemonTCGFeatureExtractor(
        env.observation_space,
        feature_variant="compact",
        use_card_table=False,
    ).eval()
    rng = np.random.default_rng(11)
    first = {}
    second = {}
    for key, space in env.observation_space.spaces.items():
        value = np.zeros((1, *space.shape), dtype=space.dtype)
        first[key] = torch.as_tensor(value)
        second[key] = torch.as_tensor(value.copy())
    for key in _zone_keys():
        random_ids = rng.integers(
            1,
            100,
            size=env.observation_space.spaces[key].shape,
            dtype=np.int32,
        )
        second[key] = torch.as_tensor(random_ids[None, ...])

    with torch.no_grad():
        first_features = extractor(first)
        second_features = extractor(second)
    assert torch.equal(first_features, second_features)


def test_sparse_zone_loss_is_normalized_near_one_for_uniform_logits():
    logits = torch.zeros((2, 2000), dtype=torch.float32)
    targets = torch.tensor([[1, 2, 2], [3, 4, 0]], dtype=torch.int64)
    loss, accuracy = CustomPPO._sparse_card_distribution_loss(logits, targets)
    assert loss.item() == pytest.approx(1.0, rel=1e-5)
    assert accuracy is not None
    assert accuracy.item() == pytest.approx(0.0)


def test_missing_privileged_targets_are_padded_for_public_inference():
    observation_space = spaces.Dict(
        {
            "vector": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            "aux_own_deck_ids": spaces.Box(0, 1999, shape=(60,), dtype=np.int32),
        }
    )
    fitted = _fit_observation_to_model_space(
        {"vector": np.ones((2, 3), dtype=np.float32)},
        observation_space,
    )
    assert fitted["aux_own_deck_ids"].shape == (2, 60)
    assert np.count_nonzero(fitted["aux_own_deck_ids"]) == 0
