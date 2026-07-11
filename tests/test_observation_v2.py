from types import SimpleNamespace

import numpy as np
import torch

from src.cg.api import AreaType, OptionType, all_card_data
from src.custom_policy import PokemonTCGFeatureExtractor, build_card_metadata
from src.env_wrapper import PokemonTCGEnv


def _player(hand=None, active=None, bench=None):
    return SimpleNamespace(
        hand=list(hand or []),
        active=list(active or []),
        bench=list(bench or []),
        discard=[],
        prize=[],
        deckCount=40,
    )


def test_play_option_resolves_hand_card_without_card_id():
    card = SimpleNamespace(id=678)
    players = [_player(hand=[card]), _player()]
    obs = SimpleNamespace(current=SimpleNamespace(players=players), select=SimpleNamespace(deck=[]))
    option = SimpleNamespace(
        type=OptionType.PLAY,
        area=None,
        index=0,
        playerIndex=0,
        cardId=None,
        energyIndex=None,
        toolIndex=None,
    )
    env = PokemonTCGEnv([6] * 60, [5] * 60)

    assert env._resolve_option_card_id(obs, option, perspective=0) == 678


def test_card_metadata_contains_rule_attributes():
    card = next(
        card
        for card in all_card_data()
        if 0 < int(card.cardId) < 2000 and int(getattr(card, "retreatCost", 0) or 0) > 0
    )
    metadata = build_card_metadata()

    assert metadata[int(card.cardId), 7] == int(card.retreatCost) / 5.0
    assert metadata[int(card.cardId), 8] == int(card.hp) / 400.0


def test_structured_feature_extractor_output_shape():
    env = PokemonTCGEnv([6] * 60, [5] * 60)
    extractor = PokemonTCGFeatureExtractor(env.observation_space, features_dim=256)
    observation = env.observation_space.sample()
    observation["action_mask"][:] = 0
    observation["action_mask"][0] = 1
    observation["option_types"][:] = 0
    observation["option_types"][0] = int(OptionType.PLAY) + 1
    observation["option_areas"][:] = 0
    observation["option_areas"][0] = int(AreaType.HAND)
    tensor_observation = {
        key: torch.as_tensor(np.asarray(value)).unsqueeze(0)
        for key, value in observation.items()
    }

    features = extractor(tensor_observation)

    assert features.shape == (1, 256)
    assert torch.isfinite(features).all()


def test_dense_reward_uses_rotated_learner_perspective():
    env = PokemonTCGEnv([6] * 60, [5] * 60, learner_perspective=1)
    old_players = [_player(), _player()]
    new_players = [_player(), _player()]
    old_players[0].prize = [1, 2]
    new_players[0].prize = [1, 2]
    old_players[1].prize = [1, 2]
    new_players[1].prize = [1]
    old_obs = SimpleNamespace(current=SimpleNamespace(players=old_players))
    new_obs = SimpleNamespace(current=SimpleNamespace(players=new_players))

    reward = env._compute_dense_reward(old_obs, new_obs, done=False)

    assert reward == env.reward_config["PRIZE_TAKEN"]
