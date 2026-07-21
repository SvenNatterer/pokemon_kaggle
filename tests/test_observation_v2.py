from types import SimpleNamespace

import numpy as np
import torch

from src.cg.api import AreaType, OptionType, all_card_data
from src.custom_policy import (
    EFFECT_INDEX,
    ENERGY_FEATURE_OFFSET,
    ENERGY_ROLE_NAMES,
    ENERGY_TYPE_COUNT,
    PokemonTCGFeatureExtractor,
    build_card_metadata,
    build_card_relations,
    _effect_metadata,
)
import src.env_wrapper as env_wrapper_module
from src.env_wrapper import (
    PokemonTCGEnv,
    bound_entity_energy_features,
    encode_energy_count,
    encode_hidden_card_count,
)


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


def test_hidden_card_count_target_preserves_duplicate_counts():
    encoded = [encode_hidden_card_count(count) for count in (0, 1, 2, 4, 20, 60)]

    assert encoded[0] == 0.0
    assert all(left < right for left, right in zip(encoded, encoded[1:]))
    assert encoded[-1] == 1.0


def test_energy_count_features_are_bounded_and_preserve_small_counts():
    encoded = [encode_energy_count(count, maximum=4) for count in (0, 1, 2, 4, 8)]

    assert encoded == [0.0, 0.25, 0.5, 1.0, 1.0]


def test_native_entity_energy_features_are_bounded_without_touching_other_features():
    features = np.zeros((12, 36), dtype=np.float32)
    features[0, 4] = 0.75
    features[0, 8] = 1.25
    features[0, 20] = 2.0

    bounded = bound_entity_energy_features(features)

    assert bounded[0, 4] == 0.75
    assert bounded[0, 8] == 1.0
    assert bounded[0, 20] == 1.0


def test_observation_uses_python_fallback_without_native_symbol(monkeypatch):
    env = PokemonTCGEnv([6] * 60, [5] * 60)
    monkeypatch.setattr(env_wrapper_module, "HAS_NATIVE_V6_OBSERVATION", False)
    monkeypatch.setattr(env, "_get_obs_python", lambda **kwargs: "python")

    def unexpected_native_call(**kwargs):
        raise AssertionError("native encoder must not run without its exported symbol")

    monkeypatch.setattr(env, "_get_obs_cpp", unexpected_native_call)

    assert env._get_obs(perspective=1, pending_selection=[2]) == "python"


def test_rules_text_is_encoded_as_factual_effect_features():
    features = _effect_metadata(
        "Discard 2 cards from your hand. Search your deck for up to 1 Pokémon. "
        "Then, draw 3 cards and heal 40 damage."
    )

    assert features[EFFECT_INDEX["draw"]] == 1.0
    assert features[EFFECT_INDEX["draw_amount"]] == 0.3
    assert features[EFFECT_INDEX["search_deck"]] == 1.0
    assert features[EFFECT_INDEX["search_pokemon"]] == 1.0
    assert features[EFFECT_INDEX["discard"]] == 1.0
    assert features[EFFECT_INDEX["discard_from_hand"]] == 1.0
    assert features[EFFECT_INDEX["heal"]] == 1.0
    assert features[EFFECT_INDEX["heal_amount"]] == 0.1


def test_energy_symbols_are_encoded_by_effect_role():
    features = _effect_metadata(
        "Search your deck for a Basic {F} Energy card. Attach a {P} Energy card "
        "from your discard pile. This card provides every type of Energy."
    )

    def energy_feature(role, energy_type):
        return ENERGY_FEATURE_OFFSET + ENERGY_ROLE_NAMES.index(role) * ENERGY_TYPE_COUNT + energy_type

    assert features[energy_feature("searched", 6)] == 1.0
    assert features[energy_feature("attached", 5)] == 1.0
    assert features[energy_feature("provided", 10)] == 1.0
    assert features[energy_feature("discarded_or_moved", 5)] == 0.0


def test_damage_formula_and_category_condition_are_structured():
    features = _effect_metadata(
        "This attack does 30 damage for each of your Ancient Pokémon in play."
    )

    assert features[EFFECT_INDEX["damage_multiplier"]] == 30 / 400
    assert features[EFFECT_INDEX["condition_ancient"]] == 1.0
    assert features[EFFECT_INDEX["owner_self"]] == 1.0


def test_temporary_restrictions_and_damage_modifiers_are_structured():
    cannot_attack = _effect_metadata("During your next turn, this Pokémon can't use attacks.")
    cannot_retreat = _effect_metadata(
        "During your opponent's next turn, the Defending Pokémon can't retreat."
    )
    self_damage = _effect_metadata("This Pokémon also does 50 damage to itself.")
    reduction = _effect_metadata("This Pokémon takes 30 less damage from attacks.")

    assert cannot_attack[EFFECT_INDEX["cannot_attack"]] == 1.0
    assert cannot_attack[EFFECT_INDEX["turn_duration"]] == 0.2
    assert cannot_retreat[EFFECT_INDEX["cannot_retreat"]] == 1.0
    assert cannot_retreat[EFFECT_INDEX["owner_opponent"]] == 1.0
    assert self_damage[EFFECT_INDEX["self_damage"]] == 50 / 400
    assert reduction[EFFECT_INDEX["damage_reduction"]] == 30 / 400


def test_copy_attack_and_ordered_deck_operations_are_structured():
    copied = _effect_metadata(
        "Choose 1 of your Benched N's Pokémon's attacks and use it as this attack."
    )
    ordered = _effect_metadata(
        "Look at the top 5 cards of your deck and put the remaining cards "
        "on the bottom of your deck in any order."
    )

    assert copied[EFFECT_INDEX["copy_attack"]] == 1.0
    assert copied[EFFECT_INDEX["target_bench"]] == 1.0
    assert ordered[EFFECT_INDEX["deck_top"]] == 1.0
    assert ordered[EFFECT_INDEX["deck_bottom"]] == 1.0
    assert ordered[EFFECT_INDEX["preserve_order"]] == 1.0


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


def test_card_relations_include_attacks_skills_and_evolution_names():
    attack_ids, skill_ids, _, own_names, evolves_from, _ = build_card_relations()
    cards = [card for card in all_card_data() if 0 < int(card.cardId) < 2000]
    card_with_attack = next(card for card in cards if getattr(card, "attacks", None))
    card_with_skill = next(card for card in cards if getattr(card, "skills", None))
    evolved = next(card for card in cards if getattr(card, "evolvesFrom", None))

    assert attack_ids[int(card_with_attack.cardId), 0] == int(card_with_attack.attacks[0])
    assert skill_ids[int(card_with_skill.cardId), 0] > 0
    assert own_names[int(evolved.cardId)] > 0
    assert evolves_from[int(evolved.cardId)] > 0


def test_count_aware_pooling_distinguishes_duplicate_cards():
    env = PokemonTCGEnv([6] * 60, [5] * 60)
    extractor = PokemonTCGFeatureExtractor(env.observation_space, features_dim=256)
    one = torch.tensor([[6, 0, 0]], dtype=torch.int64)
    three = torch.tensor([[6, 6, 6]], dtype=torch.int64)

    assert not torch.allclose(extractor._pool_card_set(one), extractor._pool_card_set(three))


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
