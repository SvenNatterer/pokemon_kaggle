from __future__ import annotations

from typing import Any

import numpy as np

from src.cg.api import AreaType, CardType, OptionType, all_attack, all_card_data
from src.agents.rule_based_agent import RuleBasedPokemonAgent


def _first_card(card_type: CardType, predicate=None) -> int:
    for card in all_card_data():
        if int(getattr(card, "cardType", -1)) != int(card_type):
            continue
        if predicate is not None and not predicate(card):
            continue
        return int(card.cardId)
    raise AssertionError(f"No card found for type {card_type}")


def _first_attack(min_damage: int = 1) -> int:
    for attack in all_attack():
        if int(getattr(attack, "damage", 0) or 0) >= min_damage:
            return int(attack.attackId)
    raise AssertionError("No attack found")


def _make_obs() -> dict[str, Any]:
    obs = {
        "vector": np.zeros(1500, dtype=np.float32),
        "action_mask": np.zeros(1000, dtype=np.int8),
        "aux_target": np.zeros(2000, dtype=np.float32),
        "entity_ids": np.zeros(12, dtype=np.int32),
        "entity_features": np.zeros((12, 36), dtype=np.float32),
        "entity_tool_ids": np.zeros(12, dtype=np.int32),
        "entity_pre_evolution_ids": np.zeros((12, 3), dtype=np.int32),
        "entity_energy_card_ids": np.zeros((12, 8), dtype=np.int32),
        "hand_ids": np.zeros(24, dtype=np.int32),
        "discard_ids": np.zeros((2, 30), dtype=np.int32),
        "revealed_ids": np.zeros(120, dtype=np.int32),
        "context_card_ids": np.zeros(2, dtype=np.int32),
        "log_card_ids": np.zeros(10, dtype=np.int32),
        "option_card_ids": np.zeros(65, dtype=np.int32),
        "option_attack_ids": np.zeros(65, dtype=np.int32),
        "option_types": np.zeros(65, dtype=np.int32),
        "option_areas": np.zeros(65, dtype=np.int32),
        "option_features": np.zeros((65, 8), dtype=np.float32),
    }
    obs["vector"][10] = 6
    obs["vector"][20] = 6
    return obs


def test_attack_beats_end_turn():
    obs = _make_obs()
    attack_id = _first_attack(80)

    obs["action_mask"][0] = 1
    obs["action_mask"][1] = 1
    obs["option_types"][0] = int(OptionType.ATTACK) + 1
    obs["option_attack_ids"][0] = attack_id
    obs["option_types"][1] = int(OptionType.END) + 1
    obs["entity_features"][0, 27] = 1.0

    agent = RuleBasedPokemonAgent()

    action = agent.choose_action(obs)

    assert action == 0


def test_setup_prefers_basic_pokemon_play():
    obs = _make_obs()
    basic_pokemon = _first_card(CardType.POKEMON, lambda card: bool(getattr(card, "basic", False)))

    obs["vector"][0] = 1
    obs["action_mask"][0] = 1
    obs["action_mask"][1] = 1
    obs["option_types"][0] = int(OptionType.PLAY) + 1
    obs["option_card_ids"][0] = basic_pokemon
    obs["option_areas"][0] = int(AreaType.HAND)
    obs["option_types"][1] = int(OptionType.END) + 1

    agent = RuleBasedPokemonAgent()

    action = agent.choose_action(obs)

    assert action == 0


def test_retreat_is_preferred_when_active_is_battered():
    obs = _make_obs()

    obs["vector"][0] = 4
    obs["vector"][10] = 4
    obs["vector"][20] = 4
    obs["action_mask"][0] = 1
    obs["action_mask"][1] = 1
    obs["option_types"][0] = int(OptionType.RETREAT) + 1
    obs["option_types"][1] = int(OptionType.END) + 1
    obs["entity_features"][0, 6] = 0.9

    agent = RuleBasedPokemonAgent()

    action = agent.choose_action(obs)

    assert action == 0


def test_predict_returns_scalar_action_and_none_state():
    obs = _make_obs()
    obs["action_mask"][0] = 1
    obs["option_types"][0] = int(OptionType.END) + 1

    agent = RuleBasedPokemonAgent()

    action, hidden = agent.predict(obs)

    assert action == 0
    assert hidden is None
