from __future__ import annotations

from typing import Any

import numpy as np

from src.cg.api import AreaType, CardType, OptionType, SelectContext, all_attack, all_card_data
from src.agents.rule_based_agent import (
    RuleBasedPokemonAgent,
    is_rule_based_model_spec,
    rule_based_spec_from_spec,
)
from src.agents.bot_loader import load_bot
from src.env.env_wrapper import OPTION_FEATURE_DIM


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


def _card_named(name: str) -> int:
    for card in all_card_data():
        if str(getattr(card, "name", "")).strip().lower().replace("’", "'") == name.lower().replace("’", "'"):
            return int(card.cardId)
    raise AssertionError(f"No card named {name}")


def _card_with_attack_text(name: str, fragment: str):
    attacks = {int(attack.attackId): attack for attack in all_attack()}
    for card in all_card_data():
        if str(card.name) != name:
            continue
        if any(fragment in str(attacks[int(attack_id)].text) for attack_id in card.attacks):
            return card
    raise AssertionError(f"No {name} card with attack text {fragment!r}")


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
        "search_ids": np.zeros(60, dtype=np.int32),
        "context_card_ids": np.zeros(3, dtype=np.int32),
        "log_card_ids": np.zeros(10, dtype=np.int32),
        "option_card_ids": np.zeros(65, dtype=np.int32),
        "option_attack_ids": np.zeros(65, dtype=np.int32),
        "option_types": np.zeros(65, dtype=np.int32),
        "option_areas": np.zeros(65, dtype=np.int32),
        "option_features": np.zeros((65, OPTION_FEATURE_DIM), dtype=np.float32),
    }
    obs["vector"][7] = 60
    obs["vector"][10] = 6
    obs["vector"][107] = 60
    obs["vector"][110] = 6
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


def test_named_profiles_are_loadable():
    for profile in ("balanced", "aggressive", "setup", "defensive"):
        spec = f"rule_based:{profile}"
        assert is_rule_based_model_spec(spec)
        assert load_bot(spec).profile == profile

    assert not is_rule_based_model_spec("rule_based:unknown")


def test_v4_archetype_specs_are_loadable_and_keep_legacy_specs_compatible():
    for archetype in (
        "alakazam", "dragapult", "abomasnow", "lucario", "kangaskhan",
        "starmie", "grimmsnarl", "archaludon", "mewtwo", "hydrapple",
        "trevenant", "zoroark",
    ):
        spec = f"rule_based:v4:{archetype}:engine"
        assert is_rule_based_model_spec(spec)
        bot = load_bot(spec)
        assert bot.version == "v4"
        assert bot.archetype == archetype
        assert bot.variant == "engine"

    assert load_bot("heuristic").version == "v3"
    assert not is_rule_based_model_spec("rule_based:v4:unknown:engine")
    assert not is_rule_based_model_spec("rule_based:v4:alakazam:unknown")


def test_v4_parameter_overrides_round_trip_through_loader():
    raw = "rule_based:v4:dragapult:tempo?attack_knockout=47&deckout_penalty=72"
    spec = rule_based_spec_from_spec(raw)
    bot = load_bot(spec.model_spec)

    assert bot.parameters.attack_knockout == 47.0
    assert bot.parameters.deckout_penalty == 72.0
    assert bot.spec.model_spec == spec.model_spec
    assert not is_rule_based_model_spec("rule_based:v4:dragapult:tempo?unknown=1")


def test_aggressive_profile_values_attack_more_than_setup_profile():
    obs = _make_obs()
    attack_id = _first_attack(80)
    obs["vector"][0] = 4
    obs["vector"][10] = 4
    obs["vector"][20] = 4
    obs["action_mask"][0] = 1
    obs["option_types"][0] = int(OptionType.ATTACK) + 1
    obs["option_attack_ids"][0] = attack_id

    aggressive = RuleBasedPokemonAgent(profile="aggressive")
    setup = RuleBasedPokemonAgent(profile="setup")
    _, aggressive_info = aggressive.choose_action(obs, return_info=True)
    _, setup_info = setup.choose_action(obs, return_info=True)

    assert aggressive_info["candidates"][0]["score"] > setup_info["candidates"][0]["score"]


def test_v4_archetype_and_variant_create_distinct_deterministic_priorities():
    obs = _make_obs()
    attack_id = _first_attack(80)
    obs["vector"][0] = 5
    obs["action_mask"][0] = 1
    obs["option_types"][0] = int(OptionType.ATTACK) + 1
    obs["option_attack_ids"][0] = attack_id

    tempo = load_bot("rule_based:v4:starmie:tempo")
    control = load_bot("rule_based:v4:kangaskhan:control")
    _, tempo_info = tempo.choose_action(obs, return_info=True)
    _, control_info = control.choose_action(obs, return_info=True)

    assert tempo_info["candidates"][0]["score"] > control_info["candidates"][0]["score"]
    assert tempo_info["archetype"] == "starmie"
    assert tempo_info["category"] == "attack"


def test_boss_orders_is_saved_without_opponent_bench():
    obs = _make_obs()
    obs["vector"][0] = 4
    obs["vector"][8] = 5
    obs["action_mask"][:2] = 1
    obs["option_types"][0] = int(OptionType.PLAY) + 1
    obs["option_card_ids"][0] = _card_named("Boss's Orders")
    obs["option_types"][1] = int(OptionType.END) + 1

    assert RuleBasedPokemonAgent().choose_action(obs) == 1


def test_boss_orders_is_played_when_opponent_has_bench():
    obs = _make_obs()
    obs["vector"][0] = 4
    obs["vector"][8] = 5
    obs["entity_features"][7, 0] = 1.0
    obs["action_mask"][:2] = 1
    obs["option_types"][0] = int(OptionType.PLAY) + 1
    obs["option_card_ids"][0] = _card_named("Boss's Orders")
    obs["option_types"][1] = int(OptionType.END) + 1

    assert RuleBasedPokemonAgent().choose_action(obs) == 0


def test_ultra_ball_discards_duplicate_before_unique_supporter():
    obs = _make_obs()
    ultra_ball = _card_named("Ultra Ball")
    energy_switch = _card_named("Energy Switch")
    boss = _card_named("Boss's Orders")
    obs["vector"][250] = int(SelectContext.DISCARD)
    obs["context_card_ids"][1] = ultra_ball
    obs["hand_ids"][:3] = [energy_switch, energy_switch, boss]
    obs["action_mask"][:2] = 1
    obs["option_types"][:2] = int(OptionType.CARD) + 1
    obs["option_card_ids"][:2] = [energy_switch, boss]

    assert RuleBasedPokemonAgent().choose_action(obs) == 0


def test_damage_counter_any_prefers_a_benched_prize_knockout():
    obs = _make_obs()
    prize_target = _first_card(CardType.POKEMON, lambda card: bool(getattr(card, "ex", False)))
    durable_target = _first_card(CardType.POKEMON, lambda card: not bool(getattr(card, "ex", False)))
    obs["vector"][250] = int(SelectContext.DAMAGE_COUNTER_ANY)
    obs["vector"][253] = 6  # Phantom Dive has six 10-damage counters to place.
    obs["entity_ids"][7:9] = [prize_target, durable_target]
    obs["entity_features"][7:9, 0] = 1.0
    obs["entity_features"][7, 4] = 0.125  # 50 HP: a one-prize knockout is available.
    obs["entity_features"][8, 4] = 0.70   # 280 HP: cannot be knocked out by Phantom Dive.
    obs["action_mask"][:2] = 1
    obs["option_types"][:2] = int(OptionType.CARD) + 1
    obs["option_card_ids"][:2] = [prize_target, durable_target]

    assert RuleBasedPokemonAgent().choose_action(obs) == 0


def test_lunatone_ability_scored_highly_with_requirements():
    obs = _make_obs()
    lunatone = _card_named("Lunatone")
    solrock = _card_named("Solrock")
    fighting_energy = 6

    # Solrock in play
    obs["entity_ids"][1] = solrock
    # Fighting energy in hand
    obs["hand_ids"][0] = fighting_energy
    obs["vector"][8] = 1

    obs["action_mask"][:2] = 1
    obs["option_types"][0] = int(OptionType.ABILITY) + 1
    obs["option_card_ids"][0] = lunatone
    obs["option_types"][1] = int(OptionType.END) + 1

    # Should choose Lunatone's ability (0) over ending the turn (1)
    assert RuleBasedPokemonAgent().choose_action(obs) == 0


def test_lunatone_ability_scored_negatively_without_requirements():
    obs = _make_obs()
    lunatone = _card_named("Lunatone")

    obs["action_mask"][:2] = 1
    obs["option_types"][0] = int(OptionType.ABILITY) + 1
    obs["option_card_ids"][0] = lunatone
    obs["option_types"][1] = int(OptionType.END) + 1

    # Should choose ending the turn (1) over Lunatone's ability (0)
    assert RuleBasedPokemonAgent().choose_action(obs) == 1


def test_prefers_trainer_play_over_attacking():
    obs = _make_obs()
    ultra_ball = _card_named("Ultra Ball")
    attack_id = _first_attack(80)

    obs["vector"][0] = 4
    obs["vector"][8] = 3

    obs["option_types"][0] = int(OptionType.ATTACK) + 1
    obs["option_attack_ids"][0] = attack_id
    obs["option_types"][1] = int(OptionType.PLAY) + 1
    obs["option_card_ids"][1] = ultra_ball

    obs["action_mask"][:2] = 1

    # Should choose Option 1 (Play Trainer) over Option 0 (Attack)
    assert RuleBasedPokemonAgent().choose_action(obs) == 1


def test_adapter_reads_opponent_resources_from_second_player_block():
    obs = _make_obs()
    obs["vector"][108] = 7
    obs["vector"][110] = 3
    # These old, incorrect offsets belong to the acting Pokémon block.
    obs["vector"][18] = 333
    obs["vector"][20] = 4

    state, _ = RuleBasedPokemonAgent().adapter.parse(obs)

    assert state.opp_hand == 7
    assert state.opp_prizes == 3


def test_midgame_phase_does_not_require_a_prize_taken():
    obs = _make_obs()
    obs["vector"][0] = 8

    state, _ = RuleBasedPokemonAgent().adapter.parse(obs)

    assert state.phase == "MIDGAME"


def test_knockout_attack_beats_non_knockout_and_trainer_setup():
    obs = _make_obs()
    obs["vector"][0] = 8
    obs["vector"][8] = 4
    obs["action_mask"][:3] = 1
    obs["option_types"][:2] = int(OptionType.ATTACK) + 1
    obs["option_types"][2] = int(OptionType.PLAY) + 1
    obs["option_card_ids"][2] = _card_named("Ultra Ball")
    obs["option_features"][0, 9] = 120 / 400
    obs["option_features"][0, 10] = 120 / 400
    obs["option_features"][0, 11] = 1
    obs["option_features"][0, 12] = 2 / 3
    obs["option_features"][1, 9] = 180 / 400
    obs["option_features"][1, 10] = 250 / 400

    assert RuleBasedPokemonAgent().choose_action(obs) == 0


def test_attach_targets_pokemon_one_energy_from_attacking():
    obs = _make_obs()
    obs["vector"][0] = 6
    obs["action_mask"][:2] = 1
    obs["option_types"][:2] = int(OptionType.ATTACH) + 1
    obs["option_features"][0, 2] = 0 / 5
    obs["option_features"][1, 2] = 1 / 5
    obs["entity_features"][:2, 0] = 1
    obs["entity_features"][:2, 35] = 0.5
    obs["entity_features"][0, 25] = 3 / 5
    obs["entity_features"][1, 25] = 1 / 5

    assert RuleBasedPokemonAgent().choose_action(obs) == 1


def test_draw_ability_is_declined_when_it_would_deck_out():
    obs = _make_obs()
    dudunsparce = _card_named("Dudunsparce")
    obs["vector"][0] = 10
    obs["vector"][7] = 2
    obs["action_mask"][:2] = 1
    obs["option_types"][0] = int(OptionType.ABILITY) + 1
    obs["option_card_ids"][0] = dudunsparce
    obs["option_types"][1] = int(OptionType.END) + 1

    assert RuleBasedPokemonAgent().choose_action(obs) == 1


def test_dynamic_damage_attack_uses_hand_size_and_detects_knockout():
    obs = _make_obs()
    alakazam = _card_with_attack_text("Alakazam", "for each card in your hand")
    attack_id = int(alakazam.attacks[0])
    obs["vector"][0] = 8
    obs["vector"][8] = 8
    obs["action_mask"][:2] = 1
    obs["option_types"][0] = int(OptionType.ATTACK) + 1
    obs["option_attack_ids"][0] = attack_id
    obs["option_types"][1] = int(OptionType.END) + 1
    obs["entity_features"][6, 0] = 1
    obs["entity_features"][6, 4] = 140 / 400
    obs["entity_ids"][6] = _card_named("Abra")

    _, decision = RuleBasedPokemonAgent().choose_action(obs, return_info=True)

    assert decision["selected"] == 0
    assert decision["selected_reason"]["attack_damage"] == 16.0
    assert decision["selected_reason"]["attack_knockout"] == 30.0


def test_empty_bench_is_secured_before_non_knockout_attack():
    obs = _make_obs()
    obs["vector"][0] = 8
    obs["action_mask"][:2] = 1
    obs["option_types"][0] = int(OptionType.ATTACK) + 1
    obs["option_features"][0, 9] = 100 / 400
    obs["option_features"][0, 10] = 200 / 400
    obs["option_types"][1] = int(OptionType.PLAY) + 1
    obs["option_card_ids"][1] = _card_named("Buddy-Buddy Poffin")

    assert RuleBasedPokemonAgent().choose_action(obs) == 1


def test_hand_recycling_supporter_is_used_to_avoid_deckout():
    obs = _make_obs()
    obs["vector"][0] = 10
    obs["vector"][7] = 0
    obs["vector"][8] = 10
    obs["action_mask"][:2] = 1
    obs["option_types"][0] = int(OptionType.PLAY) + 1
    obs["option_card_ids"][0] = _card_named("Judge")
    obs["option_types"][1] = int(OptionType.END) + 1

    assert RuleBasedPokemonAgent().choose_action(obs) == 0


def test_copied_attack_estimates_best_benched_attack_damage():
    obs = _make_obs()
    zoroark = next(card for card in all_card_data() if str(card.name) == "N’s Zoroark ex")
    zekrom = next(card for card in all_card_data() if str(card.name) == "N's Zekrom")
    obs["vector"][0] = 8
    obs["action_mask"][:2] = 1
    obs["option_types"][0] = int(OptionType.ATTACK) + 1
    obs["option_attack_ids"][0] = int(zoroark.attacks[0])
    obs["option_types"][1] = int(OptionType.END) + 1
    obs["entity_features"][1, 0] = 1
    obs["entity_ids"][1] = int(zekrom.cardId)
    obs["entity_features"][6, 0] = 1
    obs["entity_features"][6, 4] = 200 / 400
    obs["entity_ids"][6] = _card_named("Abra")

    _, decision = RuleBasedPokemonAgent().choose_action(obs, return_info=True)

    assert decision["selected"] == 0
    assert decision["selected_reason"]["attack_damage"] == 25.0
    assert decision["selected_reason"]["attack_knockout"] == 30.0


def test_retreats_healthy_weak_attacker_for_ready_high_damage_bench():
    obs = _make_obs()
    powerful_alakazam = _card_with_attack_text("Alakazam", "for each card in your hand")
    obs["vector"][0] = 8
    obs["vector"][8] = 10
    obs["action_mask"][:2] = 1
    obs["option_types"][0] = int(OptionType.RETREAT) + 1
    obs["option_types"][1] = int(OptionType.END) + 1
    obs["entity_features"][:2, 0] = 1
    obs["entity_features"][:2, 5] = 140 / 400
    obs["entity_features"][:2, 27] = 1
    obs["entity_features"][:2, 35] = 0.5
    obs["entity_ids"][0] = _card_named("Dedenne")
    obs["entity_ids"][1] = int(powerful_alakazam.cardId)

    assert RuleBasedPokemonAgent().choose_action(obs) == 0
