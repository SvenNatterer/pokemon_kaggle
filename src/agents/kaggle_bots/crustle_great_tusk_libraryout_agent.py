"""Great Tusk / Crustle library-out control bot.

Adapted for the local Kaggle-bot wrapper from SOUTA Sakurai's public notebook
``max-elo-1208-libraryout-w-crustle-great-tusk`` (version 3).  The strategy
uses Great Tusk's Land Collapse to mill, Crustle to absorb turns, and a small
Terrakion package when a knockout is the safer winning route.
"""

from __future__ import annotations

import os

from cg.api import AreaType, OptionType, Pokemon, SelectContext, to_observation_class


GREAT_TUSK = 58
DWEBBLE = 344
CRUSTLE = 345
TERRAKION = 607
FIGHTING_GONG = 1142
POKEGEAR = 1122
SWITCH = 1123
BUDDY_BUDDY_POFFIN = 1086
ULTRA_BALL = 1121
JUMBO_ICE_CREAM = 1147
POKE_PAD = 1152
BOSS_ORDERS = 1182
EXPLORERS_GUIDANCE = 1185
COLRESS_TENACITY = 1194
XEROSICS_MACHINATIONS = 1197
LISIAS_APPEAL = 1204
NEUTRALIZATION_ZONE = 1247
MIST_ENERGY = 11
ROCK_FIGHTING_ENERGY = 20
LAND_COLLAPSE = 62
GIANT_TUSK = 63
ASCENSION = 478
SUPERB_SCISSORS = 479
TERRAKION_RETALIATE = 873

DECK = [
    GREAT_TUSK, GREAT_TUSK, GREAT_TUSK, GREAT_TUSK,
    DWEBBLE, DWEBBLE, DWEBBLE, DWEBBLE,
    CRUSTLE, CRUSTLE, CRUSTLE, CRUSTLE, TERRAKION,
    FIGHTING_GONG, FIGHTING_GONG, FIGHTING_GONG, FIGHTING_GONG,
    POKE_PAD, POKE_PAD, POKE_PAD, POKE_PAD,
    BUDDY_BUDDY_POFFIN, BUDDY_BUDDY_POFFIN, BUDDY_BUDDY_POFFIN, BUDDY_BUDDY_POFFIN,
    POKEGEAR, POKEGEAR, POKEGEAR, POKEGEAR, ULTRA_BALL,
    SWITCH, SWITCH, SWITCH, SWITCH,
    XEROSICS_MACHINATIONS, XEROSICS_MACHINATIONS, XEROSICS_MACHINATIONS, XEROSICS_MACHINATIONS,
    EXPLORERS_GUIDANCE, EXPLORERS_GUIDANCE, EXPLORERS_GUIDANCE, EXPLORERS_GUIDANCE,
    BOSS_ORDERS, BOSS_ORDERS, BOSS_ORDERS, BOSS_ORDERS,
    LISIAS_APPEAL, LISIAS_APPEAL, COLRESS_TENACITY, COLRESS_TENACITY,
    NEUTRALIZATION_ZONE, JUMBO_ICE_CREAM,
    ROCK_FIGHTING_ENERGY, ROCK_FIGHTING_ENERGY, ROCK_FIGHTING_ENERGY, ROCK_FIGHTING_ENERGY,
    MIST_ENERGY, MIST_ENERGY, MIST_ENERGY, MIST_ENERGY,
]


def _card(obs, area, index, player_index):
    player = obs.current.players[player_index]
    zones = {
        AreaType.HAND: player.hand,
        AreaType.DISCARD: player.discard,
        AreaType.ACTIVE: player.active,
        AreaType.BENCH: player.bench,
        AreaType.PRIZE: player.prize,
        AreaType.STADIUM: obs.current.stadium,
        AreaType.LOOKING: obs.current.looking,
        AreaType.DECK: obs.select.deck,
    }
    zone = zones.get(area)
    return zone[index] if zone is not None and 0 <= index < len(zone) else None


def _field(player):
    return [pokemon for pokemon in player.active + player.bench if pokemon is not None]


def _active(player):
    return player.active[0] if player.active and player.active[0] is not None else None


def _in_hand(player, card_id):
    return any(card.id == card_id for card in player.hand)


def _has_field(player, card_id):
    return any(pokemon.id == card_id for pokemon in _field(player))


def _is_ex(pokemon):
    return bool(pokemon and getattr(pokemon, "ex", False))


def _context_name(context):
    return getattr(context, "name", str(context))


def _score_card(card, context, me, opponent, state):
    if card is None:
        return -100_000
    card_id = card.id
    active = _active(me)
    opponent_deck = opponent.deckCount
    if context in {"TO_HAND", "TO_DECK"}:
        if card_id == EXPLORERS_GUIDANCE and active and active.id == GREAT_TUSK and not state.supporterPlayed:
            return 100_000
        if card_id == GREAT_TUSK:
            return 90_000 if not _has_field(me, GREAT_TUSK) else 35_000
        if card_id == DWEBBLE:
            return 80_000 if not _has_field(me, DWEBBLE) else 20_000
        if card_id == CRUSTLE and _has_field(me, DWEBBLE):
            return 75_000
        if card_id in (ROCK_FIGHTING_ENERGY, MIST_ENERGY):
            return 60_000 if active and active.id == GREAT_TUSK and len(active.energies) < 2 else 18_000
        if card_id in (BOSS_ORDERS, LISIAS_APPEAL):
            return 50_000 if opponent_deck <= 12 else 12_000
        return 5_000
    if context in {"DISCARD", "DISCARD_CARD_OR_ATTACHED_CARD"}:
        if card_id in (EXPLORERS_GUIDANCE, GREAT_TUSK):
            return -50_000
        if card_id in (ROCK_FIGHTING_ENERGY, MIST_ENERGY) and active and active.id == GREAT_TUSK and len(active.energies) < 2:
            return -30_000
        if card_id in (JUMBO_ICE_CREAM, NEUTRALIZATION_ZONE):
            return 20_000
        return 10_000
    if context in {"SETUP_ACTIVE_POKEMON", "TO_ACTIVE", "SWITCH"} and isinstance(card, Pokemon):
        if card.id == GREAT_TUSK:
            return 90_000 if len(card.energies) >= 2 else 45_000
        if card.id == CRUSTLE:
            return 80_000 if any(_is_ex(pokemon) for pokemon in _field(opponent)) else 35_000
        if card.id == DWEBBLE:
            return 30_000
        return 10_000
    if context in {"SETUP_BENCH_POKEMON", "TO_BENCH", "TO_FIELD"}:
        return {GREAT_TUSK: 80_000, DWEBBLE: 70_000, TERRAKION: 20_000}.get(card_id, 1_000)
    if context in {"EVOLVES_TO", "EVOLVE"}:
        return 90_000 if card_id == CRUSTLE else 1_000
    if context == "ATTACH_FROM" and isinstance(card, Pokemon):
        return {GREAT_TUSK: 90_000, CRUSTLE: 50_000, TERRAKION: 20_000}.get(card.id, 1_000)
    return 0


def _main_score(option, obs, me, opponent, state):
    active = _active(me)
    if option.type == OptionType.ATTACK:
        attack_id = option.attackId
        if attack_id == LAND_COLLAPSE:
            return 300_000 + (100_000 if state.supporterPlayed else 0)
        if attack_id == ASCENSION:
            return 180_000 if active and active.id == DWEBBLE else 10_000
        if attack_id in (GIANT_TUSK, SUPERB_SCISSORS, TERRAKION_RETALIATE):
            return 190_000 if opponent.deckCount > 16 else 20_000
        return 5_000
    if option.type == OptionType.PLAY:
        card = _card(obs, AreaType.HAND, option.index, state.yourIndex)
        if card is None:
            return 0
        if card.id == EXPLORERS_GUIDANCE:
            return 280_000 if active and active.id == GREAT_TUSK and not state.supporterPlayed else 8_000
        if card.id in (GREAT_TUSK, DWEBBLE):
            return 100_000 if not _has_field(me, card.id) else 10_000
        if card.id in (FIGHTING_GONG, POKEGEAR, ULTRA_BALL, BUDDY_BUDDY_POFFIN):
            return 65_000
        if card.id == NEUTRALIZATION_ZONE:
            return 55_000 if any(_is_ex(pokemon) for pokemon in _field(opponent)) else 8_000
        if card.id in (BOSS_ORDERS, LISIAS_APPEAL):
            return 45_000 if opponent.deckCount <= 12 else 5_000
        if card.id == XEROSICS_MACHINATIONS:
            return 35_000
        return 8_000
    if option.type == OptionType.EVOLVE:
        return 150_000
    if option.type == OptionType.ATTACH:
        target = _card(obs, option.inPlayArea, option.inPlayIndex, state.yourIndex)
        return 100_000 if target and target.id == GREAT_TUSK else 25_000
    if option.type == OptionType.RETREAT:
        return 120_000 if active and active.id == DWEBBLE else 15_000
    if option.type == OptionType.END:
        return -100
    return 1_000


def _choose(obs):
    state = obs.current
    select = obs.select
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    context = _context_name(select.context)
    scores = []
    for option in select.option:
        if context == "MAIN":
            score = _main_score(option, obs, me, opponent, state)
        elif option.type == OptionType.CARD:
            score = _score_card(_card(obs, option.area, option.index, option.playerIndex), context, me, opponent, state)
        elif option.type == OptionType.YES:
            score = 100
        elif option.type == OptionType.NO:
            score = 0
        elif option.type == OptionType.NUMBER:
            score = -(option.number or 0) if context == "DRAW_COUNT" else (option.number or 0)
        else:
            score = 0
        scores.append(score)
    ordered = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    chosen = []
    for index in ordered:
        if len(chosen) >= select.maxCount:
            break
        if scores[index] >= 0 or len(chosen) < select.minCount:
            chosen.append(index)
    return chosen


def agent(obs_dict: dict, configuration=None) -> list[int]:
    """Return the submitted deck during setup and legal option indices otherwise."""
    try:
        obs = to_observation_class(obs_dict)
        return DECK if obs.select is None else _choose(obs)
    except Exception:
        select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
        if select is None:
            return DECK
        minimum = max(0, int(select.get("minCount", 0)))
        return list(range(min(minimum, len(select.get("option") or []))))
