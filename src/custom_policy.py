import torch
import torch.nn as nn
import numpy as np
import re
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from src.cg.api import all_attack, all_card_data
from src.env_wrapper import (
    ENTITY_FEATURE_DIM,
    ENTITY_SLOTS,
    MAX_ATTACK_ID,
    MAX_CARD_ID,
    MAX_ENCODED_OPTIONS,
    OPTION_FEATURE_DIM,
)

CARD_EMBED_DIM = 24
EFFECT_FEATURE_NAMES = (
    "draw", "draw_amount", "draw_until_hand_size", "search_deck", "search_amount",
    "search_pokemon", "search_trainer", "search_energy", "discard", "discard_amount",
    "discard_from_hand", "discard_opponent_hand", "discard_energy", "discard_stadium",
    "discard_tool", "heal", "heal_amount", "damage_counters", "damage_counter_amount",
    "switch_own", "switch_opponent", "attach_energy", "attach_from_hand",
    "attach_from_deck", "attach_from_discard", "move_energy", "return_to_hand",
    "return_to_deck", "recover_from_discard", "poison", "burn", "sleep", "paralyze",
    "confuse", "coin_flip", "bench_effect", "prevent_damage", "prevent_effects",
    "prize_more", "prize_fewer", "once_per_turn", "once_per_game", "conditional",
    "evolve", "devolve", "shuffle_hand", "shuffle_deck", "knock_out_effect",
    # Structured rules-text features. Numeric values are normalized below.
    "damage_multiplier", "damage_bonus", "self_damage", "bench_damage",
    "damage_reduction", "hp_modifier", "retreat_cost_modifier", "effect_amount",
    "coin_flip_amount", "target_amount", "turn_duration", "prize_amount",
    "cannot_attack", "cannot_retreat", "free_retreat", "ignore_weakness",
    "ignore_resistance", "ignore_effects", "cannot_play_cards", "copy_attack",
    "ignore_attack_energy_cost", "deck_top", "deck_bottom", "preserve_order",
    "reveal_opponent_hand", "manipulate_opponent_hand", "exact_amount", "up_to_amount",
    "owner_self", "owner_opponent", "target_active", "target_bench", "target_any_pokemon",
    "source_hand", "source_deck", "source_discard", "destination_hand",
    "destination_deck", "destination_discard", "destination_field",
    "condition_rule_box", "condition_ancient", "condition_future",
    "condition_team_rocket", "condition_name", "does_not_stack",
)
EFFECT_INDEX = {name: index for index, name in enumerate(EFFECT_FEATURE_NAMES)}
ENERGY_SYMBOLS = {
    "C": 0, "G": 1, "R": 2, "W": 3, "L": 4, "P": 5,
    "F": 6, "D": 7, "M": 8, "N": 9,
}
ENERGY_ROLE_NAMES = (
    "required", "searched", "attached", "discarded_or_moved", "provided", "conditional"
)
ENERGY_TYPE_COUNT = 12
ENERGY_FEATURE_OFFSET = len(EFFECT_FEATURE_NAMES)
EFFECT_METADATA_DIM = ENERGY_FEATURE_OFFSET + len(ENERGY_ROLE_NAMES) * ENERGY_TYPE_COUNT
CARD_METADATA_DIM = 54 + EFFECT_METADATA_DIM
ATTACK_EMBED_DIM = 16
ATTACK_METADATA_DIM = 14 + EFFECT_METADATA_DIM
OPTION_EMBED_DIM = 64
MAX_CARD_ATTACKS = 3
MAX_CARD_SKILLS = 3
MAX_SKILL_ID = (MAX_CARD_ID + 1) * MAX_CARD_SKILLS
EVOLUTION_EMBED_DIM = 12


def _effect_metadata(text):
    """Turn English rules text into coarse, learnable game-mechanic features.

    This deliberately encodes observable facts rather than a hand-authored card
    value.  Numeric fields are capped so unusually large printed values do not
    dominate the neural input.
    """
    result = np.zeros(EFFECT_METADATA_DIM, dtype=np.float32)
    normalized = " ".join((text or "").lower().replace("’", "'").split())
    if not normalized:
        return result

    def largest(pattern, scale, cap=1.0):
        values = [int(value) for value in re.findall(pattern, normalized)]
        return min(cap, max(values, default=0) / scale)

    def flag(name, condition):
        result[EFFECT_INDEX[name]] = float(bool(condition))

    flag("draw", "draw " in normalized)
    result[EFFECT_INDEX["draw_amount"]] = largest(r"draw (\d+)", 10.0)
    flag("draw_until_hand_size", "draw until you have" in normalized)
    flag("search_deck", "search your deck" in normalized)
    result[EFFECT_INDEX["search_amount"]] = largest(
        r"(?:up to |for )(\d+) [^.]*(?:card|pokémon|pokemon)", 10.0
    )
    flag("search_pokemon", "search your deck" in normalized and ("pokémon" in normalized or "pokemon" in normalized))
    flag("search_trainer", "search your deck" in normalized and "trainer" in normalized)
    flag("search_energy", "search your deck" in normalized and "energy" in normalized)
    flag("discard", "discard" in normalized)
    result[EFFECT_INDEX["discard_amount"]] = largest(r"discard (?:up to )?(\d+)", 10.0)
    flag("discard_from_hand", "discard" in normalized and "from your hand" in normalized)
    flag("discard_opponent_hand", "discard" in normalized and "opponent's hand" in normalized)
    flag("discard_energy", "discard" in normalized and "energy" in normalized)
    flag("discard_stadium", "discard" in normalized and "stadium" in normalized)
    flag("discard_tool", "discard" in normalized and ("tool" in normalized or "pokémon tool" in normalized))
    flag("heal", "heal " in normalized or "remove all damage" in normalized)
    result[EFFECT_INDEX["heal_amount"]] = largest(r"heal (\d+) damage", 400.0)
    flag("damage_counters", "damage counter" in normalized)
    result[EFFECT_INDEX["damage_counter_amount"]] = largest(r"put (\d+) damage counter", 40.0)
    flag("switch_own", "switch" in normalized and "your opponent's active" not in normalized)
    flag("switch_opponent", "switch" in normalized and "your opponent's active" in normalized)
    flag("attach_energy", "attach" in normalized and "energy" in normalized)
    flag("attach_from_hand", "attach" in normalized and "from your hand" in normalized)
    flag("attach_from_deck", "attach" in normalized and "from your deck" in normalized)
    flag("attach_from_discard", "attach" in normalized and "from your discard" in normalized)
    flag("move_energy", "move" in normalized and "energy" in normalized)
    flag("return_to_hand", "put" in normalized and "into" in normalized and "hand" in normalized)
    flag("return_to_deck", "into" in normalized and "deck" in normalized and "search your deck" not in normalized)
    flag("recover_from_discard", "discard pile" in normalized and ("into your hand" in normalized or "attach" in normalized))
    for name, word in (("poison", "poisoned"), ("burn", "burned"), ("sleep", "asleep"),
                       ("paralyze", "paralyzed"), ("confuse", "confused")):
        flag(name, word in normalized)
    flag("coin_flip", "flip " in normalized and "coin" in normalized)
    flag("bench_effect", "benched pokémon" in normalized or "benched pokemon" in normalized)
    flag("prevent_damage", "prevent" in normalized and "damage" in normalized)
    flag("prevent_effects", "prevent all effects" in normalized)
    flag("prize_more", "more prize card" in normalized)
    flag("prize_fewer", "fewer prize card" in normalized)
    flag("once_per_turn", "once during your turn" in normalized or "once per turn" in normalized)
    flag("once_per_game", "once per game" in normalized)
    flag("conditional", any(word in f" {normalized} " for word in (" if ", " only ", " unless ", " as long as ", " during this turn ")))
    flag("evolve", "evolve" in normalized or "evolution" in normalized)
    flag("devolve", "devolve" in normalized)
    flag("shuffle_hand", "shuffle your hand" in normalized)
    flag("shuffle_deck", "shuffle your deck" in normalized)
    flag("knock_out_effect", "will be knocked out" in normalized or "is knocked out" in normalized)

    # Damage formulas and numeric effect magnitudes. Printed attack damage stays
    # separate; these fields describe only the additional rules text.
    result[EFFECT_INDEX["damage_multiplier"]] = largest(
        r"(?:does |damage is )?(\d+) damage (?:for|times) (?:each|the number)", 400.0
    )
    result[EFFECT_INDEX["damage_bonus"]] = largest(r"(\d+) more damage", 400.0)
    result[EFFECT_INDEX["self_damage"]] = largest(
        r"(?:does|put) (\d+) (?:damage|damage counters?) (?:to|on) (?:itself|this pok[eé]mon)", 400.0
    )
    result[EFFECT_INDEX["bench_damage"]] = largest(
        r"does (\d+) damage to (?:1 of )?.*benched pok[eé]mon", 400.0
    )
    result[EFFECT_INDEX["damage_reduction"]] = largest(r"takes? (\d+) less damage", 400.0)
    result[EFFECT_INDEX["hp_modifier"]] = largest(r"(?:gets?|has) \+?(\d+) hp", 400.0)
    result[EFFECT_INDEX["retreat_cost_modifier"]] = largest(
        r"retreat cost .*? (?:is|by) \{?([0-9]+)\}?", 5.0
    )
    all_numbers = [int(value) for value in re.findall(r"\b(\d+)\b", normalized)]
    result[EFFECT_INDEX["effect_amount"]] = min(1.0, max(all_numbers, default=0) / 400.0)
    result[EFFECT_INDEX["coin_flip_amount"]] = largest(r"flip (\d+) coins?", 10.0)
    result[EFFECT_INDEX["target_amount"]] = largest(
        r"(?:choose|put|move|attach|discard) (?:up to )?(\d+)", 10.0
    )
    result[EFFECT_INDEX["turn_duration"]] = largest(r"(?:for|during) (?:the next )?(\d+) turns?", 5.0)
    if "during your next turn" in normalized or "during your opponent's next turn" in normalized:
        result[EFFECT_INDEX["turn_duration"]] = max(result[EFFECT_INDEX["turn_duration"]], 0.2)
    result[EFFECT_INDEX["prize_amount"]] = largest(r"(\d+) (?:more |fewer )?prize cards?", 6.0)

    flag("cannot_attack", any(phrase in normalized for phrase in (
        "can't use attacks", "cannot use attacks", "can't attack", "cannot attack"
    )))
    flag("cannot_retreat", "can't retreat" in normalized or "cannot retreat" in normalized)
    flag("free_retreat", "has no retreat cost" in normalized or "retreat cost is 0" in normalized)
    flag("ignore_weakness", "don't apply weakness" in normalized or "ignore weakness" in normalized)
    flag("ignore_resistance", "don't apply resistance" in normalized or "ignore resistance" in normalized)
    flag("ignore_effects", "isn't affected by any effects" in normalized or "ignore all effects" in normalized)
    flag("cannot_play_cards", "can't play" in normalized or "cannot play" in normalized)
    flag("copy_attack", any(phrase in normalized for phrase in (
        "choose 1 of", "use it as this attack", "copy an attack", "use one of"
    )) and "attack" in normalized)
    flag("ignore_attack_energy_cost", "without paying its energy cost" in normalized
         or "you don't need the necessary energy" in normalized)
    flag("deck_top", "top " in normalized and ("deck" in normalized or "cards" in normalized))
    flag("deck_bottom", "bottom" in normalized and "deck" in normalized)
    flag("preserve_order", "in any order" in normalized or "in the same order" in normalized)
    flag("reveal_opponent_hand", "opponent reveals their hand" in normalized
         or "reveal your opponent's hand" in normalized)
    flag("manipulate_opponent_hand", "opponent's hand" in normalized and any(
        word in normalized for word in ("discard", "shuffle", "put ", "choose")
    ))
    flag("exact_amount", bool(re.search(r"\b(?:choose|draw|discard|put|move|attach) \d+", normalized)))
    flag("up_to_amount", "up to " in normalized)

    flag("owner_self", "your " in normalized or "you " in normalized)
    flag("owner_opponent", "opponent" in normalized)
    flag("target_active", "active pokémon" in normalized or "active pokemon" in normalized
         or "defending pokémon" in normalized or "defending pokemon" in normalized)
    flag("target_bench", bool(re.search(r"benched(?: [^.,;]+){0,4} pok[eé]mon", normalized)))
    flag("target_any_pokemon", "any pokémon" in normalized or "any pokemon" in normalized)
    for feature, phrases in (
        ("source_hand", ("from your hand", "from their hand")),
        ("source_deck", ("from your deck", "search your deck", "top of your deck")),
        ("source_discard", ("from your discard", "discard pile")),
        ("destination_hand", ("into your hand", "to your hand")),
        ("destination_deck", ("into your deck", "bottom of your deck", "top of your deck")),
        ("destination_discard", ("into the discard", "discard pile")),
        ("destination_field", ("onto your bench", "attach it to", "put it onto")),
    ):
        flag(feature, any(phrase in normalized for phrase in phrases))
    flag("condition_rule_box", "rule box" in normalized)
    flag("condition_ancient", "ancient pokémon" in normalized or "ancient pokemon" in normalized)
    flag("condition_future", "future pokémon" in normalized or "future pokemon" in normalized)
    flag("condition_team_rocket", "team rocket" in normalized)
    flag("condition_name", " in its name" in normalized or "named " in normalized)
    flag("does_not_stack", "doesn't stack" in normalized or "does not stack" in normalized)

    sentences = re.split(r"(?<=[.!?])\s+|\n+", (text or "").replace("’", "'"))
    for sentence in sentences:
        lower = sentence.lower()
        energy_types = {ENERGY_SYMBOLS[symbol] for symbol in re.findall(r"\{([A-Z])\}", sentence) if symbol in ENERGY_SYMBOLS}
        if "every type of energy" in lower or "any type of energy" in lower:
            energy_types.add(10)
        if "team rocket's energy" in lower:
            energy_types.add(11)
        if not energy_types:
            continue
        roles = []
        if "search your deck" in lower:
            roles.append("searched")
        if "attach" in lower:
            roles.append("attached")
        removes_energy = (
            "move" in lower
            or ("discard" in lower and "from your discard" not in lower and "discard pile" not in lower)
        ) and "energy" in lower
        if removes_energy:
            roles.append("discarded_or_moved")
        if "provide" in lower and "energy" in lower:
            roles.append("provided")
        if any(phrase in lower for phrase in ("energy cost", "need ", "required to")):
            roles.append("required")
        if not roles or any(phrase in lower for phrase in ("if ", "for each", "as long as")):
            roles.append("conditional")
        for role in set(roles):
            role_offset = ENERGY_ROLE_NAMES.index(role) * ENERGY_TYPE_COUNT
            for energy_type in energy_types:
                result[ENERGY_FEATURE_OFFSET + role_offset + energy_type] = 1.0
    return result


def _integer_value(value, default=0):
    raw_value = getattr(value, "value", value)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def build_card_metadata():
    """Static, rule-relevant card attributes indexed by card ID."""
    metadata = np.zeros((MAX_CARD_ID + 1, CARD_METADATA_DIM), dtype=np.float32)
    for card in all_card_data():
        card_id = _integer_value(getattr(card, "cardId", 0))
        if not 0 < card_id <= MAX_CARD_ID:
            continue
        card_type = _integer_value(getattr(card, "cardType", 0))
        if 0 <= card_type < 7:
            metadata[card_id, card_type] = 1.0
        metadata[card_id, 7] = float(getattr(card, "retreatCost", 0) or 0) / 5.0
        metadata[card_id, 8] = float(getattr(card, "hp", 0) or 0) / 400.0

        energy_type = _integer_value(getattr(card, "energyType", 0))
        if 0 <= energy_type < 12:
            metadata[card_id, 9 + energy_type] = 1.0
        weakness = _integer_value(getattr(card, "weakness", -1), -1)
        if 0 <= weakness < 12:
            metadata[card_id, 21 + weakness] = 1.0
        resistance = _integer_value(getattr(card, "resistance", -1), -1)
        if 0 <= resistance < 12:
            metadata[card_id, 33 + resistance] = 1.0

        metadata[card_id, 45] = float(bool(getattr(card, "basic", False)))
        metadata[card_id, 46] = float(bool(getattr(card, "stage1", False)))
        metadata[card_id, 47] = float(bool(getattr(card, "stage2", False)))
        metadata[card_id, 48] = float(bool(getattr(card, "ex", False)))
        metadata[card_id, 49] = float(bool(getattr(card, "megaEx", False)))
        metadata[card_id, 50] = float(bool(getattr(card, "tera", False)))
        metadata[card_id, 51] = float(bool(getattr(card, "aceSpec", False)))
        metadata[card_id, 52] = min(1.0, len(getattr(card, "attacks", None) or []) / 3.0)
        metadata[card_id, 53] = min(1.0, len(getattr(card, "skills", None) or []) / 3.0)
        skill_text = " ".join(
            getattr(skill, "text", "") or "" for skill in (getattr(card, "skills", None) or [])
        )
        metadata[card_id, 54:] = _effect_metadata(skill_text)
    return metadata


def build_attack_metadata():
    """Damage and exact typed energy requirements indexed by attack ID."""
    metadata = np.zeros((MAX_ATTACK_ID + 1, ATTACK_METADATA_DIM), dtype=np.float32)
    for attack in all_attack():
        attack_id = _integer_value(getattr(attack, "attackId", 0))
        if not 0 < attack_id <= MAX_ATTACK_ID:
            continue
        energies = list(getattr(attack, "energies", None) or [])
        metadata[attack_id, 0] = float(getattr(attack, "damage", 0) or 0) / 400.0
        metadata[attack_id, 1] = min(1.0, len(energies) / 5.0)
        for energy in energies:
            energy_type = _integer_value(energy)
            if 0 <= energy_type < 12:
                metadata[attack_id, 2 + energy_type] += 0.2
        metadata[attack_id, 14:] = _effect_metadata(getattr(attack, "text", ""))
    return metadata


def build_card_relations():
    """Exact static attack/skill/evolution relations indexed by card ID."""
    attack_ids = np.zeros((MAX_CARD_ID + 1, MAX_CARD_ATTACKS), dtype=np.int64)
    skill_ids = np.zeros((MAX_CARD_ID + 1, MAX_CARD_SKILLS), dtype=np.int64)
    skill_metadata = np.zeros((MAX_SKILL_ID + 1, EFFECT_METADATA_DIM), dtype=np.float32)
    own_name_tokens = np.zeros(MAX_CARD_ID + 1, dtype=np.int64)
    evolves_from_tokens = np.zeros(MAX_CARD_ID + 1, dtype=np.int64)
    cards = [card for card in all_card_data() if 0 < _integer_value(getattr(card, "cardId", 0)) <= MAX_CARD_ID]
    names = sorted({
        str(name).strip().casefold()
        for card in cards
        for name in (getattr(card, "name", ""), getattr(card, "evolvesFrom", None))
        if name
    })
    name_tokens = {name: index + 1 for index, name in enumerate(names)}
    for card in cards:
        card_id = _integer_value(card.cardId)
        for index, attack_id in enumerate((getattr(card, "attacks", None) or [])[:MAX_CARD_ATTACKS]):
            attack_ids[card_id, index] = max(0, min(MAX_ATTACK_ID, _integer_value(attack_id)))
        for index, skill in enumerate((getattr(card, "skills", None) or [])[:MAX_CARD_SKILLS]):
            skill_id = card_id * MAX_CARD_SKILLS + index + 1
            skill_ids[card_id, index] = skill_id
            skill_metadata[skill_id] = _effect_metadata(getattr(skill, "text", ""))
        own_name_tokens[card_id] = name_tokens.get(str(getattr(card, "name", "")).strip().casefold(), 0)
        evolves_from = getattr(card, "evolvesFrom", None)
        if evolves_from:
            evolves_from_tokens[card_id] = name_tokens.get(str(evolves_from).strip().casefold(), 0)
    return attack_ids, skill_ids, skill_metadata, own_name_tokens, evolves_from_tokens, len(names)

class PokemonTCGFeatureExtractor(BaseFeaturesExtractor):
    """
    Legacy-compatible extractor with a structured Observation V2 path.

    Older checkpoints only contain ``vector/action_mask/aux_target`` and keep
    the exact original two-layer MLP. New checkpoints embed categorical IDs,
    preserve per-Pokemon attachments and encode each legal option with shared
    weights.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        vector_dim = observation_space.spaces['vector'].shape[0]
        required_v2_keys = {
            "entity_ids",
            "entity_features",
            "option_card_ids",
            "option_attack_ids",
            "option_features",
        }
        self.structured_v2 = required_v2_keys.issubset(observation_space.spaces)

        if not self.structured_v2:
            # Do not change module names or shapes: existing checkpoints rely on them.
            self.net = nn.Sequential(
                nn.Linear(vector_dim, 256),
                nn.ReLU(),
                nn.Linear(256, features_dim),
                nn.ReLU()
            )
            return

        self.card_embedding = nn.Embedding(MAX_CARD_ID + 1, CARD_EMBED_DIM, padding_idx=0)
        self.attack_embedding = nn.Embedding(MAX_ATTACK_ID + 1, ATTACK_EMBED_DIM, padding_idx=0)
        self.option_type_embedding = nn.Embedding(18, 8, padding_idx=0)
        self.option_area_embedding = nn.Embedding(14, 6, padding_idx=0)
        self.register_buffer("card_metadata", torch.as_tensor(build_card_metadata()))
        self.register_buffer("attack_metadata", torch.as_tensor(build_attack_metadata()))
        relations = build_card_relations()
        self.register_buffer("card_attack_ids", torch.as_tensor(relations[0]))
        self.register_buffer("card_skill_ids", torch.as_tensor(relations[1]))
        self.register_buffer("skill_metadata", torch.as_tensor(relations[2]))
        self.register_buffer("card_name_tokens", torch.as_tensor(relations[3]))
        self.register_buffer("card_evolves_from_tokens", torch.as_tensor(relations[4]))
        self.evolution_embedding = nn.Embedding(relations[5] + 1, EVOLUTION_EMBED_DIM, padding_idx=0)

        keep_mask = np.ones(vector_dim, dtype=np.float32)
        keep_mask[300:650] = 0.0
        for option_index in range(MAX_ENCODED_OPTIONS):
            base = 800 + option_index * 10 if option_index < 50 else 650 + (option_index - 50) * 10
            keep_mask[base + 1] = 0.0  # card ID
            keep_mask[base + 6] = 0.0  # attack ID
        self.register_buffer("vector_keep_mask", torch.as_tensor(keep_mask))

        self.vector_encoder = nn.Sequential(
            nn.LayerNorm(vector_dim),
            nn.Linear(vector_dim, 384),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU(),
        )
        card_repr_dim = (
            CARD_EMBED_DIM + CARD_METADATA_DIM + ATTACK_EMBED_DIM + ATTACK_METADATA_DIM
            + EFFECT_METADATA_DIM + 2 * EVOLUTION_EMBED_DIM
        )
        entity_input_dim = card_repr_dim * 4 + ENTITY_FEATURE_DIM
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        option_input_dim = (
            card_repr_dim + ATTACK_EMBED_DIM + ATTACK_METADATA_DIM + 8 + 6 + OPTION_FEATURE_DIM
        )
        self.option_encoder = nn.Sequential(
            nn.Linear(option_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, OPTION_EMBED_DIM),
            nn.ReLU(),
        )

        # Mean/max/sum/count pooling preserves multiplicity without depending on order.
        pooled_card_dim = card_repr_dim * 3 + 1
        combined_dim = (
            256
            + ENTITY_SLOTS * 64
            + pooled_card_dim
            + 2 * pooled_card_dim
            + 2 * pooled_card_dim
            + 2 * pooled_card_dim
            + pooled_card_dim
            + pooled_card_dim
            + 3 * card_repr_dim
            + 2 * OPTION_EMBED_DIM
        )
        self.net = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def _ids(self, values, maximum):
        return values.long().clamp_(0, maximum)

    def _card_repr(self, card_ids):
        card_ids = self._ids(card_ids, MAX_CARD_ID)
        attack_ids = self.card_attack_ids[card_ids]
        attack_mask = (attack_ids > 0).unsqueeze(-1)
        attack_values = torch.cat(
            [self.attack_embedding(attack_ids), self.attack_metadata[attack_ids]], dim=-1
        )
        attack_summary = (attack_values * attack_mask).sum(dim=-2) / attack_mask.sum(dim=-2).clamp_min(1)
        skill_ids = self.card_skill_ids[card_ids]
        skill_mask = (skill_ids > 0).unsqueeze(-1)
        skill_summary = (self.skill_metadata[skill_ids] * skill_mask).sum(dim=-2) / skill_mask.sum(dim=-2).clamp_min(1)
        return torch.cat(
            [
                self.card_embedding(card_ids), self.card_metadata[card_ids],
                attack_summary, skill_summary,
                self.evolution_embedding(self.card_name_tokens[card_ids]),
                self.evolution_embedding(self.card_evolves_from_tokens[card_ids]),
            ], dim=-1
        )

    @staticmethod
    def _masked_mean(values, ids):
        mask = (ids > 0).unsqueeze(-1).to(values.dtype)
        return (values * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1.0)

    def _mean_card_set(self, card_ids):
        ids = self._ids(card_ids, MAX_CARD_ID)
        return self._masked_mean(self._card_repr(ids), ids)

    def _pool_card_set(self, card_ids):
        ids = self._ids(card_ids, MAX_CARD_ID)
        values = self._card_repr(ids)
        mean = self._masked_mean(values, ids)
        mask = (ids > 0).unsqueeze(-1)
        maximum = values.masked_fill(~mask, -1e9).max(dim=-2).values
        maximum = torch.where((ids > 0).any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum))
        mask_float = (ids > 0).unsqueeze(-1).to(values.dtype)
        count = mask_float.sum(dim=-2)
        summed = (values * mask_float).sum(dim=-2) / 10.0
        normalized_count = torch.log1p(count) / np.log(121.0)
        return torch.cat([mean, maximum, summed, normalized_count], dim=-1)

    def encode_options(self, observations):
        card_ids = self._ids(observations["option_card_ids"], MAX_CARD_ID)
        attack_ids = self._ids(observations["option_attack_ids"], MAX_ATTACK_ID)
        option_types = self._ids(observations["option_types"], 17)
        option_areas = self._ids(observations["option_areas"], 13)
        return self.option_encoder(
            torch.cat(
                [
                    self._card_repr(card_ids),
                    self.attack_embedding(attack_ids),
                    self.attack_metadata[attack_ids],
                    self.option_type_embedding(option_types),
                    self.option_area_embedding(option_areas),
                    observations["option_features"].float(),
                ],
                dim=-1,
            )
        )

    def forward(self, observations):
        if not self.structured_v2:
            return self.net(observations['vector'])

        vector = observations["vector"].float() * self.vector_keep_mask
        vector_features = self.vector_encoder(vector)

        entity_ids = self._ids(observations["entity_ids"], MAX_CARD_ID)
        entity_card_repr = self._card_repr(entity_ids)
        tool_repr = self._card_repr(observations["entity_tool_ids"])
        pre_evolution_repr = self._mean_card_set(observations["entity_pre_evolution_ids"])
        energy_card_repr = self._mean_card_set(observations["entity_energy_card_ids"])
        entities = self.entity_encoder(
            torch.cat(
                [
                    entity_card_repr,
                    tool_repr,
                    pre_evolution_repr,
                    energy_card_repr,
                    observations["entity_features"].float(),
                ],
                dim=-1,
            )
        ).flatten(start_dim=-2)

        hand = self._pool_card_set(observations["hand_ids"])
        discard_ids = observations["discard_ids"]
        our_discard = self._pool_card_set(discard_ids[..., 0, :])
        opponent_discard = self._pool_card_set(discard_ids[..., 1, :])
        prize_ids = observations["prize_ids"]
        our_prizes = self._pool_card_set(prize_ids[..., 0, :])
        opponent_prizes = self._pool_card_set(prize_ids[..., 1, :])
        search = self._pool_card_set(observations["search_ids"])
        looking = self._pool_card_set(observations["looking_ids"])
        own_deck = self._pool_card_set(observations["own_deck_ids"])
        logs = self._pool_card_set(observations["log_card_ids"])
        context = self._card_repr(observations["context_card_ids"]).flatten(start_dim=-2)

        options = self.encode_options(observations)
        option_present = observations["action_mask"][..., :MAX_ENCODED_OPTIONS] > 0
        option_mean = self._masked_mean(options, option_present.long())
        option_max = options.masked_fill(~option_present.unsqueeze(-1), -1e9).max(dim=-2).values
        option_max = torch.where(
            option_present.any(dim=-1, keepdim=True), option_max, torch.zeros_like(option_max)
        )

        return self.net(
            torch.cat(
                [
                    vector_features,
                    entities,
                    hand,
                    our_discard,
                    opponent_discard,
                    our_prizes,
                    opponent_prizes,
                    search,
                    looking,
                    own_deck,
                    logs,
                    context,
                    option_mean,
                    option_max,
                ],
                dim=-1,
            )
        )

class PokemonTCGNetwork(nn.Module):
    """
    Custom Network containing LSTM memory and the three heads:
    Actor, Critic, and Auxiliary (Hand/Deck prediction).
    """
    def __init__(self, feature_dim: int, action_dim: int, aux_dim: int = 2000, hidden_dim: int = 128):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Memory Layer
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Actor Head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Critic Head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Auxiliary Head
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, aux_dim)
        )

    def forward(self, features, hidden_state=None):
        """
        features shape: (batch_size, seq_len, feature_dim) or (batch_size, feature_dim)
        """
        # Ensure 3D for LSTM
        is_2d = False
        if features.dim() == 2:
            is_2d = True
            features = features.unsqueeze(1) # (batch, 1, feature_dim)
            
        lstm_out, hidden_state = self.lstm(features, hidden_state)
        
        if is_2d:
            lstm_out = lstm_out.squeeze(1) # (batch, hidden_dim)
            
        action_logits = self.actor_head(lstm_out)
        values = self.critic_head(lstm_out)
        aux_logits = self.aux_head(lstm_out)
        
        return action_logits, values, aux_logits, hidden_state
