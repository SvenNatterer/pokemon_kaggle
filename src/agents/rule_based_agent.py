from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import logging
import random
from typing import Any

import numpy as np

from src.cg.api import AreaType, CardType, OptionType, SelectContext, all_attack, all_card_data

LOGGER = logging.getLogger(__name__)

STOP_ACTION = 999
MAX_ENCODED_OPTIONS = 65

VECTOR_TURN_INDEX = 0
VECTOR_MY_HAND_INDEX = 8
VECTOR_MY_PRIZE_INDEX = 10
VECTOR_OPP_HAND_INDEX = 18
VECTOR_OPP_PRIZE_INDEX = 20
VECTOR_SELECT_CONTEXT_INDEX = 250
VECTOR_SELECT_MIN_COUNT_INDEX = 251
VECTOR_SELECT_MAX_COUNT_INDEX = 252
VECTOR_PENDING_COUNT_INDEX = 1490
VECTOR_STOP_LEGAL_INDEX = 1491

ENTITY_ACTIVE_SLOT = 0


def is_rule_based_model_spec(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {
        "rule",
        "rule_based",
        "rule-based",
        "heuristic",
        "baseline",
    }


def _int(value: Any, default: int = 0) -> int:
    raw_value = getattr(value, "value", value)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _enum_name(value: int, enum_cls: Any, default: str = "UNKNOWN") -> str:
    try:
        return enum_cls(value).name
    except Exception:
        return default


def _clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


@dataclass(frozen=True)
class CardMeta:
    card_id: int
    card_type: int
    hp: float
    retreat_cost: int
    energy_type: int
    attack_count: int
    is_basic: bool
    is_stage1: bool
    is_stage2: bool
    is_ex: bool
    is_mega_ex: bool
    is_tera: bool
    is_ace_spec: bool

    @property
    def stage_rank(self) -> int:
        if self.is_stage2:
            return 2
        if self.is_stage1:
            return 1
        return 0


@dataclass(frozen=True)
class AttackMeta:
    attack_id: int
    damage: float
    cost_size: int
    colorless_cost: int
    energy_requirements: tuple[int, ...]


@dataclass(frozen=True)
class ParsedState:
    turn: int
    my_prizes: int
    opp_prizes: int
    my_hand: int
    opp_hand: int
    select_context: int
    min_count: int
    max_count: int
    pending_count: int
    stop_legal: bool
    entity_features: np.ndarray

    @property
    def phase(self) -> str:
        if self.turn <= 2 or self.my_prizes == 6:
            return "SETUP"
        if self.my_prizes <= 2 or self.opp_prizes <= 2:
            return "ENDGAME"
        return "MIDGAME"


@dataclass(frozen=True)
class OptionRecord:
    index: int
    option_type: int
    option_area: int
    card_id: int
    attack_id: int
    features: np.ndarray

    @property
    def type_name(self) -> str:
        return _enum_name(self.option_type, OptionType)

    @property
    def area_name(self) -> str:
        return _enum_name(self.option_area, AreaType)

    @property
    def is_stop_action(self) -> bool:
        return self.index == STOP_ACTION

    @property
    def is_padding(self) -> bool:
        return self.option_type < 0


@dataclass(frozen=True)
class ActionScore:
    index: int
    score: float
    reason: dict[str, float] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _card_meta_by_id() -> dict[int, CardMeta]:
    catalog: dict[int, CardMeta] = {}
    for card in all_card_data():
        card_id = _int(getattr(card, "cardId", 0))
        if card_id <= 0:
            continue
        catalog[card_id] = CardMeta(
            card_id=card_id,
            card_type=_int(getattr(card, "cardType", 0)),
            hp=_float(getattr(card, "hp", 0)),
            retreat_cost=_int(getattr(card, "retreatCost", 0)),
            energy_type=_int(getattr(card, "energyType", 0)),
            attack_count=len(getattr(card, "attacks", None) or []),
            is_basic=bool(getattr(card, "basic", False)),
            is_stage1=bool(getattr(card, "stage1", False)),
            is_stage2=bool(getattr(card, "stage2", False)),
            is_ex=bool(getattr(card, "ex", False)),
            is_mega_ex=bool(getattr(card, "megaEx", False)),
            is_tera=bool(getattr(card, "tera", False)),
            is_ace_spec=bool(getattr(card, "aceSpec", False)),
        )
    return catalog


@lru_cache(maxsize=1)
def _attack_meta_by_id() -> dict[int, AttackMeta]:
    catalog: dict[int, AttackMeta] = {}
    for attack in all_attack():
        attack_id = _int(getattr(attack, "attackId", 0))
        if attack_id <= 0:
            continue
        energy_requirements = tuple(_int(energy) for energy in list(getattr(attack, "energies", None) or []))
        catalog[attack_id] = AttackMeta(
            attack_id=attack_id,
            damage=_float(getattr(attack, "damage", 0)),
            cost_size=len(energy_requirements),
            colorless_cost=sum(1 for energy in energy_requirements if energy == 0),
            energy_requirements=energy_requirements,
        )
    return catalog


class PokemonObservationAdapter:
    """Translate the environment observation into lightweight bot features."""

    def parse(self, observation: dict[str, Any]) -> tuple[ParsedState, list[OptionRecord]]:
        vector = self._as_vector(observation.get("vector"))
        action_mask = self._as_vector(observation.get("action_mask"), dtype=np.int8)
        option_types = self._as_vector(observation.get("option_types"), dtype=np.int32)
        option_areas = self._as_vector(observation.get("option_areas"), dtype=np.int32)
        option_card_ids = self._as_vector(observation.get("option_card_ids"), dtype=np.int32)
        option_attack_ids = self._as_vector(observation.get("option_attack_ids"), dtype=np.int32)
        option_features = self._as_matrix(observation.get("option_features"), rows=MAX_ENCODED_OPTIONS)
        entity_features = self._as_matrix(observation.get("entity_features"), rows=12)

        state = ParsedState(
            turn=int(vector[VECTOR_TURN_INDEX]) if vector.size > VECTOR_TURN_INDEX else 0,
            my_prizes=int(vector[VECTOR_MY_PRIZE_INDEX]) if vector.size > VECTOR_MY_PRIZE_INDEX else 6,
            opp_prizes=int(vector[VECTOR_OPP_PRIZE_INDEX]) if vector.size > VECTOR_OPP_PRIZE_INDEX else 6,
            my_hand=int(vector[VECTOR_MY_HAND_INDEX]) if vector.size > VECTOR_MY_HAND_INDEX else 0,
            opp_hand=int(vector[VECTOR_OPP_HAND_INDEX]) if vector.size > VECTOR_OPP_HAND_INDEX else 0,
            select_context=int(vector[VECTOR_SELECT_CONTEXT_INDEX]) if vector.size > VECTOR_SELECT_CONTEXT_INDEX else 0,
            min_count=int(vector[VECTOR_SELECT_MIN_COUNT_INDEX]) if vector.size > VECTOR_SELECT_MIN_COUNT_INDEX else 0,
            max_count=int(vector[VECTOR_SELECT_MAX_COUNT_INDEX]) if vector.size > VECTOR_SELECT_MAX_COUNT_INDEX else 0,
            pending_count=int(vector[VECTOR_PENDING_COUNT_INDEX]) if vector.size > VECTOR_PENDING_COUNT_INDEX else 0,
            stop_legal=bool(vector[VECTOR_STOP_LEGAL_INDEX]) if vector.size > VECTOR_STOP_LEGAL_INDEX else False,
            entity_features=entity_features,
        )

        options: list[OptionRecord] = []
        for index in self._legal_indices(action_mask):
            if index == STOP_ACTION:
                options.append(
                    OptionRecord(
                        index=STOP_ACTION,
                        option_type=-1,
                        option_area=0,
                        card_id=0,
                        attack_id=0,
                        features=np.zeros(8, dtype=np.float32),
                    )
                )
                continue

            raw_type = int(option_types[index]) if index < len(option_types) else 0
            option_type = raw_type - 1
            features = option_features[index] if index < len(option_features) else np.zeros(8, dtype=np.float32)
            options.append(
                OptionRecord(
                    index=index,
                    option_type=option_type,
                    option_area=int(option_areas[index]) if index < len(option_areas) else 0,
                    card_id=int(option_card_ids[index]) if index < len(option_card_ids) else 0,
                    attack_id=int(option_attack_ids[index]) if index < len(option_attack_ids) else 0,
                    features=np.asarray(features, dtype=np.float32),
                )
            )
        return state, options

    @staticmethod
    def _as_vector(value: Any, dtype=np.float32) -> np.ndarray:
        array = np.asarray(value if value is not None else [], dtype=dtype)
        if array.ndim >= 2:
            array = array[0]
        return array

    @staticmethod
    def _as_matrix(value: Any, rows: int) -> np.ndarray:
        if value is None:
            return np.zeros((rows, 8), dtype=np.float32)
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 3:
            array = array[0]
        if array.ndim == 1:
            return np.zeros((rows, 8), dtype=np.float32)
        return array

    @staticmethod
    def _legal_indices(mask: np.ndarray) -> list[int]:
        return [int(index) for index in np.flatnonzero(mask)]


class RuleBasedPokemonAgent:
    """Transparent heuristic bot for the Pokemon TCG environment."""

    def __init__(
        self,
        temperature: float = 0.0,
        epsilon: float = 0.0,
        seed: int | None = None,
        logger: logging.Logger | None = None,
        debug: bool = False,
    ) -> None:
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")

        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.rng = random.Random(seed)
        self.logger = logger or LOGGER
        self.debug = bool(debug)
        self.adapter = PokemonObservationAdapter()
        self.last_decision: dict[str, Any] | None = None

    def predict(self, obs, state=None, episode_start=None, deterministic: bool = True):
        del state, episode_start, deterministic
        action, _ = self.choose_action(obs, return_info=True)
        return action, None

    def choose_action(self, obs: dict[str, Any], return_info: bool = False):
        parsed_state, options = self.adapter.parse(obs)
        if not options:
            decision = {
                "phase": parsed_state.phase,
                "selected": STOP_ACTION,
                "selected_reason": {"fallback": 0.0},
                "candidates": [],
            }
            self.last_decision = decision
            return (STOP_ACTION, decision) if return_info else STOP_ACTION

        scored_options = [self._score_option(parsed_state, option) for option in options]
        selected = self._select_option(parsed_state, options, scored_options)

        decision = {
            "phase": parsed_state.phase,
            "selected": selected.index,
            "selected_reason": selected.reason,
            "candidates": [
                {
                    "index": item.index,
                    "score": item.score,
                    "reason": item.reason,
                    "type": "STOP" if item.index == STOP_ACTION else _enum_name(self._option_type(options, item.index), OptionType),
                }
                for item in sorted(scored_options, key=lambda item: (item.score, -item.index), reverse=True)
            ],
        }
        self.last_decision = decision

        if self.debug:
            top = sorted(scored_options, key=lambda item: (item.score, -item.index), reverse=True)[:5]
            self.logger.debug(
                "rule bot phase=%s selected=%s score=%.3f top=%s",
                parsed_state.phase,
                selected.index,
                selected.score,
                [(item.index, round(item.score, 3), item.reason) for item in top],
            )

        return (selected.index, decision) if return_info else selected.index

    def _select_option(
        self,
        parsed_state: ParsedState,
        options: list[OptionRecord],
        scored_options: list[ActionScore],
    ) -> ActionScore:
        legal_pairs = [(option, score) for option, score in zip(options, scored_options) if not option.is_padding]
        if not legal_pairs:
            return ActionScore(index=STOP_ACTION, score=0.0, reason={"fallback": 0.0})

        if self.rng.random() < self.epsilon:
            option, score = self.rng.choice(legal_pairs)
            return ActionScore(index=option.index, score=score.score, reason=score.reason)

        if self.temperature > 0.0:
            non_stop_pairs = [(option, score) for option, score in legal_pairs if not option.is_stop_action]
            if non_stop_pairs:
                scores = np.asarray([score.score for _, score in non_stop_pairs], dtype=np.float64)
                scores = scores - scores.max()
                weights = np.exp(scores / self.temperature)
                total = float(weights.sum())
                if total > 0.0 and np.isfinite(total):
                    choice = self.rng.choices(range(len(non_stop_pairs)), weights=weights.tolist(), k=1)[0]
                    option, score = non_stop_pairs[choice]
                    return ActionScore(index=option.index, score=score.score, reason=score.reason)

        if parsed_state.stop_legal:
            best_non_stop = max(
                (score for option, score in legal_pairs if not option.is_stop_action),
                default=ActionScore(index=STOP_ACTION, score=0.0, reason={"commit": 0.0}),
                key=lambda item: (item.score, -item.index),
            )
            if best_non_stop.score <= 0.0:
                return ActionScore(index=STOP_ACTION, score=0.0, reason={"commit": 0.0})
            return best_non_stop

        return max(scored_options, key=lambda item: (item.score, -item.index))

    def _score_option(self, parsed_state: ParsedState, option: OptionRecord) -> ActionScore:
        if option.is_stop_action:
            return ActionScore(index=STOP_ACTION, score=0.0, reason={"commit": 0.0})

        score = 0.0
        reason: dict[str, float] = {}

        def add(name: str, value: float) -> None:
            nonlocal score
            if value == 0:
                return
            score += value
            reason[name] = reason.get(name, 0.0) + value

        card_meta = _card_meta_by_id().get(option.card_id)
        attack_meta = _attack_meta_by_id().get(option.attack_id)

        if option.option_type == int(OptionType.ATTACK):
            if attack_meta is not None:
                add("attack_damage", attack_meta.damage * 1.4)
                add("attack_efficiency", max(0.0, 20.0 - attack_meta.cost_size * 2.5))
                add("attack_colorless", -attack_meta.colorless_cost * 1.5)
            add("attack_phase", self._phase_attack_bonus(parsed_state.phase))
            add("attack_ready", self._active_ready_bonus(parsed_state))

        elif option.option_type == int(OptionType.RETREAT):
            add("retreat_pressure", self._retreat_pressure(parsed_state))
            add("retreat_setup_penalty", -6.0 if parsed_state.phase == "SETUP" else 0.0)

        elif option.option_type == int(OptionType.EVOLVE):
            add("evolve_value", self._evolve_value(card_meta, option, parsed_state))
            add("evolve_phase", 8.0 if parsed_state.phase != "ENDGAME" else -4.0)

        elif option.option_type == int(OptionType.ATTACH):
            add("attach_value", self._attach_value(option, parsed_state))

        elif option.option_type == int(OptionType.PLAY):
            add("play_value", self._play_value(card_meta, parsed_state))

        elif option.option_type == int(OptionType.CARD):
            add("card_value", self._card_value(card_meta, parsed_state))

        elif option.option_type == int(OptionType.YES):
            add("yes_value", self._yes_no_value(parsed_state, True))

        elif option.option_type == int(OptionType.NO):
            add("no_value", self._yes_no_value(parsed_state, False))

        elif option.option_type == int(OptionType.NUMBER):
            add("number_value", self._number_value(parsed_state, option))

        elif option.option_type == int(OptionType.END):
            add("end_turn", -10.0)
            add("end_turn_phase", -6.0 if parsed_state.phase != "ENDGAME" else -1.0)

        elif option.option_type == int(OptionType.SKILL):
            add("skill_value", 4.0)

        elif option.option_type == int(OptionType.SPECIAL_CONDITION):
            add("special_condition_value", 2.0)

        else:
            add("fallback_value", 0.5)

        if card_meta is not None and card_meta.card_type == int(CardType.POKEMON):
            add("pokemon_card", 2.0 if card_meta.is_basic and parsed_state.phase == "SETUP" else 0.5)
        if card_meta is not None and card_meta.card_type == int(CardType.SUPPORTER):
            add("supporter_card", 8.0 if parsed_state.my_hand <= 4 else 4.0)
        if card_meta is not None and card_meta.card_type == int(CardType.STADIUM):
            add("stadium_card", 5.0 if parsed_state.phase != "SETUP" else 2.0)
        if card_meta is not None and card_meta.card_type == int(CardType.ITEM):
            add("item_card", 3.0 if parsed_state.phase == "SETUP" else 2.0)
        if card_meta is not None and card_meta.card_type in {int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)}:
            add("energy_card", self._energy_card_value(option, card_meta, parsed_state))

        if score == 0.0:
            add("neutral_fallback", -0.5)

        return ActionScore(index=option.index, score=score, reason=reason)

    @staticmethod
    def _phase_attack_bonus(phase: str) -> float:
        if phase == "ENDGAME":
            return 8.0
        if phase == "SETUP":
            return -2.0
        return 4.0

    @staticmethod
    def _active_ready_bonus(parsed_state: ParsedState) -> float:
        active = parsed_state.entity_features[ENTITY_ACTIVE_SLOT] if parsed_state.entity_features.size else None
        if active is None or active.size < 29:
            return 0.0
        ready = float(active[27]) + float(active[28])
        if ready > 0.0:
            return 6.0
        if float(active[25]) <= 0.1:
            return 2.0
        return 0.0

    @staticmethod
    def _retreat_pressure(parsed_state: ParsedState) -> float:
        active = parsed_state.entity_features[ENTITY_ACTIVE_SLOT] if parsed_state.entity_features.size else None
        if active is None or active.size < 7:
            return 0.0
        hp_missing = float(active[6])
        if hp_missing > 0.7:
            return 14.0
        if hp_missing > 0.35:
            return 8.0
        return 2.0

    def _evolve_value(self, card_meta: CardMeta | None, option: OptionRecord, parsed_state: ParsedState) -> float:
        if card_meta is None:
            return 0.0
        target_slot = self._target_slot(option)
        target = self._slot_features(parsed_state, target_slot)
        target_bonus = 6.0 if target_slot == ENTITY_ACTIVE_SLOT else 2.0
        if target is not None and target.size >= 7:
            target_bonus += max(0.0, 2.0 - float(target[6]) * 2.0)
        return (
            card_meta.hp / 25.0
            + card_meta.attack_count * 3.0
            + card_meta.stage_rank * 8.0
            + (5.0 if card_meta.is_ex else 0.0)
            + target_bonus
        )

    def _attach_value(self, option: OptionRecord, parsed_state: ParsedState) -> float:
        target_slot = self._target_slot(option)
        target = self._slot_features(parsed_state, target_slot)
        if target is None or target.size < 29:
            return 0.0

        missing_hp = float(target[6])
        energy_count = float(target[20]) * 8.0 if target.size > 20 else 0.0
        attack_deficit = min(
            float(target[25]) * 5.0 if target.size > 25 else 5.0,
            float(target[26]) * 5.0 if target.size > 26 else 5.0,
        )
        ready_bonus = (float(target[27]) if target.size > 27 else 0.0) + (float(target[28]) if target.size > 28 else 0.0)

        score = 15.0
        score += max(0.0, 4.0 - attack_deficit) * 4.0
        score += max(0.0, 6.0 - energy_count) * 1.5
        score += max(0.0, 1.0 - missing_hp) * 2.0
        score += 5.0 if target_slot == ENTITY_ACTIVE_SLOT else 0.0
        score += 4.0 if parsed_state.phase == "SETUP" else 0.0
        score += 2.0 if ready_bonus > 0.0 else 0.0
        return score

    def _play_value(self, card_meta: CardMeta | None, parsed_state: ParsedState) -> float:
        if card_meta is None:
            return 0.0
        if card_meta.card_type == int(CardType.POKEMON):
            return 16.0 if card_meta.is_basic and parsed_state.phase == "SETUP" else 5.0
        if card_meta.card_type == int(CardType.SUPPORTER):
            return 20.0 if parsed_state.my_hand <= 4 else 10.0
        if card_meta.card_type == int(CardType.ITEM):
            return 10.0 if parsed_state.phase != "ENDGAME" else 6.0
        if card_meta.card_type == int(CardType.STADIUM):
            return 8.0 if parsed_state.phase != "SETUP" else 3.0
        if card_meta.card_type in {int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)}:
            return 6.0
        return 2.0

    def _card_value(self, card_meta: CardMeta | None, parsed_state: ParsedState) -> float:
        if card_meta is None:
            return 0.0
        if card_meta.card_type == int(CardType.POKEMON):
            return 12.0 if card_meta.is_basic else 6.0
        if card_meta.card_type == int(CardType.SUPPORTER):
            return 18.0 if parsed_state.my_hand <= 5 else 9.0
        if card_meta.card_type == int(CardType.ITEM):
            return 8.0
        if card_meta.card_type in {int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)}:
            return 4.0
        return 4.0

    @staticmethod
    def _yes_no_value(parsed_state: ParsedState, is_yes: bool) -> float:
        if parsed_state.select_context in {
            int(SelectContext.MULLIGAN),
            int(SelectContext.ACTIVATE),
            int(SelectContext.FIRST_EFFECT),
            int(SelectContext.COIN_HEAD),
            int(SelectContext.MORE_DEVOLVE),
        }:
            return 4.0 if is_yes else -1.0
        if parsed_state.select_context == int(SelectContext.IS_FIRST):
            return 1.0 if is_yes else 0.5
        return 1.5 if is_yes else 0.5

    @staticmethod
    def _number_value(parsed_state: ParsedState, option: OptionRecord) -> float:
        raw_count = int(round(_float(option.features[4] * 60.0))) if option.features.size >= 5 else 0
        if parsed_state.select_context in {
            int(SelectContext.DRAW_COUNT),
            int(SelectContext.DAMAGE_COUNTER_COUNT),
            int(SelectContext.REMOVE_DAMAGE_COUNTER_COUNT),
        }:
            return raw_count * 1.8
        return -raw_count * 0.3

    def _energy_card_value(self, option: OptionRecord, card_meta: CardMeta, parsed_state: ParsedState) -> float:
        target_slot = self._target_slot(option)
        target = self._slot_features(parsed_state, target_slot)
        if target is None:
            return 4.0
        energy_count = float(target[20]) * 8.0 if target.size > 20 else 0.0
        score = 8.0 + max(0.0, 4.0 - energy_count) * 2.0
        if target_slot == ENTITY_ACTIVE_SLOT:
            score += 5.0
        if parsed_state.phase == "SETUP":
            score += 3.0
        if card_meta.energy_type == 0:
            score += 1.0
        return score

    @staticmethod
    def _target_slot(option: OptionRecord) -> int:
        if option.features.size < 3:
            return ENTITY_ACTIVE_SLOT
        return _clamp_int(float(option.features[2]) * 5.0, 0, 5)

    @staticmethod
    def _slot_features(parsed_state: ParsedState, slot: int) -> np.ndarray | None:
        if parsed_state.entity_features.size == 0:
            return None
        if slot < 0 or slot >= parsed_state.entity_features.shape[0]:
            return None
        return parsed_state.entity_features[slot]

    @staticmethod
    def _option_type(options: list[OptionRecord], index: int) -> int:
        for option in options:
            if option.index == index:
                return option.option_type
        return -1
