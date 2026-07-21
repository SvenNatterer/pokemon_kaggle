"""Bounded look-ahead teacher for collecting tactical action labels.

The live battle is never mutated.  The teacher uses the engine Search API to
branch from the actor-visible observation, evaluates a small minimax tree and
returns scores for simple one-option decisions.  It is intentionally an
offline/training utility; running it for every Kaggle decision would be too
slow.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, islice
from typing import Any, Iterable

import numpy as np

from src.cg.api import (
    all_card_data,
    search_begin,
    search_end,
    search_release,
    search_step,
)


def _as_int(value: Any, default: int = 0) -> int:
    value = getattr(value, "value", value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _card_id(card: Any) -> int:
    return _as_int(getattr(card, "id", card), 0)


@dataclass(frozen=True)
class TeacherDecision:
    """One teacher label and the action values that produced it."""

    action: int
    scores: dict[int, float]
    confidence: float
    successful_hypotheses: int
    searched_nodes: int


@dataclass(frozen=True)
class LookaheadConfig:
    """Small defaults keep collection practical on CPU."""

    max_depth: int = 5
    beam_width: int = 3
    node_budget: int = 96
    max_combinations: int = 16
    terminal_value: float = 10_000.0

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.beam_width < 1:
            raise ValueError("beam_width must be at least 1")
        if self.node_budget < 1:
            raise ValueError("node_budget must be at least 1")
        if self.max_combinations < 1:
            raise ValueError("max_combinations must be at least 1")


@dataclass
class _SearchBudget:
    remaining: int
    visited: int = 0

    def spend(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        self.visited += 1
        return True


class LookaheadTeacher:
    """Evaluate single-choice root actions with a bounded minimax search.

    Search nodes after the root may contain multi-card selections.  Those are
    supported with a strict combination cap, but the returned label is limited
    to roots with ``minCount == maxCount == 1`` so it maps exactly to one V6
    policy action.
    """

    def __init__(
        self,
        config: LookaheadConfig | None = None,
        *,
        card_data_by_id: dict[int, Any] | None = None,
        search_begin_fn=search_begin,
        search_step_fn=search_step,
        search_release_fn=search_release,
        search_end_fn=search_end,
    ) -> None:
        self.config = config or LookaheadConfig()
        self.card_data_by_id = card_data_by_id or {
            _as_int(getattr(card, "cardId", 0)): card for card in all_card_data()
        }
        self._search_begin = search_begin_fn
        self._search_step = search_step_fn
        self._search_release = search_release_fn
        self._search_end = search_end_fn
        self.last_error: str | None = None

    def choose(
        self,
        raw_observation,
        encoded_observation: dict[str, Any],
        *,
        perspective: int,
        hypotheses: dict[str, list[int]] | Iterable[dict[str, list[int]]],
    ) -> TeacherDecision | None:
        """Return a teacher action, or ``None`` when the root is unsupported.

        Scores are averaged over the supplied hidden-card hypotheses.  Search
        errors fail open: a broken hypothesis is skipped and collection can
        continue.
        """

        self.last_error = None
        select = getattr(raw_observation, "select", None)
        current = getattr(raw_observation, "current", None)
        if select is None or current is None:
            return None
        if _as_int(getattr(current, "yourIndex", None), -1) != int(perspective):
            return None
        if (
            _as_int(getattr(select, "minCount", None), -1) != 1
            or _as_int(getattr(select, "maxCount", None), -1) != 1
        ):
            return None

        options = list(getattr(select, "option", None) or [])
        mask = np.asarray(encoded_observation.get("action_mask", []))
        legal_actions = [
            index
            for index in range(min(len(options), len(mask)))
            if bool(mask[index])
        ]
        # A forced choice contains no useful policy supervision.
        if len(legal_actions) < 2:
            return None

        if isinstance(hypotheses, dict):
            hypothesis_batch = [hypotheses]
        else:
            hypothesis_batch = list(hypotheses)
        if not hypothesis_batch:
            raise ValueError("At least one hidden-card hypothesis is required")

        aggregated: dict[int, list[float]] = {action: [] for action in legal_actions}
        total_nodes = 0
        successful_hypotheses = 0

        for hypothesis in hypothesis_batch:
            hypothesis_nodes = 0
            search_started = False
            try:
                root = self._search_begin(raw_observation, **hypothesis)
                search_started = True
                hypothesis_scores: dict[int, float] = {}
                per_action_budget = max(1, self.config.node_budget // len(legal_actions))
                for action in legal_actions:
                    budget = _SearchBudget(per_action_budget)
                    if not budget.spend():
                        break
                    child = self._search_step(root.searchId, [int(action)])
                    try:
                        hypothesis_scores[action] = self._minimax(
                            child,
                            perspective=int(perspective),
                            depth=1,
                            budget=budget,
                        )
                    finally:
                        self._safe_release(child.searchId)
                        hypothesis_nodes += budget.visited

                if len(hypothesis_scores) == len(legal_actions):
                    for action, score in hypothesis_scores.items():
                        aggregated[action].append(float(score))
                    successful_hypotheses += 1
            except Exception as error:  # Search must not stop data collection.
                self.last_error = f"{type(error).__name__}: {error}"
            finally:
                total_nodes += hypothesis_nodes
                if search_started:
                    try:
                        self._search_end()
                    except Exception as error:
                        self.last_error = f"{type(error).__name__}: {error}"

        if successful_hypotheses == 0:
            return None

        scores = {
            action: float(np.mean(values))
            for action, values in aggregated.items()
            if values
        }
        if len(scores) != len(legal_actions):
            return None

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        confidence = ranked[0][1] - ranked[1][1]
        return TeacherDecision(
            action=ranked[0][0],
            scores=scores,
            confidence=float(confidence),
            successful_hypotheses=successful_hypotheses,
            searched_nodes=total_nodes,
        )

    def _minimax(
        self,
        node,
        *,
        perspective: int,
        depth: int,
        budget: _SearchBudget,
    ) -> float:
        observation = node.observation
        terminal = self._terminal_score(observation, perspective, depth)
        if terminal is not None:
            return terminal
        if depth >= self.config.max_depth or budget.remaining <= 0:
            return self._heuristic_score(observation, perspective)

        selections = self._candidate_selections(observation)
        if not selections:
            return self._heuristic_score(observation, perspective)

        actor = _as_int(getattr(observation.current, "yourIndex", None), -1)
        maximizing = actor == perspective
        previewed: list[tuple[float, Any]] = []

        # Preview children once, retain the strongest beam and then recurse.
        for selection in selections:
            if not budget.spend():
                break
            child = self._search_step(node.searchId, selection)
            preview = self._terminal_score(child.observation, perspective, depth + 1)
            if preview is None:
                preview = self._heuristic_score(child.observation, perspective)
            previewed.append((float(preview), child))

        if not previewed:
            return self._heuristic_score(observation, perspective)

        previewed.sort(key=lambda item: item[0], reverse=maximizing)
        retained = previewed[: self.config.beam_width]
        discarded = previewed[self.config.beam_width :]
        for _, child in discarded:
            self._safe_release(child.searchId)

        values: list[float] = []
        for _, child in retained:
            try:
                values.append(
                    self._minimax(
                        child,
                        perspective=perspective,
                        depth=depth + 1,
                        budget=budget,
                    )
                )
            finally:
                self._safe_release(child.searchId)

        return (max if maximizing else min)(values)

    def _candidate_selections(self, observation) -> list[list[int]]:
        select = getattr(observation, "select", None)
        options = list(getattr(select, "option", None) or [])
        if select is None or not options:
            return []

        minimum = max(0, _as_int(getattr(select, "minCount", None), 0))
        maximum = min(len(options), max(0, _as_int(getattr(select, "maxCount", None), 0)))
        if maximum < minimum:
            return []

        result: list[list[int]] = []
        for count in range(minimum, maximum + 1):
            remaining = self.config.max_combinations - len(result)
            if remaining <= 0:
                break
            result.extend(
                [list(items) for items in islice(combinations(range(len(options)), count), remaining)]
            )
        return result

    def _terminal_score(self, observation, perspective: int, depth: int) -> float | None:
        current = getattr(observation, "current", None)
        if current is None:
            return None
        result = _as_int(getattr(current, "result", None), -1)
        if result == -1:
            return None
        if result == perspective:
            return self.config.terminal_value - depth
        if result == 1 - perspective:
            return -self.config.terminal_value + depth
        return 0.0

    def _heuristic_score(self, observation, perspective: int) -> float:
        current = getattr(observation, "current", None)
        players = list(getattr(current, "players", None) or [])
        if len(players) != 2:
            return 0.0
        me = players[perspective]
        opponent = players[1 - perspective]

        prize_margin = len(getattr(opponent, "prize", None) or []) - len(
            getattr(me, "prize", None) or []
        )
        hp_margin = self._field_hp(me) - self._field_hp(opponent)
        energy_margin = self._field_energy(me) - self._field_energy(opponent)
        board_margin = len(self._field(me)) - len(self._field(opponent))
        deck_margin = _as_int(getattr(me, "deckCount", None)) - _as_int(
            getattr(opponent, "deckCount", None)
        )
        active_liability_margin = self._active_prize_value(opponent) - self._active_prize_value(me)

        return float(
            350.0 * prize_margin
            + 0.08 * hp_margin
            + 6.0 * energy_margin
            + 12.0 * board_margin
            + 0.5 * deck_margin
            + 20.0 * active_liability_margin
        )

    @staticmethod
    def _field(player) -> list[Any]:
        field = list(getattr(player, "active", None) or [])
        field.extend(list(getattr(player, "bench", None) or []))
        return [pokemon for pokemon in field if pokemon is not None]

    def _field_hp(self, player) -> int:
        return sum(max(0, _as_int(getattr(pokemon, "hp", None))) for pokemon in self._field(player))

    def _field_energy(self, player) -> int:
        return sum(len(getattr(pokemon, "energies", None) or []) for pokemon in self._field(player))

    def _active_prize_value(self, player) -> int:
        active = list(getattr(player, "active", None) or [])
        if not active or active[0] is None:
            return 0
        card = self.card_data_by_id.get(_card_id(active[0]))
        if card is not None and bool(getattr(card, "megaEx", False)):
            return 3
        if card is not None and bool(getattr(card, "ex", False)):
            return 2
        return 1

    def _safe_release(self, search_id: int) -> None:
        try:
            self._search_release(search_id)
        except Exception:
            # search_end() remains the final cleanup boundary.
            pass


def _visible_card_ids(player, *, include_hand: bool) -> list[int]:
    """Return public card IDs that can be removed from a deck hypothesis."""

    visible: list[int] = []
    zones = [getattr(player, "discard", None) or []]
    if include_hand:
        zones.append(getattr(player, "hand", None) or [])
    for zone in zones:
        visible.extend(card_id for card_id in map(_card_id, zone) if card_id > 0)

    field = list(getattr(player, "active", None) or [])
    field.extend(list(getattr(player, "bench", None) or []))
    for pokemon in field:
        if pokemon is None:
            continue
        visible_id = _card_id(pokemon)
        if visible_id > 0:
            visible.append(visible_id)
        for attribute in ("preEvolution", "tools", "energyCards"):
            visible.extend(
                card_id
                for card_id in map(_card_id, getattr(pokemon, attribute, None) or [])
                if card_id > 0
            )
    return visible


def _hidden_pool(
    deck: list[int],
    visible: list[int],
    needed: int,
    rng: np.random.Generator,
) -> list[int]:
    remaining = Counter(int(card_id) for card_id in deck if int(card_id) > 0)
    for card_id in visible:
        if remaining[card_id] > 0:
            remaining[card_id] -= 1
    pool = list(remaining.elements())

    # Public state can omit ownership information (for example a stadium), so
    # fill a short pool from legal deck IDs instead of failing collection.
    legal = [int(card_id) for card_id in deck if int(card_id) > 0]
    if not legal:
        raise ValueError("A non-empty deck is required to build search hypotheses")
    while len(pool) < needed:
        pool.append(int(rng.choice(legal)))
    rng.shuffle(pool)
    return pool[:needed]


def build_search_hypotheses(
    observation,
    *,
    your_deck: list[int],
    opponent_deck: list[int],
    count: int = 4,
    rng: np.random.Generator | None = None,
    card_data_by_id: dict[int, Any] | None = None,
) -> list[dict[str, list[int]]]:
    """Sample count-correct hidden-card inputs for ``search_begin``.

    The opponent deck list is privileged teacher information.  It is used only
    to create simulations and is never added to the student's observation.
    Averaging several shuffles reduces labels that depend on one lucky hidden
    hand or prize arrangement.
    """

    if count < 1:
        raise ValueError("count must be at least 1")
    current = getattr(observation, "current", None)
    if current is None:
        raise ValueError("Cannot build hypotheses without a current battle state")
    perspective = _as_int(getattr(current, "yourIndex", None), -1)
    players = list(getattr(current, "players", None) or [])
    if perspective not in (0, 1) or len(players) != 2:
        raise ValueError("Observation has no valid acting player")

    rng = rng or np.random.default_rng()
    card_data = card_data_by_id or {
        _as_int(getattr(card, "cardId", 0)): card for card in all_card_data()
    }
    yours = players[perspective]
    opponent = players[1 - perspective]
    your_deck_count = max(0, _as_int(getattr(yours, "deckCount", None)))
    your_prize_count = len(getattr(yours, "prize", None) or [])
    opponent_deck_count = max(0, _as_int(getattr(opponent, "deckCount", None)))
    opponent_prize_count = len(getattr(opponent, "prize", None) or [])
    opponent_hand_count = max(0, _as_int(getattr(opponent, "handCount", None)))

    hidden_active = bool(
        list(getattr(opponent, "active", None) or [])
        and list(getattr(opponent, "active", None) or [])[0] is None
    )
    basic_ids = [
        card_id
        for card_id in opponent_deck
        if card_id in card_data
        and bool(getattr(card_data[card_id], "basic", False))
    ]
    if hidden_active and not basic_ids:
        raise ValueError("Opponent deck needs a Basic Pokemon for a hidden Active hypothesis")

    result: list[dict[str, list[int]]] = []
    for _ in range(count):
        your_pool = _hidden_pool(
            list(your_deck),
            _visible_card_ids(yours, include_hand=True),
            your_deck_count + your_prize_count,
            rng,
        )
        opponent_pool = _hidden_pool(
            list(opponent_deck),
            _visible_card_ids(opponent, include_hand=False),
            opponent_deck_count + opponent_prize_count + opponent_hand_count,
            rng,
        )
        your_prize = your_pool[:your_prize_count]
        predicted_your_deck = your_pool[your_prize_count:]
        opponent_hand = opponent_pool[:opponent_hand_count]
        opponent_prize_start = opponent_hand_count
        opponent_prize_end = opponent_prize_start + opponent_prize_count
        opponent_prize = opponent_pool[opponent_prize_start:opponent_prize_end]
        predicted_opponent_deck = opponent_pool[opponent_prize_end:]

        result.append(
            {
                "your_deck": predicted_your_deck,
                "your_prize": your_prize,
                "opponent_deck": predicted_opponent_deck,
                "opponent_prize": opponent_prize,
                "opponent_hand": opponent_hand,
                "opponent_active": [int(rng.choice(basic_ids))] if hidden_active else [],
            }
        )
    return result
