"""Conservative action filters for known no-effect actions.

Guardrails narrow the engine-provided action mask. They never add actions and
fail open whenever filtering would make the current selection impossible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


POWERFUL_HAND_ATTACK_ID = 1072
MIST_ENERGY_CARD_ID = 11
ROCK_FIGHTING_ENERGY_CARD_ID = 20
FIGHTING_ENERGY_TYPE = 6
TEAM_ROCKET_ARTICUNO_CARD_ID = 414
SPLASHING_DODGE_ATTACK_IDS = frozenset({244, 1266})

# Values are stable engine API enum values. Keeping the guardrail independent
# of an enum import also lets the same module run in the Kaggle submission.
LOG_ATTACK = 15
LOG_HP_CHANGE = 16
LOG_COIN = 22


def _as_int(value: Any, default: int = -1) -> int:
    if hasattr(value, "value"):
        value = value.value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class GuardrailIntervention:
    option_index: int
    rule: str
    attack_id: int
    target_serial: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InferenceGuardrails:
    """Track temporary protection and mask only guaranteed no-effect attacks."""

    def __init__(
        self,
        card_energy_types: dict[int, int] | None = None,
        basic_team_rocket_card_ids: set[int] | None = None,
    ) -> None:
        self._card_energy_types = dict(card_energy_types or {})
        self._basic_team_rocket_card_ids = set(basic_team_rocket_card_ids or ())
        self.reset()

    def set_card_energy_types(self, card_energy_types: dict[int, int]) -> None:
        """Update the public card-type lookup used by conditional card effects."""
        self._card_energy_types = dict(card_energy_types)

    def set_basic_team_rocket_card_ids(self, card_ids: set[int]) -> None:
        """Update IDs protected by Team Rocket's Articuno's Repelling Veil."""
        self._basic_team_rocket_card_ids = set(card_ids)

    def reset(self) -> None:
        self._fully_protected_turn_by_serial: dict[int, int] = {}

    @staticmethod
    def _last_splashing_dodge_result(logs) -> tuple[int, int, bool] | None:
        pending_attack: tuple[int, int] | None = None
        latest_result: tuple[int, int, bool] | None = None

        for log in logs or []:
            log_type = _as_int(getattr(log, "type", None))
            if log_type == LOG_ATTACK:
                attack_id = _as_int(getattr(log, "attackId", None))
                if attack_id in SPLASHING_DODGE_ATTACK_IDS:
                    pending_attack = (
                        _as_int(getattr(log, "playerIndex", None)),
                        _as_int(getattr(log, "serial", None)),
                    )
                else:
                    pending_attack = None
            elif log_type == LOG_COIN and pending_attack is not None:
                coin_player = _as_int(getattr(log, "playerIndex", None))
                if coin_player == pending_attack[0]:
                    latest_result = (
                        pending_attack[0],
                        pending_attack[1],
                        bool(getattr(log, "head", False)),
                    )
                    pending_attack = None

        return latest_result

    def _update_temporary_protection(self, obs) -> None:
        current = getattr(obs, "current", None)
        if current is None:
            self.reset()
            return

        turn = _as_int(getattr(current, "turn", None), 0)
        actor = _as_int(getattr(current, "yourIndex", None))
        self._fully_protected_turn_by_serial = {
            serial: protected_turn
            for serial, protected_turn in self._fully_protected_turn_by_serial.items()
            if protected_turn == turn
        }

        result = self._last_splashing_dodge_result(getattr(obs, "logs", None))
        if result is None:
            return

        attack_player, attacker_serial, heads = result
        # When the observation has already passed control to the opponent, the
        # "during your opponent's next turn" protection is active right now.
        if heads and attacker_serial >= 0 and actor != attack_player:
            self._fully_protected_turn_by_serial[attacker_serial] = turn

    @staticmethod
    def _has_attached_card(pokemon, card_id: int) -> bool:
        return any(
            _as_int(getattr(card, "id", card)) == card_id
            for card in (getattr(pokemon, "energyCards", None) or [])
        )

    def _is_fighting_pokemon(self, pokemon) -> bool:
        card_id = _as_int(getattr(pokemon, "id", None))
        return self._card_energy_types.get(card_id) == FIGHTING_ENERGY_TYPE

    @staticmethod
    def _selection_still_completable(obs, mask, pending_selection) -> bool:
        select = getattr(obs, "select", None)
        options = list(getattr(select, "option", None) or [])
        if select is None or not options:
            return True

        pending_count = len(pending_selection or [])
        minimum = max(0, _as_int(getattr(select, "minCount", None), 0))
        required = max(0, minimum - pending_count)
        selectable = int(np.count_nonzero(mask[: min(len(options), len(mask))]))
        return selectable >= required and bool(np.any(mask))

    def apply(
        self,
        obs,
        encoded_obs: dict[str, Any],
        *,
        perspective: int,
        pending_selection=(),
    ) -> tuple[dict[str, Any], list[GuardrailIntervention]]:
        """Return a copied observation with safely narrowed action legality."""
        self._update_temporary_protection(obs)

        current = getattr(obs, "current", None)
        select = getattr(obs, "select", None)
        original_mask = encoded_obs.get("action_mask")
        if (
            current is None
            or select is None
            or original_mask is None
            or _as_int(getattr(current, "yourIndex", None)) != int(perspective)
        ):
            return encoded_obs, []

        players = list(getattr(current, "players", None) or [])
        opponent_index = 1 - int(perspective)
        if opponent_index < 0 or opponent_index >= len(players):
            return encoded_obs, []
        opponent_active = list(getattr(players[opponent_index], "active", None) or [])
        target = opponent_active[0] if opponent_active else None
        if target is None:
            return encoded_obs, []

        target_serial = _as_int(getattr(target, "serial", None))
        target_card_id = _as_int(getattr(target, "id", None))
        current_turn = _as_int(getattr(current, "turn", None), 0)
        protected_by_splashing_dodge = (
            self._fully_protected_turn_by_serial.get(target_serial) == current_turn
        )
        protected_by_mist_energy = self._has_attached_card(target, MIST_ENERGY_CARD_ID)
        protected_by_rock_fighting_energy = (
            self._has_attached_card(target, ROCK_FIGHTING_ENERGY_CARD_ID)
            and self._is_fighting_pokemon(target)
        )
        opponent_bench = list(getattr(players[opponent_index], "bench", None) or [])
        repelling_veil_in_play = any(
            pokemon is not None
            and _as_int(getattr(pokemon, "id", None)) == TEAM_ROCKET_ARTICUNO_CARD_ID
            for pokemon in opponent_active + opponent_bench
        )
        protected_by_repelling_veil = (
            repelling_veil_in_play
            and target_card_id in self._basic_team_rocket_card_ids
        )
        if not any(
            (
                protected_by_splashing_dodge,
                protected_by_mist_energy,
                protected_by_rock_fighting_energy,
                protected_by_repelling_veil,
            )
        ):
            return encoded_obs, []

        guarded_mask = np.asarray(original_mask).copy()
        interventions: list[GuardrailIntervention] = []
        for index, option in enumerate(list(getattr(select, "option", None) or [])):
            if index >= len(guarded_mask) or not guarded_mask[index]:
                continue
            attack_id = _as_int(getattr(option, "attackId", None))
            if attack_id != POWERFUL_HAND_ATTACK_ID:
                continue

            if protected_by_splashing_dodge:
                rule = "powerful_hand_blocked_by_splashing_dodge"
            elif protected_by_mist_energy:
                rule = "powerful_hand_blocked_by_mist_energy"
            elif protected_by_rock_fighting_energy:
                rule = "powerful_hand_blocked_by_rock_fighting_energy"
            else:
                rule = "powerful_hand_blocked_by_repelling_veil"
            guarded_mask[index] = 0
            interventions.append(
                GuardrailIntervention(
                    option_index=index,
                    rule=rule,
                    attack_id=attack_id,
                    target_serial=target_serial,
                )
            )

        if not interventions:
            return encoded_obs, []
        if not self._selection_still_completable(obs, guarded_mask, pending_selection):
            return encoded_obs, []

        guarded_obs = dict(encoded_obs)
        guarded_obs["action_mask"] = guarded_mask
        return guarded_obs, interventions


class SampledSearchGuardrails:
    """Preview a small sample of risky training actions with the local Search API.

    The initial risk classifier is intentionally narrow: only a legal Powerful
    Hand attack is considered risky. A decision is sampled at most once, and a
    Search/API error fails open so training can continue with the engine mask.
    """

    _KNOWN_BASIC_POKEMON_ID = POWERFUL_HAND_ATTACK_ID
    _KNOWN_BASIC_ENERGY_ID = 1

    def __init__(self, sample_rate: float = 0.0) -> None:
        sample_rate = float(sample_rate)
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("Search guardrail sample rate must be between 0 and 1")
        self.sample_rate = sample_rate
        self.reset()

    def reset(self) -> None:
        self.risky_state_count = 0
        self.sampled_state_count = 0
        self.search_begin_count = 0
        self.search_step_count = 0
        self.failure_count = 0
        self.intervention_count = 0
        self.last_error: str | None = None
        self.last_interventions: list[GuardrailIntervention] = []
        self._decision_cache: dict[tuple[Any, ...], tuple[GuardrailIntervention, ...]] = {}

    @property
    def effective_sample_rate(self) -> float:
        if self.risky_state_count == 0:
            return 0.0
        return self.sampled_state_count / self.risky_state_count

    @staticmethod
    def _target_for(obs, perspective: int):
        current = getattr(obs, "current", None)
        players = list(getattr(current, "players", None) or [])
        opponent_index = 1 - int(perspective)
        if opponent_index < 0 or opponent_index >= len(players):
            return None
        active = list(getattr(players[opponent_index], "active", None) or [])
        return active[0] if active else None

    @staticmethod
    def _pokemon_on_field(player, serial: int):
        field = list(getattr(player, "active", None) or [])
        field.extend(list(getattr(player, "bench", None) or []))
        return next(
            (pokemon for pokemon in field if _as_int(getattr(pokemon, "serial", None)) == serial),
            None,
        )

    @classmethod
    def _search_hypotheses(cls, obs) -> dict[str, list[int]]:
        """Build count-correct placeholders for public, immediate attack previews.

        Powerful Hand's immediate damage outcome depends on the public field, not
        on hidden card identities. These placeholders are therefore only suitable
        for this narrow one-step preview, not for general MCTS rollouts.
        """
        current = obs.current
        actor = _as_int(current.yourIndex)
        yours = current.players[actor]
        opponent = current.players[1 - actor]
        opponent_active = list(getattr(opponent, "active", None) or [])
        hidden_active = bool(opponent_active and opponent_active[0] is None)
        return {
            "your_deck": [cls._KNOWN_BASIC_ENERGY_ID] * int(yours.deckCount),
            "your_prize": [cls._KNOWN_BASIC_ENERGY_ID] * len(yours.prize),
            "opponent_deck": [cls._KNOWN_BASIC_POKEMON_ID] * int(opponent.deckCount),
            "opponent_prize": [cls._KNOWN_BASIC_ENERGY_ID] * len(opponent.prize),
            "opponent_hand": [cls._KNOWN_BASIC_ENERGY_ID] * int(opponent.handCount),
            "opponent_active": [cls._KNOWN_BASIC_POKEMON_ID] if hidden_active else [],
        }

    @staticmethod
    def _decision_key(obs, perspective: int, pending_selection) -> tuple[Any, ...]:
        current = obs.current
        return (
            getattr(obs, "search_begin_input", None),
            _as_int(getattr(current, "turn", None), 0),
            _as_int(getattr(current, "turnActionCount", None), 0),
            int(perspective),
            tuple(pending_selection or ()),
        )

    @staticmethod
    def _apply_cached(
        obs,
        encoded_obs: dict[str, Any],
        interventions: tuple[GuardrailIntervention, ...],
        pending_selection,
    ) -> tuple[dict[str, Any], list[GuardrailIntervention]]:
        if not interventions:
            return encoded_obs, []
        guarded_mask = np.asarray(encoded_obs["action_mask"]).copy()
        for intervention in interventions:
            if 0 <= intervention.option_index < len(guarded_mask):
                guarded_mask[intervention.option_index] = 0
        if not InferenceGuardrails._selection_still_completable(
            obs, guarded_mask, pending_selection
        ):
            return encoded_obs, []
        guarded_obs = dict(encoded_obs)
        guarded_obs["action_mask"] = guarded_mask
        return guarded_obs, list(interventions)

    @staticmethod
    def _is_no_effect(child_obs, opponent_index: int, target_serial: int, before_hp: int) -> bool:
        current = getattr(child_obs, "current", None)
        players = list(getattr(current, "players", None) or [])
        if opponent_index < 0 or opponent_index >= len(players):
            return False

        target_after = SampledSearchGuardrails._pokemon_on_field(
            players[opponent_index], target_serial
        )
        # A missing target was knocked out or otherwise moved, so the attack was
        # not a no-op. For Powerful Hand, unchanged HP means no useful effect.
        if target_after is None or _as_int(getattr(target_after, "hp", None)) != before_hp:
            return False

        target_hp_logs = [
            log
            for log in (getattr(child_obs, "logs", None) or [])
            if _as_int(getattr(log, "type", None)) == LOG_HP_CHANGE
            and _as_int(getattr(log, "serial", None)) == target_serial
        ]
        # The engine emits an explicit zero HP change when protection cancels
        # Powerful Hand. If logs are absent, fail open instead of guessing.
        return bool(target_hp_logs) and all(
            _as_int(getattr(log, "value", None), 0) == 0 for log in target_hp_logs
        )

    def apply(
        self,
        obs,
        encoded_obs: dict[str, Any],
        *,
        perspective: int,
        rng,
        pending_selection=(),
    ) -> tuple[dict[str, Any], list[GuardrailIntervention]]:
        self.last_interventions = []
        current = getattr(obs, "current", None)
        select = getattr(obs, "select", None)
        original_mask = encoded_obs.get("action_mask")
        if (
            self.sample_rate <= 0.0
            or current is None
            or select is None
            or original_mask is None
            or getattr(obs, "search_begin_input", None) is None
            or _as_int(getattr(current, "yourIndex", None)) != int(perspective)
            or pending_selection
            or _as_int(getattr(select, "minCount", None), 0) != 1
            or _as_int(getattr(select, "maxCount", None), 0) != 1
        ):
            return encoded_obs, []

        mask = np.asarray(original_mask)
        risky_options = [
            index
            for index, option in enumerate(list(getattr(select, "option", None) or []))
            if index < len(mask)
            and bool(mask[index])
            and _as_int(getattr(option, "attackId", None)) == POWERFUL_HAND_ATTACK_ID
        ]
        target = self._target_for(obs, perspective)
        if not risky_options or target is None:
            return encoded_obs, []

        key = self._decision_key(obs, perspective, pending_selection)
        cached = self._decision_cache.get(key)
        if cached is not None:
            guarded, interventions = self._apply_cached(
                obs, encoded_obs, cached, pending_selection
            )
            self.last_interventions = interventions
            return guarded, interventions

        self.risky_state_count += 1
        if float(rng.random()) >= self.sample_rate:
            self._decision_cache[key] = ()
            return encoded_obs, []

        self.sampled_state_count += 1
        target_serial = _as_int(getattr(target, "serial", None))
        before_hp = _as_int(getattr(target, "hp", None))
        opponent_index = 1 - int(perspective)
        interventions: list[GuardrailIntervention] = []
        search_started = False
        try:
            from src.cg.api import search_begin, search_end, search_step

            root = search_begin(obs, **self._search_hypotheses(obs))
            search_started = True
            self.search_begin_count += 1
            for option_index in risky_options:
                child = search_step(root.searchId, [option_index])
                self.search_step_count += 1
                if self._is_no_effect(
                    child.observation, opponent_index, target_serial, before_hp
                ):
                    interventions.append(
                        GuardrailIntervention(
                            option_index=option_index,
                            rule="powerful_hand_zero_effect_search",
                            attack_id=POWERFUL_HAND_ATTACK_ID,
                            target_serial=target_serial,
                        )
                    )
        except Exception as error:
            self.failure_count += 1
            self.last_error = f"{type(error).__name__}: {error}"
            interventions = []
        finally:
            if search_started:
                try:
                    search_end()
                except Exception as error:
                    self.failure_count += 1
                    self.last_error = f"{type(error).__name__}: {error}"
                    interventions = []

        cached_interventions = tuple(interventions)
        guarded, accepted = self._apply_cached(
            obs, encoded_obs, cached_interventions, pending_selection
        )
        if len(accepted) != len(cached_interventions):
            cached_interventions = ()
            accepted = []
            guarded = encoded_obs
        self._decision_cache[key] = cached_interventions
        self.intervention_count += len(accepted)
        self.last_interventions = accepted
        return guarded, accepted
