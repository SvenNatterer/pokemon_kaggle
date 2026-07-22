"""Versioned configuration for the deterministic rule-based opponent league."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import re
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode


RULE_BASED_ALIASES = {"rule", "rule_based", "rule-based", "heuristic", "baseline"}
RULE_BASED_PROFILES = {"balanced", "aggressive", "setup", "defensive"}
RULE_BASED_VARIANTS = {"balanced", "tempo", "engine", "control"}


def normalize_archetype(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "generic").casefold()).strip("_")
    aliases = {
        "mega_abomasnow_ex": "abomasnow",
        "mega_lucario_ex": "lucario",
        "mega_kangaskhan_ex": "kangaskhan",
        "mega_starmie_ex": "starmie",
        "marnies_grimmsnarl_ex": "grimmsnarl",
        "marnie_s_grimmsnarl_ex": "grimmsnarl",
        "archaludon_ex": "archaludon",
        "dragapult_ex": "dragapult",
        "ns_zoroark_ex": "zoroark",
        "n_s_zoroark_ex": "zoroark",
        "team_rockets_mewtwo_ex": "mewtwo",
        "team_rocket_s_mewtwo_ex": "mewtwo",
        "hops_trevenant": "trevenant",
        "hop_s_trevenant": "trevenant",
        "hydrapple_ex": "hydrapple",
    }
    return aliases.get(normalized, normalized or "generic")


@dataclass(frozen=True)
class ArchetypePlan:
    attack: float = 1.0
    setup: float = 1.0
    defense: float = 1.0
    resource: float = 1.0
    prize_trade: float = 1.0
    spread_damage: float = 1.0
    desired_bench: int = 2
    deck_reserve: int = 4


ARCHETYPE_PLANS: dict[str, ArchetypePlan] = {
    "generic": ArchetypePlan(),
    "alakazam": ArchetypePlan(attack=1.05, setup=1.12, resource=1.15, desired_bench=3, deck_reserve=6),
    "dragapult": ArchetypePlan(attack=1.08, setup=1.08, prize_trade=1.10, spread_damage=1.45, desired_bench=3),
    "abomasnow": ArchetypePlan(attack=1.12, setup=1.12, defense=0.92, desired_bench=2),
    "lucario": ArchetypePlan(attack=1.16, setup=1.08, resource=1.04, desired_bench=2),
    "kangaskhan": ArchetypePlan(attack=0.96, defense=1.22, resource=1.10, deck_reserve=8),
    "starmie": ArchetypePlan(attack=1.22, setup=0.96, defense=0.86, prize_trade=1.12, desired_bench=2),
    "grimmsnarl": ArchetypePlan(attack=1.04, setup=1.18, defense=1.08, resource=1.12, desired_bench=3),
    "archaludon": ArchetypePlan(attack=1.08, setup=1.20, resource=1.08, desired_bench=3),
    "mewtwo": ArchetypePlan(attack=1.12, setup=1.10, resource=1.08, desired_bench=3),
    "hydrapple": ArchetypePlan(attack=1.05, setup=1.22, resource=1.08, desired_bench=4),
    "trevenant": ArchetypePlan(attack=1.00, defense=1.12, resource=1.16, deck_reserve=7),
    "zoroark": ArchetypePlan(attack=1.12, setup=1.16, resource=1.06, desired_bench=4),
}


VARIANT_MULTIPLIERS: dict[str, dict[str, float]] = {
    "balanced": {},
    "tempo": {"attack": 1.12, "setup": 0.96, "defense": 0.90, "resource": 0.96},
    "engine": {"attack": 0.96, "setup": 1.13, "defense": 1.00, "resource": 1.08},
    "control": {"attack": 0.94, "setup": 1.02, "defense": 1.16, "resource": 1.12},
}


@dataclass(frozen=True)
class RuleParameters:
    """Bounded, serializable coefficients available to league tuning."""

    attack_damage_scale: float = 0.10
    attack_knockout: float = 30.0
    attack_prize: float = 14.0
    attack_win_game: float = 80.0
    attack_pressure_cap: float = 8.0
    attack_board_empty_penalty: float = 22.0
    attack_board_endgame_penalty: float = 5.0
    attack_ready_bonus: float = 6.0
    setup_sequence_penalty: float = 10000.0
    basic_empty_bench_bonus: float = 30.0
    basic_thin_bench_bonus: float = 12.0
    attach_one_away: float = 34.0
    attach_two_away: float = 24.0
    attach_three_away: float = 16.0
    retreat_damage_weight: float = 24.0
    retreat_condition_weight: float = 9.0
    deckout_penalty: float = 50.0
    damage_counter_knockout: float = 80.0
    damage_counter_prize: float = 20.0
    search_ex_bonus: float = 5.0

    def with_overrides(self, values: Mapping[str, Any] | None) -> "RuleParameters":
        if not values:
            return self
        valid = {item.name for item in fields(self)}
        unknown = sorted(set(values) - valid)
        if unknown:
            raise ValueError(f"unknown rule parameter(s): {', '.join(unknown)}")
        updates = {key: float(value) for key, value in values.items()}
        if any(not -100000.0 <= value <= 100000.0 for value in updates.values()):
            raise ValueError("rule parameter override is outside the safe range")
        return replace(self, **updates)


@dataclass(frozen=True)
class RuleBotSpec:
    version: str = "v3"
    profile: str = "balanced"
    archetype: str = "generic"
    variant: str = "balanced"
    parameter_overrides: tuple[tuple[str, float], ...] = ()

    @property
    def model_spec(self) -> str:
        if self.version == "v3" and self.archetype == "generic":
            base = f"rule_based:{self.profile}"
        elif self.version == "v4":
            base = f"rule_based:v4:{self.archetype}:{self.variant}"
        else:
            base = f"rule_based:v6:{self.archetype}:{self.variant}"
        if not self.parameter_overrides:
            return base
        return f"{base}?{urlencode(self.parameter_overrides)}"

    @property
    def parameters(self) -> RuleParameters:
        return RuleParameters().with_overrides(dict(self.parameter_overrides))


def parse_rule_based_spec(value: Any) -> RuleBotSpec | None:
    normalized = str(value or "").strip().casefold()
    if not normalized:
        return None
    base, _separator, query = normalized.partition("?")
    try:
        overrides = tuple((key, float(raw)) for key, raw in parse_qsl(query, keep_blank_values=False))
        RuleParameters().with_overrides(dict(overrides))
    except (TypeError, ValueError):
        return None
    parts = base.split(":")
    if parts[0] not in RULE_BASED_ALIASES:
        return None
    if len(parts) == 1:
        return RuleBotSpec(parameter_overrides=overrides)
    if len(parts) == 2 and parts[1] in RULE_BASED_PROFILES:
        return RuleBotSpec(version="v3", profile=parts[1], parameter_overrides=overrides)
    if len(parts) not in {2, 3, 4} or parts[1] not in {"v4", "v6"}:
        return None
    version = parts[1]
    archetype = normalize_archetype(parts[2] if len(parts) >= 3 else "generic")
    variant = parts[3] if len(parts) == 4 else "balanced"
    if archetype not in ARCHETYPE_PLANS or variant not in RULE_BASED_VARIANTS:
        return None
    profile = {
        "balanced": "balanced",
        "tempo": "aggressive",
        "engine": "setup",
        "control": "defensive",
    }[variant]
    return RuleBotSpec(
        version=version,
        profile=profile,
        archetype=archetype,
        variant=variant,
        parameter_overrides=overrides,
    )


def archetype_plan(name: Any) -> ArchetypePlan:
    return ARCHETYPE_PLANS[normalize_archetype(name)]
