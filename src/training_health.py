"""Shared health summaries and eligibility gates for PPO runs and evaluations."""

from __future__ import annotations

import math
from typing import Any, Iterable


HEALTH_SCHEMA_VERSION = 1
# V6 encodes option slots 0..64. The final histogram bucket represents every
# larger raw engine option count while retaining a compact IPC-safe payload.
MAX_TRACKED_OPTION_COUNT = 65
OPTION_COUNT_HISTOGRAM_SIZE = MAX_TRACKED_OPTION_COUNT + 2
HEALTH_COUNTER_KEYS = (
    "invalid_learner_actions",
    "option_overflows",
    "engine_errors",
)


def empty_option_count_histogram() -> list[int]:
    return [0] * OPTION_COUNT_HISTOGRAM_SIZE


def normalize_option_count_histogram(values: Any) -> list[int]:
    """Return the fixed-size, non-negative option-count histogram."""
    normalized = empty_option_count_histogram()
    if not isinstance(values, (list, tuple)):
        return normalized
    for index, value in enumerate(values[:OPTION_COUNT_HISTOGRAM_SIZE]):
        try:
            normalized[index] = max(0, int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def merge_option_count_histograms(histograms: Iterable[Any]) -> list[int]:
    merged = empty_option_count_histogram()
    for values in histograms:
        for index, count in enumerate(normalize_option_count_histogram(values)):
            merged[index] += count
    return merged


def option_count_percentiles(histogram: Any) -> dict[str, int | None]:
    """Compute bounded option-count percentiles from the compact histogram."""
    values = normalize_option_count_histogram(histogram)
    total = sum(values)
    if total <= 0:
        return {name: None for name in ("p50", "p90", "p95", "p99")}

    result: dict[str, int] = {}
    for name, quantile in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)):
        target = max(1, math.ceil(total * quantile))
        cumulative = 0
        for count, frequency in enumerate(values):
            cumulative += frequency
            if cumulative >= target:
                result[name] = count
                break
    return result


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def summarize_health(
    *,
    episodes: int = 0,
    learner_decisions: int = 0,
    invalid_learner_actions: int = 0,
    option_overflows: int = 0,
    engine_errors: int = 0,
    max_option_count_seen: int = 0,
    option_count_histogram: Any = None,
    opponent_episodes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build the persisted health record from raw counters."""
    histogram = normalize_option_count_histogram(option_count_histogram)
    learner_decisions = _non_negative_int(learner_decisions)
    if learner_decisions == 0:
        learner_decisions = sum(histogram)
    episodes = _non_negative_int(episodes)
    invalid_learner_actions = _non_negative_int(invalid_learner_actions)
    option_overflows = _non_negative_int(option_overflows)
    engine_errors = _non_negative_int(engine_errors)
    max_option_count_seen = _non_negative_int(max_option_count_seen)
    return {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "episodes": episodes,
        "learner_decisions": learner_decisions,
        "invalid_learner_actions": invalid_learner_actions,
        "invalid_learner_action_rate": (
            invalid_learner_actions / learner_decisions if learner_decisions else 0.0
        ),
        "option_overflows": option_overflows,
        "engine_errors": engine_errors,
        "max_option_count_seen": max_option_count_seen,
        "option_count_histogram": histogram,
        "option_count_percentiles": option_count_percentiles(histogram),
        "opponent_episodes": dict(sorted((opponent_episodes or {}).items())),
    }


def health_gate(health: dict[str, Any] | None, *, crashes: int = 0) -> dict[str, Any]:
    """Return the hard P0 eligibility decision for one run or evaluation."""
    health = health or {}
    crashes = _non_negative_int(crashes)
    violations: list[str] = []
    if crashes:
        violations.append(f"evaluation_crashes={crashes}")
    for key in HEALTH_COUNTER_KEYS:
        value = _non_negative_int(health.get(key, 0))
        if value:
            violations.append(f"{key}={value}")
    return {
        "passed": not violations,
        "violations": violations,
        "policy": "zero_tolerance_engine_errors_invalid_actions_and_option_overflows",
    }
