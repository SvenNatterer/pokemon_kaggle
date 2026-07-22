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

from stable_baselines3.common.callbacks import BaseCallback

class TrainingHealthCallback(BaseCallback):
    def __init__(self, fail_fast: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self.fail_fast = bool(fail_fast)
        self.triggered = False
        self.stop_reason = ""
        self.invalid_learner_actions = 0
        self.option_overflows = 0
        self.engine_errors = 0
        self.learner_decisions = 0
        self.max_option_count_seen = 0
        self.option_count_histogram = [0] * OPTION_COUNT_HISTOGRAM_SIZE
        self.opponent_episodes = {}
        self.episodes = 0

    def _on_step(self) -> bool:
        if self.locals and "infos" in self.locals:
            infos = self.locals["infos"]
            dones = self.locals.get("dones", [False] * len(infos))
            for idx, info in enumerate(infos):
                if not isinstance(info, dict):
                    continue
                inv = info.get("invalid_learner_action_count", 0)
                if inv > 0:
                    self.invalid_learner_actions += inv
                    if self.fail_fast:
                        self.triggered = True
                        self.stop_reason = f"invalid_learner_actions={inv}"
                        return False
                self.option_overflows += info.get("option_overflow_count", 0)
                self.engine_errors += info.get("engine_error_count", 0)
                self.learner_decisions += info.get("learner_decision_count", 0)
                self.max_option_count_seen = max(self.max_option_count_seen, info.get("max_option_count_seen", 0))
                if "learner_option_count_histogram" in info:
                    self.option_count_histogram = merge_option_count_histograms(
                        [self.option_count_histogram, info["learner_option_count_histogram"]]
                    )
                if dones[idx] and "opponent_label" in info:
                    lbl = info["opponent_label"]
                    self.opponent_episodes[lbl] = self.opponent_episodes.get(lbl, 0) + 1
                    self.episodes += 1
        return True

    def _on_rollout_end(self) -> None:
        rec = summarize_health(
            episodes=self.episodes,
            learner_decisions=self.learner_decisions,
            invalid_learner_actions=self.invalid_learner_actions,
            option_overflows=self.option_overflows,
            engine_errors=self.engine_errors,
            max_option_count_seen=self.max_option_count_seen,
            option_count_histogram=self.option_count_histogram,
            opponent_episodes=self.opponent_episodes,
        )
        gate = health_gate(rec)
        if hasattr(self, "model") and self.model and hasattr(self.model, "logger") and self.model.logger:
            self.model.logger.record("health/gate_passed", 1.0 if gate["passed"] else 0.0)

    def summary(self) -> dict[str, Any]:
        return summarize_health(
            episodes=self.episodes,
            learner_decisions=self.learner_decisions,
            invalid_learner_actions=self.invalid_learner_actions,
            option_overflows=self.option_overflows,
            engine_errors=self.engine_errors,
            max_option_count_seen=self.max_option_count_seen,
            option_count_histogram=self.option_count_histogram,
            opponent_episodes=self.opponent_episodes,
        )
