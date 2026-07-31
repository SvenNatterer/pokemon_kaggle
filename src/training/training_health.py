"""Shared health summaries and eligibility gates for PPO runs and evaluations."""

from __future__ import annotations

import math
import numpy as np
from collections import Counter
from typing import Any, Iterable, List, Dict

try:
    import wandb
except ImportError:
    wandb = None


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


class FeatureMetricsCallback(BaseCallback):
    """
    Callback that logs Prize Mapping & Opponent Archetype Prediction performance metrics to WandB.
    
    Metrics logged to WandB:
    - features/prize_certainty_ratio: % of episode steps where prize cards are 100% thinned/known.
    - features/prize_entropy: Mean information entropy of prize card estimation.
    - features/archetype_confidence: Mean top-1 probability of opponent archetype prediction.
    - features/archetype_accuracy: Prediction accuracy against known opponent decks.
    - train/archetype_accuracy: Archetype accuracy logged under train namespace.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.prize_certainty_history: List[float] = []
        self.prize_entropy_history: List[float] = []
        self.archetype_conf_history: List[float] = []
        self.archetype_acc_history: List[float] = []

    def _on_step(self) -> bool:
        # Collect step info from env infos buffer if available
        if hasattr(self.training_env, "env_method"):
            try:
                for info in self.training_env.env_method("get_feature_metrics"):
                    if isinstance(info, dict):
                        if "prize_certainty" in info:
                            self.prize_certainty_history.append(float(info["prize_certainty"]))
                        if "prize_entropy" in info:
                            self.prize_entropy_history.append(float(info["prize_entropy"]))
                        if "archetype_conf" in info:
                            self.archetype_conf_history.append(float(info["archetype_conf"]))
                        if "archetype_acc" in info:
                            self.archetype_acc_history.append(float(info["archetype_acc"]))
            except Exception:
                pass
        return True

    def _on_rollout_end(self) -> None:
        if self.prize_certainty_history:
            self.logger.record("features/prize_certainty_ratio", np.mean(self.prize_certainty_history))
            self.prize_certainty_history.clear()

        if self.prize_entropy_history:
            self.logger.record("features/prize_entropy", np.mean(self.prize_entropy_history))
            self.prize_entropy_history.clear()

        if self.archetype_conf_history:
            self.logger.record("features/archetype_confidence", np.mean(self.archetype_conf_history))
            self.archetype_conf_history.clear()

        if self.archetype_acc_history:
            mean_acc = np.mean(self.archetype_acc_history)
            self.logger.record("features/archetype_accuracy", mean_acc)
            self.logger.record("train/archetype_accuracy", mean_acc)
            self.archetype_acc_history.clear()


class GuardrailMetricsCallback(BaseCallback):
    """
    Callback that logs total guardrail intervention count and per-rule breakdown to WandB.
    
    Metrics logged to WandB:
    - guardrails/total_interventions: Total cumulative count of guardrail interventions across environments.
    - guardrails/rule/<rule_name>: Cumulative interventions by specific guardrail rule.
    - guardrails/interventions_barplot: WandB Bar plot visualizing intervention sources.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._previous_results: list[dict[str, Any]] = []
        self._collection_error: str | None = None

    def _on_step(self) -> bool:
        return True

    def _collect_results(self) -> list[dict[str, Any]]:
        self._collection_error = None
        results = []
        if hasattr(self.training_env, "env_method"):
            try:
                results = self.training_env.env_method("get_guardrail_metrics")
            except Exception as error:
                self._collection_error = f"{type(error).__name__}: {error}"
                results = []
        elif hasattr(self.training_env, "get_guardrail_metrics"):
            try:
                results = [self.training_env.get_guardrail_metrics()]
            except Exception as error:
                self._collection_error = f"{type(error).__name__}: {error}"
                results = []
        elif hasattr(self.training_env, "envs"):
            try:
                results = [
                    e.get_guardrail_metrics()
                    for e in getattr(self.training_env, "envs", [])
                    if hasattr(e, "get_guardrail_metrics")
                ]
            except Exception as error:
                self._collection_error = f"{type(error).__name__}: {error}"
                results = []
        return [result for result in results if isinstance(result, dict)]

    @staticmethod
    def _counter_delta(current: float, previous: float) -> float:
        # A worker restart resets its monotonic counters; treat the new value
        # as the complete delta instead of producing a negative rollout.
        return current - previous if current >= previous else current

    @classmethod
    def summarize_results(
        cls,
        results: list[dict[str, Any]],
        previous_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        previous_results = previous_results or []
        scalar_keys = (
            "decisions_total",
            "proposals_total",
            "accepted_total",
            "rolled_back_total",
            "shadow_total",
            "search_failures_total",
        )
        totals = {key: 0.0 for key in scalar_keys}
        deltas = {key: 0.0 for key in scalar_keys}
        known_rules: set[str] = set()
        rule_totals: Counter[str] = Counter()
        rule_deltas: Counter[str] = Counter()

        for index, result in enumerate(results):
            previous = previous_results[index] if index < len(previous_results) else {}
            for key in scalar_keys:
                current_value = float(result.get(key, 0.0))
                previous_value = float(previous.get(key, 0.0))
                totals[key] += current_value
                deltas[key] += cls._counter_delta(current_value, previous_value)

            current_rules = result.get("by_rule", {})
            previous_rules = previous.get("by_rule", {})
            if not isinstance(current_rules, dict):
                current_rules = {}
            if not isinstance(previous_rules, dict):
                previous_rules = {}
            known_rules.update(str(name) for name in result.get("known_rules", ()))
            known_rules.update(str(name) for name in current_rules)
            for name in known_rules | {str(rule) for rule in current_rules}:
                current_value = float(current_rules.get(name, 0.0))
                previous_value = float(previous_rules.get(name, 0.0))
                rule_totals[name] += current_value
                rule_deltas[name] += cls._counter_delta(current_value, previous_value)

        accepted_delta = deltas["accepted_total"]
        decision_delta = deltas["decisions_total"]
        return {
            "totals": totals,
            "deltas": deltas,
            "known_rules": sorted(known_rules),
            "by_rule": {
                name: float(rule_totals.get(name, 0.0)) for name in sorted(known_rules)
            },
            "by_rule_rollout": {
                name: float(rule_deltas.get(name, 0.0)) for name in sorted(known_rules)
            },
            "intervention_rate_per_1000": (
                1000.0 * accepted_delta / decision_delta if decision_delta > 0 else 0.0
            ),
        }

    @staticmethod
    def barplot_rows(summary: dict[str, Any]) -> list[list[Any]]:
        counts = summary.get("by_rule", {})
        return [
            [name, float(counts.get(name, 0.0))]
            for name in sorted(
                summary.get("known_rules", ()),
                key=lambda rule: (-float(counts.get(rule, 0.0)), str(rule)),
            )
        ]

    def _on_rollout_end(self) -> None:
        results = self._collect_results()
        summary = self.summarize_results(results, self._previous_results)
        self._previous_results = results
        totals = summary["totals"]
        deltas = summary["deltas"]

        # Keep the legacy key while adding unambiguous total/rollout series.
        self.logger.record("guardrails/total_interventions", totals["accepted_total"])
        self.logger.record("guardrails/interventions_total", totals["accepted_total"])
        self.logger.record("guardrails/interventions_rollout", deltas["accepted_total"])
        self.logger.record(
            "guardrails/intervention_rate_per_1000",
            summary["intervention_rate_per_1000"],
        )
        self.logger.record("guardrails/proposals_rollout", deltas["proposals_total"])
        self.logger.record(
            "guardrails/proposals_rolled_back",
            deltas["rolled_back_total"],
        )
        self.logger.record("guardrails/shadow_proposals", deltas["shadow_total"])
        self.logger.record(
            "guardrails/search_failures",
            deltas["search_failures_total"],
        )
        self.logger.record(
            "guardrails/metrics_collection_error",
            float(self._collection_error is not None),
        )
        for rule_name, count in summary["by_rule"].items():
            # Full rule names are useful in W&B/TensorBoard but collide after
            # Stable-Baselines' console formatter truncates them to 36 chars.
            self.logger.record(
                f"guardrails/rule/{rule_name}",
                count,
                exclude="stdout",
            )

        rows = self.barplot_rows(summary)
        if wandb is not None and wandb.run is not None and rows:
            try:
                table = wandb.Table(data=rows, columns=["Rule", "Interventions"])
                barplot = wandb.plot.bar(
                    table,
                    "Rule",
                    "Interventions",
                    title="Accepted Guardrail Interventions by Rule",
                )
                wandb.log(
                    {"guardrails/interventions_barplot": barplot},
                    commit=False,
                )
                self.logger.record("guardrails/barplot_logging_error", 0.0)
            except Exception as error:
                self.logger.record("guardrails/barplot_logging_error", 1.0)
                if self.verbose:
                    print(f"Guardrail W&B barplot logging failed: {type(error).__name__}: {error}")
