from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.training.training_health import (
    GuardrailMetricsCallback,
    OPTION_COUNT_HISTOGRAM_SIZE,
    health_gate,
    summarize_health,
)


def test_guardrail_metrics_use_worker_deltas_and_keep_zero_count_rules_visible():
    previous = [
        {
            "decisions_total": 90,
            "accepted_total": 1,
            "known_rules": ["rule_a", "rule_b", "rule_zero"],
            "by_rule": {"rule_a": 1, "rule_b": 0, "rule_zero": 0},
        },
        {
            "decisions_total": 40,
            "accepted_total": 0,
            "known_rules": ["rule_a", "rule_b", "rule_zero"],
            "by_rule": {"rule_a": 0, "rule_b": 0, "rule_zero": 0},
        },
    ]
    current = [
        {
            "decisions_total": 100,
            "proposals_total": 3,
            "accepted_total": 2,
            "rolled_back_total": 1,
            "known_rules": ["rule_a", "rule_b", "rule_zero"],
            "by_rule": {"rule_a": 2, "rule_b": 0, "rule_zero": 0},
        },
        {
            "decisions_total": 50,
            "proposals_total": 1,
            "accepted_total": 1,
            "rolled_back_total": 0,
            "known_rules": ["rule_a", "rule_b", "rule_zero"],
            "by_rule": {"rule_a": 0, "rule_b": 1, "rule_zero": 0},
        },
    ]

    summary = GuardrailMetricsCallback.summarize_results(current, previous)

    assert summary["totals"]["accepted_total"] == 3
    assert summary["deltas"]["accepted_total"] == 2
    assert summary["deltas"]["decisions_total"] == 20
    assert summary["intervention_rate_per_1000"] == 100
    assert summary["by_rule"] == {"rule_a": 2.0, "rule_b": 1.0, "rule_zero": 0.0}
    assert GuardrailMetricsCallback.barplot_rows(summary) == [
        ["rule_a", 2.0],
        ["rule_b", 1.0],
        ["rule_zero", 0.0],
    ]


def test_guardrail_callback_logs_wandb_barplot_with_all_registered_rules():
    class Logger:
        def __init__(self):
            self.values = {}

        def record(self, key, value, **kwargs):
            self.values[key] = value

    class Callback(GuardrailMetricsCallback):
        def _collect_results(self):
            return [{
                "decisions_total": 10,
                "proposals_total": 2,
                "accepted_total": 1,
                "rolled_back_total": 1,
                "known_rules": ["accepted_rule", "zero_rule"],
                "by_rule": {"accepted_rule": 1, "zero_rule": 0},
            }]

    callback = Callback()
    callback.model = SimpleNamespace(logger=Logger())
    callback.num_timesteps = 512
    fake_wandb = SimpleNamespace(
        run=object(),
        Table=Mock(return_value="table"),
        plot=SimpleNamespace(bar=Mock(return_value="barplot")),
        log=Mock(),
    )

    with patch("src.training.training_health.wandb", fake_wandb):
        callback._on_rollout_end()

    fake_wandb.Table.assert_called_once_with(
        data=[["accepted_rule", 1.0], ["zero_rule", 0.0]],
        columns=["Rule", "Interventions"],
    )
    fake_wandb.log.assert_called_once_with(
        {"guardrails/interventions_barplot": "barplot"},
        commit=False,
    )
    assert callback.model.logger.values["guardrails/interventions_total"] == 1.0
    assert callback.model.logger.values["guardrails/interventions_rollout"] == 1.0
    assert callback.model.logger.values["guardrails/barplot_logging_error"] == 0.0


def test_health_summary_reports_bounded_option_percentiles_and_opponents():
    histogram = [0] * OPTION_COUNT_HISTOGRAM_SIZE
    histogram[2] = 8
    histogram[5] = 2
    health = summarize_health(
        episodes=3,
        learner_decisions=10,
        max_option_count_seen=5,
        option_count_histogram=histogram,
        opponent_episodes={"rule_bot": 3},
    )

    assert health["option_count_percentiles"] == {
        "p50": 2,
        "p90": 5,
        "p95": 5,
        "p99": 5,
    }
    assert health["opponent_episodes"] == {"rule_bot": 3}
    assert health_gate(health)["passed"]


def test_health_gate_rejects_each_p0_corruption_signal():
    health = summarize_health(
        invalid_learner_actions=1,
        option_overflows=2,
        engine_errors=3,
    )

    gate = health_gate(health, crashes=1)

    assert not gate["passed"]
    assert gate["violations"] == [
        "evaluation_crashes=1",
        "invalid_learner_actions=1",
        "option_overflows=2",
        "engine_errors=3",
    ]


def test_training_health_callback_aggregates_cumulative_worker_snapshots():
    from src.training.train import TrainingHealthCallback

    class Logger:
        def __init__(self):
            self.values = {}

        def record(self, key, value):
            self.values[key] = value

    class Model:
        def __init__(self):
            self.logger = Logger()

    histogram = [0] * OPTION_COUNT_HISTOGRAM_SIZE
    histogram[3] = 4
    callback = TrainingHealthCallback()
    callback.model = Model()
    callback.locals = {
        "dones": [True, False],
        "infos": [
            {
                "invalid_learner_action_count": 0,
                "option_overflow_count": 0,
                "engine_error_count": 0,
                "learner_decision_count": 4,
                "max_option_count_seen": 3,
                "learner_option_count_histogram": histogram,
                "opponent_label": "pool_a",
            },
            {
                "invalid_learner_action_count": 0,
                "option_overflow_count": 0,
                "engine_error_count": 0,
                "learner_decision_count": 2,
                "max_option_count_seen": 2,
                "learner_option_count_histogram": [0, 0, 2] + [0] * (OPTION_COUNT_HISTOGRAM_SIZE - 3),
            },
        ],
    }

    assert callback._on_step()
    callback._on_rollout_end()
    summary = callback.summary()

    assert summary["learner_decisions"] == 6
    assert summary["option_count_percentiles"]["p50"] == 3
    assert summary["opponent_episodes"] == {"pool_a": 1}
    assert callback.model.logger.values["health/gate_passed"] == 1.0


def test_training_health_callback_stops_on_invalid_learner_action():
    from src.training.train import TrainingHealthCallback

    callback = TrainingHealthCallback(fail_fast=True)
    callback.locals = {
        "dones": [False],
        "infos": [{"invalid_learner_action_count": 1}],
    }

    assert not callback._on_step()
    assert callback.triggered
    assert "invalid_learner_actions=1" in callback.stop_reason
