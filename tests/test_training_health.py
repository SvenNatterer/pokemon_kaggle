from src.training.training_health import (
    OPTION_COUNT_HISTOGRAM_SIZE,
    health_gate,
    summarize_health,
)


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
