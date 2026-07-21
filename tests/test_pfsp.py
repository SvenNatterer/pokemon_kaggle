import math

import pytest

from src.pfsp import OpponentRecord, PFSPLite


def test_initial_manifest_weights_are_unchanged_until_first_segment():
    controller = PFSPLite(
        ["historical", "rule", "recent"],
        [6.0, 3.0, 1.0],
        max_probability=0.7,
    )

    assert controller.current_probabilities == pytest.approx([0.6, 0.3, 0.1])
    controller.observe("historical", 1)
    assert controller.current_probabilities == pytest.approx([0.6, 0.3, 0.1])


def test_pfsp_prefers_under_sampled_and_competitive_opponents_with_bounds():
    controller = PFSPLite(
        ["competitive", "easy", "hard", "unseen"],
        [1.0, 1.0, 1.0, 1.0],
        random_fraction=0.20,
        max_probability=0.45,
    )
    for index in range(100):
        controller.observe("competitive", 1 if index % 2 == 0 else -1)
        controller.observe("easy", 1 if index < 95 else -1)
        controller.observe("hard", -1 if index < 95 else 1)

    probabilities, segment = controller.finish_segment()
    by_label = dict(zip(controller.labels, probabilities))

    assert by_label["unseen"] == pytest.approx(0.45)
    assert by_label["competitive"] > by_label["easy"]
    assert by_label["competitive"] > by_label["hard"]
    assert all(probability >= 0.05 for probability in probabilities)
    assert all(probability <= 0.45 for probability in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)
    assert segment["games"] == 300


def test_segment_and_cumulative_results_track_draws_as_half_a_win():
    controller = PFSPLite(["a", "b"], [1.0, 1.0], max_probability=0.7)
    controller.observe("a", 1)
    controller.observe("a", -1)
    controller.observe("a", 0)
    controller.observe("b", 0)

    _, segment = controller.finish_segment()
    summary = controller.summary()

    assert segment["opponents"]["a"]["games"] == 3
    assert segment["opponents"]["a"]["wins"] == 1
    assert segment["opponents"]["a"]["losses"] == 1
    assert segment["opponents"]["a"]["draws"] == 1
    assert summary["opponents"]["a"]["effective_win_rate"] == pytest.approx(0.5)
    assert summary["completed_segments"] == 1
    assert controller.segment_games == 0


def test_posterior_uncertainty_shrinks_with_more_games():
    small = OpponentRecord(wins=1, losses=1, games=2)
    large = OpponentRecord(wins=50, losses=50, games=100)

    _, small_uncertainty = small.posterior(prior_games=4.0)
    _, large_uncertainty = large.posterior(prior_games=4.0)

    assert math.isfinite(small_uncertainty)
    assert large_uncertainty < small_uncertainty


def test_unknown_labels_and_invalid_outcomes_are_not_counted():
    controller = PFSPLite(["known"], [1.0], max_probability=1.0)

    assert not controller.observe("unknown", 1)
    assert not controller.observe("known", 2)
    assert controller.segment_games == 0


def test_restore_preserves_cumulative_state_and_starts_a_fresh_segment():
    original = PFSPLite(
        ["competitive", "easy"],
        [1.0, 1.0],
        random_fraction=0.20,
        max_probability=0.70,
    )
    original.observe("competitive", 1)
    original.observe("competitive", -1)
    original.observe("easy", 1)
    original.finish_segment()

    restored = PFSPLite(
        ["competitive", "easy"],
        [1.0, 1.0],
        random_fraction=0.20,
        max_probability=0.70,
    )
    restored.restore(original.summary())

    assert restored.completed_segments == 1
    assert restored.current_probabilities == pytest.approx(
        original.current_probabilities
    )
    assert restored.records["competitive"].games == 2
    assert restored.records["easy"].wins == 1
    assert restored.segment_games == 0


def test_restore_rejects_a_different_opponent_pool():
    controller = PFSPLite(["a"], [1.0], max_probability=1.0)

    with pytest.raises(ValueError, match="labels"):
        controller.restore(
            {
                "probabilities": {"different": 1.0},
                "opponents": {"different": {"games": 0}},
            }
        )


def test_environment_weight_update_only_changes_future_sampling_weights():
    from src.env_wrapper import PokemonTCGEnv

    env = PokemonTCGEnv.__new__(PokemonTCGEnv)
    env.opponent_pool = [
        {"label": "a", "weight": 0.8},
        {"label": "b", "weight": 0.2},
    ]
    env.current_opponent_index = 0
    env.current_opponent_label = "a"

    env.set_opponent_weights([0.25, 0.75])

    assert [entry["weight"] for entry in env.opponent_pool] == [0.25, 0.75]
    assert env.current_opponent_index == 0
    assert env.current_opponent_label == "a"


def test_environment_rejects_invalid_weight_updates():
    from src.env_wrapper import PokemonTCGEnv

    env = PokemonTCGEnv.__new__(PokemonTCGEnv)
    env.opponent_pool = [{"weight": 1.0}, {"weight": 1.0}]

    with pytest.raises(ValueError):
        env.set_opponent_weights([1.0])
    with pytest.raises(ValueError):
        env.set_opponent_weights([1.0, -1.0])
    with pytest.raises(ValueError):
        env.set_opponent_weights([0.0, 0.0])


def test_callback_updates_workers_at_boundary_and_skips_engine_errors(tmp_path):
    from src.train import PFSPLiteCallback

    class FakeLogger:
        def __init__(self):
            self.values = {}

        def record(self, key, value):
            self.values[key] = value

    class FakeVecEnv:
        def __init__(self):
            self.calls = []

        def env_method(self, method, weights):
            self.calls.append((method, weights))

    class FakeModel:
        def __init__(self, env):
            self.env = env
            self.logger = FakeLogger()

        def get_env(self):
            return self.env

    state_path = tmp_path / "model.pfsp.json"
    callback = PFSPLiteCallback(
        [{"label": "a", "weight": 1.0}, {"label": "b", "weight": 1.0}],
        state_path=str(state_path),
        segment_episodes=1,
        max_probability=0.7,
    )
    env = FakeVecEnv()
    callback.model = FakeModel(env)
    callback.locals = {
        "dones": [True, True],
        "infos": [
            {"opponent_label": "a", "learner_result": 1},
            {
                "opponent_label": "b",
                "learner_result": 0,
                "engine_error": "discard this game",
            },
        ],
    }

    assert callback._on_step()
    callback._on_rollout_end()

    assert callback.controller.records["a"].games == 1
    assert callback.controller.records["b"].games == 0
    assert env.calls[0][0] == "set_opponent_weights"
    assert sum(env.calls[0][1]) == pytest.approx(1.0)
    assert state_path.exists()


def test_callback_uses_unique_short_metric_labels_for_long_opponent_names(tmp_path):
    from src.train import PFSPLiteCallback

    labels = [
        "Dragapult ex rule-based balanced",
        "Dragapult ex rule-based benchmark",
    ]
    callback = PFSPLiteCallback(
        [{"label": label, "weight": 1.0} for label in labels],
        state_path=str(tmp_path / "model.pfsp.json"),
        segment_episodes=1,
        max_probability=0.7,
    )

    metric_labels = list(callback.metric_labels.values())

    assert len(set(metric_labels)) == len(labels)
    assert all(metric.startswith("opp_") for metric in metric_labels)
    assert all(len(f"pfsp/{metric}/probability") < 36 for metric in metric_labels)


def test_callback_restores_weights_before_resumed_training(tmp_path):
    from src.train import PFSPLiteCallback

    class FakeVecEnv:
        def __init__(self):
            self.calls = []

        def env_method(self, method, weights):
            self.calls.append((method, list(weights)))

    class FakeModel:
        def __init__(self, env):
            self.env = env

        def get_env(self):
            return self.env

    state_path = tmp_path / "model.pfsp.json"
    first = PFSPLiteCallback(
        [{"label": "a", "weight": 1.0}, {"label": "b", "weight": 1.0}],
        state_path=str(state_path),
        segment_episodes=1,
        max_probability=0.7,
    )
    first.controller.observe("a", 1)
    first.controller.finish_segment()
    first.persist()

    resumed = PFSPLiteCallback(
        [{"label": "a", "weight": 1.0}, {"label": "b", "weight": 1.0}],
        state_path=str(state_path),
        segment_episodes=1,
        max_probability=0.7,
    )
    env = FakeVecEnv()
    resumed.model = FakeModel(env)
    resumed._init_callback()

    assert resumed.restored is True
    assert resumed.controller.completed_segments == 1
    assert env.calls == [
        ("set_opponent_weights", resumed.controller.current_probabilities)
    ]
