import math

import pytest

from src.league.pfsp import OpponentRecord, PFSPLite


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
    from src.env.env_wrapper import PokemonTCGEnv


