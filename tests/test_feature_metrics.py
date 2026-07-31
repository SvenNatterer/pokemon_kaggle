import pytest
import numpy as np
from unittest.mock import MagicMock
from src.training.training_health import FeatureMetricsCallback


def test_feature_metrics_callback():
    callback = FeatureMetricsCallback()
    mock_model = MagicMock()
    mock_logger = MagicMock()
    mock_model.logger = mock_logger
    callback.init_callback(mock_model)
    
    # Simulate step metrics
    callback.prize_certainty_history = [1.0, 0.0, 1.0]
    callback.prize_entropy_history = [0.2, 0.5, 0.1]
    callback.archetype_conf_history = [0.85, 0.90]
    callback.archetype_acc_history = [1.0, 1.0]
    
    callback._on_rollout_end()
    
    # Verify records recorded to logger
    mock_logger.record.assert_any_call("features/prize_certainty_ratio", pytest.approx(0.6666, abs=1e-3))
    mock_logger.record.assert_any_call("features/prize_entropy", pytest.approx(0.2666, abs=1e-3))
    mock_logger.record.assert_any_call("features/archetype_confidence", pytest.approx(0.875, abs=1e-3))
    mock_logger.record.assert_any_call("features/archetype_accuracy", 1.0)


def test_disable_archetype_prediction_flag():
    from src.env.env_wrapper import PokemonTCGEnv
    env = PokemonTCGEnv(
        my_deck=[1, 2, 3],
        opponent_deck=[4, 5, 6],
        enable_archetype_prediction=False,
    )
    assert env.enable_archetype_prediction is False
    mock_obs = MagicMock()
    env._update_feature_metrics(mock_obs)
    assert not hasattr(env, "_last_archetype_conf")
    assert not hasattr(env, "_last_archetype_acc")
