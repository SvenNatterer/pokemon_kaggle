import argparse
import sys

import pytest


class _CapturedParser(RuntimeError):
    pass


def _train_defaults(monkeypatch):
    from src import train

    captured = {}

    def capture_defaults(parser):
        captured.update(
            {
                action.dest: action.default
                for action in parser._actions
                if action.dest != "help"
            }
        )
        raise _CapturedParser

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_defaults)
    with pytest.raises(_CapturedParser):
        train.train()
    return captured


def test_default_profile_is_the_proven_v6_compact_baseline(monkeypatch):
    from src import train

    defaults = _train_defaults(monkeypatch)

    assert defaults["timesteps"] == 1_000_000
    assert defaults["num_envs"] == 7
    assert defaults["n_steps"] == 2048
    assert defaults["batch_size"] == 1024
    assert defaults["n_epochs"] == 2
    assert defaults["lr"] == pytest.approx(1e-4)
    assert defaults["ent_coef"] == pytest.approx(0.008)
    assert defaults["clip_range"] == pytest.approx(0.12)
    assert defaults["target_kl"] == pytest.approx(0.03)
    assert defaults["aux_coef"] == pytest.approx(0.1)

    assert defaults["policy_version"] == "v6"
    assert defaults["feature_variant"] == "compact"
    assert defaults["belief_actor"] is True
    assert defaults["belief_dim"] == 64
    assert defaults["belief_detach"] is True
    assert defaults["card_table"] is True
    assert defaults["inference_guardrails"] is True
    assert defaults["rotate_perspective"] is True

    assert defaults["scalar_obs"] is False
    assert defaults["scalar_embeddings"] is False
    assert "sparse_rewards" not in defaults
    assert "potential_rewards" not in defaults
    assert defaults["adaptive_stop"] is False
    assert defaults["pfsp_lite"] is True
    assert defaults["search_guardrail_rate"] == 0.0
    assert defaults["health_gate"] is True
    assert defaults["reserved_opponents"] == [
        "decks/holdout_opponents.json",
        "decks/validation_opponents.json",
    ]
    assert train.TRAINING_USES_POTENTIAL_REWARDS is True


def test_reward_logging_excludes_state_potential_but_keeps_paid_difference():
    from src.training.train import RewardBreakdownCallback

    class Logger:
        def __init__(self):
            self.values = {}

        def record(self, key, value):
            self.values[key] = value

    class Model:
        def __init__(self):
            self.logger = Logger()

    callback = RewardBreakdownCallback()
    callback.model = Model()
    callback.locals = {
        "dones": [True],
        "infos": [
            {
                "reward_breakdown": {
                    "potential": 3.5,
                    "potential_diff": 0.25,
                    "prize_win": 1.0,
                }
            }
        ],
    }

    assert callback._on_step()
    callback._on_rollout_end()

    assert "rewards/potential" not in callback.model.logger.values
    assert callback.model.logger.values["rewards/potential_diff"] == pytest.approx(0.25)
    assert not any(key.startswith("monitor/") for key in callback.model.logger.values)
    assert callback.model.logger.values["rollout/win_rate"] == pytest.approx(1.0)


def test_endless_training_uses_one_non_overflowing_learn_budget():
    from src.training.train import endless_learn_budget

    current_timesteps = 2_250_752
    budget = endless_learn_budget(current_timesteps)

    assert budget + current_timesteps == sys.maxsize
