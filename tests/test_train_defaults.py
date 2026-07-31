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
    assert defaults["opp_pool"] == train.DEFAULT_TRAINING_POOL

    assert defaults["policy_version"] == "v6"
    assert defaults["feature_variant"] == "compact"
    assert defaults["entity_relation_mode"] == "baseline"
    assert defaults["wandb_mode"] == "online"
    assert defaults["extra_metrics"] is True
    assert defaults["belief_actor"] is True
    assert defaults["belief_dim"] == 64
    assert defaults["belief_detach"] is True
    assert defaults["card_table"] is True
    assert defaults["inference_guardrails"] is True
    assert defaults["inference_guardrail_mode"] == "active"
    assert defaults["rotate_perspective"] is True

    assert defaults["scalar_obs"] is False
    assert defaults["scalar_embeddings"] is False
    assert defaults["sparse_rewards"] is False
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
                "winner": 0,
                "learner_perspective": 0,
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


def test_reward_logging_uses_engine_outcome_instead_of_positive_shaped_return():
    from src.training.train import RewardBreakdownCallback

    class Logger:
        def __init__(self):
            self.values = {}

        def record(self, key, value):
            self.values[key] = value

    class Model:
        def __init__(self):
            self.logger = Logger()
            self.ep_info_buffer = [{"r": 2.0}]

    callback = RewardBreakdownCallback()
    callback.model = Model()
    callback.locals = {
        "dones": [True, True],
        "infos": [
            {
                "winner": 0,
                "learner_perspective": 1,
                "reward_breakdown": {"gameplan_bonus": 3.0, "loss": -1.0},
            },
            {
                "terminal_info": {
                    "winner": 1,
                    "learner_perspective": 1,
                    "reward_breakdown": {"prize_win": 1.0},
                }
            },
        ],
    }

    assert callback._on_step()
    callback._on_rollout_end()

    assert "rollout/ep_rew_max" not in callback.model.logger.values
    assert "rollout/ep_rew_min" not in callback.model.logger.values
    assert callback.model.logger.values["rollout/win_rate"] == pytest.approx(0.5)


def test_pfsp_logs_minimum_win_rate_under_rollout_namespace():
    from src.training.train import PFSPCallback

    class Logger:
        def __init__(self):
            self.values = {}

        def record(self, key, value):
            self.values[key] = value

    class Model:
        def __init__(self):
            self.logger = Logger()

    callback = PFSPCallback(
        [{"label": "strong", "weight": 1.0}, {"label": "weak", "weight": 1.0}],
        window_games=10,
    )
    callback.model = Model()
    callback.controller.observe("strong", 1)
    callback.controller.observe("weak", -1)

    callback._on_rollout_end()

    assert callback.model.logger.values["rollout/min_win_rate"] == pytest.approx(0.0)


def test_endless_training_uses_one_non_overflowing_learn_budget():
    from src.training.train import endless_learn_budget

    current_timesteps = 2_250_752
    budget = endless_learn_budget(current_timesteps)

    assert budget + current_timesteps == sys.maxsize


def test_exp_018_lookahead_config_forwarding():
    from src.training.train import parse_args_with_config
    from src.env.env_wrapper import PokemonTCGEnv

    args = parse_args_with_config([
        "--config",
        "configs/experiments/exp_018_value_distillation.yaml",
    ])

    assert hasattr(args, "lookahead_config")
    assert args.lookahead_config == {
        "max_depth": 1,
        "beam_width": 2,
        "node_budget": 8,
        "max_combinations": 16,
    }

    env = PokemonTCGEnv(
        [1]*60, [1]*60,
        enable_lookahead_teacher=True,
        lookahead_config=args.lookahead_config
    )
    assert env.lookahead_teacher is not None
    assert env.lookahead_teacher.config.max_depth == 1
    assert env.lookahead_teacher.config.node_budget == 8
    assert env.lookahead_teacher.config.beam_width == 2


@pytest.mark.parametrize(
    ("config_path", "expected_mode"),
    [
        ("configs/experiments/exp_019_masked_entity_attention.yaml", "masked"),
        ("configs/experiments/exp_020_relational_entity_attention.yaml", "relational"),
        ("configs/experiments/exp_021_two_step_relational_attention.yaml", "two_step"),
        ("configs/experiments/exp_023_python_object_relational_attention.yaml", "python_object_relational"),
    ],
)
def test_relational_experiment_configs_select_the_requested_encoder(config_path, expected_mode):
    from src.training.train import parse_args_with_config

    args = parse_args_with_config(["--config", config_path])

    assert args.entity_relation_mode == expected_mode
    assert args.wandb_mode == "online"


def test_guardrail_shadow_config_enables_shadow_telemetry_and_search_sampling():
    from src.training.train import parse_args_with_config

    args = parse_args_with_config([
        "--config",
        "configs/experiments/exp_022_guardrails_shadow.yaml",
    ])

    assert args.inference_guardrails is True
    assert args.inference_guardrail_mode == "shadow"
    assert args.search_guardrail_rate == pytest.approx(0.075)
    assert args.wandb_mode == "online"


def test_exp_021_explicitly_uses_active_hard_guardrails_without_search():
    from src.training.train import parse_args_with_config

    args = parse_args_with_config([
        "--config",
        "configs/experiments/exp_021_two_step_relational_attention.yaml",
    ])

    assert args.entity_relation_mode == "two_step"
    assert args.inference_guardrails is True
    assert args.inference_guardrail_mode == "active"
    assert args.search_guardrail_rate == 0.0
    assert args.wandb_mode == "online"


def test_exp_025_config_disables_non_representation_helpers():
    from src.training.train import parse_args_with_config

    args = parse_args_with_config([
        "--config",
        "configs/experiments/exp_025_strategic_vector_v1.yaml",
    ])

    assert args.scalar_obs is True
    assert args.feature_variant == "strategic_vector_v1"
    assert args.aux_coef == 0.0
    assert args.distill_coef == 0.0
    assert args.value_distill_coef == 0.0
    assert args.enable_lookahead_teacher is False
    assert args.teacher_sample_rate == 0.0
    assert args.belief_actor is False
    assert args.card_table is False
    assert args.inference_guardrails is False
    assert args.inference_guardrail_mode == "off"
    assert args.search_guardrail_rate == 0.0
    assert args.enable_archetype_prediction is False
    assert args.extra_metrics is False
    assert args.sparse_rewards is True
    assert args.wandb_mode == "online"


def test_default_training_config_uses_conservative_checkpoint_evaluation():
    from src.training.train import parse_args_with_config

    args = parse_args_with_config(["--config", "configs/experiments/default_training.yaml"])

    assert args.enable_archetype_prediction is False
    assert args.checkpoint_eval_steps == 250_000
    assert args.checkpoint_eval_games_per_opponent == 50
