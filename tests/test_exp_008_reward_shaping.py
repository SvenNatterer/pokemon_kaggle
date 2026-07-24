import pytest
from src.training.train import parse_args_with_config
from src.env.env_wrapper import PokemonTCGEnv


def test_exp_008_reward_config_parsing():
    args = parse_args_with_config(["--config", "configs/experiments/exp_008_reward_shaping.yaml"])
    assert hasattr(args, "reward_config")
    rewards = args.reward_config
    assert rewards["STEP_PENALTY"] == pytest.approx(-0.001)
    assert rewards["DECK_OUT_PENALTY"] == pytest.approx(-0.50)
    assert rewards["DECK_SHRINK"] == pytest.approx(-0.005)
    assert rewards["DECK_LOW_COUNT_MULT"] == pytest.approx(4.0)


def test_env_applies_reward_config():
    deck = [1] * 60
    custom_rewards = {
        "STEP_PENALTY": -0.002,
        "DECK_OUT_PENALTY": -0.75,
        "DECK_SHRINK": -0.01,
        "DECK_LOW_COUNT_MULT": 5.0,
    }
    env = PokemonTCGEnv(
        my_deck=deck,
        opponent_deck=deck,
        reward_config=custom_rewards,
    )
    assert env.reward_config["STEP_PENALTY"] == pytest.approx(-0.002)
    assert env.reward_config["DECK_OUT_PENALTY"] == pytest.approx(-0.75)
    assert env.reward_config["DECK_SHRINK"] == pytest.approx(-0.01)
    assert env.reward_config["DECK_LOW_COUNT_MULT"] == pytest.approx(5.0)
