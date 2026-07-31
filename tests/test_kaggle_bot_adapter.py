"""Unit tests for Kaggle Python script bot adapter."""

from pathlib import Path
import pytest
import numpy as np

from src.agents.bot_loader import load_bot
from src.agents.kaggle_bots.wrapper import KagglePythonAgentWrapper, is_python_script_agent_spec
from scripts.collect_lookahead_teacher import _validate_expert


def test_is_python_script_agent_spec():
    assert is_python_script_agent_spec("python_script:src/agents/kaggle_bots/alakazam_v8_agent.py")
    assert is_python_script_agent_spec("src/agents/kaggle_bots/battlecore_agent.py")
    assert not is_python_script_agent_spec("rule_based:v4:aggro")
    assert not is_python_script_agent_spec("models/ppo_v6.zip")


def test_load_kaggle_bot():
    bot = load_bot("python_script:src/agents/kaggle_bots/alakazam_v8_agent.py")
    assert isinstance(bot, KagglePythonAgentWrapper)
    assert hasattr(bot, "action_space")
    assert bot.action_space.n == 66


def test_validate_expert_accepts_kaggle_bot():
    bot = load_bot("python_script:src/agents/kaggle_bots/alakazam_v8_agent.py")
    # Should not raise
    _validate_expert(bot, "python_script:src/agents/kaggle_bots/alakazam_v8_agent.py")


def test_kaggle_bot_predict_fallback():
    bot = load_bot("python_script:src/agents/kaggle_bots/alakazam_v8_agent.py")
    action, state = bot.predict({"invalid_key": 123})
    assert isinstance(action, np.ndarray)
    assert action.shape == ()
    assert 0 <= int(action) < 66
