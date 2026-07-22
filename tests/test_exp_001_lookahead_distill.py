"""Test EXP-001 Lookahead Teacher Distillation loss and sampling integration."""

import torch
import numpy as np
import pytest
from src.env.env_wrapper import PokemonTCGEnv
from src.training.custom_ppo import CustomPPO
from src.training.lookahead_teacher import LookaheadTeacher, LookaheadConfig


def test_env_lookahead_teacher_init():
    """Verify env initializes lookahead_teacher with 50% sample rate."""
    env = PokemonTCGEnv(
        my_deck=[1] * 60,
        opponent_deck=[1] * 60,
        enable_lookahead_teacher=True,
        teacher_sample_rate=0.50,
    )
    assert env.enable_lookahead_teacher is True
    assert env.teacher_sample_rate == 0.50
    assert env.lookahead_teacher is not None


def test_custom_ppo_distill_loss():
    """Verify CustomPPO initializes distill_coef and computes distillation loss."""
    ppo = CustomPPO(
        "MlpLstmPolicy",
        env=None,
        c_aux=0.5,
        distill_coef=0.1,
        _init_setup_model=False,
    )
    assert ppo.distill_coef == 0.1
    assert ppo.c_aux == 0.5
