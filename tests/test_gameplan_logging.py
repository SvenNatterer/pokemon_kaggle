import os
import pytest
import numpy as np
from stable_baselines3.common.callbacks import CallbackList

from src.env.env_wrapper import PokemonTCGEnv, V6_ACTION_SPACE_SIZE
from src.models.inference_guardrails import InferenceGuardrails, TR38GameplanEvaluator
from src.training.train import RewardBreakdownCallback, make_env, read_deck
from src.training.training_health import GuardrailMetricsCallback
from src.training.custom_ppo import CustomPPO, PokemonTCGRecurrentPolicy


def test_tr38_gameplan_evaluator_direct():
    deck_path = "decks/deck_bank/bank_38.csv"
    deck = read_deck(deck_path)
    env = PokemonTCGEnv(deck, deck, inference_guardrails=True)

    class DummyPlayer:
        def __init__(self):
            self.active = [type("Active", (), {"id": 431, "card_id": 431})()]
            self.bench = [type("Bench", (), {"id": 401, "card_id": 401})()]
            self.hand = [type("Hand", (), {"id": 431, "card_id": 431})()]
            self.maxBench = 3

    class DummyCurrent:
        def __init__(self):
            self.players = [DummyPlayer(), DummyPlayer()]

    class DummyObs:
        def __init__(self):
            self.current = DummyCurrent()

    class DummyOption:
        def __init__(self, opt_type, card_id=None, index=None, is_bench=False, is_ability=False):
            self.type = opt_type
            self.card_id = card_id
            self.cardId = card_id
            self.index = index
            self.isBench = is_bench
            self.isAbility = is_ability

    evaluator = TR38GameplanEvaluator()
    obs = DummyObs()

    # 1. Bench action with TR Pokemon (card_id 431)
    bench_opt = DummyOption(opt_type=7, card_id=431, is_bench=True)
    reward_bench = evaluator.evaluate(env, obs, bench_opt)
    assert reward_bench > 0.0

    # 2. Ability action with Spidops (card_id 401)
    ability_opt = DummyOption(opt_type=10, card_id=401, is_ability=True)
    reward_ability = evaluator.evaluate(env, obs, ability_opt)
    assert reward_ability > 0.0


def test_lightweight_model_logs_gameplan_and_interventions(tmp_path):
    deck_path = "decks/deck_bank/bank_38.csv"
    env_fn = make_env(
        deck_path,
        deck_path,
        opp_model_path=None,
        action_space_size=V6_ACTION_SPACE_SIZE,
        inference_guardrails=True,
    )
    env = env_fn()

    model = CustomPPO(
        policy=PokemonTCGRecurrentPolicy,
        env=env,
        learning_rate=1e-4,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        gamma=0.99,
        verbose=0,
        tensorboard_log=str(tmp_path / "tb_logs"),
    )

    reward_cb = RewardBreakdownCallback()
    guardrail_cb = GuardrailMetricsCallback()
    callbacks = CallbackList([reward_cb, guardrail_cb])

    model.learn(total_timesteps=16, callback=callbacks)

    reward_cb._on_rollout_end()
    guardrail_cb._on_rollout_end()
    logged_keys = set(model.logger.name_to_value.keys())

    assert "rewards/gameplan_bonus" in logged_keys
    assert "rewards/gameplan_violation_penalty" in logged_keys
    assert "guardrails/total_interventions" in logged_keys

    env.close()
