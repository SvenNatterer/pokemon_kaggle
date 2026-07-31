from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.training import model_factory


def _args(**overrides):
    values = {
        "feature_variant": "compact",
        "card_table": True,
        "entity_relation_mode": "baseline",
        "belief_actor": True,
        "belief_dim": 64,
        "belief_detach": True,
        "lr": 1e-4,
        "n_steps": 512,
        "batch_size": 1024,
        "n_epochs": 2,
        "ent_coef": 0.008,
        "clip_range": 0.12,
        "target_kl": 0.03,
        "aux_coef": 0.1,
        "distill_coef": 0.1,
        "seed": 11,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fresh_model_factory_preserves_default_training_construction(monkeypatch):
    captured = {}
    expected_model = object()

    def capture_model(*positional, **keywords):
        captured["positional"] = positional
        captured["keywords"] = keywords
        return expected_model

    monkeypatch.setattr(model_factory, "CustomPPO", capture_model)
    env = object()

    result = model_factory.build_fresh_custom_ppo(env, _args())

    assert result is expected_model
    assert captured["positional"] == (model_factory.PokemonTCGRecurrentPolicy, env)
    assert captured["keywords"] == {
        "verbose": 1,
        "learning_rate": 1e-4,
        "n_steps": 512,
        "batch_size": 1024,
        "n_epochs": 2,
        "gamma": 0.999,
        "ent_coef": 0.008,
        "clip_range": 0.12,
        "target_kl": 0.03,
        "c_aux": 0.1,
        "distill_coef": 0.1,
        "value_distill_coef": 0.0,
        "seed": 11,
        "device": "cpu",
        "tensorboard_log": "logs/",
        "policy_kwargs": {
            "features_extractor_class": model_factory.PokemonTCGFeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "feature_variant": "compact",
                "use_card_table": True,
                "entity_relation_mode": "baseline",
            },
            "use_belief_actor": True,
            "belief_dim": 64,
            "detach_belief_actor": True,
        },
    }


def test_fresh_model_factory_preserves_custom_network_dimensions(monkeypatch):
    captured = {}

    def capture_model(*positional, **keywords):
        captured.update(keywords)
        return object()

    monkeypatch.setattr(model_factory, "CustomPPO", capture_model)

    model_factory.build_fresh_custom_ppo(
        object(),
        _args(features_dim=512, hidden_dim=256, value_distill_coef=0.25),
    )

    assert captured["value_distill_coef"] == pytest.approx(0.25)
    assert captured["policy_kwargs"]["features_extractor_kwargs"]["features_dim"] == 512
    assert captured["policy_kwargs"]["net_arch"] == {
        "pi": [256, 256],
        "vf": [256, 256],
    }


def test_reset_policy_optimizer_discards_prior_optimizer_state():
    module = torch.nn.Linear(2, 1)
    old_optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    old_optimizer.zero_grad()
    module(torch.ones(1, 2)).sum().backward()
    old_optimizer.step()
    assert old_optimizer.state

    policy = SimpleNamespace(
        optimizer=old_optimizer,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"eps": 1e-5},
        parameters=module.parameters,
    )

    new_optimizer = model_factory.reset_policy_optimizer(policy, 2e-4)

    assert new_optimizer is policy.optimizer
    assert new_optimizer is not old_optimizer
    assert not new_optimizer.state
    assert new_optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)
    assert new_optimizer.defaults["eps"] == pytest.approx(1e-5)


def test_save_model_atomically_publishes_complete_zip(tmp_path):
    class FakeModel:
        def save(self, path):
            Path(path).write_bytes(b"complete-model")

    target = model_factory.save_model_atomically(FakeModel(), tmp_path / "bc_model")

    assert target == tmp_path / "bc_model.zip"
    assert target.read_bytes() == b"complete-model"
    assert list(tmp_path.glob("*.tmp-*.zip")) == []
