import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from src.training.behavioral_cloning import (
    load_demonstration_dataset,
    split_episodes,
    train_behavioral_cloning,
)
from scripts.train_behavioral_cloning import (
    build_sidecar_payload,
    parse_args,
    validate_dataset_provenance,
)


def _write_episode(
    root: Path,
    episode_id: int,
    *,
    length: int = 5,
    illegal_label: bool = False,
    invalid_outcome: bool = False,
    invalid_loss_mask: bool = False,
) -> dict[str, str | int]:
    vector = np.stack(
        [
            np.asarray([float((episode_id + step) % 2), float(step + 1)], dtype=np.float32)
            for step in range(length)
        ]
    )
    action_mask = np.zeros((length, 66), dtype=np.int8)
    action_mask[:, :2] = 1
    action = (vector[:, 0] > 0.5).astype(np.int64)
    if illegal_label:
        action[0] = 2
    loss_mask = np.ones(length, dtype=np.bool_)
    if invalid_loss_mask:
        loss_mask[0] = 0
    arrays = {
        "obs__vector": vector,
        "obs__action_mask": action_mask,
        "target__aux_target": np.tile(
            np.asarray([0.0, 0.5, 1.0], dtype=np.float32), (length, 1)
        ),
        "action": action,
        "loss_mask": loss_mask,
        "episode_id": np.full(length, episode_id, dtype=np.int64),
        "step": np.arange(length, dtype=np.int32),
        "episode_start": np.asarray([1] + [0] * (length - 1), dtype=np.bool_),
        "done": np.asarray([0] * (length - 1) + [1], dtype=np.bool_),
        "perspective": np.full(length, episode_id % 2, dtype=np.int8),
        "outcome": np.full(length, 2 if invalid_outcome else episode_id % 2, dtype=np.int8),
        "legal_count": np.full(length, 2, dtype=np.int16),
        "label_source": np.full(length, "rule_based:balanced", dtype=np.str_),
        "teacher_confidence": np.ones(length, dtype=np.float32),
    }
    path = root / f"episode_{episode_id:03d}.npz"
    np.savez(path, **arrays)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"episode_id": episode_id, "path": path.name, "sha256": digest}


def _write_dataset(root: Path, count: int = 4, **episode_kwargs) -> Path:
    entries = [
        _write_episode(root, episode_id, **episode_kwargs)
        for episode_id in range(count)
    ]
    manifest = {
        "schema_version": 1,
        "action_space_size": 66,
        "episodes": entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_actor = nn.LSTM(input_size=2, hidden_size=4, batch_first=True)
        self.action_head = nn.Linear(4, 66)
        self.aux_head = nn.Linear(4, 3)
        self.optimizer_class = torch.optim.Adam
        self.optimizer_kwargs = {"eps": 1e-5}
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=1e-4, **self.optimizer_kwargs
        )
        self.incoming_states: list[tuple[bool, float]] = []

    def set_training_mode(self, training: bool) -> None:
        self.train(training)

    def evaluate_behavior_cloning(
        self,
        obs,
        lstm_states,
        episode_starts,
        *,
        compute_aux=False,
    ):
        hidden, cell = lstm_states
        outputs = []
        for index in range(obs["vector"].shape[0]):
            starts = bool(episode_starts[index].item())
            self.incoming_states.append((starts, float(hidden.abs().sum().item())))
            keep = 1.0 - episode_starts[index].reshape(1, 1, 1)
            hidden = hidden * keep
            cell = cell * keep
            output, (hidden, cell) = self.lstm_actor(
                obs["vector"][index].reshape(1, 1, 2).float(),
                (hidden, cell),
            )
            outputs.append(output.reshape(1, 4))
        latent = torch.cat(outputs, dim=0)
        logits = self.action_head(latent)
        mask = obs["action_mask"].float()
        masked_logits = logits + (1.0 - mask) * -1e8
        distribution = torch.distributions.Categorical(logits=masked_logits)
        aux_logits = self.aux_head(latent) if compute_aux else None
        return distribution, masked_logits, aux_logits, (hidden, cell)


def _tiny_model():
    policy = _TinyPolicy()
    observation_space = gym.spaces.Dict(
        {
            "vector": gym.spaces.Box(-10, 10, shape=(2,), dtype=np.float32),
            "action_mask": gym.spaces.Box(0, 1, shape=(66,), dtype=np.int8),
            "aux_target": gym.spaces.Box(0, 1, shape=(3,), dtype=np.float32),
            "teacher_action": gym.spaces.Box(-1, 66, shape=(1,), dtype=np.int32),
            "teacher_value": gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        }
    )
    return SimpleNamespace(
        policy=policy,
        observation_space=observation_space,
        action_space=gym.spaces.Discrete(66),
        device="cpu",
        max_grad_norm=0.5,
        lr_schedule=lambda _progress: 1e-4,
        num_timesteps=123,
        _n_updates=9,
    )


def test_loader_is_episode_lazy_and_splits_complete_episodes(tmp_path, monkeypatch):
    dataset_path = _write_dataset(tmp_path)

    def reject_global_concatenation(*_args, **_kwargs):
        raise AssertionError("dataset loading must not concatenate every shard")

    with monkeypatch.context() as context:
        context.setattr(np, "concatenate", reject_global_concatenation)
        dataset = load_demonstration_dataset(dataset_path)

    assert len(dataset.episodes) == 4
    assert dataset.transition_count == 20
    assert not hasattr(dataset.episodes[0], "observations")
    train_a, validation_a = split_episodes(dataset.episodes, 0.25, seed=17)
    train_b, validation_b = split_episodes(dataset.episodes, 0.25, seed=17)

    assert [episode.episode_id for episode in train_a] == [
        episode.episode_id for episode in train_b
    ]
    assert [episode.episode_id for episode in validation_a] == [
        episode.episode_id for episode in validation_b
    ]
    assert {episode.episode_id for episode in train_a}.isdisjoint(
        episode.episode_id for episode in validation_a
    )
    assert all(
        episode.load().episode_start.tolist() == [True, False, False, False, False]
        for episode in dataset.episodes
    )


def test_loader_rejects_checksum_mismatch(tmp_path):
    _write_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episodes"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_demonstration_dataset(tmp_path)


def test_loader_requires_a_checksum_for_every_episode(tmp_path):
    _write_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episodes"][0].pop("sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid SHA-256"):
        load_demonstration_dataset(tmp_path)


def test_loader_rejects_manifest_sample_count_mismatch(tmp_path):
    _write_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["num_episodes"] = 4
    manifest["num_samples"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="declares 999 samples"):
        load_demonstration_dataset(tmp_path)


def test_loader_rejects_illegal_expert_action(tmp_path):
    _write_dataset(tmp_path, count=2, illegal_label=True)

    with pytest.raises(ValueError, match="expert action is illegal"):
        load_demonstration_dataset(tmp_path)


@pytest.mark.parametrize(
    ("episode_kwargs", "message"),
    [
        ({"invalid_outcome": True}, "outcome must contain only"),
        ({"invalid_loss_mask": True}, "loss_mask must equal"),
    ],
)
def test_loader_rejects_invalid_trajectory_metadata(tmp_path, episode_kwargs, message):
    _write_dataset(tmp_path, count=2, **episode_kwargs)

    with pytest.raises(ValueError, match=message):
        load_demonstration_dataset(tmp_path)


def test_recurrent_training_carries_state_between_chunks_and_resets_for_ppo(tmp_path):
    dataset = load_demonstration_dataset(_write_dataset(tmp_path))
    model = _tiny_model()
    original_optimizer = model.policy.optimizer

    result = train_behavioral_cloning(
        model,
        dataset,
        epochs=2,
        sequence_length=2,
        learning_rate=1e-2,
        aux_coef=0.1,
        validation_fraction=0.25,
        patience=2,
        seed=3,
    )

    assert result.best_epoch in {1, 2}
    assert np.isfinite(result.best_validation_nll)
    assert result.validation_metrics[-1].labelled_steps > 0
    assert np.isfinite(result.validation_metrics[-1].entropy)
    assert np.isfinite(result.validation_metrics[-1].aux_loss)
    assert model.num_timesteps == 0
    assert model._n_updates == 0
    assert model.policy.optimizer is not original_optimizer
    assert not model.policy.optimizer.state

    # Every true episode start receives a zero state. At least one later chunk
    # receives a carried, detached non-zero state.
    starts = [state_norm for is_start, state_norm in model.policy.incoming_states if is_start]
    continuations = [
        state_norm for is_start, state_norm in model.policy.incoming_states if not is_start
    ]
    assert starts and all(state_norm == pytest.approx(0.0) for state_norm in starts)
    assert any(state_norm > 0.0 for state_norm in continuations)

    payload = build_sidecar_payload(
        args=SimpleNamespace(config=None, epochs=2, seed=3),
        dataset=dataset,
        result=result,
        output_path=tmp_path / "model.zip",
        checkpoint_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )
    assert payload["dataset"]["manifest_sha256"] == "b" * 64
    assert payload["split"]["train_episode_ids"] == list(result.train_episode_ids)
    assert len(payload["training"]["train_history"]) == len(result.train_metrics)
    assert len(payload["training"]["validation_history"]) == len(
        result.validation_metrics
    )


def test_cli_config_supplies_paths_and_explicit_cli_values_win(tmp_path):
    config = tmp_path / "bc.yaml"
    config.write_text(
        "\n".join(
            [
                "dataset: data/demonstrations",
                "deck: decks/learner.csv",
                "opp_deck: decks/opponent.csv",
                "output_model: models/bc.zip",
                "epochs: 3",
                "lr: 0.0002",
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config), "--epochs", "7"])

    assert args.dataset == "data/demonstrations"
    assert args.output_model == "models/bc.zip"
    assert args.epochs == 7
    assert args.learning_rate == pytest.approx(2e-4)


def test_training_binds_decks_and_reserved_checks_to_dataset_provenance(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    dataset = load_demonstration_dataset(_write_dataset(dataset_root, count=2))
    learner_deck = tmp_path / "learner.csv"
    opponent_deck = tmp_path / "opponent.csv"
    other_deck = tmp_path / "other.csv"
    learner_deck.write_text("\n".join(str(index) for index in range(60)), encoding="utf-8")
    opponent_deck.write_text("\n".join(str(index + 100) for index in range(60)), encoding="utf-8")
    other_deck.write_text("\n".join(str(index + 200) for index in range(60)), encoding="utf-8")
    dataset.manifest["metadata"] = {
        "collector": "scripts/collect_lookahead_teacher.py",
        "scalar_obs": False,
        "feature_variant": "compact",
        "deck_sha256": hashlib.sha256(learner_deck.read_bytes()).hexdigest(),
        "opponent_deck_sha256": hashlib.sha256(opponent_deck.read_bytes()).hexdigest(),
        "reserved_opponent_manifests": [
            "/frozen/validation_opponents.json",
            "/frozen/holdout_opponents.json",
        ],
    }

    validate_dataset_provenance(
        dataset,
        deck=str(learner_deck),
        opponent_deck=str(opponent_deck),
        scalar_obs=False,
        feature_variant="compact",
    )

    with pytest.raises(ValueError, match="opponent_deck_sha256"):
        validate_dataset_provenance(
            dataset,
            deck=str(learner_deck),
            opponent_deck=str(other_deck),
            scalar_obs=False,
            feature_variant="compact",
        )

    dataset.manifest["metadata"]["reserved_opponent_manifests"] = [
        "/frozen/validation_opponents.json"
    ]
    with pytest.raises(ValueError, match="holdout_opponents.json"):
        validate_dataset_provenance(
            dataset,
            deck=str(learner_deck),
            opponent_deck=str(opponent_deck),
            scalar_obs=False,
            feature_variant="compact",
        )
