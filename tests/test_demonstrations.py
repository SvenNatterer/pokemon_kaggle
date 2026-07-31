from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.demonstrations import (
    ACTION_SPACE_SIZE,
    DemonstrationDatasetWriter,
    actor_observation,
    read_deck,
    validate_legal_action,
)
from scripts.collect_lookahead_teacher import build_parser


def _observation(*legal_actions: int) -> dict[str, np.ndarray]:
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    mask[list(legal_actions)] = 1
    return {
        "vector": np.zeros(1500, dtype=np.float32),
        "action_mask": mask,
        "aux_target": np.ones(2000, dtype=np.float32),
        "teacher_action": np.asarray([-1], dtype=np.int32),
        "teacher_value": np.asarray([0.0], dtype=np.float32),
        "entity_ids": np.zeros(12, dtype=np.int32),
        "entity_features": np.zeros((12, 36), dtype=np.float32),
        "option_card_ids": np.zeros(65, dtype=np.int32),
        "option_attack_ids": np.zeros(65, dtype=np.int32),
        "option_types": np.zeros(65, dtype=np.int32),
        "option_areas": np.zeros(65, dtype=np.int32),
        "option_features": np.zeros((65, 21), dtype=np.float32),
    }


def test_writer_publishes_complete_episode_without_target_leakage(tmp_path):
    writer = DemonstrationDatasetWriter(tmp_path / "dataset", metadata={"expert": "rule"})
    writer.start_episode(perspective=1)
    writer.append(_observation(0, 1), action=1, label_source="expert")
    q_values = np.arange(ACTION_SPACE_SIZE, dtype=np.float32)
    writer.append(
        _observation(0),
        action=0,
        label_source="lookahead",
        teacher_confidence=2.5,
        teacher_q=q_values,
    )
    shard = writer.commit_episode(outcome=-1)

    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["action_space_size"] == ACTION_SPACE_SIZE
    assert manifest["shards"] == ["shards/episode-000000.npz"]
    assert manifest["num_episodes"] == 1
    assert manifest["num_samples"] == 2

    with np.load(shard, allow_pickle=False) as arrays:
        assert "obs__aux_target" not in arrays
        assert "obs__teacher_action" not in arrays
        assert "obs__teacher_value" not in arrays
        assert arrays["target__aux_target"].shape == (2, 2000)
        assert arrays["action"].tolist() == [1, 0]
        assert arrays["loss_mask"].tolist() == [True, False]
        assert arrays["episode_id"].tolist() == [0, 0]
        assert arrays["step"].tolist() == [0, 1]
        assert arrays["episode_start"].tolist() == [True, False]
        assert arrays["done"].tolist() == [False, True]
        assert arrays["perspective"].tolist() == [1, 1]
        assert arrays["outcome"].tolist() == [-1, -1]
        assert arrays["legal_count"].tolist() == [2, 1]
        assert arrays["label_source"].tolist() == ["expert", "lookahead"]
        assert arrays["teacher_confidence"].tolist() == pytest.approx([0.0, 2.5])
        assert np.isnan(arrays["teacher_q"][0]).all()
        np.testing.assert_array_equal(arrays["teacher_q"][1], q_values)


def test_episode_ids_and_steps_are_contiguous_across_shards(tmp_path):
    writer = DemonstrationDatasetWriter(tmp_path / "dataset")
    for expected_episode_id in range(2):
        assert writer.start_episode(perspective=0) == expected_episode_id
        writer.append(_observation(3), action=3, label_source="expert")
        shard = writer.commit_episode(outcome=1)
        with np.load(shard, allow_pickle=False) as arrays:
            assert arrays["episode_id"].tolist() == [expected_episode_id]
            assert arrays["step"].tolist() == [0]


@pytest.mark.parametrize(
    ("mask_update", "message"),
    [
        (lambda mask: mask.__setitem__(0, 2), "binary"),
        (lambda mask: mask.fill(0), "no legal action"),
    ],
)
def test_validate_legal_action_rejects_invalid_masks(mask_update, message):
    observation = _observation(0)
    mask_update(observation["action_mask"])
    with pytest.raises(ValueError, match=message):
        validate_legal_action(observation, 0)


def test_writer_rejects_masked_label(tmp_path):
    writer = DemonstrationDatasetWriter(tmp_path / "dataset")
    writer.start_episode(perspective=0)
    with pytest.raises(ValueError, match="masked/illegal"):
        writer.append(_observation(0), action=1, label_source="expert")


def test_collector_default_is_large_enough_for_train_validation_split():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--expert",
            "rule_based:balanced",
            "--deck",
            "learner.csv",
            "--opp-deck",
            "opponent.csv",
            "--out",
            "dataset",
        ]
    )

    assert args.games >= 2


def test_collector_accepts_the_strategic_scalar_observation_pair():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--expert",
            "python_script:src/agents/kaggle_bots/alakazam_v8_agent.py",
            "--deck",
            "learner.csv",
            "--opp-deck",
            "opponent.csv",
            "--out",
            "dataset",
            "--scalar-obs",
            "--feature-variant",
            "strategic_vector_v1",
        ]
    )

    assert args.scalar_obs is True
    assert args.feature_variant == "strategic_vector_v1"


def test_actor_observation_accepts_the_scalar_v6_contract():
    observation = _observation(0, 1)
    scalar = actor_observation(
        {
            "vector": observation["vector"],
            "action_mask": observation["action_mask"],
            "aux_target": observation["aux_target"],
        }
    )

    assert set(scalar) == {"vector", "action_mask"}


def test_read_deck_accepts_csv_first_column_and_requires_sixty_cards(tmp_path):
    deck = tmp_path / "deck.csv"
    deck.write_text("".join(f"{card},ignored\n" for card in range(1, 61)))
    assert read_deck(deck) == list(range(1, 61))

    deck.write_text("1\n")
    with pytest.raises(ValueError, match="60 cards"):
        read_deck(deck)
