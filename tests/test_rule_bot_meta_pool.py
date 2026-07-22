from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_rule_bots import load_pool, training_probabilities
from src.agents.rule_based_agent import is_rule_based_model_spec
from src.utils import resolve_deck_path, resolve_pool_path


ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = resolve_pool_path("rule_bot_meta_pool_v1.json")
GENERALIZATION_PATH = resolve_pool_path("rule_bot_generalization_v1.json")
TRAINING_POOL_PATH = resolve_pool_path("rule_bot_training_pool_v1.json")


def _reserved_decks(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(entry["deck_path"]) for entry in payload.get("opponents", [])}


def test_meta_pool_is_complete_weighted_and_loadable():
    payload, bots = load_pool(POOL_PATH)

    assert len(bots) == 20
    assert sum(entry["meta_weight"] for entry in bots) == pytest.approx(1.0)
    assert {entry["archetype"] for entry in bots if entry["group"] == "core"} == {
        "alakazam", "dragapult", "abomasnow", "lucario", "kangaskhan",
        "starmie", "grimmsnarl", "archaludon",
    }
    assert all(is_rule_based_model_spec(entry["model"]) for entry in bots)
    probabilities = training_probabilities(payload, bots)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert max(probabilities.values()) <= payload["sampling"]["maximum_single_bot_probability"]


def test_meta_pool_does_not_reuse_reserved_validation_or_holdout_decks():
    _payload, bots = load_pool(POOL_PATH)
    pool_decks = {entry["deck"] for entry in bots}
    reserved = _reserved_decks(resolve_pool_path("validation_opponents.json"))
    reserved |= _reserved_decks(resolve_pool_path("holdout_opponents.json"))

    assert pool_decks.isdisjoint(reserved)


def test_reconstructed_kaggle_decks_have_exactly_sixty_cards():
    for relative in (
        "deck_grimmsnarl_ex_kaggle.csv",
        "deck_archaludon_ex_kaggle.csv",
        "deck_starmie_ex_kaggle_a.csv",
        "deck_starmie_ex_kaggle_b.csv",
        "deck_abomasnow_ex_kaggle_validation.csv",
        "deck_lucario_ex_kaggle_validation.csv",
        "deck_starmie_ex_kaggle_validation.csv",
        "deck_grimmsnarl_ex_kaggle_validation.csv",
        "deck_archaludon_ex_kaggle_validation.csv",
    ):
        cards = [line for line in resolve_deck_path(relative).read_text(encoding="utf-8").splitlines() if line]
        assert len(cards) == 60


def test_generalization_decks_are_disjoint_from_training_and_reserved_sets():
    _pool_payload, pool_bots = load_pool(POOL_PATH)
    _generalization_payload, generalization_bots = load_pool(GENERALIZATION_PATH)
    pool_decks = {entry["deck"] for entry in pool_bots}
    generalization_decks = {entry["deck"] for entry in generalization_bots}
    reserved = _reserved_decks(resolve_pool_path("validation_opponents.json"))
    reserved |= _reserved_decks(resolve_pool_path("holdout_opponents.json"))

    assert len(generalization_bots) == 8
    assert generalization_decks.isdisjoint(pool_decks)
    assert generalization_decks.isdisjoint(reserved)


def test_runtime_training_pool_matches_derived_meta_probabilities():
    payload, bots = load_pool(POOL_PATH)
    expected = training_probabilities(payload, bots)
    runtime = json.loads(TRAINING_POOL_PATH.read_text(encoding="utf-8"))

    assert len(runtime) == len(bots)
    assert sum(entry["weight"] for entry in runtime) == pytest.approx(1.0)
    assert {entry["label"] for entry in runtime} == set(expected)
    for entry in runtime:
        assert entry["weight"] == pytest.approx(expected[entry["label"]], abs=1e-9)
