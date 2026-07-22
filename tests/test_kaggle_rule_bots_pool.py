from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.agents.bot_loader import load_bot
from src.agents.rule_based_agent import RuleBasedPokemonAgent
from src.utils import resolve_deck_path, resolve_pool_path


POOL_PATH = resolve_pool_path("kaggle_rule_bots_dev_pool.json")


def test_kaggle_rule_bots_pool_file_exists_and_contains_seven_bots():
    assert POOL_PATH.is_file()
    payload = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 7

    labels = {entry["label"] for entry in payload}
    assert len(labels) == 7
    assert any("Mega Lucario" in label for label in labels)
    assert any("Mega Abomasnow" in label for label in labels)
    assert any("Dragapult" in label for label in labels)
    assert any("Iono" in label for label in labels)
    assert any("Alakazam" in label for label in labels)
    assert any("Battlecore" in label for label in labels)
    assert any("Conservative" in label for label in labels)


def test_kaggle_rule_bots_decks_all_have_sixty_cards():
    payload = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    for entry in payload:
        deck_path = resolve_deck_path(entry["deck"])
        assert deck_path.is_file(), f"Deck file missing: {entry['deck']}"
        lines = [line.strip() for line in deck_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 60, f"Deck {entry['deck']} has {len(lines)} cards instead of 60"


def test_kaggle_rule_bots_agents_can_all_be_loaded():
    payload = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    for entry in payload:
        model_path = entry["model"]
        bot = load_bot(model_path)
        assert bot is not None, f"Could not load bot for {model_path}"
        assert hasattr(bot, "predict"), f"Loaded bot {model_path} lacks predict method"


def test_model_can_battle_against_kaggle_dev_pool_opponent():
    payload = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    model = RuleBasedPokemonAgent(spec="rule_based:balanced")

    dummy_obs = {
        "vector": [0] * 1500,
        "action_mask": [1, 0, 0, 1],
    }

    action_model, _ = model.predict(dummy_obs)
    assert action_model is not None

    for target in payload:
        opponent_bot = load_bot(target["model"])
        action_opp, _ = opponent_bot.predict(dummy_obs)
        assert action_opp is not None, f"Bot {target['label']} returned None action"
