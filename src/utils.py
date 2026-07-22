"""Shared utility functions for file operations, JSON handling, and naming."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from src.league.model_paths import parse_deck_model_path

import math

ROOT = Path(__file__).resolve().parents[1]


def wilson_lower_bound(wins: int, losses: int, draws: int, z: float = 1.96) -> float:
    """Wilson 95% lower bound; each draw counts as half a success."""
    total = wins + losses + draws
    if total <= 0:
        return 0.0
    successes = wins + 0.5 * draws
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    spread = z * math.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total)
    return max(0.0, min(1.0, (centre - spread) / denominator))


def utc_now() -> str:
    """Return ISO formatted current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Durably write JSON by flushing a sibling temp file before os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read a JSON file safely, returning default on missing/corrupted file."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def resolve_pool_path(pool_path: str | Path) -> Path:
    """Resolve pool path checking decks/pools/ first, then decks/."""
    p = Path(pool_path)
    if p.is_file():
        return p
    in_pools = ROOT / "decks" / "pools" / p.name
    if in_pools.is_file():
        return in_pools
    in_decks = ROOT / "decks" / p.name
    if in_decks.is_file():
        return in_decks
    return p


def resolve_deck_path(deck_path: str | Path) -> Path:
    """Resolve deck path checking decks/deck_bank/ first, then decks/."""
    p = Path(deck_path)
    if p.is_file():
        return p
    in_bank = ROOT / "decks" / "deck_bank" / p.name
    if in_bank.is_file():
        return in_bank
    in_decks = ROOT / "decks" / p.name
    if in_decks.is_file():
        return in_decks
    return p



def deck_id_for_path(deck_path: str) -> str:
    """Return the deck-name key for a regular or bank deck path."""
    stem = Path(deck_path).stem
    return stem[5:] if stem.startswith("deck_") else stem


def deck_name_for_path(deck_path: str) -> str:
    """Return the archetype name generated from the strongest Pokemon."""
    deck_id = deck_id_for_path(deck_path)
    names = read_json(ROOT / "decks" / "deck_names.json", {})
    name = str(names.get(deck_id) or "").strip()
    if name == "Hydrapple ex":
        return "Ogerpon"
    return name


def deck_display_name_for_path(deck_path: str) -> str:
    """Return a friendly label like `Ogerpon` for UI and logs."""
    return deck_name_for_path(deck_path) or "Unknown"


def model_display_name_for_path(model_path: str, deck_path: str) -> str:
    """Return a display label like `V5 Mega Lucario ex` for PPO bots."""
    parsed = parse_deck_model_path(model_path) or {}
    prefix = str(parsed.get("prefix") or "").strip()
    version = "PPO"
    if prefix.startswith("ppo_v"):
        version = f"V{prefix.split('_v', 1)[1].split('_', 1)[0]}"
    elif prefix == "ppo_deck":
        version = "PPO"
    archetype = deck_name_for_path(deck_path) or "Unknown"
    return f"{version} {archetype}"
