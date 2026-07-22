"""
pokemon_kaggle package - modularized structure
"""

import sys

from src import env, models, training, league, arena, agents, data

# Backward-compatibility aliases
from src.env import env_wrapper, adaptive_stopping
from src.models import custom_policy, inference_guardrails
from src.training import custom_ppo, train, training_health, lookahead_teacher

sys.modules["src.custom_ppo"] = custom_ppo
sys.modules["src.custom_policy"] = custom_policy
from src.league import pfsp, tournament, model_paths, experiment_registry
from src.arena import arena_core, arena_match, arena_worker, arena_control, arena_utils, evaluation_worker, evaluate_single, server
from src.agents import bot_loader
from src.data import deck_sources, scrape_decks, download_all_decks, limitless_deck_scraper, generate_replay

from src.env.env_wrapper import PokemonTCGEnv
from src.models.custom_policy import PokemonTCGFeatureExtractor
from src.league.model_paths import resolve_deck_model_path, discover_deck_models
from src.arena.arena_core import discover_participants, ArenaStore, rank_participants

__all__ = [
    "env",
    "models",
    "training",
    "league",
    "arena",
    "agents",
    "data",
    "env_wrapper",
    "adaptive_stopping",
    "custom_policy",
    "inference_guardrails",
    "custom_ppo",
    "train",
    "training_health",
    "lookahead_teacher",
    "pfsp",
    "tournament",
    "model_paths",
    "experiment_registry",
    "arena_core",
    "arena_match",
    "arena_worker",
    "arena_control",
    "arena_utils",
    "evaluation_worker",
    "evaluate_single",
    "server",
    "bot_loader",
    "deck_sources",
    "scrape_decks",
    "download_all_decks",
    "limitless_deck_scraper",
    "generate_replay",
    "PokemonTCGEnv",
    "PokemonTCGFeatureExtractor",
    "resolve_deck_model_path",
    "discover_deck_models",
    "discover_participants",
    "ArenaStore",
    "rank_participants",
]
