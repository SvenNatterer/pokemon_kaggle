"""Measure Python vs native C++ observation encoding throughput."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.cg.game import battle_finish
from src.env_wrapper import PokemonTCGEnv, V6_ACTION_SPACE_SIZE


ROOT = Path(__file__).resolve().parents[2]


def load_deck(path: Path) -> list[int]:
    cards = [int(line.split(",", 1)[0]) for line in path.read_text().splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"{path} contains {len(cards)} cards; expected 60")
    return cards


def close_env(env: PokemonTCGEnv) -> None:
    if env.current_obs_dict is not None:
        battle_finish()
        env.current_obs_dict = None


def encoder_benchmark(env: PokemonTCGEnv, iterations: int) -> tuple[float, float]:
    kwargs = {
        "perspective": env.learner_perspective,
        "pending_selection": env.pending_selection,
    }
    for _ in range(20):
        env._get_obs_python(**kwargs)
        env._get_obs_cpp(**kwargs)

    started = time.perf_counter()
    for _ in range(iterations):
        env._get_obs_python(**kwargs)
    python_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for _ in range(iterations):
        env._get_obs_cpp(**kwargs)
    cpp_seconds = time.perf_counter() - started
    return iterations / python_seconds, iterations / cpp_seconds


def env_benchmark(deck_a: list[int], deck_b: list[int], steps: int, native: bool) -> float:
    env = PokemonTCGEnv(
        deck_a,
        deck_b,
        action_space_size=V6_ACTION_SPACE_SIZE,
        rotate_perspective=True,
    )
    if not native:
        env._get_obs = env._get_obs_python
    rng = np.random.default_rng(20260715)
    completed = 0
    observation, _ = env.reset(seed=20260715)
    started = time.perf_counter()
    try:
        while completed < steps:
            legal = np.flatnonzero(observation["action_mask"])
            if legal.size == 0:
                raise RuntimeError("No legal action during benchmark")
            observation, _, terminated, truncated, _ = env.step(int(rng.choice(legal)))
            completed += 1
            if terminated or truncated:
                observation, _ = env.reset(seed=20260715 + completed)
    finally:
        elapsed = time.perf_counter() - started
        close_env(env)
    return completed / elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-iterations", type=int, default=5000)
    parser.add_argument("--env-steps", type=int, default=2000)
    args = parser.parse_args()

    deck_a = load_deck(ROOT / "decks" / "deck_98.csv")
    deck_b = load_deck(ROOT / "decks" / "deck_0.csv")
    env = PokemonTCGEnv(deck_a, deck_b, action_space_size=V6_ACTION_SPACE_SIZE)
    try:
        env.reset(seed=20260715)
        python_obs_s, cpp_obs_s = encoder_benchmark(env, args.encoder_iterations)
    finally:
        close_env(env)

    python_env_s = env_benchmark(deck_a, deck_b, args.env_steps, native=False)
    cpp_env_s = env_benchmark(deck_a, deck_b, args.env_steps, native=True)

    print(f"Encoder Python: {python_obs_s:,.0f} obs/s")
    print(f"Encoder C++:    {cpp_obs_s:,.0f} obs/s ({cpp_obs_s / python_obs_s:.2f}x)")
    print(f"Env Python:     {python_env_s:,.0f} learner steps/s")
    print(f"Env C++:        {cpp_env_s:,.0f} learner steps/s ({cpp_env_s / python_env_s:.2f}x)")


if __name__ == "__main__":
    main()
