"""Side-by-side validation for the Python and C++ V6 observation encoders."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.cg.game import battle_finish
from src.env_wrapper import PokemonTCGEnv, V6_ACTION_SPACE_SIZE


ROOT = Path(__file__).resolve().parents[2]


def load_deck(path: Path) -> list[int]:
    values = [int(line.strip().split(",")[0]) for line in path.read_text().splitlines() if line.strip()]
    if len(values) != 60:
        raise ValueError(f"{path} contains {len(values)} cards; expected 60")
    return values


def assert_observations_match(env: PokemonTCGEnv, game: int, step: int) -> None:
    kwargs = {
        "perspective": env.learner_perspective,
        "pending_selection": env.pending_selection,
    }
    python_obs = env._get_obs_python(**kwargs)
    cpp_obs = env._get_obs_cpp(**kwargs)
    if python_obs.keys() != cpp_obs.keys():
        raise AssertionError(
            f"game={game} step={step}: key mismatch: "
            f"python={sorted(python_obs)} cpp={sorted(cpp_obs)}"
        )

    for key in python_obs:
        python_value = np.asarray(python_obs[key])
        cpp_value = np.asarray(cpp_obs[key])
        if python_value.shape != cpp_value.shape:
            raise AssertionError(
                f"game={game} step={step} key={key}: "
                f"shape {python_value.shape} != {cpp_value.shape}"
            )
        if np.issubdtype(python_value.dtype, np.floating):
            matches = np.allclose(python_value, cpp_value, rtol=0.0, atol=1e-6)
        else:
            matches = np.array_equal(python_value, cpp_value)
        if not matches:
            mismatch = np.argwhere(~np.isclose(python_value, cpp_value, rtol=0.0, atol=1e-6))
            location = tuple(mismatch[0]) if mismatch.size else ()
            detail = ""
            if key == "option_features" and location:
                option_index = int(location[0])
                detail = (
                    f" option_card={python_obs['option_card_ids'][option_index]}"
                    f" attack={python_obs['option_attack_ids'][option_index]}"
                    f" entities={python_obs['entity_ids'].tolist()}"
                )
            raise AssertionError(
                f"game={game} step={step} key={key} index={location}: "
                f"python={python_value[location]} cpp={cpp_value[location]}{detail}"
            )


def run(games: int, seed: int, max_steps: int) -> None:
    deck_a = load_deck(ROOT / "decks" / "deck_98.csv")
    deck_b = load_deck(ROOT / "decks" / "deck_0.csv")
    env = PokemonTCGEnv(
        deck_a,
        deck_b,
        action_space_size=V6_ACTION_SPACE_SIZE,
        rotate_perspective=True,
    )
    rng = np.random.default_rng(seed)
    checked_states = 0
    try:
        for game in range(games):
            env.reset(seed=seed + game)
            for step in range(max_steps):
                assert_observations_match(env, game, step)
                checked_states += 1
                observation = env._get_obs_cpp(perspective=env.learner_perspective)
                legal_actions = np.flatnonzero(observation["action_mask"])
                if legal_actions.size == 0:
                    raise AssertionError(f"game={game} step={step}: no legal action")
                action = int(rng.choice(legal_actions))
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            else:
                raise AssertionError(f"game={game}: exceeded {max_steps} learner steps")
    finally:
        if env.current_obs_dict is not None:
            battle_finish()
            env.current_obs_dict = None
    print(f"PASS: {checked_states} states across {games} random games matched array-for-array")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-steps", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.games, args.seed, args.max_steps)
