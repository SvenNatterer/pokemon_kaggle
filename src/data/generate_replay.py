import os
import sys
import json

import argparse
import numpy as np

# Add src to pythonpath so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from src.env.env_wrapper import LEGACY_ACTION_SPACE_SIZE, PokemonTCGEnv, _fit_observation_to_model_space, read_sample_deck
from src.cg.game import visualize_data
import pandas as pd
from src.agents.rule_based_agent import is_rule_based_model_spec
from src.agents.bot_loader import load_bot
from src.utils import atomic_write_json, deck_display_name_for_path, resolve_deck_path

def read_deck(deck_path):
    resolved = resolve_deck_path(deck_path) if deck_path else None
    if not resolved or not resolved.exists():
        return read_sample_deck()
    df = pd.read_csv(resolved, header=None)
    return df[0].tolist()

def generate_replay(model_a_path, deck_a_path, model_b_path, deck_b_path, out_path):
    print(f"Loading models for replay...")
    deck_a = read_deck(deck_a_path)
    deck_b = read_deck(deck_b_path)
    
    model = None
    if is_rule_based_model_spec(model_a_path) or (model_a_path and os.path.exists(model_a_path)):
        print(f"Loading {model_a_path}...")
        model = load_bot(model_a_path)
        print("Model loaded successfully.")
    else:
        print("Model not found! Generating replay using random actions instead.")
        
    action_space_size = int(
        getattr(getattr(model, "action_space", None), "n", LEGACY_ACTION_SPACE_SIZE)
    )
    env = PokemonTCGEnv(
        my_deck=deck_a,
        opponent_deck=deck_b,
        opponent_model_path=model_b_path,
        action_space_size=action_space_size,
    )

    print("Resetting env...")
    obs, info = env.reset()
    done = False
    step = 0
    
    print("Simulating game...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        model_space = getattr(model, "observation_space", None)
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        while not done:
            if model is not None:
                if model_space is not None:
                    obs_for_model = _fit_observation_to_model_space(obs, model_space)
                else:
                    obs_for_model = obs
                action, lstm_state = model.predict(
                    obs_for_model,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=False,
                )
                episode_start = np.zeros((1,), dtype=bool)
            else:
                valid_actions = [i for i, mask in enumerate(obs["action_mask"]) if mask == 1]
                import random
                action = valid_actions[0] if valid_actions else 0
                if valid_actions:
                    action = random.choice(valid_actions)
                    
            # Extract and save data BEFORE the step, just in case the step crashes the C++ engine
            try:
                # print(f"Calling visualize_data at step {step}...")
                json_data = visualize_data()
                data = json.loads(json_data)
                if len(data) > 0:
                    deck_name_a = deck_display_name_for_path(deck_a_path) if deck_a_path else deck_a_path
                    deck_name_b = deck_display_name_for_path(deck_b_path) if deck_b_path else deck_b_path
                    data[0]["metadata"] = {
                        "p0_name": deck_name_a,
                        "p1_name": deck_name_b,
                        "p0_deck": deck_a_path,
                        "p1_deck": deck_b_path,
                    }
                    # print(f"Writing {len(json_data)} bytes...")
                    atomic_write_json(out_path, data)
                else:
                    atomic_write_json(out_path, data)
            except Exception as e:
                print("Error extracting data:", e)
                
            print(f"Taking step {step} with action {action}...")
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            
        print(f"Game finished in {step} steps. Reward: {reward}")
    except Exception as e:
        print(f"Game crashed early at step {step}! Extracting replay up to this point. Error: {e}")
    
    print("Extracting final visualizer data...")
    try:
        json_data = visualize_data()
        data = json.loads(json_data)
        if len(data) > 0:
            deck_name_a = deck_display_name_for_path(deck_a_path) if deck_a_path else deck_a_path
            deck_name_b = deck_display_name_for_path(deck_b_path) if deck_b_path else deck_b_path

            data[0]["metadata"] = {
                "p0_name": deck_name_a,
                "p1_name": deck_name_b,
                "p0_deck": deck_a_path,
                "p1_deck": deck_b_path,
            }
            atomic_write_json(out_path, data)
        else:
            atomic_write_json(out_path, data)
    except Exception:
        pass
        
    print(f"Replay saved to {out_path}!")
    env.close()

def slugify(text: str) -> str:
    cleaned = "".join(c if c.isalnum() else "_" for c in str(text).lower())
    return "_".join(filter(None, cleaned.split("_")))


def generate_replay_batch(
    model_a_path: str,
    deck_a_path: str,
    pool_path: str | None = None,
    model_b_path: str | None = None,
    deck_b_path: str | None = None,
    out_dir: str | None = None,
    num_games: int | None = None,
) -> list[str]:
    from pathlib import Path

    out_directory = Path(out_dir) if out_dir else Path("replays") / "batch_eval"
    out_directory.mkdir(parents=True, exist_ok=True)
    generated_files: list[str] = []

    opponents = []
    if pool_path and Path(pool_path).exists():
        pool_file = Path(pool_path)
        print(f"Loading opponent pool from {pool_file}...")
        pool_data = json.loads(pool_file.read_text(encoding="utf-8"))
        if isinstance(pool_data, list):
            for idx, bot in enumerate(pool_data, start=1):
                label = bot.get("label", f"bot_{idx}")
                opponents.append({
                    "label": label,
                    "model": bot.get("model", ""),
                    "deck": bot.get("deck", ""),
                })
    elif model_b_path or deck_b_path:
        opponents.append({
            "label": "opponent",
            "model": model_b_path or "",
            "deck": deck_b_path or "",
        })

    if not opponents:
        # Fallback: create default opponent sample based on deck bank if available
        deck_bank_dir = Path("decks") / "deck_bank"
        if deck_bank_dir.exists():
            deck_files = sorted(list(deck_bank_dir.glob("*.csv")))
            n = num_games if num_games else min(5, len(deck_files))
            for i in range(n):
                d_file = deck_files[i % len(deck_files)]
                opponents.append({
                    "label": d_file.stem,
                    "model": "rule_based:random",
                    "deck": str(d_file),
                })
        else:
            opponents.append({
                "label": "random_bot",
                "model": "rule_based:random",
                "deck": deck_a_path,
            })

    print(f"Generating replays for {len(opponents)} matches...")
    print(f"Candidate Model: {model_a_path}")
    print(f"Candidate Deck:  {deck_a_path}")
    print(f"Output Directory:{out_directory}\n")

    for idx, opp in enumerate(opponents, start=1):
        slug = slugify(opp["label"])
        out_file = out_directory / f"replay_vs_{slug}.json"
        if len(opponents) > 1 and (out_directory / f"replay_{idx:02d}_vs_{slug}.json").exists():
            out_file = out_directory / f"replay_{idx:02d}_vs_{slug}.json"

        print(f"[{idx}/{len(opponents)}] Playing vs {opp['label']} ({slug})...")
        try:
            generate_replay(
                model_a_path=str(model_a_path),
                deck_a_path=str(deck_a_path),
                model_b_path=str(opp["model"]),
                deck_b_path=str(opp["deck"]),
                out_path=str(out_file),
            )
            print(f"  -> Saved replay to {out_file}\n")
            generated_files.append(str(out_file))
        except Exception as err:
            print(f"  -> Replay generation failed for {opp['label']}: {err}\n", file=sys.stderr)

    return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=str, default="")
    parser.add_argument("--deck-a", type=str, default="")
    parser.add_argument("--model-b", type=str, default="")
    parser.add_argument("--deck-b", type=str, default="")
    parser.add_argument("--pool", type=str, default="")
    parser.add_argument("--out", type=str, default="replays/replay.json")
    args = parser.parse_args()

    if args.pool:
        generate_replay_batch(
            model_a_path=args.model_a,
            deck_a_path=args.deck_a,
            pool_path=args.pool,
            out_dir=args.out,
        )
    else:
        out_path = args.out
        if not os.path.isabs(out_path):
            out_path = os.path.join(os.path.dirname(__file__), "..", out_path)
        generate_replay(args.model_a, args.deck_a, args.model_b, args.deck_b, out_path)

