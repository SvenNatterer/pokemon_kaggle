import os
import sys
import json

import argparse
import numpy as np

# Add src to pythonpath so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from src.env_wrapper import PokemonTCGEnv, _fit_observation_to_model_space, read_sample_deck
from cg.game import visualize_data
import pandas as pd

def read_deck(deck_path):
    if not deck_path or not os.path.exists(deck_path):
        return read_sample_deck()
    df = pd.read_csv(deck_path, header=None)
    return df[0].tolist()

def generate_replay(model_a_path, deck_a_path, model_b_path, deck_b_path, out_path):
    print(f"Loading models for replay...")
    deck_a = read_deck(deck_a_path)
    deck_b = read_deck(deck_b_path)
    
    env = PokemonTCGEnv(my_deck=deck_a, opponent_deck=deck_b, opponent_model_path=model_b_path)
    
    from src.custom_ppo import CustomPPO
    def load_model_smart(path, env=None):
        try:
            return CustomPPO.load(path, env=env)
        except Exception as e:
            if env:
                try:
                    return PPO.load(path, env=env)
                except Exception:
                    pass
            try:
                return CustomPPO.load(path)
            except Exception:
                return PPO.load(path)
            
    model = None
    if model_a_path and os.path.exists(model_a_path):
        print(f"Loading {model_a_path}...")
        model = load_model_smart(model_a_path, env=env)
        print("Model loaded successfully.")
    else:
        print("Model not found! Generating replay using random actions instead.")
        
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
                    deck_name_a = deck_a_path
                    deck_name_b = deck_b_path
                    if os.path.exists("decks/deck_names.json"):
                        try:
                            with open("decks/deck_names.json", "r") as f:
                                names = json.load(f)
                                bname_a = os.path.basename(deck_a_path)[:-4]
                                ida = bname_a[5:] if bname_a.startswith("deck_") else bname_a
                                deck_name_a = f"D{ida.replace('bank_', '')} " + names.get(ida, "Unknown")
                                
                                bname_b = os.path.basename(deck_b_path)[:-4]
                                idb = bname_b[5:] if bname_b.startswith("deck_") else bname_b
                                deck_name_b = f"D{idb.replace('bank_', '')} " + names.get(idb, "Unknown")
                        except: pass
                    data[0]["metadata"] = {"p0_name": deck_name_a, "p1_name": deck_name_b}
                    # print(f"Writing {len(json_data)} bytes...")
                    with open(out_path, "w") as f:
                        json.dump(data, f)
                else:
                    with open(out_path, "w") as f:
                        f.write(json_data)
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
            deck_name_a = deck_a_path
            deck_name_b = deck_b_path
            if os.path.exists("decks/deck_names.json"):
                try:
                    with open("decks/deck_names.json", "r") as f:
                        names = json.load(f)
                        bname_a = os.path.basename(deck_a_path)[:-4]
                        ida = bname_a[5:] if bname_a.startswith("deck_") else bname_a
                        deck_name_a = f"D{ida.replace('bank_', '')} " + names.get(ida, "Unknown")
                        
                        bname_b = os.path.basename(deck_b_path)[:-4]
                        idb = bname_b[5:] if bname_b.startswith("deck_") else bname_b
                        deck_name_b = f"D{idb.replace('bank_', '')} " + names.get(idb, "Unknown")
                except: pass
            
            data[0]["metadata"] = {"p0_name": deck_name_a, "p1_name": deck_name_b}
            with open(out_path, "w") as f:
                json.dump(data, f)
        else:
            with open(out_path, "w") as f:
                f.write(json_data)
    except Exception:
        pass
        
    print(f"Replay saved to {out_path}!")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=str, default="")
    parser.add_argument("--deck-a", type=str, default="")
    parser.add_argument("--model-b", type=str, default="")
    parser.add_argument("--deck-b", type=str, default="")
    parser.add_argument("--out", type=str, default="replays/replay.json")
    args = parser.parse_args()
    
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(__file__), "..", out_path)
        
    generate_replay(args.model_a, args.deck_a, args.model_b, args.deck_b, out_path)
