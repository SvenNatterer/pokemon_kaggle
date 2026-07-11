import json
import os
import glob
import sys

sys.path.append(os.path.abspath('.'))
from src.model_paths import discover_deck_models

def clean():
    # 1. Get base models
    base_models = [m["name"] for m in discover_deck_models(include_variants=False)]
    print(f"Base models: {base_models}")

    # 2. Clean JSON files
    for filename in ["decks/elo_ratings.json", "decks/games_played.json", "decks/pairwise_winrates.json", "decks/current_generation_winrates.json"]:
        if not os.path.exists(filename): continue
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
            except:
                continue
        
        cleaned = {}
        for k, v in data.items():
            if k in base_models:
                if isinstance(v, dict):
                    cleaned[k] = {k2: v2 for k2, v2 in v.items() if k2 in base_models}
                else:
                    cleaned[k] = v
        
        with open(filename, 'w') as f:
            json.dump(cleaned, f, indent=2)

    # 3. Move variant zip files to backup
    os.makedirs("models/backup", exist_ok=True)
    all_zips = glob.glob("models/*.zip")
    for z in all_zips:
        name = os.path.basename(z)[:-4]
        if name not in base_models and name not in ['ppo_base_brain', 'ppo_v4_base_brain']:
            print(f"Moving variant: {z}")
            os.rename(z, os.path.join("models/backup", os.path.basename(z)))

if __name__ == '__main__':
    clean()
