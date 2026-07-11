import json
import os
import glob
import re
import sys
import random
import shutil

def run():
    if not os.path.exists("decks/elo_ratings.json"):
        print("No elo_ratings.json found.")
        return

    with open("decks/elo_ratings.json", "r") as f:
        elos = json.load(f)

    if os.path.exists("decks/games_played.json"):
        with open("decks/games_played.json", "r") as f:
            games_played = json.load(f)
    else:
        games_played = {}

    if os.path.exists("decks/pairwise_winrates.json"):
        with open("decks/pairwise_winrates.json", "r") as f:
            pairwise = json.load(f)
    else:
        pairwise = {}

    # Get models in active directory
    active_zips = glob.glob("models/*.zip")
    active_model_names = [os.path.basename(z)[:-4] for z in active_zips]

    # Group by deck ID
    deck_groups = {}
    for model_name in active_model_names:
        m = re.match(r"^(ppo(?:_v4)?)_deck_((?:bank_)?\d+)", model_name)
        if m:
            deck_id = m.group(2)
            if deck_id not in deck_groups:
                deck_groups[deck_id] = []
            elo = elos.get(model_name, 1000)
            deck_groups[deck_id].append((model_name, elo))
        else:
            if "other" not in deck_groups:
                deck_groups["other"] = []
            deck_groups["other"].append((model_name, elos.get(model_name, 1000)))

    # Evaluate models and find champions
    best_models = set()
    to_eliminate = set()
    established_decks = []
    challenger_decks = []

    print("=== TOURNAMENT EVALUATION ===")
    for deck_id, models in deck_groups.items():
        if deck_id == "other":
            for m in models:
                best_models.add(m[0])
            continue

        models.sort(key=lambda x: x[1], reverse=True)
        best = models[0]
        best_name = best[0]
        best_elo = best[1]
        best_played = games_played.get(best_name, 0)
        
        print(f"Deck {deck_id}: BEST is {best_name} (Elo: {best_elo:.1f}, Games: {best_played})")
        best_models.add(best_name)

        if best_played >= 100:
            established_decks.append((deck_id, best_name, best_elo))
        else:
            challenger_decks.append((deck_id, best_name, best_elo))
        
        for other in models[1:]:
            other_name = other[0]
            played = games_played.get(other_name, 0)
            if played < 100:
                print(f"  - Keeping: {other_name} (Elo: {other[1]:.1f}) - Only {played} games played")
                best_models.add(other_name)
            else:
                to_eliminate.add(other_name)
                print(f"  - Eliminating: {other_name} (Elo: {other[1]:.1f})")

    # Step 2: Enforce Top 10 rule
    print("\n=== ROSTER MANAGEMENT ===")
    
    # Sort established decks by Elo
    established_decks.sort(key=lambda x: x[2], reverse=True)
    
    # If we have too many established decks, kick the weakest ones
    while len(established_decks) > 10:
        deck_id, name, elo = established_decks.pop()
        print(f"RELEGATION: Kicking established deck {deck_id} ({name}, Elo: {elo:.1f}) to backup!")
        to_eliminate.add(name)
        best_models.remove(name)

    print(f"Active Roster: {len(established_decks)} established, {len(challenger_decks)} challengers.")

    # Step 3: Promote a random deck if there's room
    newly_promoted_model = None
    if len(established_decks) + len(challenger_decks) <= 10:
        print("Roster has room. Looking for a new random challenger from backup...")
        os.makedirs("models/backup", exist_ok=True)
        backup_zips = glob.glob("models/backup/*.zip")
        backup_model_names = [os.path.basename(z)[:-4] for z in backup_zips]
        
        # Group backup models by deck
        backup_deck_groups = {}
        for b_name in backup_model_names:
            m = re.match(r"^(ppo(?:_v4)?)_deck_((?:bank_)?\d+)", b_name)
            if m:
                backup_deck_groups.setdefault(m.group(2), []).append(b_name)
        
        # Filter out decks that are already in the active roster (best_models)
        active_deck_ids = set()
        for active_model in best_models:
            m = re.match(r"^(ppo(?:_v4)?)_deck_((?:bank_)?\d+)", active_model)
            if m:
                active_deck_ids.add(m.group(2))
                
        available_backup_decks = [d for d in backup_deck_groups.keys() if d not in active_deck_ids]
        
        if available_backup_decks:
            chosen_deck_id = random.choice(available_backup_decks)
            # Pick the best model (by previous elo if exists) or just the first
            chosen_models = backup_deck_groups[chosen_deck_id]
            # Try to pick the one with highest elo from json if it still has one, else just random
            chosen_models.sort(key=lambda x: elos.get(x, 0), reverse=True)
            chosen_model = chosen_models[0]
            
            print(f"PROMOTION: Selected deck {chosen_deck_id} ({chosen_model}) to challenge the Top 10!")
            
            # We will move it from backup to models
            src_path = os.path.join("models/backup", chosen_model + ".zip")
            dst_path = os.path.join("models", chosen_model + ".zip")
            shutil.move(src_path, dst_path)
            
            # Reset stats
            elos[chosen_model] = 1000
            games_played[chosen_model] = 0
            if chosen_model in pairwise:
                del pairwise[chosen_model]
            
            # Mark it as active so it doesn't get cleaned up
            best_models.add(chosen_model)
            newly_promoted_model = chosen_model
        else:
            print("No available backup decks found that aren't already active.")

    print("\nStarting cleanup...")
    os.makedirs("models/backup", exist_ok=True)
    
    # Clean up JSON files
    json_files = [
        "decks/elo_ratings.json", 
        "decks/games_played.json", 
        "decks/pairwise_winrates.json", 
        "decks/current_generation_winrates.json"
    ]
    
    # We must save the elos, games_played, and pairwise dicts we modified
    for filename in json_files:
        if not os.path.exists(filename): continue
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
            except:
                continue
        
        # Override with our updated dicts if applicable
        if filename == "decks/elo_ratings.json":
            data = elos
        elif filename == "decks/games_played.json":
            data = games_played
        elif filename == "decks/pairwise_winrates.json":
            data = pairwise

        cleaned = {}
        for k, v in data.items():
            if k in best_models:
                if isinstance(v, dict):
                    cleaned[k] = {k2: v2 for k2, v2 in v.items() if k2 in best_models}
                else:
                    cleaned[k] = v
        
        with open(filename, 'w') as f:
            json.dump(cleaned, f, indent=2)

    # Move zip files
    all_zips = glob.glob("models/*.zip")
    import time
    now = time.time()
    for z in all_zips:
        try:
            if (now - os.path.getmtime(z)) < 7200:
                print(f"Skipping recently modified model (actively training): {z}")
                continue
        except:
            pass
            
        name = os.path.basename(z)[:-4]
        # Only move if we explicitly marked it for elimination OR it's a known deck format but not in best
        m = re.match(r"^(ppo(?:_v4)?)_deck_((?:bank_)?\d+)", name)
        if m:
            if name in to_eliminate or (name not in best_models and name != newly_promoted_model):
                print(f"Moving to backup: {z}")
                os.rename(z, os.path.join("models/backup", os.path.basename(z)))

    print("Cleanup complete!")

if __name__ == '__main__':
    run()
