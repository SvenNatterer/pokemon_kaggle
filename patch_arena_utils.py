import re
import os

with open('src/arena_utils.py', 'r') as f:
    content = f.read()

# 1. Add get_active_models function below get_active_decks
if 'def get_active_models():' not in content:
    get_models_func = """
def get_active_models():
    from model_paths import discover_deck_models
    return [m["name"] for m in discover_deck_models(include_variants=True)]
"""
    content = content.replace('def get_active_decks():', get_models_func + '\ndef get_active_decks():')

# 2. Modify update_active_decks to update_active_models logic
update_func_replacement = """def update_active_decks(match_results=None):
    from model_paths import discover_deck_models
    active_models = [m["name"] for m in discover_deck_models(include_variants=True)]
    elos = get_elo_ratings()
    winrates = get_pairwise_winrates()
    games_played = get_games_played()
    
    # Sort models by Elo
    sorted_models = sorted(active_models, key=lambda m: elos.get(m, 1200.0), reverse=True)
    
    # Create Markdown table
    md = "# 🏆 Pokémon TCG AI Leaderboard\\n\\n"
    md += "| Rank | Model Name | Elo | Winrate | Matches |\\n"
    md += "|------|------------|-----|---------|---------|\\n"
    
    import json
    deck_names_map = {}
    if os.path.exists("decks/deck_names.json"):
        try:
            with open("decks/deck_names.json", "r") as f:
                deck_names_map = json.load(f)
        except: pass
        
    for rank, model_name in enumerate(sorted_models, 1):
        match = re.search(r'deck_(\d+)', model_name)
        deck_id = match.group(1) if match else "0"
        d_name = deck_names_map.get(deck_id, f"Deck {deck_id}")
        gp = games_played.get(model_name, 0)
        
        # Calculate winrate
        total_wins = 0
        total_matches = 0
        if model_name in winrates:
            for opp, stats in winrates[model_name].items():
                total_wins += stats[0]
                total_matches += stats[1]
        
        wr = (total_wins / total_matches * 100) if total_matches > 0 else 0.0
        
        elo_val = elos.get(model_name, 1200.0)
        md += f"| **{rank}** | {model_name} ({d_name}) | **{int(elo_val)}** | {wr:.1f}% | {total_matches} |\\n"
        
    with open("decks/active_decks.json", "w") as f:
        json.dump(sorted_models, f)
        
    if match_results:
        md += f"\\n## Aktuelle Match-Ergebnisse\\n"
        for res in reversed(match_results):
            md += f"- {res}\\n"
            
    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        f.write(md)
"""

# replace the old update_active_decks
content = re.sub(r'def update_active_decks\(match_results=None\):.*?def run_tournament\(', update_func_replacement + '\n\ndef run_tournament(', content, flags=re.DOTALL)

# 3. Modify run_continuous_match
run_match_replacement = """def run_continuous_match():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    active_models = get_active_models()
    if len(active_models) < 2:
        print("Not enough active models to run a match.")
        import time
        time.sleep(5)
        return
        
    import random
    model_a, model_b = random.sample(active_models, 2)
    
    import re
    def get_deck_id(model_name):
        match = re.search(r'deck_(\d+)', model_name)
        return match.group(1) if match else "0"
        
    id_a = get_deck_id(model_a)
    id_b = get_deck_id(model_b)
    
    csv_a = f"decks/deck_{id_a}.csv"
    csv_b = f"decks/deck_{id_b}.csv"
    
    zip_a = f"models/{model_a}.zip"
    zip_b = f"models/{model_b}.zip"
    
    if not os.path.exists(zip_a) or not os.path.exists(zip_b):
        print(f"Missing models: {zip_a} or {zip_b}")
        return
        
    name_a = get_deck_name(csv_a)
    name_b = get_deck_name(csv_b)
    
    # Optional live update status text
    try:
        with open("decks/status.json", "w") as f:
            json.dump({
                "status": "online",
                "progress": 0,
                "current_action": f"⚔️ Arena: {model_a} vs {model_b}"
            }, f)
    except: pass
    
    print(f"Match: {model_a} vs {model_b}")
    
    # Run evaluation script (which uses load_model_smart)
    # Important: evaluate_single.py currently expects deck_id as argument, 
    # BUT wait, evaluate_single.py uses id_a to resolve model path!
    # Let's change how we call evaluate_single.py.
    # Actually, evaluate_single.py takes id_a and id_b.
    # We must patch evaluate_single.py or we can just pass the model names 
    # instead of deck_ids to it!
"""

content = re.sub(r'def run_continuous_match\(\):.*?print\(f"Match: Deck \{id_a\} vs Deck \{id_b\}"\)', run_match_replacement, content, flags=re.DOTALL)

with open('src/arena_utils.py', 'w') as f:
    f.write(content)
