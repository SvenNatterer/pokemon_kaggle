import re
import os

with open('src/arena_utils.py', 'r') as f:
    content = f.read()

# Add get_active_models
if 'def get_active_models():' not in content:
    content = content.replace('def get_active_decks():', 'def get_active_models():\n    from model_paths import discover_deck_models\n    return [m["name"] for m in discover_deck_models(include_variants=True)]\n\ndef get_active_decks():')


# Replace update_live_leaderboard
leaderboard_str = """def update_live_leaderboard(winrates, completed, total, current_action="", match_results=None):
    elos = get_elo_ratings()
    
    # We want ALL active models to be tracked
    from model_paths import discover_deck_models
    all_models = [m["name"] for m in discover_deck_models(include_variants=True)]
    
    active_ids = list(set(list(winrates.keys()) + all_models))
    for d_id in active_ids:
        if str(d_id) not in elos:
            elos[str(d_id)] = 1200.0
            
    # Sort by Elo
    sorted_ids = sorted(active_ids, key=lambda x: elos.get(str(x), 1200.0), reverse=True)
    
    games_played = get_games_played()
    
    md = f"---\\n"
    md += f"requestFeedback: false\\n"
    md += f"summary: Live Leaderboard\\n"
    md += f"userFacing: true\\n"
    md += f"---\\n\\n"
    md += f"# 🏆 Live Evolution Tournament Leaderboard\\n\\n"
    md += f"**Status:** {current_action} ({completed}/{total} Paarungen evaluiert)\\n\\n"
    
    status_data = {
        "action": current_action,
        "completed": completed,
        "total": total
    }
    with open("decks/status.json", "w") as f:
        json.dump(status_data, f)
    md += "| Platz | Deck | Elo | Win-Rate | Gespielte Matches |\\n"
    md += "| --- | --- | --- | --- | --- |\\n"
    deck_names_map = {}
    if os.path.exists("decks/deck_names.json"):
        try:
            with open("decks/deck_names.json", "r") as f:
                deck_names_map = json.load(f)
        except: pass

    import glob
    for deck_file in glob.glob("decks/deck_*.csv") + glob.glob("decks/ghost_pool/deck_*.csv"):
        try:
            d_id = str(deck_file.split('_')[-1].split('.')[0])
            deck_names_map[d_id] = get_deck_name(deck_file)
        except: pass

    for rank, model_id in enumerate(sorted_ids, 1):
        import re
        match = re.search(r'deck_(\d+)', str(model_id))
        deck_num = match.group(1) if match else "0"
        d_name = deck_names_map.get(str(deck_num), f"Deck {deck_num}")
        gp = games_played.get(model_id, 0)
        
        # Calculate winrate from pairwise
        total_wins = 0
        total_matches = 0
        if model_id in winrates:
            for opp, stats in winrates[model_id].items():
                total_wins += stats[0]
                total_matches += stats[1]
                
        wr = (total_wins / total_matches * 100) if total_matches > 0 else 0.0
        elo_val = elos.get(str(model_id), 1200.0)
        md += f"| **{rank}** | {model_id} ({d_name}) | **{int(elo_val)}** | {wr:.1f}% | {total_matches} |\\n"
        
    with open("decks/deck_names.json", "w") as f:
        json.dump(deck_names_map, f)
        
    active_ids_str = [str(model_id) for model_id in sorted_ids]
    with open("decks/active_decks.json", "w") as f:
        json.dump(active_ids_str, f)
        
    if match_results:
        md += f"\\n## Aktuelle Match-Ergebnisse\\n"
        for res in reversed(match_results):
            md += f"- {res}\\n"
            
    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        f.write(md)
"""

content = re.sub(r'def update_live_leaderboard.*?def run_tournament', leaderboard_str + '\n\ndef run_tournament', content, flags=re.DOTALL)

# Replace run_continuous_match
match_str = """def run_continuous_match():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    active = get_active_models()
    if len(active) < 2:
        print("Not enough active models to run a match.")
        import time
        time.sleep(5)
        return
        
    import random
    id_a, id_b = random.sample(active, 2)
    
    import re
    match_a = re.search(r'deck_(\d+)', id_a)
    deck_num_a = match_a.group(1) if match_a else "0"
    match_b = re.search(r'deck_(\d+)', id_b)
    deck_num_b = match_b.group(1) if match_b else "0"
    
    csv_a = f"decks/deck_{deck_num_a}.csv"
    csv_b = f"decks/deck_{deck_num_b}.csv"
    zip_a = f"models/{id_a}.zip"
    zip_b = f"models/{id_b}.zip"
    
    if not os.path.exists(zip_a) or not os.path.exists(zip_b):
        return
        
    name_a = get_deck_name(csv_a)
    name_b = get_deck_name(csv_b)
    
    try:
        with open("decks/status.json", "w") as f:
            json.dump({
                "status": "online",
                "progress": 0,
                "current_action": f"⚔️ Arena: {id_a} vs {id_b}"
            }, f)
    except: pass
    
    print(f"Match: {id_a} vs {id_b}")
"""

content = re.sub(r'def run_continuous_match\(\):.*?print\(f"Match: Deck \{id_a\} vs Deck \{id_b\}"\)', match_str, content, flags=re.DOTALL)

with open('src/arena_utils.py', 'w') as f:
    f.write(content)
