import json
import os
import sys
import glob
import subprocess
import shutil

# All unique deck URLs from the top 25 on LimitlessTCG
MASTER_URLS = [
    "https://limitlesstcg.com/decks/list/28249",
    "https://limitlesstcg.com/decks/list/28251",
    "https://limitlesstcg.com/decks/list/28259",
    "https://limitlesstcg.com/decks/list/28263",
    "https://limitlesstcg.com/decks/list/27611",
    "https://limitlesstcg.com/decks/list/28265",
    "https://limitlesstcg.com/decks/list/28269",
    "https://limitlesstcg.com/decks/list/27612",
    "https://limitlesstcg.com/decks/list/28275",
    "https://limitlesstcg.com/decks/list/28278",
    "https://limitlesstcg.com/decks/list/28279",
    "https://limitlesstcg.com/decks/list/28283",
    "https://limitlesstcg.com/decks/list/28289",
    "https://limitlesstcg.com/decks/list/28291",
    "https://limitlesstcg.com/decks/list/28294",
    "https://limitlesstcg.com/decks/list/28302",
    "https://limitlesstcg.com/decks/list/28311",
    "https://limitlesstcg.com/decks/list/28314",
    "https://limitlesstcg.com/decks/list/28316",
    "https://limitlesstcg.com/decks/list/28319",
    "https://limitlesstcg.com/decks/list/28320",
    "https://limitlesstcg.com/decks/list/28324",
    "https://limitlesstcg.com/decks/list/27713",
    "https://limitlesstcg.com/decks/list/28331",
    "https://limitlesstcg.com/decks/list/28332",
    "https://limitlesstcg.com/decks/list/27922",
    "https://limitlesstcg.com/decks/list/28335",
    "https://limitlesstcg.com/decks/list/28340",
    "https://limitlesstcg.com/decks/list/28346",
    "https://limitlesstcg.com/decks/list/28359",
    "https://limitlesstcg.com/decks/list/28361",
    "https://limitlesstcg.com/decks/list/28364",
    "https://limitlesstcg.com/decks/list/28367",
    "https://limitlesstcg.com/decks/list/28368",
    "https://limitlesstcg.com/decks/list/26267",
    "https://limitlesstcg.com/decks/list/28371",
    "https://limitlesstcg.com/decks/list/28372",
    "https://limitlesstcg.com/decks/list/28376",
    "https://limitlesstcg.com/decks/list/28377",
    "https://limitlesstcg.com/decks/list/28378",
    "https://limitlesstcg.com/decks/list/28384",
    "https://limitlesstcg.com/decks/list/27516",
    "https://limitlesstcg.com/decks/list/28126",
    "https://limitlesstcg.com/decks/list/28398",
    "https://limitlesstcg.com/decks/list/28400",
    "https://limitlesstcg.com/decks/list/28403",
    "https://limitlesstcg.com/decks/list/28405",
    "https://limitlesstcg.com/decks/list/28411",
    "https://limitlesstcg.com/decks/list/28412",
    "https://limitlesstcg.com/decks/list/28417",
    "https://limitlesstcg.com/decks/list/28420",
    "https://limitlesstcg.com/decks/list/28426",
    "https://limitlesstcg.com/decks/list/28428",
    "https://limitlesstcg.com/decks/list/28431",
    "https://limitlesstcg.com/decks/list/27146",
    "https://limitlesstcg.com/decks/list/27056",
    "https://limitlesstcg.com/decks/list/28436",
    "https://limitlesstcg.com/decks/list/28443",
    "https://limitlesstcg.com/decks/list/27615",
    "https://limitlesstcg.com/decks/list/28453",
    "https://limitlesstcg.com/decks/list/28455",
    "https://limitlesstcg.com/decks/list/28456",
    "https://limitlesstcg.com/decks/list/28459",
    "https://limitlesstcg.com/decks/list/27614",
    "https://limitlesstcg.com/decks/list/28463",
    "https://limitlesstcg.com/decks/list/28472",
    "https://limitlesstcg.com/decks/list/28474",
    "https://limitlesstcg.com/decks/list/28476",
    "https://limitlesstcg.com/decks/list/28477",
    "https://limitlesstcg.com/decks/list/28479",
    "https://limitlesstcg.com/decks/list/28482",
    "https://limitlesstcg.com/decks/list/28484",
    "https://limitlesstcg.com/decks/list/28487",
    "https://limitlesstcg.com/decks/list/27926",
    "https://limitlesstcg.com/decks/list/28504",
    "https://limitlesstcg.com/decks/list/28505",
    "https://limitlesstcg.com/decks/list/28507",
    "https://limitlesstcg.com/decks/list/28510",
    "https://limitlesstcg.com/decks/list/28511",
    "https://limitlesstcg.com/decks/list/28515",
    "https://limitlesstcg.com/decks/list/28516",
    "https://limitlesstcg.com/decks/list/28518",
    "https://limitlesstcg.com/decks/list/26533",
    "https://limitlesstcg.com/decks/list/28529",
    "https://limitlesstcg.com/decks/list/28534",
    "https://limitlesstcg.com/decks/list/28537",
    "https://limitlesstcg.com/decks/list/28549",
    "https://limitlesstcg.com/decks/list/28551",
    "https://limitlesstcg.com/decks/list/28553",
    "https://limitlesstcg.com/decks/list/28554",
    "https://limitlesstcg.com/decks/list/28555",
    "https://limitlesstcg.com/decks/list/28556",
    "https://limitlesstcg.com/decks/list/27631",
    "https://limitlesstcg.com/decks/list/28561",
    "https://limitlesstcg.com/decks/list/28563",
    "https://limitlesstcg.com/decks/list/28568",
    "https://limitlesstcg.com/decks/list/28569",
    "https://limitlesstcg.com/decks/list/28572",
    "https://limitlesstcg.com/decks/list/28573",
    "https://limitlesstcg.com/decks/list/28578"
]

MAX_ROSTER_SIZE = 10
ELIMINATE_COUNT = 3
TIMESTEPS_PER_GEN = 100000
NUM_MATCH_GAMES = 5

def get_active_decks():
    decks = glob.glob("decks/deck_*.csv")
    # Return sorted by ID
    return sorted(decks, key=lambda x: int(x.split('_')[-1].split('.')[0]))

def get_highest_id():
    import glob
    active = glob.glob("decks/deck_*.csv")
    ghosts = glob.glob("decks/ghost_pool/deck_*.csv")
    all_decks = active + ghosts
    if not all_decks:
        return 0
    return max([int(x.split('_')[-1].split('.')[0]) for x in all_decks])

def scrape_missing_decks():
    active = get_active_decks()
    missing = MAX_ROSTER_SIZE - len(active)
    if missing <= 0:
        return
        
    print(f"Roster has {len(active)} decks. Scraping {missing} new decks from Limitless...")
    next_id = get_highest_id() + 1
    
    # Write a temporary scrape_temp.py to fetch the next decks
    # We will pick URLs that we haven't scraped yet. 
    # For simplicity, we just use the next_id as an index to MASTER_URLS.
    
    import scrape_decks
    import shutil
    import json
    
    # Identify the current champion to inherit weights from
    champion_id = None
    if os.path.exists("decks/active_decks.json"):
        try:
            with open("decks/active_decks.json", "r") as f:
                actives = json.load(f)
                if actives:
                    champion_id = actives[0] # Rank 1
        except: pass
    
    added = 0
    attempts = 0
    import glob
    bank_decks = glob.glob("decks/deck_bank/bank_*.csv")
    if bank_decks:
        bank_decks.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        max_attempts = len(bank_decks) * 2
    else:
        max_attempts = 0
        
    while added < missing and attempts < max_attempts:
        bank_idx = (next_id - 1 + attempts) % len(bank_decks)
        source_csv = bank_decks[bank_idx]
        
        new_deck_id = next_id + added
        print(f"Pulling {source_csv} from offline bank as Deck {new_deck_id}...")
        
        try:
            shutil.copyfile(source_csv, f"decks/deck_{new_deck_id}.csv")
            success = True
        except Exception as e:
            print(f"Failed to copy from bank: {e}")
            success = False
            
        if success:
            new_deck_id = next_id + added
            added += 1
            print(f" -> Success! Deck {new_deck_id} added.")
            
            # Weight Transfer / Inheritance
            source_model = None
            msg = ""
            if os.path.exists("models/ppo_base_brain.zip"):
                source_model = "models/ppo_base_brain.zip"
                msg = f"🧠 Deck {new_deck_id} startet mit dem Base Brain Fundament!"
            elif champion_id and os.path.exists(f"models/ppo_deck_{champion_id}.zip"):
                source_model = f"models/ppo_deck_{champion_id}.zip"
                msg = f"🧬 Deck {new_deck_id} erbt das Gehirn von Champion Deck {champion_id}!"
                
            if source_model:
                try:
                    shutil.copyfile(source_model, f"models/ppo_deck_{new_deck_id}.zip")
                    print(f" -> {msg}")
                    with open("decks/status.json", "w") as f:
                        json.dump({"action": msg, "completed": 1, "total": 1}, f)
                    import time; time.sleep(3)
                except Exception as e:
                    print(f" -> Failed to inherit weights: {e}")
            
        else:
            print(f" -> Failed/Rejected. Skipping to next URL...")
            
        attempts += 1
        
    # FALLBACK: If scraping failed, clone an existing deck to ensure we always have enough decks
    if added < missing:
        print(f"Warning: Could not scrape {missing - added} decks. Using fallback (cloning active decks)...")
        while added < missing:
            new_deck_id = next_id + added
            added += 1
            
            fallback_source_csv = None
            if champion_id and os.path.exists(f"decks/deck_{champion_id}.csv"):
                fallback_source_csv = f"decks/deck_{champion_id}.csv"
            else:
                actives = get_active_decks()
                if actives:
                    fallback_source_csv = actives[0]
                    
            if fallback_source_csv:
                shutil.copyfile(fallback_source_csv, f"decks/deck_{new_deck_id}.csv")
                print(f" -> Fallback Success! Cloned {fallback_source_csv} as Deck {new_deck_id}.")
                
                # Weight Transfer / Inheritance
                source_model = None
                msg = ""
                if os.path.exists("models/ppo_base_brain.zip"):
                    source_model = "models/ppo_base_brain.zip"
                    msg = f"🧠 Deck {new_deck_id} erbt das Base Brain Fundament (Fallback)!"
                elif champion_id and os.path.exists(f"models/ppo_deck_{champion_id}.zip"):
                    source_model = f"models/ppo_deck_{champion_id}.zip"
                    msg = f"🧬 Deck {new_deck_id} erbt das Gehirn von Champion Deck {champion_id} (Fallback)!"
                    
                if source_model:
                    try:
                        shutil.copyfile(source_model, f"models/ppo_deck_{new_deck_id}.zip")
                        print(f" -> {msg}")
                        with open("decks/status.json", "w") as f:
                            json.dump({"action": msg, "completed": 1, "total": 1}, f)
                        import time; time.sleep(3)
                    except Exception as e:
                        print(f" -> Failed to inherit weights: {e}")
            else:
                print(f" -> Critical Fallback Failure: No active decks to clone from!")  
        attempts += 1
        
    if added < missing:
        print(f"[!] Warning: Could only find {added} valid decks out of {missing} needed after {max_attempts} attempts.")

def get_all_opponent_candidates(deck_id):
    active = get_active_decks()
    ghosts = glob.glob("decks/ghost_pool/deck_*.csv")
    candidates = []
    for d in active + ghosts:
        did = d.split('_')[-1].split('.')[0]
        if did != deck_id:
            candidates.append(d)
    return candidates

def sample_opponent(deck_id):
    candidates = get_all_opponent_candidates(deck_id)
    if not candidates:
        return f"decks/deck_{deck_id}.csv" # Self-play fallback
        
    elos = get_elo_ratings()
    my_elo = get_elo(deck_id, elos)
                        
    import numpy as np
    import random
    
    weights = []
    for d in candidates:
        did = d.split('_')[-1].split('.')[0]
        opp_elo = get_elo(did, elos)
        
        # Skill-Based Matchmaking: higher weight for similar Elo
        # using exponential decay based on Elo difference (scale 200)
        diff = abs(my_elo - opp_elo)
        weight = np.exp(-diff / 200.0)
        weights.append(weight)
        
    weights = np.array(weights)
    weights /= weights.sum()
    
    chosen = np.random.choice(candidates, p=weights)
    return chosen

def train_active_decks(last_scores):
    active = get_active_decks()
    print("\n--- Training Generation ---")
    
    current_scores = {}
    for deck in active:
        deck_id = deck.split('_')[-1].split('.')[0]
        current_scores[deck_id] = last_scores.get(deck_id, 0.0)

    for i, deck in enumerate(active):
        deck_id = deck.split('_')[-1].split('.')[0]
        model_name = f"ppo_deck_{deck_id}"
        
        # Matchmaker: pick an opponent
        opp_deck = sample_opponent(deck_id)
        opp_id = os.path.basename(opp_deck).split('_')[-1].split('.')[0]
        opp_model_name = f"ppo_deck_{opp_id}"
        
        # Update leaderboard to show training status
        d_name = get_deck_name(deck)
        opp_name = get_deck_name(opp_deck)
        update_live_leaderboard(
            current_scores, 
            i, len(active), 
            f"🧠 Training: {d_name} vs {opp_name} (100.000 Schritte)"
        )
        
        # Check if opponent model is in active or ghost pool
        opp_model_path = f"models/{opp_model_name}.zip"
        if not os.path.exists(opp_model_path):
            opp_model_path = f"models/ghost_pool/{opp_model_name}.zip"
            if not os.path.exists(opp_model_path):
                opp_model_path = "" # Will fallback to self play inside train.py if missing
                
        print(f"Training Deck {deck_id} against {opp_id}...")
        
        cmd = [
            "python", "src/train.py",
            "--deck", deck,
            "--model-name", model_name,
            "--timesteps", str(TIMESTEPS_PER_GEN),
            "--opp-deck", opp_deck
        ]
        if opp_model_path:
            cmd.extend(["--opp-model", opp_model_path])
            
        subprocess.run(cmd)

ARTIFACT_PATH = "leaderboard.md"
GAMES_PLAYED_FILE = "decks/games_played.json"
ELO_RATINGS_FILE = "decks/elo_ratings.json"

def get_games_played():
    if os.path.exists(GAMES_PLAYED_FILE):
        with open(GAMES_PLAYED_FILE, "r") as f:
            return json.load(f)
    return {}

def save_games_played(gp):
    with open(GAMES_PLAYED_FILE, "w") as f:
        json.dump(gp, f)

def get_elo_ratings():
    if os.path.exists(ELO_RATINGS_FILE):
        with open(ELO_RATINGS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_elo_ratings(elos):
    with open(ELO_RATINGS_FILE, "w") as f:
        json.dump(elos, f)

def get_elo(deck_id, elos=None):
    if elos is None:
        elos = get_elo_ratings()
    return elos.get(str(deck_id), 1200.0)

def calculate_new_elo(elo_a, elo_b, score_a, k=32):
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    return elo_a + k * (score_a - expected_a)

_ID_TO_NAME_CACHE = None

def get_deck_name(deck_csv):
    global _ID_TO_NAME_CACHE
    try:
        if _ID_TO_NAME_CACHE is None:
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from src.cg.api import all_card_data
            cards = all_card_data()
            _ID_TO_NAME_CACHE = {c.cardId: c.name for c in cards}
            
        id_to_name = _ID_TO_NAME_CACHE
        
        # We manually label Deck 2 to preserve its legacy title
        if "deck_0" in deck_csv:
            return "Base Brain"
        if "deck_2" in deck_csv:
            return "Dragapult Dusknoir"
            
        with open(deck_csv, 'r') as f:
            for line in f:
                card_id = int(line.strip())
                name = id_to_name.get(card_id, "")
                if name and "Energy" not in name and "Ball" not in name and "Research" not in name and "Boss" not in name and "Arven" not in name and "Iono" not in name and "Nest" not in name and "Poffin" not in name:
                    return name
        return "Unknown"
    except:
        return "Unknown"

def update_live_leaderboard(winrates, completed, total, current_action="", match_results=None):
    elos = get_elo_ratings()
    active_ids = list(winrates.keys())
    for d_id in active_ids:
        if str(d_id) not in elos:
            elos[str(d_id)] = 1200.0
            
    # Sort by Elo instead of winrate
    sorted_ids = sorted(active_ids, key=lambda x: elos.get(str(x), 1200.0), reverse=True)
    
    games_played = get_games_played()
    
    md = f"---\n"
    md += f"requestFeedback: false\n"
    md += f"summary: Live Leaderboard\n"
    md += f"userFacing: true\n"
    md += f"---\n\n"
    md += f"# 🏆 Live Evolution Tournament Leaderboard\n\n"
    md += f"**Status:** {current_action} ({completed}/{total} Paarungen evaluiert)\n\n"
    
    status_data = {
        "action": current_action,
        "completed": completed,
        "total": total
    }
    with open("decks/status.json", "w") as f:
        json.dump(status_data, f)
    md += "| Platz | Deck | Elo | Win-Rate | Gespielte Matches |\n"
    md += "| --- | --- | --- | --- | --- |\n"
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

    for rank, deck_id in enumerate(sorted_ids, 1):
        d_name = deck_names_map.get(str(deck_id), f"Deck {deck_id}")
        gp = games_played.get(deck_id, 0)
        wr = winrates.get(deck_id, 0.0)
        elo_val = elos.get(str(deck_id), 1200.0)
        md += f"| **{rank}** | Deck {deck_id} ({d_name}) | **{int(elo_val)}** | {wr:.1f}% | {gp} |\n"
        
    with open("decks/deck_names.json", "w") as f:
        json.dump(deck_names_map, f)
        
    active_ids_str = [str(deck_id) for deck_id in sorted_ids]
    with open("decks/active_decks.json", "w") as f:
        json.dump(active_ids_str, f)
        
    if match_results:
        md += f"\n## Aktuelle Match-Ergebnisse\n"
        for res in reversed(match_results):
            md += f"- {res}\n"
            
    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        f.write(md)

def run_continuous_match():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    active = get_active_decks()
    if len(active) < 2:
        print("Not enough active decks to run a match.")
        import time
        time.sleep(5)
        return
        
    import random
    deck_a_file, deck_b_file = random.sample(active, 2)
    
    id_a = deck_a_file.split('_')[-1].split('.')[0]
    id_b = deck_b_file.split('_')[-1].split('.')[0]
    
    csv_a = deck_a_file
    zip_a = f"models/ppo_deck_{id_a}.zip"
    
    csv_b = deck_b_file
    zip_b = f"models/ppo_deck_{id_b}.zip"
    
    if not os.path.exists(zip_a) or not os.path.exists(zip_b):
        return
        
    name_a = get_deck_name(deck_a_file)
    name_b = get_deck_name(deck_b_file)
    
    # Optional live update status text
    try:
        with open("decks/status.json", "w") as f:
            json.dump({
                "status": "online",
                "progress": 0,
                "current_action": f"⚔️ Arena: {name_a} vs {name_b}"
            }, f)
    except: pass
    
    print(f"Match: Deck {id_a} vs Deck {id_b}")
    
    result = subprocess.run(
        ["python", "src/evaluate_single.py", zip_a, csv_a, zip_b, csv_b, str(NUM_MATCH_GAMES)],
        capture_output=True, text=True
    )
    
    wins_a = 0
    wins_b = 0
    pw_a = 0
    dw_a = 0
    bw_a = 0
    pw_b = 0
    dw_b = 0
    bw_b = 0
    if result.returncode != 0:
        print(f"Evaluation crashed! (C++ error). Counting as 0 wins.")
    else:
        for line in result.stdout.split('\n'):
            if line.startswith("RESULT:"):
                parts = line.split(":")[1].split(",")
                wins_a = int(parts[0])
                wins_b = int(parts[1]) # draws are ignored for winrate calculation
                try:
                    pw_a = int(parts[3])
                    dw_a = int(parts[4])
                    bw_a = int(parts[5])
                    pw_b = int(parts[6])
                    dw_b = int(parts[7])
                    bw_b = int(parts[8])
                except:
                    pass
                break
                
    # Elo Update
    elos = get_elo_ratings()
    elo_a = get_elo(id_a, elos)
    elo_b = get_elo(id_b, elos)
    
    draws = NUM_MATCH_GAMES - wins_a - wins_b
    score_a_total = wins_a + 0.5 * draws
    score_b_total = wins_b + 0.5 * draws
    
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    expected_b = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))
    
    K = 32
    new_elo_a = elo_a + K * (score_a_total - expected_a * NUM_MATCH_GAMES)
    new_elo_b = elo_b + K * (score_b_total - expected_b * NUM_MATCH_GAMES)
    
    elos[id_a] = new_elo_a
    elos[id_b] = new_elo_b
    save_elo_ratings(elos)
                
    games_played = get_games_played()
    games_played[id_a] = games_played.get(id_a, 0) + NUM_MATCH_GAMES
    games_played[id_b] = games_played.get(id_b, 0) + NUM_MATCH_GAMES
    save_games_played(games_played)
    
    # Track pairwise for Matchmaker (and dashboard)
    pairwise_file = "decks/pairwise_winrates.json"
    if os.path.exists(pairwise_file):
        with open(pairwise_file, "r") as f:
            pairwise = json.load(f)
    else:
        pairwise = {}
        
    if id_a not in pairwise: pairwise[id_a] = {}
    if id_b not in pairwise: pairwise[id_b] = {}
    
    # We store (wins, total_games)
    a_vs_b = pairwise[id_a].get(id_b, [0, 0])
    a_vs_b[0] += wins_a
    a_vs_b[1] += wins_a + wins_b
    pairwise[id_a][id_b] = a_vs_b
    
    b_vs_a = pairwise[id_b].get(id_a, [0, 0])
    b_vs_a[0] += wins_b
    b_vs_a[1] += wins_a + wins_b
    pairwise[id_b][id_a] = b_vs_a
    
    with open(pairwise_file, "w") as f:
        json.dump(pairwise, f)
        
    # Track for win conditions as well (in current_generation_winrates.json)
    current_file = "decks/current_generation_winrates.json"
    if os.path.exists(current_file):
        with open(current_file, "r") as f:
            current_pw = json.load(f)
    else:
        current_pw = {}
        
    if id_a not in current_pw: current_pw[id_a] = {}
    if id_b not in current_pw: current_pw[id_b] = {}
    
    # Format: [wins_a, total_games, pw_a, dw_a, wins_b, pw_b, dw_b, bw_a, bw_b]
    ca_vs_b = current_pw[id_a].get(id_b, [0, 0, 0, 0, 0, 0, 0, 0, 0])
    while len(ca_vs_b) < 9: ca_vs_b.append(0)
    
    ca_vs_b[0] += wins_a
    ca_vs_b[1] += wins_a + wins_b
    ca_vs_b[2] += pw_a
    ca_vs_b[3] += dw_a
    ca_vs_b[4] += wins_b
    ca_vs_b[5] += pw_b
    ca_vs_b[6] += dw_b
    ca_vs_b[7] += bw_a
    ca_vs_b[8] += bw_b
    current_pw[id_a][id_b] = ca_vs_b
    
    cb_vs_a = current_pw[id_b].get(id_a, [0, 0, 0, 0, 0, 0, 0, 0, 0])
    while len(cb_vs_a) < 9: cb_vs_a.append(0)
    
    cb_vs_a[0] += wins_b
    cb_vs_a[1] += wins_a + wins_b
    cb_vs_a[2] += pw_b
    cb_vs_a[3] += dw_b
    cb_vs_a[4] += wins_a
    cb_vs_a[5] += pw_a
    cb_vs_a[6] += dw_a
    cb_vs_a[7] += bw_b
    cb_vs_a[8] += bw_a
    current_pw[id_b][id_a] = cb_vs_a
    
    with open(current_file, "w") as f:
        json.dump(current_pw, f)
        
    # Re-calculate overall scores to update the static artifact/board
    scores = {d.split('_')[-1].split('.')[0]: 0 for d in active}
    total_games = {d.split('_')[-1].split('.')[0]: 0 for d in active}
    for da in current_pw:
        for db in current_pw[da]:
            scores[da] += current_pw[da][db][0]
            total_games[da] += current_pw[da][db][1]
            
    final_winrates = {}
    for d_id in scores:
        if total_games.get(d_id, 0) > 0:
            final_winrates[d_id] = (scores[d_id] / total_games[d_id]) * 100
        else:
            final_winrates[d_id] = 0.0
            
    # Quick live leaderboard update
    if wins_a > wins_b:
        res_str = f"**Deck {id_a}** vs Deck {id_b} -> **{wins_a}:{wins_b}**"
    elif wins_b > wins_a:
        res_str = f"Deck {id_a} vs **Deck {id_b}** -> **{wins_b}:{wins_a}**"
    else:
        res_str = f"Deck {id_a} vs Deck {id_b} -> Unentschieden ({wins_a}:{wins_b})"
        
    # You can keep track of recent match results by appending it to a file
    # For now we just print it
    print(res_str)
    
    # We update the artifact (which isn't strictly necessary with the web UI, but keeps compatibility)
    update_live_leaderboard(final_winrates, 1, 1, "Fortlaufende Arena", [res_str])
    
    # Small pause
    import time
    time.sleep(1)

def champion_bonus_training():
    active = get_active_decks()
    if len(active) < 2:
        return
        
    elos = get_elo_ratings()
    active_elos = {d.split('_')[-1].split('.')[0]: get_elo(d.split('_')[-1].split('.')[0], elos) for d in active}
    sorted_elos = sorted(active_elos.items(), key=lambda x: x[1], reverse=True)
    
    best_deck_id = sorted_elos[0][0]
    opp_id = sorted_elos[1][0]
    
    deck = f"decks/deck_{best_deck_id}.csv"
    opp_deck = f"decks/deck_{opp_id}.csv"
    model_name = f"ppo_deck_{best_deck_id}"
    opp_model_path = f"models/ppo_deck_{opp_id}.zip"
    
    print(f"\n--- 👑 Champion Bonus Training ---")
    print(f"Deck {best_deck_id} (Rank 1) receives 100k extra steps against Deck {opp_id} (Rank 2)!")
    
    d_name = get_deck_name(deck)
    opp_name = get_deck_name(opp_deck)
    update_live_leaderboard(
        {}, 
        0, 1, 
        f"👑 Champion Bonus: {d_name} vs {opp_name} (100k Steps)"
    )
    
    cmd = [
        "python", "src/train.py",
        "--deck", deck,
        "--model-name", model_name,
        "--timesteps", str(TIMESTEPS_PER_GEN),
        "--opp-deck", opp_deck
    ]
    if os.path.exists(opp_model_path):
        cmd.extend(["--opp-model", opp_model_path])
        
    subprocess.run(cmd)

def eliminate_weakest(scores):
    if not scores:
        return
        
    elos = get_elo_ratings()
    active_elos = {d_id: get_elo(d_id, elos) for d_id in scores.keys()}
    
    sorted_elos = sorted(active_elos.items(), key=lambda x: x[1])
    print("\n--- Leaderboard (by Elo) ---")
    for rank, (deck_id, elo) in enumerate(reversed(sorted_elos), 1):
        print(f"{rank}. Deck {deck_id} - Elo: {int(elo)}")
        
    # Eliminate the lowest Elos
    if len(sorted_elos) <= ELIMINATE_COUNT:
        eliminated = sorted_elos[:-1] # Leave at least the best one
    else:
        eliminated = sorted_elos[:ELIMINATE_COUNT]
        
    if not eliminated:
        print("\n--- Eliminating ---")
        print("Nichts zu eliminieren.")
        return
        
    print("\n--- Eliminating ---")
    for deck_id, elo in eliminated:
        print(f"Eliminating Deck {deck_id} (Elo: {int(elo)})")
        csv_path = f"decks/deck_{deck_id}.csv"
        zip_path = f"models/ppo_deck_{deck_id}.zip"
        
        os.makedirs("decks/ghost_pool", exist_ok=True)
        os.makedirs("models/ghost_pool", exist_ok=True)
        
        ghost_csv = f"decks/ghost_pool/deck_{deck_id}.csv"
        ghost_zip = f"models/ghost_pool/ppo_deck_{deck_id}.zip"
        
        if os.path.exists(csv_path): 
            os.rename(csv_path, ghost_csv)
        if os.path.exists(zip_path): 
            os.rename(zip_path, ghost_zip)

