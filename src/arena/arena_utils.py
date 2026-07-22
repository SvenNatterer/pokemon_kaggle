import json
import os
import sys
import glob
import subprocess
import shutil
import time
import random
from src.league.model_paths import parse_deck_model_path
from src.league.model_paths import (
    default_deck_model_path,
    iter_existing_deck_model_paths,
    resolve_deck_model_base,
    resolve_deck_model_path,
)
from src.agents.rule_based_agent import is_rule_based_model_spec


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
MATCH_TIMEOUT_SECONDS = 60
SPECIAL_ARENA_CONFIG = "decks/arena_specials.json"
DEFAULT_SPECIAL_ARENA_CONTENDERS = [
    {
        "id": "rule_based",
        "label": "Rule Bot (Riolu)",
        "deck": "decks/deck_bank/bank_18.csv",
        "model": "rule_based",
        "weight": 1.0,
        "enabled": True,
    }
]


def get_special_arena_contenders():
    contenders = []
    source = DEFAULT_SPECIAL_ARENA_CONTENDERS
    match_rate = 0.25
    if os.path.exists(SPECIAL_ARENA_CONFIG):
        try:
            with open(SPECIAL_ARENA_CONFIG, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                source = data.get("contenders", source)
                match_rate = max(0.0, min(1.0, float(data.get("match_rate", match_rate))))
            elif isinstance(data, list):
                source = data
        except Exception:
            pass

    for entry in source:
        if not isinstance(entry, dict):
            continue
        if not entry.get("enabled", True):
            continue
        deck = entry.get("deck")
        model = entry.get("model", entry.get("model_path", ""))
        contender_id = entry.get("id") or entry.get("label") or "rule_based"
        if not deck or not os.path.exists(deck):
            continue
        contenders.append({
            "id": contender_id,
            "label": entry.get("label", contender_id),
            "deck": deck,
            "model": model,
            "weight": max(0.0, float(entry.get("weight", 1.0))),
        })
    return contenders, match_rate


def _resolve_participant(participant_id, special_lookup):
    if participant_id in special_lookup:
        return special_lookup[participant_id]

    parsed = None
    try:
        parsed = parse_deck_model_path(f"models/{participant_id}.zip")
    except Exception:
        parsed = None

    if not parsed:
        return None

    deck_num = parsed["deck_id"]
    deck_path = f"decks/deck_bank/{deck_num}.csv" if str(deck_num).startswith("bank_") else f"decks/deck_{deck_num}.csv"
    return {
        "id": participant_id,
        "label": participant_id,
        "deck": deck_path,
        "model": f"models/{participant_id}.zip",
    }

import json
from pathlib import Path

ARENA_AGENTS_FILE = Path("decks/arena_agents.json")


def get_deck_path(deck_id):
    deck_id = str(deck_id)

    if deck_id.startswith("bank_"):
        return f"decks/deck_bank/{deck_id}.csv"

    return f"decks/deck_{deck_id}.csv"


def get_arena_participants():
    participants = []

    # Normale PPO-Modelle
    for model_path in Path("models").glob("*.zip"):
        parsed = parse_deck_model_path(str(model_path))

        if not parsed:
            continue

        deck_id = parsed["deck_id"]

        participants.append({
            "id": model_path.stem,
            "name": model_path.stem,
            "agent_type": "ppo",
            "model_path": str(model_path),
            "deck_path": get_deck_path(deck_id),
        })

    # Nicht trainierte Agenten, beispielsweise Rule Bots
    if ARENA_AGENTS_FILE.exists():
        with ARENA_AGENTS_FILE.open("r", encoding="utf-8") as file:
            config = json.load(file)

        for agent in config.get("agents", []):
            if not agent.get("enabled", True):
                continue

            participants.append({
                "id": agent["id"],
                "name": agent.get("name", agent["id"]),
                "agent_type": agent["agent_type"],
                "model_path": None,
                "deck_path": agent["deck_path"],
            })

    return participants

def get_active_models():
    from src.league.model_paths import discover_deck_models
    return [m["name"] for m in discover_deck_models(include_variants=True)]

def get_active_decks():
    decks = glob.glob("decks/deck_*.csv")
    valid_decks = [d for d in decks if d.split('_')[-1].split('.')[0].isdigit()]
    # Return sorted by ID
    return sorted(valid_decks, key=lambda x: int(x.split('_')[-1].split('.')[0]))

def get_highest_id():
    import glob
    active = glob.glob("decks/deck_*.csv")
    ghosts = glob.glob("decks/ghost_pool/deck_*.csv")
    all_decks = active + ghosts
    valid_decks = [d for d in all_decks if d.split('_')[-1].split('.')[0].isdigit()]
    if not valid_decks:
        return 0
    return max([int(x.split('_')[-1].split('.')[0]) for x in valid_decks])

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
            champion_model = resolve_deck_model_path(champion_id) if champion_id else ""
            if os.path.exists("models/ppo_base_brain.zip"):
                source_model = "models/ppo_base_brain.zip"
                msg = f"🧠 Deck {new_deck_id} startet mit dem Base Brain Fundament!"
            elif champion_model:
                source_model = champion_model
                msg = f"🧬 Deck {new_deck_id} erbt das Gehirn von Champion Deck {champion_id}!"
                
            if source_model:
                try:
                    shutil.copyfile(source_model, default_deck_model_path(new_deck_id))
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
                champion_model = resolve_deck_model_path(champion_id) if champion_id else ""
                if os.path.exists("models/ppo_base_brain.zip"):
                    source_model = "models/ppo_base_brain.zip"
                    msg = f"🧠 Deck {new_deck_id} erbt das Base Brain Fundament (Fallback)!"
                elif champion_model:
                    source_model = champion_model
                    msg = f"🧬 Deck {new_deck_id} erbt das Gehirn von Champion Deck {champion_id} (Fallback)!"
                    
                if source_model:
                    try:
                        shutil.copyfile(source_model, default_deck_model_path(new_deck_id))
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
        model_name = resolve_deck_model_base(deck_id)
        
        # Matchmaker: pick an opponent
        opp_deck = sample_opponent(deck_id)
        opp_id = os.path.basename(opp_deck).split('_')[-1].split('.')[0]
        
        # Update leaderboard to show training status
        d_name = get_deck_name(deck)
        opp_name = get_deck_name(opp_deck)
        update_live_leaderboard(
            current_scores, 
            i, len(active), 
            f"🧠 Training: {d_name} vs {opp_name} (100.000 Schritte)"
        )
        
        opp_model_path = resolve_deck_model_path(opp_id, include_ghost=True)
                
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
        try:
            with open(GAMES_PLAYED_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_games_played(gp):
    with open(GAMES_PLAYED_FILE, "w") as f:
        json.dump(gp, f)

def get_elo_ratings():
    if os.path.exists(ELO_RATINGS_FILE):
        try:
            with open(ELO_RATINGS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
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
        if "deck_4" in deck_csv:
            return "v4 Base Brain"
            
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
    
    # We want ALL active models to be tracked
    from src.league.model_paths import discover_deck_models
    all_models = [m["name"] for m in discover_deck_models(include_variants=True)]
    
    active_ids = list(set(list(winrates.keys()) + all_models))
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
    for deck_file in glob.glob("decks/deck_*.csv") + glob.glob("decks/ghost_pool/deck_*.csv") + glob.glob("decks/deck_bank/bank_*.csv"):
        try:
            base = os.path.basename(deck_file).replace('.csv', '')
            d_id = base.replace('deck_', '') if base.startswith('deck_') else base
            deck_names_map[d_id] = get_deck_name(deck_file)
        except: pass

    for rank, model_id in enumerate(sorted_ids, 1):
        from src.league.model_paths import parse_deck_model_path
        parsed = parse_deck_model_path(f"models/{model_id}.zip")
        deck_num = parsed["deck_id"] if parsed else "0"
        d_name = deck_names_map.get(str(deck_num), f"Deck {deck_num}")
        gp = games_played.get(model_id, 0)
        
        wr = winrates.get(model_id, 0.0)
        
        elo_val = elos.get(str(model_id), 1200.0)
        md += f"| **{rank}** | {model_id} ({d_name}) | **{int(elo_val)}** | {wr:.1f}% | {gp} |\\n"
        
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


def run_tournament(num_matches=None):
    matches_played = 0
    while num_matches is None or matches_played < num_matches:
        run_continuous_match()
        matches_played += 1

FILTER_COUNTER = 0

def auto_filter_enabled():
    value = os.environ.get("POKEMON_ARENA_AUTO_FILTER", "")
    return value.lower() in {"1", "true", "yes", "on"}

def maybe_run_auto_filter(threshold, reason):
    global FILTER_COUNTER
    FILTER_COUNTER += 1
    if FILTER_COUNTER < threshold:
        return

    FILTER_COUNTER = 0
    if not auto_filter_enabled():
        print(f"Arena: automatic filtering disabled; run scripts/manual_filter_models.py manually ({reason}).")
        return

    print("Arena: Running automatic roster management (filter_best_models.py)...")
    subprocess.run(["venv/bin/python", "filter_best_models.py"])

def manage_queue():
    import os
    import random
    import shutil
    from src.league.model_paths import discover_deck_models
    
    games = get_games_played()
    active = get_active_models()
    
    # Identify challengers
    challengers = [m for m in active if games.get(m, 0) < 100]
    challengers_count = len(challengers)
    
    if challengers_count < 3:
        # Only pull valid models (ignoring junk/temp files)
        queue_models = discover_deck_models(model_dir="models/queue")
        if queue_models:
            chosen = random.choice(queue_models)["path"]
            name = os.path.basename(chosen)[:-4]
            print(f"Queue: Introducing {name} to the arena! (Currently {challengers_count} challengers active)")
            shutil.move(chosen, os.path.join("models", os.path.basename(chosen)))
    elif challengers_count > 3:
        import time
        now = time.time()
        
        def is_recently_modified(name):
            try:
                mtime = os.path.getmtime(os.path.join("models", name + ".zip"))
                return (now - mtime) < 7200 # 2 hours
            except:
                return False

        # Protect models that are likely actively training
        pushable_challengers = [c for c in challengers if not is_recently_modified(c)]
        
        # Sort by games played
        pushable_challengers.sort(key=lambda m: games.get(m, 0))
        
        excess = challengers_count - 3
        to_push = pushable_challengers[:excess]
        os.makedirs("models/queue", exist_ok=True)
        
        for name in to_push:
            for m in discover_deck_models(model_dir="models"):
                if m["name"] == name:
                    print(f"Queue: Pushing {name} back to queue to maintain limit of 3!")
                    shutil.move(m["path"], os.path.join("models/queue", os.path.basename(m["path"])))
                    break

def run_continuous_match():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    manage_queue()

    active = get_active_models()
    if len(active) < 2:
        print("Not enough active models to run a match.")
        time.sleep(5)
        return
        
    games = get_games_played()
    active_games = {m: games.get(m, 0) for m in active}
    min_games = min(active_games.values())
    lowest_models = [m for m in active if active_games[m] == min_games]
    
    id_a = random.choice(lowest_models)
    remaining = [m for m in active if m != id_a]
    id_b = random.choice(remaining)
    
    from src.league.model_paths import parse_deck_model_path
    parsed_a = parse_deck_model_path(f"models/{id_a}.zip")
    parsed_b = parse_deck_model_path(f"models/{id_b}.zip")
    deck_num_a = parsed_a["deck_id"] if parsed_a else "0"
    deck_num_b = parsed_b["deck_id"] if parsed_b else "0"
    
    csv_a = f"decks/deck_bank/{deck_num_a}.csv" if str(deck_num_a).startswith("bank_") else f"decks/deck_{deck_num_a}.csv"
    csv_b = f"decks/deck_bank/{deck_num_b}.csv" if str(deck_num_b).startswith("bank_") else f"decks/deck_{deck_num_b}.csv"
    zip_a = f"models/{id_a}.zip"
    zip_b = f"models/{id_b}.zip"
    
    if not zip_a or not zip_b:
        return
        
    name_a = get_deck_name(csv_a)
    name_b = get_deck_name(csv_b)
    
    # Optional live update status text
    try:
        with open("decks/status.json", "w") as f:
            json.dump({
                "status": "online",
                "progress": 0,
                "current_action": f"⚔️ Arena: {name_a} vs {name_b}"
            }, f)
    except: pass
    
    print(f"Match: {id_a} vs {id_b}")

    # Publish the roster immediately so the dashboard is useful while the first
    # (potentially slow) evaluation is still running.
    empty_pairwise = {model_id: {} for model_id in active}
    for data_file in (
        "decks/pairwise_winrates.json",
        "decks/current_generation_winrates.json",
    ):
        if not os.path.exists(data_file):
            with open(data_file, "w") as f:
                json.dump(empty_pairwise, f)
    update_live_leaderboard({}, 0, 1, f"Arena: {name_a} vs {name_b}")

    try:
        result = subprocess.run(
            [sys.executable, "src/arena/evaluate_single.py", zip_a, csv_a, zip_b, csv_b, str(NUM_MATCH_GAMES)],
            capture_output=True,
            text=True,
            timeout=MATCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        message = f"Match timeout after {MATCH_TIMEOUT_SECONDS}s: {id_a} vs {id_b}"
        print(message)
        update_live_leaderboard({}, 0, 1, message)
        time.sleep(1)
        return
    
    wins_a = 0
    wins_b = 0
    pw_a = 0
    dw_a = 0
    bw_a = 0
    pw_b = 0
    dw_b = 0
    bw_b = 0
    if "Error" in result.stderr or "Evaluation crashed!" in result.stdout or "CHILD ERROR" in result.stdout:
        print("Evaluation crashed! Counting as 0 wins.")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
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
    
    games_played_dict = get_games_played()
    gp_a = games_played_dict.get(id_a, 0)
    gp_b = games_played_dict.get(id_b, 0)
    
    def get_k_factor(games):
        if games < 50: return 32
        if games < 150: return 24
        return 16
        
    K_a = get_k_factor(gp_a)
    K_b = get_k_factor(gp_b)
    
    new_elo_a = elo_a + K_a * (score_a_total - expected_a * NUM_MATCH_GAMES)
    new_elo_b = elo_b + K_b * (score_b_total - expected_b * NUM_MATCH_GAMES)
    
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
        try:
            with open(pairwise_file, "r") as f:
                pairwise = json.load(f)
        except:
            pairwise = {}
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
        try:
            with open(current_file, "r") as f:
                current_pw = json.load(f)
        except:
            current_pw = {}
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
    scores = {d: 0 for d in active}
    total_games = {d: 0 for d in active}
    for da in current_pw:
        if da not in scores:
            continue
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
        res_str = f"**{id_a}** vs {id_b} -> **{wins_a}:{wins_b}**"
    elif wins_b > wins_a:
        res_str = f"{id_a} vs **{id_b}** -> **{wins_b}:{wins_a}**"
    else:
        res_str = f"{id_a} vs {id_b} -> Unentschieden ({wins_a}:{wins_b})"
        
    # You can keep track of recent match results by appending it to a file
    # For now we just print it
    print(res_str)
    
    # We update the artifact (which isn't strictly necessary with the web UI, but keeps compatibility)
    update_live_leaderboard(final_winrates, 1, 1, "Fortlaufende Arena", [res_str])
    
    # Small pause
    time.sleep(1)
    
    maybe_run_auto_filter(15, "continuous arena")
    


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
    model_name = resolve_deck_model_base(best_deck_id)
    opp_model_path = resolve_deck_model_path(opp_id)
    
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
    if opp_model_path:
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
        
        os.makedirs("decks/ghost_pool", exist_ok=True)
        os.makedirs("models/ghost_pool", exist_ok=True)
        
        ghost_csv = f"decks/ghost_pool/deck_{deck_id}.csv"
        
        if os.path.exists(csv_path): 
            os.rename(csv_path, ghost_csv)
        for zip_path in list(iter_existing_deck_model_paths(deck_id)):
            ghost_zip = os.path.join("models/ghost_pool", os.path.basename(zip_path))
            os.rename(zip_path, ghost_zip)
