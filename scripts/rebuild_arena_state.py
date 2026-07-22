import os
import sys
import json
from pathlib import Path

# Adjust path to import from src
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.arena.arena_core import (
    discover_participants,
    read_json,
    atomic_write_json,
    rank_participants,
    ArenaStore,
    DEFAULT_ELO,
)
from src.arena.arena_match import _k_factor, load_holdout_results

def main():
    # 1. Load active participants
    participants = discover_participants()
    p_ids = {p.bot_id for p in participants}
    print(f"Loaded {len(p_ids)} active participants: {sorted(list(p_ids))}")

    # 2. Clean runtime bot display names
    bot_names_path = BASE_DIR / "arena_data" / "bot_names.json"
    bot_names = read_json(bot_names_path, {})
    cleaned_bot_names = {k: v for k, v in bot_names.items() if k in p_ids}
    print(f"Cleaned bot names: {len(bot_names)} -> {len(cleaned_bot_names)}")
    atomic_write_json(bot_names_path, cleaned_bot_names)

    # 3. Clean arena_data/bot_health.json
    bot_health_path = BASE_DIR / "arena_data" / "bot_health.json"
    bot_health = read_json(bot_health_path, {})
    cleaned_bot_health = {k: v for k, v in bot_health.items() if k in p_ids}
    print(f"Cleaned bot health entries: {len(bot_health)} -> {len(cleaned_bot_health)}")
    atomic_write_json(bot_health_path, cleaned_bot_health)

    # 4. Clean arena_data/matches.json and recalculate Elo
    matches_path = BASE_DIR / "arena_data" / "matches.json"
    matches = read_json(matches_path, [])
    filtered_matches = [m for m in matches if m.get("bot_a") in p_ids and m.get("bot_b") in p_ids]
    print(f"Filtered matches count: {len(matches)} -> {len(filtered_matches)}")

    # Recalculate Elos from scratch for the remaining matches
    elos = {p_id: DEFAULT_ELO for p_id in p_ids}
    games = {p_id: 0 for p_id in p_ids}

    recalculated_matches = []
    for m in filtered_matches:
        if m.get("error_status"):
            recalculated_matches.append(m)
            continue
        a, b = m["bot_a"], m["bot_b"]
        wins_a, wins_b, draws = int(m["wins_a"]), int(m["wins_b"]), int(m["draws"])
        total = wins_a + wins_b + draws
        
        elo_a_before = elos[a]
        elo_b_before = elos[b]
        
        expected_a = 1.0 / (1.0 + 10 ** ((elo_b_before - elo_a_before) / 400.0))
        score_a = wins_a + 0.5 * draws
        
        # Calculate delta based on games played before this match
        delta_a = _k_factor(games[a]) * (score_a - expected_a * total)
        
        elo_a_after = elo_a_before + delta_a
        elo_b_after = elo_b_before - delta_a
        
        elos[a] = elo_a_after
        elos[b] = elo_b_after
        games[a] += total
        games[b] += total
        
        new_m = dict(m)
        new_m["elo_a_before"] = elo_a_before
        new_m["elo_b_before"] = elo_b_before
        new_m["elo_a_after"] = elo_a_after
        new_m["elo_b_after"] = elo_b_after
        recalculated_matches.append(new_m)

    atomic_write_json(matches_path, recalculated_matches)
    print(f"Saved {len(recalculated_matches)} recalculated matches.")

    # 5. Clean arena_data/evaluations.json
    evals_path = BASE_DIR / "arena_data" / "evaluations.json"
    evals = read_json(evals_path, [])
    cleaned_evals = []
    for ev in evals:
        b_id = ev.get("bot_id")
        ids = [i.strip() for i in str(b_id).split(",") if i.strip()]
        tracked_ids = [i for i in ids if i in p_ids]
        if tracked_ids:
            ev["bot_id"] = ",".join(tracked_ids)
            if "results" in ev and isinstance(ev["results"], dict):
                res = ev["results"]
                if "summary" in res and isinstance(res["summary"], list):
                    res["summary"] = [row for row in res["summary"] if row.get("candidate") in p_ids]
                if "matches" in res and isinstance(res["matches"], list):
                    res["matches"] = [row for row in res["matches"] if row.get("candidate") in p_ids]
            if "summary" in ev and isinstance(ev["summary"], list):
                ev["summary"] = [row for row in ev["summary"] if row.get("candidate") in p_ids]
            cleaned_evals.append(ev)

    print(f"Cleaned evaluations list: {len(evals)} -> {len(cleaned_evals)}")
    atomic_write_json(evals_path, cleaned_evals)

    # 6. Clean arena_data/evaluation.json (active state)
    eval_path = BASE_DIR / "arena_data" / "evaluation.json"
    eval_state = read_json(eval_path, {})
    if "bot_id" in eval_state:
        ids = [i.strip() for i in str(eval_state["bot_id"]).split(",") if i.strip()]
        tracked_ids = [i for i in ids if i in p_ids]
        if tracked_ids:
            eval_state["bot_id"] = ",".join(tracked_ids)
        else:
            eval_state = {"state": "idle"}
        atomic_write_json(eval_path, eval_state)
        print("Cleaned arena_data/evaluation.json")

    # 7. Re-generate leaderboard.json
    store = ArenaStore()
    board = rank_participants(participants, recalculated_matches, load_holdout_results())
    store.save_leaderboard(board)
    print("Saved clean leaderboard.json.")

    # 8. Re-generate the optional Markdown leaderboard artifact
    md = "---\n"
    md += "requestFeedback: false\n"
    md += "summary: Live Leaderboard\n"
    md += "userFacing: true\n"
    md += "---\n\n"
    md += "# 🏆 Live Evolution Tournament Leaderboard\n\n"
    md += f"**Status:** Arena: Fortlaufende Arena (recalculated/cleaned)\n\n"
    md += "| Platz | Deck | Elo | Win-Rate | Gespielte Matches |\n"
    md += f"| --- | --- | --- | --- | --- |\n"

    for row in board:
        rank = row["rank"]
        bot_id = row["bot_id"]
        display_name = row["display_name"]
        elo_val = row["elo"]
        wr = row["arena_winrate"] * 100
        gp = row["matches"]
        md += f"| **{rank}** | {bot_id} ({display_name}) | **{int(elo_val)}** | {wr:.1f}% | {gp} |\n"

    with open(BASE_DIR / "arena_data" / "leaderboard.md", "w") as f:
        f.write(md)
    print("Wrote updated arena_data/leaderboard.md.")

if __name__ == "__main__":
    main()
