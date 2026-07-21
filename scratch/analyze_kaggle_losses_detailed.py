import os
import glob
import json
import csv
from collections import Counter
import statistics

# Load card data for archetype inference
CARD_DATA_PATH = "pokemon-tcg-ai-battle/EN_Card_Data.csv"
SUBMISSIONS_METADATA_PATH = "replays/kaggle/submissions.json"

class CardInfo:
    def __init__(self, card_id, name, stage, hp, kind):
        self.card_id = card_id
        self.name = name
        self.stage = stage
        self.hp = hp
        self.kind = kind

def load_card_data(path):
    cards = {}
    if not os.path.exists(path):
        print(f"Warning: card data not found at {path}")
        return cards
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                card_id = int(row["Card ID"])
            except:
                continue
            stage = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
            try:
                hp = int(row.get("HP", ""))
            except:
                hp = 0
            
            # Determine kind
            lowered = stage.casefold()
            if "energy" in lowered:
                kind = "Energy"
            elif lowered.endswith("pokémon") or lowered.endswith("pokemon"):
                kind = "Pokémon"
            else:
                kind = "Trainer"
                
            cards[card_id] = CardInfo(card_id, row.get("Card Name", ""), stage, hp, kind)
    return cards

def pokemon_strength(card):
    stage = card.stage.casefold()
    if stage.startswith("stage 2"):
        stage_bonus = 200
    elif stage.startswith("stage 1"):
        stage_bonus = 100
    else:
        stage_bonus = 0
    return stage_bonus + card.hp

def infer_archetype(cards_counter, card_data):
    candidates = []
    for card_id, count in cards_counter.items():
        card = card_data.get(card_id)
        if card and card.kind == "Pokémon":
            candidates.append((card, count))
    if not candidates:
        return "Unknown"
    card, _count = max(
        candidates,
        key=lambda item: (
            pokemon_strength(item[0]),
            item[1],
            " ex" in item[0].name.casefold(),
            item[0].name.casefold(),
        ),
    )
    return card.name

def iter_card_instances(value, player_index):
    if isinstance(value, dict):
        if {"id", "serial", "playerIndex"}.issubset(value):
            try:
                if int(value["playerIndex"]) == player_index:
                    yield int(value["serial"]), int(value["id"])
            except:
                pass
        if {"cardId", "serial", "playerIndex"}.issubset(value):
            try:
                if int(value["playerIndex"]) == player_index:
                    yield int(value["serial"]), int(value["cardId"])
            except:
                pass
        for child in value.values():
            yield from iter_card_instances(child, player_index)
    elif isinstance(value, list):
        for child in value:
            yield from iter_card_instances(child, player_index)

def reconstruct_cards(steps, player_index):
    by_serial = {}
    if not isinstance(steps, list):
        return Counter()
    for step in steps:
        if not isinstance(step, list):
            continue
        for agent_state in step:
            if not isinstance(agent_state, dict):
                continue
            for serial, card_id in iter_card_instances(agent_state, player_index):
                by_serial[serial] = card_id
    return Counter(by_serial.values())

def load_submissions_metadata():
    if os.path.exists(SUBMISSIONS_METADATA_PATH):
        try:
            with open(SUBMISSIONS_METADATA_PATH, "r") as f:
                return json.load(f).get("submissions", {})
        except:
            pass
    return {}

def main():
    card_data = load_card_data(CARD_DATA_PATH)
    subs_metadata = load_submissions_metadata()
    
    # We want to analyze directories in replays/kaggle
    kaggle_dir = "replays/kaggle"
    sub_dirs = sorted([d for d in os.listdir(kaggle_dir) if d.isdigit() and os.path.isdir(os.path.join(kaggle_dir, d))], key=int)
    
    overall_losses = []
    submission_stats = {}
    
    print("Analyzing Kaggle replays...")
    
    for sub_id in sub_dirs:
        sub_path = os.path.join(kaggle_dir, sub_id)
        replay_files = glob.glob(os.path.join(sub_path, "*-replay.json"))
        if not replay_files:
            continue
            
        desc = subs_metadata.get(sub_id, {}).get("description", "Unknown")
        status = subs_metadata.get(sub_id, {}).get("status", "Unknown")
        
        # Only analyze completed submissions
        if status != "COMPLETE" and sub_id not in ["54723890", "54784364", "54793285"]:
            # Some folders might not have complete status in submissions.json but have replays
            if len(replay_files) < 5:
                continue
        
        wins = 0
        losses = 0
        draws = 0
        
        sub_losses = []
        
        for f in replay_files:
            try:
                with open(f, "r") as fh:
                    data = json.load(fh)
                if not data or "steps" not in data or len(data["steps"]) == 0:
                    continue
                    
                info = data.get("info", {})
                names = info.get("TeamNames") or [a.get("Name") for a in info.get("Agents", [])]
                if not names or len(names) != 2:
                    continue
                
                # Identify own index ("Sven Natterer")
                own_idx = 0
                if names[1] == "Sven Natterer":
                    own_idx = 1
                elif names[0] != "Sven Natterer":
                    # If both are different, look for Sven Natterer in names
                    if "Sven Natterer" in names:
                        own_idx = names.index("Sven Natterer")
                    else:
                        own_idx = 0  # Fallback
                
                opp_idx = 1 - own_idx
                opp_name = names[opp_idx]
                
                rewards = data.get("rewards")
                if not rewards or len(rewards) != 2:
                    continue
                    
                own_reward = rewards[own_idx]
                opp_reward = rewards[opp_idx]
                
                # Determine outcome
                outcome = "unknown"
                if own_reward > opp_reward:
                    outcome = "win"
                    wins += 1
                elif own_reward < opp_reward:
                    outcome = "loss"
                    losses += 1
                else:
                    outcome = "draw"
                    draws += 1
                
                # Reconstruct opponent cards for archetype inference
                opp_cards = reconstruct_cards(data.get("steps"), opp_idx)
                opp_archetype = infer_archetype(opp_cards, card_data)
                
                if outcome == "loss":
                    # Infer loss reason from the final step game state
                    # Find the last step that has current state
                    final_obs = None
                    for step in reversed(data["steps"]):
                        if isinstance(step, list) and len(step) > own_idx:
                            agent_state = step[own_idx]
                            if agent_state and isinstance(agent_state, dict):
                                obs = agent_state.get("observation")
                                if obs and isinstance(obs, dict) and "current" in obs:
                                    final_obs = obs["current"]
                                    break
                    
                    reason = "Unknown"
                    own_deck_len = -1
                    opp_prize_len = -1
                    own_active_len = -1
                    own_bench_len = -1
                    
                    if final_obs and "players" in final_obs and len(final_obs["players"]) == 2:
                        own_state = final_obs["players"][own_idx]
                        opp_state = final_obs["players"][opp_idx]
                        
                        own_deck_len = own_state.get("deckCount", -1)
                        opp_prize_len = len(opp_state.get("prize", []))
                        own_active_len = len(own_state.get("active", [])) if own_state.get("active") is not None else 0
                        own_bench_len = len(own_state.get("bench", [])) if own_state.get("bench") is not None else 0
                        
                        # Classification logic
                        if own_deck_len == 0:
                            reason = "Deck Out"
                        elif opp_prize_len == 0:
                            reason = "Prize KO"
                        elif own_active_len == 0 and own_bench_len == 0:
                            reason = "Benched Out"
                        else:
                            # Let us check prize and deck again. If prize has None but we lost,
                            # it could be prize KO if they took their last prize card in the transition.
                            # But wait, in the final step, is it possible that opponent prize is 0?
                            # Sometimes the game terminates and the prize array is empty.
                            # If own_active_len is 0 but own_bench_len > 0: it means we benched out because
                            # we had no active pokemon and had to promote one, but couldn't or game ended.
                            # Let us classify based on deck size first, then prize cards, then active
                            if own_deck_len == 0:
                                reason = "Deck Out"
                            elif own_active_len == 0 and own_bench_len == 0:
                                reason = "Benched Out"
                            else:
                                reason = "Prize KO" # Fallback if we lost but deck > 0 and board exists
                                
                    loss_info = {
                        "submission_id": sub_id,
                        "description": desc,
                        "episode_id": data.get("id") or f.split("-")[1],
                        "opponent": opp_name,
                        "opponent_archetype": opp_archetype,
                        "reason": reason,
                        "turns": len(data["steps"]),
                        "own_deck_len": own_deck_len,
                        "opp_prize_len": opp_prize_len,
                        "own_active_len": own_active_len,
                        "own_bench_len": own_bench_len
                    }
                    sub_losses.append(loss_info)
                    overall_losses.append(loss_info)
            except Exception as e:
                # print(f"Error parsing {f}: {e}")
                pass
                
        total_games = wins + losses + draws
        win_rate = wins / total_games if total_games > 0 else 0
        submission_stats[sub_id] = {
            "description": desc,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total_games": total_games,
            "win_rate": win_rate,
            "losses_details": sub_losses
        }
        
    # Print high level summary
    print("\n============================================================")
    print("SUBMISSION PERFORMANCE SUMMARY")
    print("============================================================")
    for sub_id, stats in sorted(submission_stats.items(), key=lambda x: int(x[0])):
        if stats["total_games"] == 0: continue
        print(f"Sub ID {sub_id:<10} | Games: {stats['total_games']:>3} | Win Rate: {stats['win_rate']:>6.1%} (W: {stats['wins']}, L: {stats['losses']}, D: {stats['draws']}) | {stats['description'][:50]}")

    # Now focus on submissions that are newer and stronger (better than 450 points, meaning newer submissions)
    # The most advanced ones are: 54723890 (Base A), 54784364 (V6 Compact Alakazam), 54793285 (V6 Compact PFSP)
    high_tier_subs = ["54723890", "54784364", "54793285"]
    
    print("\n============================================================")
    print("LOSS REASON ANALYSIS FOR HIGH-TIER SUBMISSIONS (>450 Elo)")
    print("Submissions analyzed: 54723890, 54784364, 54793285")
    print("============================================================")
    
    high_tier_losses = [l for l in overall_losses if l["submission_id"] in high_tier_subs]
    
    if not high_tier_losses:
        print("No losses found in high-tier submissions!")
    else:
        reasons_counter = Counter([l["reason"] for l in high_tier_losses])
        print(f"Total Losses: {len(high_tier_losses)}")
        for r, count in reasons_counter.most_common():
            pct = count / len(high_tier_losses)
            print(f"  - {r:<15}: {count:>3} ({pct:>5.1%})")
            
        print("\nBreakdown of Losses by Opponent Archetype:")
        opp_arch_counter = Counter([l["opponent_archetype"] for l in high_tier_losses])
        for arch, count in opp_arch_counter.most_common():
            pct = count / len(high_tier_losses)
            print(f"  - {arch:<25}: {count:>3} ({pct:>5.1%})")
            
        print("\nAverage Game Duration (Turns) for each Loss Reason:")
        durations_by_reason = {}
        for l in high_tier_losses:
            r = l["reason"]
            if r not in durations_by_reason:
                durations_by_reason[r] = []
            durations_by_reason[r].append(l["turns"])
            
        for r, turns in durations_by_reason.items():
            avg_t = statistics.mean(turns)
            min_t = min(turns)
            max_t = max(turns)
            print(f"  - {r:<15}: Avg {avg_t:>5.1f} turns (range: {min_t}-{max_t})")
            
        print("\nDetailed list of Losses against top opponent archetypes:")
        # Group by Opponent Archetype and Loss Reason
        groups = {}
        for l in high_tier_losses:
            key = (l["opponent_archetype"], l["reason"])
            groups[key] = groups.get(key, 0) + 1
            
        for (arch, r), count in sorted(groups.items(), key=lambda x: -x[1]):
            print(f"  - {arch:<25} lost via {r:<15}: {count} times")

    # Let's write the complete output to a markdown report artifact!
    write_markdown_report(submission_stats, high_tier_subs, card_data)

def write_markdown_report(submission_stats, high_tier_subs, card_data):
    report_path = "reports/kaggle_losses_analysis.md"
    lines = []
    lines.append("# Kaggle Replay Loss Analysis")
    lines.append("")
    lines.append("This report analyzes the downloaded Kaggle replays, focusing specifically on our advanced submissions (Elo > 450).")
    lines.append("")
    
    lines.append("## Submission Performance Overview")
    lines.append("")
    lines.append("| Submission ID | Description | Total Games | Win Rate | Wins | Losses | Draws |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for sub_id, stats in sorted(submission_stats.items(), key=lambda x: int(x[0])):
        if stats["total_games"] == 0: continue
        lines.append(f"| {sub_id} | {stats['description']} | {stats['total_games']} | {stats['win_rate']:.1%} | {stats['wins']} | {stats['losses']} | {stats['draws']} |")
    lines.append("")
    
    # Analyze high tier
    high_tier_losses = []
    for sub_id in high_tier_subs:
        if sub_id in submission_stats:
            high_tier_losses.extend(submission_stats[sub_id]["losses_details"])
            
    lines.append("## High-Tier Submissions (>450 Elo) Loss Analysis")
    lines.append(f"Analyzed submissions: {', '.join(high_tier_subs)}")
    lines.append("")
    
    if not high_tier_losses:
        lines.append("No losses recorded in these submissions.")
    else:
        lines.append("### Loss Reasons Distribution")
        lines.append("")
        reasons_counter = Counter([l["reason"] for l in high_tier_losses])
        lines.append("| Loss Reason | Count | Percentage |")
        lines.append("| --- | --- | --- |")
        for r, count in reasons_counter.most_common():
            pct = count / len(high_tier_losses)
            lines.append(f"| {r} | {count} | {pct:.1%} |")
        lines.append("")
        
        lines.append("### Losses by Opponent Archetype")
        lines.append("")
        opp_arch_counter = Counter([l["opponent_archetype"] for l in high_tier_losses])
        lines.append("| Opponent Archetype | Count | Percentage |")
        lines.append("| --- | --- | --- |")
        for arch, count in opp_arch_counter.most_common():
            pct = count / len(high_tier_losses)
            lines.append(f"| {arch} | {count} | {pct:.1%} |")
        lines.append("")
        
        lines.append("### Average Duration (Turns) by Loss Reason")
        lines.append("")
        durations_by_reason = {}
        for l in high_tier_losses:
            r = l["reason"]
            if r not in durations_by_reason:
                durations_by_reason[r] = []
            durations_by_reason[r].append(l["turns"])
            
        lines.append("| Loss Reason | Avg Turns | Range |")
        lines.append("| --- | --- | --- |")
        for r, turns in durations_by_reason.items():
            avg_t = statistics.mean(turns)
            lines.append(f"| {r} | {avg_t:.1f} | {min(turns)}-{max(turns)} |")
        lines.append("")
        
        lines.append("### Detailed Matchups & Loss Reasons")
        lines.append("")
        lines.append("| Opponent Archetype | Loss Reason | Count |")
        lines.append("| --- | --- | --- |")
        groups = {}
        for l in high_tier_losses:
            key = (l["opponent_archetype"], l["reason"])
            groups[key] = groups.get(key, 0) + 1
        for (arch, r), count in sorted(groups.items(), key=lambda x: -x[1]):
            lines.append(f"| {arch} | {r} | {count} |")
            
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nMarkdown report written to {report_path}")

if __name__ == "__main__":
    main()
