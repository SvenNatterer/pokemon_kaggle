import json
import os
import glob
import statistics

replay_dir = "replays"
files = glob.glob(os.path.join(replay_dir, "**", "*.json"), recursive=True)

failures = []

reasons = {
    1: "Opponent took all Prize cards",
    2: "0 deck cards (Deck Out)",
    3: "No Pokemon in Active Spot (Benched Out)",
    4: "Card Effect"
}

for f in files:
    try:
        with open(f, "r") as fh:
            data = json.load(fh)
        if len(data) == 0: continue
            
        last_step = data[-1]
        players = data[0].get("metadata", {})
        
        p0_name = players.get("p0_name", "Unknown0")
        p1_name = players.get("p1_name", "Unknown1")
        
        final_logs = last_step.get("logs", [])
        
        winner = None
        reason = None
        for log in final_logs:
            if log.get("type") == "Result":
                res = log.get("result")
                rea = log.get("reason")
                if res == 0:
                    winner = p0_name
                    loser = p1_name
                elif res == 1:
                    winner = p1_name
                    loser = p0_name
                else:
                    winner = "Draw"
                    loser = "Draw"
                reason = reasons.get(rea, str(rea))
                
        if winner and winner != "Draw":
            failures.append({
                "loser": loser,
                "winner": winner,
                "reason": reason,
                "steps": len(data)
            })
    except Exception as e:
        pass

# Group by loser and reason
reason_stats = {}
for fail in failures:
    loser = fail["loser"]
    r = fail["reason"]
    steps = fail["steps"]
    
    if loser not in reason_stats: reason_stats[loser] = {}
    if r not in reason_stats[loser]: reason_stats[loser][r] = []
    
    reason_stats[loser][r].append(steps)

for loser, stats in reason_stats.items():
    print(f"\n{loser} losses:")
    for r, step_list in stats.items():
        avg_steps = statistics.mean(step_list)
        print(f"  {r}: {len(step_list)} times (Avg steps: {avg_steps:.1f})")

