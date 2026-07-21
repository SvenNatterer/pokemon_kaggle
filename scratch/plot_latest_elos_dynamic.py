import os
import datetime
import requests
import matplotlib.pyplot as plt
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate with Kaggle API
api = KaggleApi()
api.authenticate()

# Fetch submissions for the competition
print("Fetching submissions for competition 'pokemon-tcg-ai-battle'...")
subs = api.competition_submissions("pokemon-tcg-ai-battle")

# Filter completed submissions with valid score
valid_subs = []
for s in subs:
    score_str = getattr(s, "public_score", None)
    status = getattr(s, "status", "")
    # Depending on SDK, status might be an enum or string
    status_str = str(status).upper()
    if score_str is not None and "COMPLETE" in status_str:
        try:
            score = float(score_str)
            date = s.date
            ref = s.ref
            desc = getattr(s, "description", "") or f"Sub {ref}"
            valid_subs.append({
                "ref": ref,
                "date": date,
                "score": score,
                "desc": desc
            })
        except (TypeError, ValueError):
            continue

# Sort by date ascending to get timeline correct, then pick the last 5
valid_subs.sort(key=lambda x: x["date"])
last_5 = valid_subs[-5:]

print(f"Found {len(last_5)} recent completed submissions:")
for sub in last_5:
    print(f"Ref: {sub['ref']}, Date: {sub['date']}, Final Elo: {sub['score']}, Desc: {sub['desc']}")

if not last_5:
    print("Error: No completed submissions found.")
    exit(1)

# API config for fetching episodes
url = 'https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Content-Type': 'application/json'
}

output_dir = "/Users/svennatterer/.gemini/antigravity-ide/brain/7c6b5bef-3989-4c0d-9502-8347cdce6d2a"
os.makedirs(output_dir, exist_ok=True)

def parse_time(t_str):
    t_str = t_str.replace('Z', '')
    if '.' in t_str:
        parts = t_str.split('.')
        fraction = parts[1][:6]  # truncate to microseconds
        t_str = parts[0] + '.' + fraction
    return datetime.datetime.fromisoformat(t_str)

all_data = {}

for sub in last_5:
    sub_id = sub["ref"]
    print(f"\nFetching episodes for submission {sub_id}...")
    r = requests.post(url, json={'submissionId': sub_id}, headers=headers)
    if r.status_code != 200:
        print(f"Failed to fetch {sub_id}: HTTP {r.status_code}")
        continue
    
    data = r.json()
    episodes = data.get('episodes', [])
    if not episodes:
        print(f"No episodes returned for {sub_id}")
        continue
    
    # Sort episodes by creation time
    sorted_eps = []
    for ep in episodes:
        try:
            c_time = parse_time(ep['createTime'])
            sorted_eps.append((c_time, ep))
        except Exception as e:
            sorted_eps.append((ep['id'], ep))
            
    sorted_eps.sort(key=lambda x: x[0])
    
    scores = []
    times = []
    wins = 0
    losses = 0
    draws = 0
    
    for idx, (t, ep) in enumerate(sorted_eps):
        our_agent = None
        for agent in ep.get('agents', []):
            if agent.get('submissionId') == sub_id:
                our_agent = agent
                break
        
        if our_agent is None:
            continue
            
        reward = our_agent.get('reward', 0)
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
            
        # Record initial score before the first match
        if idx == 0 and 'initialScore' in our_agent:
            scores.append(our_agent['initialScore'])
            if isinstance(t, datetime.datetime):
                times.append(t - datetime.timedelta(minutes=5))
            else:
                times.append(0)
            
        if 'updatedScore' in our_agent:
            scores.append(our_agent['updatedScore'])
            times.append(t)
            
    if scores:
        all_data[sub_id] = {
            "desc": sub["desc"],
            "date": sub["date"],
            "scores": scores,
            "times": times,
            "stats": {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "total": wins + losses + draws
            }
        }
        print(f"Loaded {len(scores)} Elo history steps for {sub_id}")

# 1. Plot Individual Charts
for sub_id, info in all_data.items():
    scores = info["scores"]
    times = info["times"]
    desc = info["desc"]
    stats = info["stats"]
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor('#1a1a24')
    fig.patch.set_facecolor('#111115')
    ax.grid(color='#2d2d3a', linestyle='--', linewidth=0.5)
    
    games = list(range(len(scores)))
    ax.plot(games, scores, color='#ff4655', linewidth=2.5, marker='o', markersize=4, label='Elo')
    
    # Annotate final score
    final_score = scores[-1]
    ax.annotate(f"Final: {final_score:.1f}", 
                xy=(games[-1], final_score), 
                xytext=(-15, 15),
                textcoords='offset points',
                color='white', weight='bold',
                arrowprops=dict(arrowstyle="->", color='#ff4655', connectionstyle="arc3,rad=.2"))
                
    # Annotate max score
    max_idx = np.argmax(scores)
    max_score = scores[max_idx]
    if max_idx != len(scores) - 1:
        ax.annotate(f"Max: {max_score:.1f}", 
                    xy=(max_idx, max_score), 
                    xytext=(0, 15),
                    textcoords='offset points',
                    color='#00e676', weight='bold', ha='center',
                    arrowprops=dict(arrowstyle="->", color='#00e676'))

    # Title & labels
    ax.set_title(f"Elo Verlauf: {desc[:50]}... (ID: {sub_id})", color='white', fontsize=12, weight='bold', pad=15)
    ax.set_xlabel("Anzahl Spiele (Episodes)", color='#888899', fontsize=11, labelpad=8)
    ax.set_ylabel("Elo Rating", color='#888899', fontsize=11, labelpad=8)
    
    ymin, ymax = min(scores), max(scores)
    yrange = ymax - ymin if ymax > ymin else 10
    ax.set_ylim(ymin - yrange * 0.15, ymax + yrange * 0.2)
    
    ax.tick_params(colors='#888899', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#2d2d3a')
        
    wr = (stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0
    stats_text = (
        f"Spiele gesamt: {stats['total']}\n"
        f"Siege: {stats['wins']} | Niederlagen: {stats['losses']}\n"
        f"Winrate: {wr:.1f}%\n"
        f"Max Elo: {ymax:.1f} | Min Elo: {ymin:.1f}"
    )
    ax.text(0.02, 0.05, stats_text, transform=ax.transAxes, color='#a0a0b0', fontsize=9.5,
            bbox=dict(facecolor='#111115', alpha=0.85, edgecolor='#2d2d3a', boxstyle='round,pad=0.4'))
            
    plt.tight_layout()
    img_path = os.path.join(output_dir, f"kaggle_elo_{sub_id}.png")
    plt.savefig(img_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Saved individual plot for {sub_id} to {img_path}")

# 2. Combined Plot by Game Count
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_facecolor('#1a1a24')
fig.patch.set_facecolor('#111115')
ax.grid(color='#2d2d3a', linestyle='--', linewidth=0.5)

colors = ['#00e5ff', '#ff4655', '#00e676', '#ffd600', '#d500f9']

for idx, (sub_id, info) in enumerate(all_data.items()):
    scores = info["scores"]
    desc = info["desc"]
    games = list(range(len(scores)))
    ax.plot(games, scores, color=colors[idx % len(colors)], linewidth=2, 
            label=f"Sub {sub_id}: {desc[:30]}... ({int(scores[-1])} Elo)")

ax.set_title("Vergleich Elo-Entwicklung der letzten 5 Submissions (nach Spielen)", color='white', fontsize=14, weight='bold', pad=15)
ax.set_xlabel("Anzahl Spiele ab Einreichung", color='#888899', fontsize=11, labelpad=8)
ax.set_ylabel("Elo Rating", color='#888899', fontsize=11, labelpad=8)
ax.tick_params(colors='#888899', labelsize=9)

legend = ax.legend(facecolor='#111115', edgecolor='#2d2d3a', loc='upper left')
for text in legend.get_texts():
    text.set_color('#a0a0b0')
    text.set_fontsize(9)

for spine in ax.spines.values():
    spine.set_color('#2d2d3a')

plt.tight_layout()
combined_games_path = os.path.join(output_dir, "kaggle_elos_combined_games.png")
plt.savefig(combined_games_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"Saved combined games plot to {combined_games_path}")

# 3. Combined Plot by Relative Time (Hours)
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_facecolor('#1a1a24')
fig.patch.set_facecolor('#111115')
ax.grid(color='#2d2d3a', linestyle='--', linewidth=0.5)

for idx, (sub_id, info) in enumerate(all_data.items()):
    scores = info["scores"]
    times = info["times"]
    desc = info["desc"]
    
    # Calculate relative hours since first match
    rel_hours = []
    if times and isinstance(times[0], datetime.datetime):
        t0 = times[0]
        for t in times:
            diff = t - t0
            rel_hours.append(diff.total_seconds() / 3600.0)
    else:
        rel_hours = list(range(len(scores)))
        
    ax.plot(rel_hours, scores, color=colors[idx % len(colors)], linewidth=2, 
            label=f"Sub {sub_id}: {desc[:30]}... ({int(scores[-1])} Elo)")

ax.set_title("Vergleich Elo-Entwicklung der letzten 5 Submissions (zeitlicher Verlauf)", color='white', fontsize=14, weight='bold', pad=15)
ax.set_xlabel("Zeit seit erster Episode (in Stunden)", color='#888899', fontsize=11, labelpad=8)
ax.set_ylabel("Elo Rating", color='#888899', fontsize=11, labelpad=8)
ax.tick_params(colors='#888899', labelsize=9)

legend = ax.legend(facecolor='#111115', edgecolor='#2d2d3a', loc='upper left')
for text in legend.get_texts():
    text.set_color('#a0a0b0')
    text.set_fontsize(9)

for spine in ax.spines.values():
    spine.set_color('#2d2d3a')

plt.tight_layout()
combined_time_path = os.path.join(output_dir, "kaggle_elos_combined_time.png")
plt.savefig(combined_time_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"Saved combined time plot to {combined_time_path}")
