import os
import datetime
import requests
import matplotlib.pyplot as plt
import numpy as np

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Content-Type': 'application/json'
}
url = 'https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes'

subs_info = [
    {"id": 54834368, "desc": "V6 Compact Hydrapple bank_70"},
    {"id": 54815682, "desc": "Alakazam V6 compact PFSP 2M"},
    {"id": 54793285, "desc": "V6 Compact bank_54 scratch PFSP"},
    {"id": 54784364, "desc": "V6 Compact Alakazam 1M"},
    {"id": 54723890, "desc": "Base A | V6 energy scaling"}
]

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

for sub in subs_info:
    sub_id = sub["id"]
    print(f"Fetching episodes for {sub_id}...")
    r = requests.post(url, json={'submissionId': sub_id}, headers=headers)
    if r.status_code != 200:
        print(f"Failed to fetch {sub_id}: {r.status_code}")
        continue
    
    data = r.json()
    episodes = data.get('episodes', [])
    if not episodes:
        print(f"No episodes for {sub_id}")
        continue
    
    # Sort episodes by creation time ascending
    sorted_eps = []
    for ep in episodes:
        try:
            c_time = parse_time(ep['createTime'])
            sorted_eps.append((c_time, ep))
        except Exception as e:
            # Fallback to ID sorting if date parsing fails
            sorted_eps.append((ep['id'], ep))
            
    sorted_eps.sort(key=lambda x: x[0])
    
    # Extract Elo scores
    scores = []
    times = []
    wins = 0
    losses = 0
    draws = 0
    
    for idx, (t, ep) in enumerate(sorted_eps):
        # find our agent
        our_agent = None
        for agent in ep.get('agents', []):
            if agent.get('submissionId') == sub_id:
                our_agent = agent
                break
        
        if our_agent is None:
            continue
            
        # Record outcome
        reward = our_agent.get('reward', 0)
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
            
        # Initial score for first match
        if idx == 0 and 'initialScore' in our_agent:
            scores.append(our_agent['initialScore'])
            times.append(t - datetime.timedelta(minutes=5)) # slightly before first match
            
        if 'updatedScore' in our_agent:
            scores.append(our_agent['updatedScore'])
            times.append(t)
            
    all_data[sub_id] = {
        "desc": sub["desc"],
        "scores": scores,
        "times": times,
        "stats": {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total": wins + losses + draws
        }
    }

# 1. Plot individual charts
for sub_id, info in all_data.items():
    scores = info["scores"]
    times = info["times"]
    desc = info["desc"]
    stats = info["stats"]
    
    if not scores:
        continue
        
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor('#1a1a24')
    fig.patch.set_facecolor('#111115')
    ax.grid(color='#2d2d3a', linestyle='--', linewidth=0.5)
    
    # Plot line
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
    ax.set_title(f"Elo Verlauf: {desc} (ID: {sub_id})", color='white', fontsize=14, weight='bold', pad=15)
    ax.set_xlabel("Anzahl Spiele (Episodes)", color='#888899', fontsize=11, labelpad=8)
    ax.set_ylabel("Elo Rating", color='#888899', fontsize=11, labelpad=8)
    
    # Set y limits nicely
    ymin, ymax = min(scores), max(scores)
    yrange = ymax - ymin if ymax > ymin else 10
    ax.set_ylim(ymin - yrange * 0.15, ymax + yrange * 0.2)
    
    # Tick colors
    ax.tick_params(colors='#888899', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#2d2d3a')
        
    # Text box with stats
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
    print(f"Saved plot for {sub_id} to {img_path}")

# 2. Plot combined chart
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_facecolor('#1a1a24')
fig.patch.set_facecolor('#111115')
ax.grid(color='#2d2d3a', linestyle='--', linewidth=0.5)

colors = ['#00e5ff', '#ff4655', '#00e676', '#ffd600', '#d500f9']

for idx, (sub_id, info) in enumerate(all_data.items()):
    scores = info["scores"]
    desc = info["desc"]
    if not scores:
        continue
    games = list(range(len(scores)))
    ax.plot(games, scores, color=colors[idx % len(colors)], linewidth=2, label=f"Sub {sub_id}: {desc} ({int(scores[-1])} Elo)")

ax.set_title("Vergleich Elo-Entwicklung der letzten 5 Submissions", color='white', fontsize=15, weight='bold', pad=15)
ax.set_xlabel("Anzahl Spiele ab Einreichung", color='#888899', fontsize=11, labelpad=8)
ax.set_ylabel("Elo Rating", color='#888899', fontsize=11, labelpad=8)
ax.tick_params(colors='#888899', labelsize=9)

# Legend positioning and styling
legend = ax.legend(facecolor='#111115', edgecolor='#2d2d3a', loc='upper left')
for text in legend.get_texts():
    text.set_color('#a0a0b0')
    text.set_fontsize(9)

for spine in ax.spines.values():
    spine.set_color('#2d2d3a')

plt.tight_layout()
combined_path = os.path.join(output_dir, "kaggle_elos_combined.png")
plt.savefig(combined_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"Saved combined plot to {combined_path}")
