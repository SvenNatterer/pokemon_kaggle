import os
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

subs = api.competition_submissions("pokemon-tcg-ai-battle")

# filter successful ones
valid_subs = []
for s in subs:
    score_str = getattr(s, "public_score", None)
    if score_str is not None:
        try:
            score = float(score_str)
            date = s.date
            desc = getattr(s, "description", "") or f"Sub {s.ref}"
            valid_subs.append({
                "ref": s.ref,
                "date": date,
                "score": score,
                "desc": desc
            })
        except (TypeError, ValueError):
            continue

# sort by date ascending
valid_subs.sort(key=lambda x: x["date"])

# get last 5
last_5 = valid_subs[-5:]

print(f"Found {len(last_5)} recent completed submissions:")
for sub in last_5:
    print(f"Ref: {sub['ref']}, Date: {sub['date']}, Score: {sub['score']}, Desc: {sub['desc']}")

if not last_5:
    print("No completed submissions found.")
    exit(0)

# Now plot
fig, ax = plt.subplots(figsize=(10, 6))

# Set facecolors
ax.set_facecolor('#1a1a24')
fig.patch.set_facecolor('#111115')
ax.grid(color='#2d2d3a', linestyle='--', linewidth=0.5)

# Extract data
dates = [s["date"].strftime("%d.%m\n%H:%M") for s in last_5]
scores = [s["score"] for s in last_5]

# Plot line and scatter
line, = ax.plot(dates, scores, marker='o', color='#ff4655', linewidth=3, markersize=10, label='Kaggle Elo')

# Annotate points with scores
min_score = min(scores)
max_score = max(scores)
score_range = max_score - min_score if max_score > min_score else 10
offset = score_range * 0.08

for i, score in enumerate(scores):
    ax.annotate(f"{int(score)}", (dates[i], score + offset), color='white', weight='bold', fontsize=12, ha='center')

# Title and labels
ax.set_title("Kaggle Elo - Letzte 5 Submissions", color='white', fontsize=16, weight='bold', pad=20)
ax.set_xlabel("Datum", color='#888899', fontsize=12, labelpad=10)
ax.set_ylabel("Elo Rating", color='#888899', fontsize=12, labelpad=10)

# Set reasonable y-limits
ax.set_ylim(min_score - score_range * 0.2, max_score + score_range * 0.3)

# Tick colors
ax.tick_params(colors='#888899', labelsize=10)
for spine in ax.spines.values():
    spine.set_color('#2d2d3a')

# Add descriptions as text boxes
for i, s in enumerate(last_5):
    # Short description
    desc_text = f"Sub {s['ref']} ({s['date'].strftime('%d.%m')}): {s['desc'][:45]}"
    if len(s['desc']) > 45:
        desc_text += "..."
    ax.text(0.02, 0.94 - i*0.06, desc_text, transform=ax.transAxes, color='#a0a0b0', fontsize=9.5, 
            bbox=dict(facecolor='#111115', alpha=0.8, edgecolor='#2d2d3a', boxstyle='round,pad=0.3'))

plt.tight_layout()

# Save to the artifacts folder
output_dir = "/Users/svennatterer/.gemini/antigravity-ide/brain/7c6b5bef-3989-4c0d-9502-8347cdce6d2a"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "kaggle_elos.png")
plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
print(f"Saved plot to {output_path}")
