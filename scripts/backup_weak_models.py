import json
import os
import shutil

THRESHOLD = 1170
BACKUP_DIR = "models/backup"
ELO_FILE = "decks/elo_ratings.json"

os.makedirs(BACKUP_DIR, exist_ok=True)

with open(ELO_FILE, "r") as f:
    elos = json.load(f)

to_remove = [k for k, v in elos.items() if v < THRESHOLD]

moved_files = []
for model_name in to_remove:
    # Look for the zip file
    zip_path = f"models/{model_name}.zip"
    if os.path.exists(zip_path):
        dest_path = os.path.join(BACKUP_DIR, f"{model_name}.zip")
        shutil.move(zip_path, dest_path)
        moved_files.append(zip_path)
    
    # Also remove from elo ratings
    del elos[model_name]

# Save updated elos
with open(ELO_FILE, "w") as f:
    json.dump(elos, f, indent=4)

print(f"Moved {len(moved_files)} weak models to {BACKUP_DIR}:")
for p in moved_files:
    print(f"  - {p}")
