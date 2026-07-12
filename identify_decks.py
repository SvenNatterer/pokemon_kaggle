import glob
import csv
import json
import os

card_dict = {}
with open('pokemon-tcg-ai-battle/EN_Card_Data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            card_id = int(row[0])
            name = row[1]
            card_type = row[4] # "Basic Pokémon", "Trainer", etc
            hp = row[8]
            card_dict[card_id] = {'name': name, 'type': card_type, 'hp': hp}
        except:
            pass

def pokemon_strength(card):
    """Prefer evolved, high-HP attackers over their setup Pokemon."""
    card_type = card['type']
    stage_bonus = 0
    if card_type.startswith("Stage 2"):
        stage_bonus = 200
    elif card_type.startswith("Stage 1"):
        stage_bonus = 100
    try:
        hp = int(card['hp'])
    except (TypeError, ValueError):
        hp = 0
    return stage_bonus + hp


def normalize_deck_name(name):
    if name == "Hydrapple ex":
        return "Ogerpon"
    return name


decks = glob.glob('decks/deck_*.csv') + glob.glob('decks/deck_bank/bank_*.csv')
deck_names = {}
for deck_file in decks:
    stem = os.path.splitext(os.path.basename(deck_file))[0]
    deck_id = stem[5:] if stem.startswith('deck_') else stem
    
    with open(deck_file, 'r') as f:
        lines = f.readlines()
        
    counts = {}
    for line in lines:
        try:
            cid = int(line.strip())
            if cid in card_dict and "Pokémon" in card_dict[cid]['type']:
                card = card_dict[cid]
                # Tools contain "Pokémon" in their type too, but are not attackers.
                if card['type'].endswith('Pokémon'):
                    entry = counts.setdefault(card['name'], {'count': 0, 'card': card})
                    entry['count'] += 1
        except:
            pass

    sorted_pkmn = sorted(
        counts.items(),
        key=lambda item: (pokemon_strength(item[1]['card']), item[1]['count']),
        reverse=True,
    )
    if sorted_pkmn:
        deck_names[deck_id] = normalize_deck_name(sorted_pkmn[0][0])
    else:
        deck_names[deck_id] = "Unknown"

# Training historically looks up bank decks by their numeric suffix only.
for deck_id, name in list(deck_names.items()):
    if deck_id.startswith('bank_'):
        deck_names.setdefault(deck_id[5:], name)

with open('decks/deck_names.json', 'w', encoding='utf-8') as f:
    json.dump(deck_names, f, indent=2, ensure_ascii=False, sort_keys=True)
    f.write('\n')

for deck_id, name in sorted(deck_names.items()):
    print(f"{deck_id}: {name}")
