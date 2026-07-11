import glob
import csv

card_dict = {}
with open('pokemon-tcg-ai-battle/EN_Card_Data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            card_id = int(row[0])
            name = row[1]
            card_type = row[4] # "Basic Pokémon", "Trainer", etc
            card_dict[card_id] = {'name': name, 'type': card_type}
        except:
            pass

decks = glob.glob('decks/deck_bank/bank_*.csv')
deck_names = {}
for deck_file in decks:
    deck_id = deck_file.split('_')[-1].split('.')[0]
    
    with open(deck_file, 'r') as f:
        lines = f.readlines()
        
    counts = {}
    for line in lines:
        try:
            cid = int(line.strip())
            if cid in card_dict and "Pokémon" in card_dict[cid]['type']:
                counts[card_dict[cid]['name']] = counts.get(card_dict[cid]['name'], 0) + 1
        except:
            pass
            
    # sort by count descending
    sorted_pkmn = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    if sorted_pkmn:
        deck_names[deck_id] = sorted_pkmn[0][0] + (" / " + sorted_pkmn[1][0] if len(sorted_pkmn) > 1 else "")
    else:
        deck_names[deck_id] = "Unknown"

# print top 20
for k in sorted(deck_names.keys(), key=lambda x: int(x))[:30]:
    print(f"Bank {k}: {deck_names[k]}")
