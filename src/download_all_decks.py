import os
import sys
import urllib.request
import re
import ssl

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from src.cg.api import all_card_data
    from src.auto_tourney import MASTER_URLS
except ImportError:
    print("Could not import required modules.")
    sys.exit(1)

def fetch_and_save(url, bank_id):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    print(f"Scraping {url} -> bank_{bank_id}.csv ...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req, context=ctx).read().decode('utf-8')
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return False
    
    pattern = re.compile(r'<span class="card-count">(\d+)</span>\s*<span class="card-name">([^<]+)</span>')
    matches = pattern.findall(html)
    
    if not matches:
        print(f"No cards found for {url}")
        return False
        
    cards = all_card_data()
    name_map = {c.name.lower(): c.cardId for c in cards}
    
    name_map["psychic energy"] = 5
    name_map["basic {p} energy"] = 5
    name_map["water energy"] = 3
    name_map["basic {w} energy"] = 3
    name_map["fire energy"] = 2
    name_map["basic {r} energy"] = 2
    name_map["lightning energy"] = 4
    name_map["basic {l} energy"] = 4
    name_map["fighting energy"] = 6
    name_map["basic {f} energy"] = 6
    name_map["darkness energy"] = 7
    name_map["basic {d} energy"] = 7
    name_map["metal energy"] = 8
    name_map["basic {m} energy"] = 8
    name_map["grass energy"] = 1
    name_map["basic {g} energy"] = 1
    name_map["telepathic psychic energy"] = 19
    
    for c in cards:
        if "’" in c.name:
            name_map[c.name.replace("’", "'").lower()] = c.cardId
    
    deck_list = []
    for count_str, name in matches:
        count = int(count_str)
        name_lower = name.lower().replace("&#039;", "'").replace("&amp;", "&")
        
        card_id = name_map.get(name_lower)
        if card_id is None:
            for c in cards:
                if name_lower in c.name.lower() or c.name.lower() in name_lower:
                    card_id = c.cardId
                    break
                    
        if card_id is None:
            print(f"  [!] Missing: {name} - Deck rejected.")
            return False
            
        for _ in range(count):
            deck_list.append(card_id)
            
    if len(deck_list) != 60:
        print(f"  [!] Invalid card count ({len(deck_list)} instead of 60) - Deck rejected.")
        return False
    
    os.makedirs("decks/deck_bank", exist_ok=True)
    with open(f"decks/deck_bank/bank_{bank_id}.csv", "w") as f:
        for cid in deck_list:
            f.write(f"{cid}\n")
            
    print(f"Successfully saved bank_{bank_id}.csv")
    return True

def main():
    success_count = 0
    for i, url in enumerate(MASTER_URLS, 1):
        if fetch_and_save(url, i):
            success_count += 1
            
    print(f"\nDone! Downloaded {success_count}/{len(MASTER_URLS)} decks into decks/deck_bank/")

if __name__ == "__main__":
    main()
