import os
import sys

# Add src to pythonpath so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cg.api import all_card_data

def main():
    cards = all_card_data()
    print(f"Total cards in database: {len(cards)}")
    
    decklist = [
        ("Mega Kangaskhan ex", 4),
        ("Meowth ex", 4),
        ("Lillie's Clefairy ex", 4),
        ("Latias ex", 3),
        ("Wellspring Mask Ogerpon ex", 2),
        ("Fezandipiti ex", 2),
        ("Moltres", 1),
        ("Chien-Pao", 1),
        ("Koraidon ex", 1),
        ("Crispin", 4),
        ("Boss's Orders", 3),
        ("Ciphermaniac's Codebreaking", 2),
        ("Cyrano", 1),
        ("Ultra Ball", 4),
        ("Dusk Ball", 4),
        ("Wondrous Patch", 3),
        ("Prime Catcher", 1),
        ("Lillie's Pearl", 2),
        ("Area Zero Underdepths", 4),
        ("Psychic Energy", 4),
        ("Water Energy", 2),
        ("Fighting Energy", 2),
        ("Telepathic Psychic Energy", 1),
        ("Fire Energy", 1)
    ]
    
    deck_ids = []
    missing = []
    
    for c in cards:
        if c.cardType == 5: # BASIC_ENERGY
            print(f"Basic Energy: {c.name} (ID: {c.cardId})")

if __name__ == "__main__":
    main()
