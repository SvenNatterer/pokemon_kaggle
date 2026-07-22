import sys
import json
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.cg.api import all_card_data

cards = all_card_data()
card_names = {c.cardId: c.name for c in cards}

def parse_replay(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    print(f"=== BATTLE LOG: {os.path.basename(file_path)} ===")
    
    for step in data:
        logs = step.get("logs", [])
        for log in logs:
            ltype = log.get("type")
            pid = log.get("playerIndex")
            pname = f"Player {pid}"
            card_id = log.get("cardId")
            card_name = card_names.get(card_id, "Unknown Card") if card_id else ""
            
            if ltype == "TurnStart":
                print(f"\n--- Turn started for {pname} ---")
            elif ltype == "Draw":
                pass # print(f"[{pname}] Draws a card ({card_name})")
            elif ltype == "Play":
                print(f"[{pname}] Plays {card_name}")
            elif ltype == "Attach":
                print(f"[{pname}] Attaches {card_name}")
            elif ltype == "Evolve":
                print(f"[{pname}] Evolves into {card_name}")
            elif ltype == "Attack":
                print(f"[{pname}] Attacks! (Using {card_name})")
            elif ltype == "HpChange":
                amount = log.get("amount", 0)
                if amount < 0:
                    print(f"  -> Takes {-amount} damage!")
                elif amount > 0:
                    print(f"  -> Heals {amount} HP!")
            elif ltype == "Result":
                print(f"\n*** BATTLE FINISHED! Winner: Player {log.get('winner', 'Unknown')} ***")
            elif ltype == "TurnEnd":
                pass
            elif ltype in ["Shuffle", "HasBasicPokemon", "MoveCard", "Switch"]:
                pass # skip verbose logs
            else:
                pass # print(f"[{pname}] {ltype} {card_name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse_replay(sys.argv[1])
    else:
        print("Usage: python parse_replay.py <path_to_replay.json>")
