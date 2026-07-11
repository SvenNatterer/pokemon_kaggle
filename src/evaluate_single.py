import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tournament import evaluate_vs_baseline, evaluate_vs_opponent

if __name__ == "__main__":
    if len(sys.argv) == 6:
        model_a = sys.argv[1]
        deck_a = sys.argv[2]
        model_b = sys.argv[3]
        deck_b = sys.argv[4]
        num_games = int(sys.argv[5])
        
        try:
            result, details = evaluate_vs_opponent(model_a, deck_a, model_b, deck_b, num_games, return_details=True)
            wins, losses, draws, pw1, dw1, bw1, pw2, dw2, bw2 = result
            print(f"RESULT:{wins},{losses},{draws},{pw1},{dw1},{bw1},{pw2},{dw2},{bw2}")
            print(f"DETAIL:{json.dumps(details, sort_keys=True)}")
        except Exception as e:
            print(f"CHILD ERROR: {e}")
    else:
        model_path = sys.argv[1]
        deck_path = sys.argv[2]
        num_games = int(sys.argv[3])
        
        try:
            wins = evaluate_vs_baseline(model_path, deck_path, num_games)
            print(f"WINS:{wins}")
        except Exception as e:
            print(f"CHILD ERROR: {e}")
