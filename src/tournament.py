import os
import sys
import pandas as pd

# Add src to pythonpath so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env_wrapper import PokemonTCGEnv
from src.custom_ppo import CustomPPO

def read_deck(deck_path):
    df = pd.read_csv(deck_path, header=None)
    return df[0].tolist()

def simulate_match(model1_path, deck1_path, model2_path, deck2_path, num_games=10):
    """
    Simulates matches between two CustomPPO models.
    Returns the number of wins for model1.
    """
    deck1 = read_deck(deck1_path)
    deck2 = read_deck(deck2_path)
    
    print(f"Evaluating {model1_path}...")
    wins1 = evaluate_vs_baseline(model1_path, deck1_path, num_games)
    
    print(f"Evaluating {model2_path}...")
    wins2 = evaluate_vs_baseline(model2_path, deck2_path, num_games)
    
    return wins1, wins2

def evaluate_vs_baseline(model_path, deck_path, num_games=10):
    deck = read_deck(deck_path)
    env = PokemonTCGEnv(deck, deck)
    model = CustomPPO.load(model_path, env=env)
    
    wins = 0
    for i in range(num_games):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        if reward > 0:
            wins += 1
            
    return wins

def evaluate_vs_opponent(model1_path, deck1_path, model2_path, deck2_path, num_games=10):
    deck1 = read_deck(deck1_path)
    deck2 = read_deck(deck2_path)
    
    model2 = CustomPPO.load(model2_path)
    
    env = PokemonTCGEnv(my_deck=deck1, opponent_deck=deck2, opponent_model_path=model2_path)
    model1 = CustomPPO.load(model1_path, env=env)
    
    wins = 0
    losses = 0
    draws = 0
    
    prize_wins_1 = 0
    deckout_wins_1 = 0
    prize_wins_2 = 0
    deckout_wins_2 = 0
    
    for i in range(num_games):
        obs, info = env.reset()
        done = False
        final_info = {}
        while not done:
            action, _states = model1.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            final_info = info
            
        reason = final_info.get('win_reason', 'other')
            
        if reward > 0:
            wins += 1
            if reason == 'prize': prize_wins_1 += 1
            elif reason == 'deckout': deckout_wins_1 += 1
        elif reward < 0:
            losses += 1
            if reason == 'prize': prize_wins_2 += 1
            elif reason == 'deckout': deckout_wins_2 += 1
        else:
            draws += 1
            
    return wins, losses, draws, prize_wins_1, deckout_wins_1, prize_wins_2, deckout_wins_2

def main():
    decks = [f"decks/deck_{i}.csv" for i in range(1, 9)]
    models = [f"models/ppo_deck_{i}.zip" for i in range(1, 9)]
    deck_names = [
        "Lillie's Clefairy", "Dragapult Dusknoir", "Slowking", "Ogerpon Box",
        "Crustle", "Dragapult Blaziken", "Rocket's Mewtwo", "Dragapult"
    ]
    
    scores = {name: 0 for name in deck_names}
    
    print("======================================================")
    print("🏆 POKEMON TCG AGENT TOURNAMENT 🏆")
    print("======================================================")
    
    # We will evaluate each agent's winrate over 20 games against the baseline
    num_eval_games = 20
    
    for i in range(len(models)):
        print(f"\nTesting {deck_names[i]} ({models[i]})...")
        wins = evaluate_vs_baseline(models[i], decks[i], num_eval_games)
        win_rate = (wins / num_eval_games) * 100
        scores[deck_names[i]] = win_rate
        print(f"-> Win rate: {win_rate}% ({wins}/{num_eval_games})")
        
    print("\n======================================================")
    print("FINAL LEADERBOARD")
    print("======================================================")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, score) in enumerate(sorted_scores, 1):
        print(f"{rank}. {name}: {score}% Win Rate")

if __name__ == "__main__":
    main()
