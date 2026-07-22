import os
import sys
import pandas as pd
import numpy as np

# Add src to pythonpath so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env.env_wrapper import LEGACY_ACTION_SPACE_SIZE, PokemonTCGEnv, _fit_observation_to_model_space
from src.training.custom_ppo import CustomPPO
from src.league.model_paths import discover_deck_models, resolve_deck_model_path, strip_zip_suffix
from src.agents.bot_loader import load_bot


def model_action_space_size(model):
    size = int(getattr(getattr(model, "action_space", None), "n", LEGACY_ACTION_SPACE_SIZE))
    return size


def build_evaluation_env(
    learner_deck,
    opponent_deck,
    opponent_model_path,
    learner_perspective,
    action_space_size=LEGACY_ACTION_SPACE_SIZE,
):
    """Keep deck ownership stable; the environment places decks by perspective."""
    kwargs = dict(
        my_deck=learner_deck,
        opponent_deck=opponent_deck,
        opponent_model_path=opponent_model_path,
        learner_perspective=learner_perspective,
    )
    if action_space_size != LEGACY_ACTION_SPACE_SIZE:
        kwargs["action_space_size"] = action_space_size
    return PokemonTCGEnv(**kwargs)

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
    model = load_bot(model_path)
    env = PokemonTCGEnv(deck, deck, action_space_size=model_action_space_size(model))
    
    wins = 0
    for i in range(num_games):
        obs, info = env.reset()
        done = False
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        model_space = getattr(model, "observation_space", None)
        while not done:
            if model_space is not None:
                obs_for_model = _fit_observation_to_model_space(obs, model_space)
            else:
                obs_for_model = obs
            action, lstm_state = model.predict(
                obs_for_model,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=True,
            )
            episode_start = np.zeros((1,), dtype=bool)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        if info.get("winner") == 0:
            wins += 1
            
    return wins

def evaluate_vs_opponent(model1_path, deck1_path, model2_path, deck2_path, num_games=10, return_details=False):
    deck1 = read_deck(deck1_path)
    deck2 = read_deck(deck2_path)
    
    wins = 0
    losses = 0
    draws = 0
    
    prize_wins_1 = 0
    deckout_wins_1 = 0
    benchout_wins_1 = 0
    prize_wins_2 = 0
    deckout_wins_2 = 0
    benchout_wins_2 = 0
    total_turns = 0
    reason_counts = {}
    candidate_win_reasons = {}
    opponent_win_reasons = {}
    perspective_results = {
        "player_0": {"games": 0, "wins": 0, "losses": 0, "draws": 0, "turns": 0},
        "player_1": {"games": 0, "wins": 0, "losses": 0, "draws": 0, "turns": 0},
    }
    
    def run_direction(
        learner_model_path,
        learner_deck,
        opponent_model_path,
        opponent_deck,
        games,
        learner_perspective,
    ):
        nonlocal wins, losses, draws
        nonlocal prize_wins_1, deckout_wins_1, benchout_wins_1
        nonlocal prize_wins_2, deckout_wins_2, benchout_wins_2
        nonlocal total_turns

        learner_model = load_bot(learner_model_path)
        env = build_evaluation_env(
            learner_deck,
            opponent_deck,
            opponent_model_path,
            learner_perspective,
            model_action_space_size(learner_model),
        )
        model_space = getattr(learner_model, "observation_space", None)

        try:
            for _ in range(games):
                obs, info = env.reset()
                done = False
                lstm_state = None
                episode_start = np.ones((1,), dtype=bool)
                turns = 0
                while not done:
                    obs_for_model = (
                        _fit_observation_to_model_space(obs, model_space)
                        if model_space is not None else obs
                    )
                    action, lstm_state = learner_model.predict(
                        obs_for_model,
                        state=lstm_state,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    episode_start = np.zeros((1,), dtype=bool)
                    obs, _, terminated, truncated, info = env.step(action)
                    turns += 1
                    done = terminated or truncated

                total_turns += turns
                perspective = perspective_results[f"player_{learner_perspective}"]
                perspective["games"] += 1
                perspective["turns"] += turns

                engine_winner = info.get("winner", -1)
                reason = info.get("win_reason", "other")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                candidate_won = (engine_winner == learner_perspective)
                reference_won = (engine_winner == 1 - learner_perspective)

                if candidate_won:
                    wins += 1
                    perspective["wins"] += 1
                    candidate_win_reasons[reason] = candidate_win_reasons.get(reason, 0) + 1
                    if reason == "prize": prize_wins_1 += 1
                    elif reason == "deckout": deckout_wins_1 += 1
                    elif reason == "benchout": benchout_wins_1 += 1
                elif reference_won:
                    losses += 1
                    perspective["losses"] += 1
                    opponent_win_reasons[reason] = opponent_win_reasons.get(reason, 0) + 1
                    if reason == "prize": prize_wins_2 += 1
                    elif reason == "deckout": deckout_wins_2 += 1
                    elif reason == "benchout": benchout_wins_2 += 1
                else:
                    draws += 1
                    perspective["draws"] += 1
        finally:
            env.close()

    games_as_player_0 = (num_games + 1) // 2
    games_as_player_1 = num_games // 2

    run_direction(
        model1_path,
        deck1,
        model2_path,
        deck2,
        games_as_player_0,
        learner_perspective=0,
    )
    if games_as_player_1:
        run_direction(
            model1_path,
            deck1,
            model2_path,
            deck2,
            games_as_player_1,
            learner_perspective=1,
        )
            
    result = (wins, losses, draws, prize_wins_1, deckout_wins_1, benchout_wins_1, prize_wins_2, deckout_wins_2, benchout_wins_2)
    if return_details:
        for values in perspective_results.values():
            games = values["games"]
            values["mean_turns"] = values["turns"] / games if games else 0.0
        return result, {
            "total_turns": total_turns,
            "mean_turns": total_turns / num_games if num_games else 0.0,
            "reason_counts": reason_counts,
            "candidate_win_reasons": candidate_win_reasons,
            "opponent_win_reasons": opponent_win_reasons,
            "perspective": perspective_results,
        }
    return result


def discover_tournament_entries():
    entries = []
    for model in discover_deck_models():
        deck_path = f"decks/deck_{model['deck_id']}.csv"
        if not os.path.exists(deck_path):
            continue

        label = model["name"]
        entries.append({
            "label": label,
            "model_path": model["path"],
            "deck_path": deck_path,
        })

    return entries


def main():
    entries = discover_tournament_entries()
    if not entries:
        fallback_ids = range(1, 9)
        entries = [
            {
                "label": f"ppo_deck_{i}",
                "model_path": resolve_deck_model_path(i) or f"models/ppo_deck_{i}.zip",
                "deck_path": f"decks/deck_{i}.csv",
            }
            for i in fallback_ids
        ]
    
    scores = {entry["label"]: 0 for entry in entries}
    
    print("======================================================")
    print("🏆 POKEMON TCG AGENT TOURNAMENT 🏆")
    print("======================================================")
    
    # We will evaluate each agent's winrate over 20 games against the baseline
    num_eval_games = 20
    
    for entry in entries:
        print(f"\nTesting {entry['label']} ({entry['model_path']})...")
        wins = evaluate_vs_baseline(entry["model_path"], entry["deck_path"], num_eval_games)
        win_rate = (wins / num_eval_games) * 100
        scores[entry["label"]] = win_rate
        print(f"-> Win rate: {win_rate}% ({wins}/{num_eval_games})")
        
    print("\n======================================================")
    print("FINAL LEADERBOARD")
    print("======================================================")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, score) in enumerate(sorted_scores, 1):
        print(f"{rank}. {name}: {score}% Win Rate")

if __name__ == "__main__":
    main()
