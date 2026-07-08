import sys
import os

# Register Kaggle Environment
sys.path.append(os.path.abspath('pokemon-tcg-ai-battle'))
import pokemon_kaggle

from kaggle_environments import make

# We use the agent inside tests/test_sub
def agent_1(obs, config):
    sys.path.insert(0, os.path.abspath('tests/test_sub'))
    import main
    return main.agent(obs)

def agent_2(obs, config):
    sys.path.insert(0, os.path.abspath('tests/test_sub'))
    import main
    return main.agent(obs)

import pandas as pd

# Load Deck 100
deck_100 = pd.read_csv("decks/deck_100.csv", header=None)[0].tolist()

for i in range(1, 6):
    print(f"\n--- Starting Match {i}/5 ---")
    env = make("cabt", configuration={"decks": [deck_100, deck_100]}, debug=True)
    env.run([agent_1, agent_2])
    print(f"Match {i} Success! statuses:", env.statuses)
    
    # Generate official HTML visualizer
    html_path = os.path.join(os.path.abspath('replays'), f'official_replay_{i}.html')
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, "w") as f:
        f.write(env.render(mode="html"))
    print(f"Saved official visualizer to: {html_path}")
