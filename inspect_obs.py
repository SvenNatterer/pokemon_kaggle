import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'pokemon-tcg-ai-battle/sample_submission/sample_submission')))
from cg.game import battle_start, battle_select
from cg.api import to_observation_class

df = pd.read_csv("decks/deck_23.csv", header=None)
deck = df[0].tolist()
obs_dict, _ = battle_start(deck, deck)

for _ in range(10):
    obs = to_observation_class(obs_dict)
    opts = len(obs.select.option)
    if opts > 0:
        obs_dict = battle_select([0])
    
obs = to_observation_class(obs_dict)
p0 = obs.current.players[0]
p1 = obs.current.players[1]
print("P0 HandCount:", p0.handCount, "Hand:", p0.hand)
print("P1 HandCount:", p1.handCount, "Hand:", p1.hand)
