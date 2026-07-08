import sys
import os

sys.path.insert(0, os.path.abspath('.'))
import main
from cg.game import battle_start

print("Imports successful!")

# Initialize a battle to get a real observation dict
my_deck = main.read_deck_csv()
obs_dict_1, _ = battle_start(my_deck, my_deck)

# Run the agent
try:
    action = main.agent(obs_dict_1)
    print("Agent returned:", action)
except Exception as e:
    import traceback
    traceback.print_exc()
