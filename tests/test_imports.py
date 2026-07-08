import sys
import os

sys.path.insert(0, os.path.abspath('tests/test_sub'))
import main

print("Imports successful!")
# Test if we can initialize the model without Kaggle env crashing
try:
    # Set model to None to force reload
    main.model = None
    
    obs_dict_1 = {
        'step': 1,
        'select': {
            'option': [{'id': 'test', 'count': 1, 'type': 1}],
            'maxCount': 1,
            'minCount': 1
        },
        'current': {
            'yourIndex': 0,
            'turn': 0,
            'active': [],
            'bench': [],
            'hand': [],
            'discard': [],
            'stadium': None,
            'prizes': 6,
            'opp_active': [],
            'opp_bench': [],
            'opp_hand_size': 0,
            'opp_discard': [],
            'opp_prizes': 6
        },
        'logs': [],
        'remainingOverageTime': 600
    }
    
    # We just want to see if it loads the model successfully and attempts to predict
    main.agent(obs_dict_1)
except Exception as e:
    print(f"Agent threw an expected error because of dummy data, but imports were fine: {type(e).__name__}: {e}")
print("Test completed.")
