import os
import sys

# Add src to sys path
sys.path.append(os.path.abspath("src"))

from env_wrapper import PokemonTCGEnv, to_observation_class
from src.cg.sim import Battle, lib, V6ObservationBuffer
import ctypes
import numpy as np

def test():
    # Initialize game
    print("Testing GetV6Observation C++ API parity with Python...")
    
    # We will step through a real battle and compare the output.
    env = PokemonTCGEnv(
        my_deck=[150] * 60,
        opponent_deck=[150] * 60,
    )
    
    # Reset env
    obs, info = env.reset()
    
    buf = V6ObservationBuffer()
    pending = (ctypes.c_int * 1)(0)
    
    lib.GetV6Observation(Battle.battle_ptr, 0, pending, 0, ctypes.byref(buf))
    
    print(f"entity_ids[0] returned from C++: {buf.entity_ids[0]}")
    if buf.entity_ids[0] == 42:
        print("SUCCESS! API binding works.")
        
        # Test Parity for some fields
        print(f"Action Mask: {list(buf.action_mask[:10])}")
        print(f"Entity features slot 0 [0:10]: {list(buf.entity_features[:10])}")
    else:
        print("FAILED: Value is not 42.")

if __name__ == "__main__":
    test()
