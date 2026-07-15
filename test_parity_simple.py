import os
import sys

sys.path.append(os.path.abspath("src"))

from src.cg.sim import Battle, lib, V6ObservationBuffer
import ctypes
import numpy as np
import json

def test():
    # Parse all cards
    lib.AllCard.restype = ctypes.c_char_p
    all_cards_json = lib.AllCard().decode('utf-8')
    cards = json.loads(all_cards_json)
    
    basic_id = -1
    energy_id = -1
    for c in cards:
        if c.get("cardType") == 0 and c.get("basic") == True and basic_id == -1:
            basic_id = c["cardId"]
        if c.get("cardType") == 5 and energy_id == -1:
            energy_id = c["cardId"]
            
    print(f"Basic Pokemon ID: {basic_id}, Energy ID: {energy_id}")
    
    deck0 = [basic_id] * 4 + [energy_id] * 56
    deck1 = deck0[:]
    
    cards_arr = deck0 + deck1
    arg = (ctypes.c_int * len(cards_arr))(*cards_arr)
    
    start_data = lib.BattleStart(arg)
    Battle.battle_ptr = start_data.battlePtr
    
    if not Battle.battle_ptr:
        print("Failed to start battle!")
        return
        
    sd = lib.GetBattleData(Battle.battle_ptr)
    obs_json = sd.json.decode('utf-8')
    print("Initial obs json length:", len(obs_json))
    
    buf = V6ObservationBuffer()
    pending = (ctypes.c_int * 1)(0)
    
    lib.GetV6Observation(Battle.battle_ptr, 0, pending, 0, ctypes.byref(buf))
    
    print(f"entity_ids[0] returned from C++: {buf.entity_ids[0]}")
    if buf.entity_ids[0] != 0:
        print("SUCCESS! API binding works.")
        print(f"Action Mask: {list(buf.action_mask[:10])}")
        print(f"Entity features slot 0 [0:10]: {list(buf.entity_features[:10])}")
        print(f"Entity IDs: {list(buf.entity_ids)}")
    else:
        print("FAILED: Value is 0.")
        
    lib.BattleFinish(Battle.battle_ptr)

if __name__ == "__main__":
    test()
