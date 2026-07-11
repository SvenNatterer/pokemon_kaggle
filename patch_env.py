import re

with open("src/env_wrapper.py", "r") as f:
    content = f.read()

# 1. Replace the __init__ part
old_init = """        # Build map of cardId -> required energies from attacks dynamically
        try:
            attacks_map = {a.attackId: a.energies for a in all_attack()}
            self.pokemon_required_energies = {}
            for card in all_card_data():
                if card.cardType == CardType.POKEMON:
                    req = set()
                    for attack_id in card.attacks:
                        energies = attacks_map.get(attack_id, [])
                        for e in energies:
                            e_val = int(e) if not hasattr(e, 'value') else e.value
                            if e_val != 0:
                                req.add(e_val)
                    if not req:
                        self.pokemon_required_energies[card.cardId] = None # None means any energy in deck is correct
                    else:
                        self.pokemon_required_energies[card.cardId] = req
        except Exception as e:
            print("Failed to build dynamic pokemon energy mapping:", e)
            self.pokemon_required_energies = {}"""

new_init = """        # Build map of cardId -> exact attack costs dynamically
        try:
            attacks_map = {a.attackId: a.energies for a in all_attack()}
            self.pokemon_attack_costs = {}
            for card in all_card_data():
                if card.cardType == CardType.POKEMON:
                    costs = []
                    for attack_id in card.attacks:
                        energies = attacks_map.get(attack_id, [])
                        cost = [int(e) if not hasattr(e, 'value') else e.value for e in energies]
                        costs.append(cost)
                    self.pokemon_attack_costs[card.cardId] = costs
        except Exception as e:
            print("Failed to build dynamic pokemon energy mapping:", e)
            self.pokemon_attack_costs = {}"""

content = content.replace(old_init, new_init)

# 2. Replace the reward logic part
old_reward = """            if new_p0.active and old_p0.active:
                old_active_energies = len(old_p0.active[0].energies)
                new_active_energies = len(new_p0.active[0].energies)
                if new_active_energies > old_active_energies:
                    active_attached = True
                    
                    # Figure out WHICH energy type was attached to active
                    old_e_list = list(old_p0.active[0].energies)
                    new_e_list = list(new_p0.active[0].energies)
                    
                    for e in old_e_list:
                        if e in new_e_list:
                            new_e_list.remove(e)
                        
                    if len(new_e_list) > 0:
                        added_e = new_e_list[0]
                        
                        active_pokemon_id = new_p0.active[0].id if new_p0.active and new_p0.active[0] else None
                        req_energies = None
                        if active_pokemon_id is not None:
                            req_energies = self.pokemon_required_energies.get(active_pokemon_id)
                            
                        if req_energies is None:
                            # Fallback/Colorless: Alle im Deck vorkommenden Energietypen sind korrekt
                            req_energies = self.valid_energy_types
                        else:
                            # We always allow Colorless (0)
                            req_energies = set(req_energies) | {0}
                            
                        if added_e in req_energies:
                            correct_energy_type = True

            if active_attached and correct_energy_type:
                reward += delta_total_energy * 0.25  # Big reward for correct energy on active
            elif active_attached and not correct_energy_type:
                reward -= delta_total_energy * 0.15  # Penalty for wrong energy
            else:
                reward += delta_total_energy * 0.05  # Small reward for bench energy"""

new_reward = """            useful_energy_attached = False
            
            if new_p0.active and old_p0.active:
                old_active_energies = len(old_p0.active[0].energies)
                new_active_energies = len(new_p0.active[0].energies)
                if new_active_energies > old_active_energies:
                    active_attached = True
                    
                    old_e_list = list(old_p0.active[0].energies)
                    new_e_list = list(new_p0.active[0].energies)
                    
                    active_pokemon_id = new_p0.active[0].id
                    costs = self.pokemon_attack_costs.get(active_pokemon_id, [])
                    
                    def calc_deficit(attached, cost):
                        attached_counts = {}
                        for e in attached:
                            e_val = int(e) if not hasattr(e, 'value') else e.value
                            attached_counts[e_val] = attached_counts.get(e_val, 0) + 1
                        
                        cost_specific = [e for e in cost if e != 0]
                        cost_colorless = sum(1 for e in cost if e == 0)
                        
                        missing_specific = 0
                        for req_e in cost_specific:
                            if attached_counts.get(req_e, 0) > 0:
                                attached_counts[req_e] -= 1
                            else:
                                missing_specific += 1
                                
                        remaining_attached = sum(attached_counts.values())
                        missing_colorless = max(0, cost_colorless - remaining_attached)
                        return missing_specific + missing_colorless

                    for cost in costs:
                        def_before = calc_deficit(old_e_list, cost)
                        def_after = calc_deficit(new_e_list, cost)
                        if def_after < def_before:
                            useful_energy_attached = True
                            break

            if active_attached and useful_energy_attached:
                reward += delta_total_energy * 0.25  # Big reward for useful energy on active
            elif active_attached and not useful_energy_attached:
                reward -= delta_total_energy * 0.15  # Penalty for useless/over-attached energy
            else:
                reward += delta_total_energy * 0.05  # Small reward for bench energy"""

content = content.replace(old_reward, new_reward)

with open("src/env_wrapper.py", "w") as f:
    f.write(content)

print("Patched env_wrapper.py")
