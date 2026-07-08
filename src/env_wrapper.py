import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

from cg.game import battle_start, battle_select, battle_finish
from cg.api import AreaType, OptionType, SelectContext, to_observation_class, Observation
def _pokemon_key(pokemon):
    if pokemon is None:
        return None
    serial = getattr(pokemon, "serial", None)
    return serial if serial is not None else getattr(pokemon, "id", None)

def _same_pokemon(a, b):
    a_key = _pokemon_key(a)
    return a_key is not None and a_key == _pokemon_key(b)

def select_action_indices(obs, action=None, perspective=0):
    select = obs.select
    valid_options = len(select.option) if select and select.option else 0
    if valid_options == 0:
        return []

    min_count = max(0, select.minCount if select else 0)
    max_count = min(valid_options, max(0, select.maxCount if select else 0))
    if max_count == 0:
        return []

    try:
        action = int(action) if action is not None else None
    except Exception:
        action = None
        
    preferred = action if action is not None and 0 <= action < valid_options else None
    min_required = min(min_count, max_count)
    if preferred is None and min_required == 0:
        return []

    selected = []
    if preferred is not None:
        selected.append(preferred)

    for index in range(valid_options):
        if len(selected) >= min_required:
            break
        if index not in selected:
            selected.append(index)

    return [int(index) for index in selected[:max_count]]

class PokemonTCGEnv(gym.Env):
    def __init__(self, my_deck: list[int], opponent_deck: list[int], opponent_model_path=None):
        super().__init__()
        self.my_deck = my_deck
        self.opponent_deck = opponent_deck
        self.opponent_model_path = opponent_model_path
        self.opponent_model = None
        
        # Action space: a discrete selection of an option from the available ones.
        self.max_options = 1000 
        self.action_space = spaces.Discrete(self.max_options)
        
        # Observation space: 1500-dim vector + action mask + aux target
        self.vector_dim = 1500
        self.aux_dim = 2000
        self.observation_space = spaces.Dict({
            "vector": spaces.Box(low=-1000, high=1000, shape=(self.vector_dim,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.max_options,), dtype=np.int8),
            "aux_target": spaces.Box(low=0, high=1, shape=(self.aux_dim,), dtype=np.float32)
        })
        
        self.current_obs_dict = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Lazy load the model in the worker process to avoid pickling issues
        if self.opponent_model_path is not None and self.opponent_model is None:
            # We import here to avoid circular imports if necessary
            from stable_baselines3 import PPO
            try:
                # We can't easily use CustomPPO here without circular imports, 
                # but standard PPO can load the policy weights just fine for inference!
                self.opponent_model = PPO.load(self.opponent_model_path)
            except Exception as e:
                print(f"Failed to load opponent model from {self.opponent_model_path}: {e}")
                self.opponent_model_path = None
        
        if self.current_obs_dict is not None:
            battle_finish()
            
        self.current_obs_dict, _ = battle_start(self.my_deck, self.opponent_deck)
        self._process_opponent_turns()
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action) # Convert numpy scalar to python int
        old_obs = to_observation_class(self.current_obs_dict)
        
        # Track dense reward starting with time penalty
        step_reward = -0.005
        
        valid_options = len(old_obs.select.option) if old_obs.select and old_obs.select.option else 0
        if action < 0 or action >= valid_options:
            step_reward -= 0.1 # Invalid action penalty
            
        if valid_options > 0 and 0 <= action < valid_options:
            chosen_option = old_obs.select.option[action]
            if chosen_option.type == getattr(OptionType, 'ATTACK', 11):
                step_reward += 0.1
            elif chosen_option.type == getattr(OptionType, 'END', 14):
                step_reward -= 0.02
                
        action_list = select_action_indices(old_obs, action, perspective=0)
        
        self.current_obs_dict = battle_select(action_list)
        
        done = self._process_opponent_turns()
        
        new_obs = to_observation_class(self.current_obs_dict)
        step_reward += self._compute_dense_reward(old_obs, new_obs, done)
        
        return self._get_obs(), step_reward, done, False, self._get_info()
        
    def _compute_dense_reward(self, old_obs, new_obs, done):
        if done:
            if new_obs.current and new_obs.current.result == 0:
                # Unterscheide zwischen Prize-Win und Deck-Out-Win
                if len(new_obs.current.players[0].prize) == 0:
                    return 10.0 # Prize Win (scaled down)
                else:
                    return 2.0 # Deck-Out Win
            elif new_obs.current and new_obs.current.result == 1:
                loss_reward = -2.0 # Loss
                # Deck-Out Penalty
                if new_obs.current.players[0].deckCount == 0:
                    loss_reward -= 3.0 # Penalty for deck-out
                return loss_reward
            return 0.0 # Draw
            
        if not old_obs.current or not new_obs.current:
            return 0.0
            
        reward = 0.0
        old_p0 = old_obs.current.players[0]
        old_p1 = old_obs.current.players[1]
        new_p0 = new_obs.current.players[0]
        new_p1 = new_obs.current.players[1]
        
        # Prize Card Taken/Lost
        delta_prize_0 = len(old_p0.prize) - len(new_p0.prize)
        if delta_prize_0 > 0:
            reward += delta_prize_0 * 1.0
            
        delta_prize_1 = len(old_p1.prize) - len(new_p1.prize)
        if delta_prize_1 > 0:
            reward -= delta_prize_1 * 1.0
            
        # Deck shrink penalty
        delta_deck_0 = old_p0.deckCount - new_p0.deckCount
        if delta_deck_0 > 0:
            reward -= delta_deck_0 * 0.01

        # Damage Deltas (BOOSTED for Aggro)
        def sum_hp(p):
            hp = 0
            if p.active and p.active[0]: hp += p.active[0].hp
            for b in p.bench:
                if b: hp += b.hp
            return hp
            
        delta_my_hp = sum_hp(old_p0) - sum_hp(new_p0)
        if delta_my_hp > 0:
            reward -= delta_my_hp * 0.02
            
        delta_opp_hp = sum_hp(old_p1) - sum_hp(new_p1)
        if delta_opp_hp > 0:
            reward += delta_opp_hp * 0.03  # Dealing damage is heavily rewarded
            
        # Energy Attached
        def sum_energies(p):
            e = 0
            if p.active and p.active[0]: e += len(p.active[0].energies)
            for b in p.bench:
                if b: e += len(b.energies)
            return e
            
        delta_total_energy = sum_energies(new_p0) - sum_energies(old_p0)
        if delta_total_energy > 0:
            active_attached = False
            correct_energy_type = False
            if old_p0.active and old_p0.active[0] and new_p0.active and new_p0.active[0]:
                if _same_pokemon(old_p0.active[0], new_p0.active[0]):
                    old_e_count = len(old_p0.active[0].energies)
                    new_e_count = len(new_p0.active[0].energies)
                    if new_e_count > old_e_count:
                        active_attached = True
                        old_e_list = [int(e) if not hasattr(e, 'value') else e.value for e in old_p0.active[0].energies]
                        new_e_list = [int(e) if not hasattr(e, 'value') else e.value for e in new_p0.active[0].energies]
                        for e in old_e_list:
                            if e in new_e_list:
                                new_e_list.remove(e)
                        
                        if len(new_e_list) > 0:
                            added_e = new_e_list[0]
                            # Deck 98 is a Grass deck, so required energy is GRASS (1) or COLORLESS (0)
                            req_energies = [1, 0] 
                            if added_e in req_energies:
                                correct_energy_type = True

            if active_attached and correct_energy_type:
                reward += delta_total_energy * 0.25  # Big reward for correct energy on active
            elif active_attached and not correct_energy_type:
                reward -= delta_total_energy * 0.15  # Penalty for wrong energy
            else:
                reward += delta_total_energy * 0.05  # Small reward for bench energy
            
        # Opponent Active KO (Big bonus for knocking out)
        old_opp_active = old_p1.active[0] if old_p1.active else None
        new_opp_active = new_p1.active[0] if new_p1.active else None
        if old_opp_active is not None and old_opp_active.hp > 0:
            if new_opp_active is None or not _same_pokemon(old_opp_active, new_opp_active) or new_opp_active.hp <= 0:
                reward += 0.3  # KO Bonus
                
        # Step Penalty (Time penalty)
        reward -= 0.001
                
        return reward

    def _process_opponent_turns(self):
        """Simulate the opponent's turns randomly or with a model until it is our turn or the game ends."""
        done = False
        while True:
            obs = to_observation_class(self.current_obs_dict)
            
            # Check if game is over
            if obs.current is not None and obs.current.result != -1:
                done = True
                break
                
            # If it's our turn, stop simulating opponent
            if obs.current is not None and obs.current.yourIndex == 0:
                break
                
            if self.opponent_model is not None:
                opponent_obs = self._get_obs(perspective=1)
                
                # Expand dims to match what stable-baselines expects
                for k in opponent_obs:
                    opponent_obs[k] = np.expand_dims(opponent_obs[k], axis=0)
                
                action, _ = self.opponent_model.predict(opponent_obs, deterministic=True)
                action = int(action.item())
                
                action_list = select_action_indices(obs, action, perspective=1)
                self.current_obs_dict = battle_select(action_list)
                
            else:
                # Deterministic fallback for opponent turns.
                action_list = select_action_indices(obs, 0, perspective=1)
                self.current_obs_dict = battle_select(action_list)
            
        return done

    def _get_obs(self, perspective=0):
        obs = to_observation_class(self.current_obs_dict)
        mask = np.zeros(self.max_options, dtype=np.int8)
        if obs.select and obs.select.option:
            num_opts = min(len(obs.select.option), self.max_options)
            mask[:num_opts] = 1
            
        vec = np.zeros((self.vector_dim,), dtype=np.float32)
        if obs.current is not None:
            state = obs.current
            # Continuous & Boolean Stats (0-299)
            vec[0] = state.turn
            vec[1] = float(abs(state.yourIndex - perspective))
            vec[2] = state.firstPlayer
            vec[3] = float(state.supporterPlayed)
            vec[4] = float(state.stadiumPlayed)
            vec[5] = float(state.energyAttached)
            vec[6] = float(state.retreated)
            
            idx = 7
            player_list = [state.players[perspective], state.players[1 - perspective]]
            for p in player_list:
                vec[idx] = p.deckCount
                vec[idx+1] = p.handCount
                vec[idx+2] = p.benchMax
                vec[idx+3] = len(p.prize)
                vec[idx+4] = len(p.discard)
                vec[idx+5] = float(p.poisoned)
                vec[idx+6] = float(p.burned)
                vec[idx+7] = float(p.asleep)
                vec[idx+8] = float(p.paralyzed)
                vec[idx+9] = float(p.confused)
                idx += 10
                
                # Active
                if p.active and p.active[0] is not None:
                    vec[idx] = p.active[0].hp
                    vec[idx+1] = p.active[0].maxHp
                    vec[idx+2] = float(p.active[0].appearThisTurn)
                    for e in p.active[0].energies:
                        e_idx = int(e) if not hasattr(e, 'value') else e.value
                        if 0 <= e_idx < 12:
                            vec[idx+3+e_idx] += 1.0
                idx += 15
                
                # Bench
                for i in range(5):
                    if i < len(p.bench) and p.bench[i] is not None:
                        vec[idx] = p.bench[i].hp
                        vec[idx+1] = p.bench[i].maxHp
                        vec[idx+2] = float(p.bench[i].appearThisTurn)
                        for e in p.bench[i].energies:
                            e_idx = int(e) if not hasattr(e, 'value') else e.value
                            if 0 <= e_idx < 12:
                                vec[idx+3+e_idx] += 1.0
                    idx += 15

            # Logs Continuous (256-275)
            if obs.logs:
                log_len = min(len(obs.logs), 5)
                for i in range(log_len):
                    log = obs.logs[-(i+1)]
                    vec[256 + i*4] = float(log.playerIndex if log.playerIndex is not None else 0)
                    vec[257 + i*4] = float(log.value if log.value is not None else 0)
                    vec[258 + i*4] = float(log.head if log.head is not None else 0)
                    vec[259 + i*4] = float(log.reason if log.reason is not None else 0)

            # SelectData Context (250-255)
            vec[255] = state.turnActionCount
            if obs.select:
                vec[250] = float(obs.select.context)
                vec[251] = float(obs.select.minCount)
                vec[252] = float(obs.select.maxCount)
                vec[253] = float(obs.select.remainDamageCounter)
                vec[254] = float(obs.select.remainEnergyCost)
                
                if obs.select.contextCard and obs.select.contextCard is not None:
                    vec[409] = float(obs.select.contextCard.id)
                if obs.select.effect and obs.select.effect is not None:
                    vec[410] = float(obs.select.effect.id)

            # Card IDs (300-410)
            if state.stadium and state.stadium[0] is not None:
                vec[330] = float(state.stadium[0].id)
            
            p_us = state.players[perspective]
            p_them = state.players[1 - perspective]
            
            # Us Field
            if p_us.active and p_us.active[0] is not None:
                vec[300] = float(p_us.active[0].id)
                if p_us.active[0].tools: vec[331] = float(p_us.active[0].tools[0].id)
            for i in range(5):
                if i < len(p_us.bench) and p_us.bench[i] is not None:
                    vec[301 + i] = float(p_us.bench[i].id)
                    if p_us.bench[i].tools: vec[332 + i] = float(p_us.bench[i].tools[0].id)
            
            # Them Field
            if p_them.active and p_them.active[0] is not None:
                vec[343] = float(p_them.active[0].id)
                if p_them.active[0].tools: vec[337] = float(p_them.active[0].tools[0].id)
            for i in range(5):
                if i < len(p_them.bench) and p_them.bench[i] is not None:
                    vec[344 + i] = float(p_them.bench[i].id)
                    if p_them.bench[i].tools: vec[338 + i] = float(p_them.bench[i].tools[0].id)
            
            # Us Hand
            if p_us.hand is not None:
                for i in range(min(len(p_us.hand), 24)):
                    vec[306 + i] = float(p_us.hand[i].id)
                    
            # Us Discard
            if p_us.discard is not None:
                disc = p_us.discard[-30:] if len(p_us.discard) > 30 else p_us.discard
                for i in range(len(disc)):
                    vec[349 + i] = float(disc[i].id)
                    
            # Them Discard
            if p_them.discard is not None:
                disc = p_them.discard[-30:] if len(p_them.discard) > 30 else p_them.discard
                for i in range(len(disc)):
                    vec[379 + i] = float(disc[i].id)

            # Deck (411-470)
            if obs.select and obs.select.deck is not None:
                for i in range(min(len(obs.select.deck), 60)):
                    vec[411 + i] = float(obs.select.deck[i].id)

            # Looking (471-530)
            if state.looking is not None:
                for i in range(min(len(state.looking), 60)):
                    if state.looking[i] is not None:
                        vec[471 + i] = float(state.looking[i].id)

            # Prize (531-542)
            for i in range(min(len(p_us.prize), 6)):
                if p_us.prize[i] is not None:
                    vec[531 + i] = float(p_us.prize[i].id)
            for i in range(min(len(p_them.prize), 6)):
                if p_them.prize[i] is not None:
                    vec[537 + i] = float(p_them.prize[i].id)

            # Pre-Evolutions and Energy Cards (543-638)
            def extract_pokemon_cards(p_state, pre_evo_base, nrg_base):
                p_list = []
                if p_state.active and p_state.active[0] is not None: p_list.append(p_state.active[0])
                p_list.extend([x for x in p_state.bench if x is not None])
                
                e_idx = 0
                pe_idx = 0
                for pok in p_list:
                    if hasattr(pok, 'preEvolution') and pok.preEvolution:
                        for c in pok.preEvolution:
                            if pe_idx < 18:
                                vec[pre_evo_base + pe_idx] = float(c.id)
                                pe_idx += 1
                    if hasattr(pok, 'energyCards') and pok.energyCards:
                        for c in pok.energyCards:
                            if e_idx < 30:
                                vec[nrg_base + e_idx] = float(c.id)
                                e_idx += 1

            extract_pokemon_cards(p_us, 543, 579)
            extract_pokemon_cards(p_them, 561, 609)

            # Log Cards (639-648) and Enums (1300-1314)
            if obs.logs:
                log_len = min(len(obs.logs), 5)
                for i in range(log_len):
                    log = obs.logs[-(i+1)]
                    vec[639 + i*2] = float(log.cardId if log.cardId is not None else 0)
                    vec[640 + i*2] = float(log.cardIdTarget if log.cardIdTarget is not None else 0)
                    
                    vec[1300 + i*3] = float(log.type if log.type is not None else 0)
                    vec[1301 + i*3] = float(log.fromArea if log.fromArea is not None else 0)
                    vec[1302 + i*3] = float(log.toArea if log.toArea is not None else 0)

            # Options (800-1299)
            if obs.select and obs.select.option:
                opt_len = min(len(obs.select.option), 50)
                for i in range(opt_len):
                    opt = obs.select.option[i]
                    base = 800 + i*10
                    vec[base] = float(opt.type)
                    vec[base+1] = float(opt.cardId if opt.cardId is not None else 0)
                    vec[base+2] = float(opt.area if opt.area is not None else 0)
                    vec[base+3] = float(opt.index if opt.index is not None else 0)
                    vec[base+4] = float(opt.inPlayArea if opt.inPlayArea is not None else 0)
                    vec[base+5] = float(opt.inPlayIndex if opt.inPlayIndex is not None else 0)
                    vec[base+6] = float(opt.attackId if opt.attackId is not None else 0)
                    vec[base+7] = float(opt.specialConditionType if opt.specialConditionType is not None else 0)
                    vec[base+8] = float(opt.playerIndex if opt.playerIndex is not None else 0)
                    vec[base+9] = float(opt.number if opt.number is not None else (opt.count if hasattr(opt, 'count') and opt.count is not None else 0))
                
        # Auxiliary Target: Predict the hidden cards of the opponent
        hidden_counts = {}
        for card_id in self.opponent_deck:
            hidden_counts[card_id] = hidden_counts.get(card_id, 0) + 1
            
        if obs.current is not None:
            p1 = obs.current.players[1]
            visible = []
            if p1.active and p1.active[0]:
                visible.append(p1.active[0].id)
                for e in p1.active[0].energies: 
                    visible.append(e.id if hasattr(e, 'id') else int(e))
            for b in p1.bench:
                if b:
                    visible.append(b.id)
                    for e in b.energies: 
                        visible.append(e.id if hasattr(e, 'id') else int(e))
            for d in p1.discard:
                if d: 
                    visible.append(d.id if hasattr(d, 'id') else int(d))
                
            for vid in visible:
                if vid in hidden_counts and hidden_counts[vid] > 0:
                    hidden_counts[vid] -= 1
                    
        aux_target = np.zeros(2000, dtype=np.float32)
        for card_id, count in hidden_counts.items():
            if card_id < 2000 and count > 0:
                aux_target[card_id] = 1.0
                
        return {"vector": vec, "action_mask": mask, "aux_target": aux_target}
        
    def _get_info(self):
        info = {}
        try:
            obs = to_observation_class(self.current_obs_dict)
            if obs and obs.current:
                p0 = obs.current.players[0]
                p1 = obs.current.players[1]
                if len(p0.prize) == 0 or len(p1.prize) == 0:
                    info['win_reason'] = 'prize'
                elif p0.deckCount == 0 or p1.deckCount == 0:
                    info['win_reason'] = 'deckout'
                else:
                    info['win_reason'] = 'other'
        except Exception:
            pass
        return info

    def close(self):
        if self.current_obs_dict is not None:
            battle_finish()
            self.current_obs_dict = None

def read_sample_deck():
    path = os.path.join(os.path.dirname(__file__), "..", "pokemon-tcg-ai-battle", "sample_submission", "sample_submission", "deck.csv")
    with open(path, "r") as f:
        deck = [int(line.strip()) for line in f if line.strip()]
    return deck
