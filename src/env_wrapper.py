import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, Observation

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
        
        # Observation space: 100-dim vector + action mask + aux target
        self.vector_dim = 100
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
        
        # Basic parsing: choose action
        valid_options = len(old_obs.select.option) if old_obs.select and old_obs.select.option else 0
        min_c = old_obs.select.minCount if old_obs.select else 0
        max_c = old_obs.select.maxCount if old_obs.select else 0
        
        if max_c == 0:
            action_list = []
        elif action >= valid_options:
            # Fallback to random if invalid action
            step_reward -= 0.1 # Invalid action penalty
            sample_size = min(valid_options, max_c)
            action_list = random.sample(list(range(valid_options)), sample_size)
        else:
            action_list = [action]
            if len(action_list) < min_c:
                remaining = min_c - 1
                available = [i for i in range(valid_options) if i != action]
                if remaining <= len(available):
                    action_list += random.sample(available, remaining)
            action_list = action_list[:max_c]
            
        action_list = [int(x) for x in action_list]
        
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
            reward += delta_prize_0 * 0.5
            
        delta_prize_1 = len(old_p1.prize) - len(new_p1.prize)
        if delta_prize_1 > 0:
            reward -= delta_prize_1 * 0.5
            
        # Deck shrink penalty
        delta_deck_0 = old_p0.deckCount - new_p0.deckCount
        if delta_deck_0 > 0:
            reward -= delta_deck_0 * 0.01

        # Damage Deltas
        def sum_hp(p):
            hp = 0
            if p.active and p.active[0]: hp += p.active[0].hp
            for b in p.bench:
                if b: hp += b.hp
            return hp
            
        delta_my_hp = sum_hp(old_p0) - sum_hp(new_p0)
        if delta_my_hp > 0:
            reward -= delta_my_hp * 0.005
            
        delta_opp_hp = sum_hp(old_p1) - sum_hp(new_p1)
        if delta_opp_hp > 0:
            reward += delta_opp_hp * 0.005
            
        # Energy Attached
        def sum_energies(p):
            e = 0
            if p.active and p.active[0]: e += len(p.active[0].energies)
            for b in p.bench:
                if b: e += len(b.energies)
            return e
            
        delta_my_energy = sum_energies(new_p0) - sum_energies(old_p0)
        if delta_my_energy > 0:
            reward += delta_my_energy * 0.01
            
        # Opponent Active KO
        old_opp_active = old_p1.active[0] if old_p1.active else None
        new_opp_active = new_p1.active[0] if new_p1.active else None
        if old_opp_active is not None and old_opp_active.hp > 0:
            if new_opp_active is None or new_opp_active.id != old_opp_active.id or new_opp_active.hp <= 0:
                reward += 0.05
                
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
                
                valid_options = len(obs.select.option)
                min_c = obs.select.minCount
                max_c = obs.select.maxCount
                
                if max_c == 0:
                    action_list = []
                elif action >= valid_options:
                    sample_size = min(valid_options, max_c)
                    action_list = random.sample(list(range(valid_options)), sample_size)
                else:
                    action_list = [action]
                    if len(action_list) < min_c:
                        remaining = min_c - 1
                        available = [i for i in range(valid_options) if i != action]
                        if remaining <= len(available):
                            action_list += random.sample(available, remaining)
                    action_list = action_list[:max_c]
                
                action_list = [int(x) for x in action_list]
                self.current_obs_dict = battle_select(action_list)
                
            else:
                # Random selection for opponent
                valid_options = len(obs.select.option)
                max_c = obs.select.maxCount
                sample_size = min(valid_options, max_c)
                action_list = random.sample(list(range(valid_options)), sample_size)
                action_list = [int(x) for x in action_list]
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
                    vec[idx+2] = len(p.active[0].energies)
                idx += 3
                
                # Bench
                for i in range(5):
                    if i < len(p.bench) and p.bench[i] is not None:
                        vec[idx + i*3] = p.bench[i].hp
                        vec[idx + i*3 + 1] = p.bench[i].maxHp
                        vec[idx + i*3 + 2] = len(p.bench[i].energies)
                idx += 15
                
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
