import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

from cg.game import battle_start, battle_select, battle_finish
from cg.api import AreaType, OptionType, SelectContext, to_observation_class, Observation, all_card_data, all_attack, CardType, LogType

RESULT_REASON_PRIZE = 1
RESULT_REASON_DECK_OUT = 2
RESULT_REASON_BENCH_OUT = 3
STOP_ACTION = 999
MAX_ENCODED_OPTIONS = 65

DEFAULT_REWARD_CONFIG = {
    "STEP_PENALTY": 0.0,
    "INVALID_ACTION": -0.05,
    "ATTACK_BONUS": 0.0,
    "END_TURN": 0.0,
    "PRIZE_WIN": 1.0,
    "DECK_OUT_WIN": 1.0,
    "BENCH_OUT_WIN": 1.0,
    "LOSS": -1.0,
    "DECK_OUT_PENALTY": 0.0,
    "SWITCH_PENALTY": 0.0,
    "PRIZE_TAKEN": 0.005,
    "PRIZE_LOST": -0.005,
    "DECK_SHRINK": 0.0,
    "DECK_LOW_COUNT_MULT": 0.0,
    "DAMAGE_TAKEN": -0.00005,
    "DAMAGE_DEALT": 0.00005,
    "ENERGY_ATTACHED": 0.0,
    "ENERGY_ACTIVE_MULT": 0.0,
    "ENERGY_USELESS_MULT": 0.0,
    "TIME_PENALTY": 0.0,
}

def _pokemon_key(pokemon):
    if pokemon is None:
        return None
    serial = getattr(pokemon, "serial", None)
    return serial if serial is not None else getattr(pokemon, "id", None)

def _same_pokemon(a, b):
    a_key = _pokemon_key(a)
    return a_key is not None and a_key == _pokemon_key(b)

def advance_selection(obs, action, pending_indices=None, stop_action=STOP_ACTION):
    """Advance one autoregressive selection step.

    Returns ``(selected_indices, committed, invalid)``.  The engine should only
    receive the selected indices when ``committed`` is true.
    """
    pending = list(pending_indices or [])
    select = getattr(obs, "select", None)
    options = list(getattr(select, "option", None) or [])
    valid_options = min(len(options), stop_action)
    min_count = min(valid_options, max(0, int(getattr(select, "minCount", 0) or 0)))
    max_count = min(valid_options, max(0, int(getattr(select, "maxCount", 0) or 0)))

    if max_count == 0:
        return [], True, False

    try:
        action = int(action)
    except (TypeError, ValueError):
        return pending, False, True

    if action == stop_action:
        return pending, len(pending) >= min_count, len(pending) < min_count

    if action < 0 or action >= valid_options or action in pending:
        return pending, False, True

    pending.append(action)
    return pending, len(pending) >= max_count, False

def _fit_array_to_space(value, space):
    array = np.asarray(value)
    if not isinstance(space, spaces.Box):
        return array

    target_shape = space.shape
    if array.shape[-len(target_shape):] == target_shape:
        return array.astype(space.dtype, copy=False)

    target_size = int(np.prod(target_shape))
    if array.ndim == len(target_shape) + 1:
        batch_shape = array.shape[:-len(target_shape)]
        flat = array.reshape(*batch_shape, -1)
        fitted = np.zeros((*batch_shape, target_size), dtype=space.dtype)
        copy_size = min(flat.shape[-1], target_size)
        fitted[..., :copy_size] = flat[..., :copy_size]
        return fitted.reshape(*batch_shape, *target_shape)

    flat = array.reshape(-1)
    fitted = np.zeros((target_size,), dtype=space.dtype)
    copy_size = min(flat.shape[0], target_size)
    fitted[:copy_size] = flat[:copy_size]
    return fitted.reshape(target_shape)

def _fit_observation_to_model_space(observation, observation_space):
    if isinstance(observation_space, spaces.Dict):
        fitted = {}
        for key, space in observation_space.spaces.items():
            if key in observation:
                fitted[key] = _fit_array_to_space(observation[key], space)
        return fitted

    if isinstance(observation_space, spaces.Box) and isinstance(observation, dict):
        return _fit_array_to_space(observation.get("vector", np.array([], dtype=np.float32)), observation_space)

    return observation

def _terminal_result_reason(obs):
    for log in reversed(getattr(obs, "logs", []) or []):
        try:
            is_result_log = int(getattr(log, "type", -1)) == int(LogType.RESULT)
        except (TypeError, ValueError):
            is_result_log = False
        if is_result_log:
            return getattr(log, "reason", None)
    return None

class PokemonTCGEnv(gym.Env):
    def __init__(
        self,
        my_deck: list[int],
        opponent_deck: list[int],
        opponent_model_path=None,
        reward_config=None,
        sparse_rewards=False,
        opponent_pool=None,
        learner_perspective=0,
        rotate_perspective=False,
    ):
        super().__init__()
        self.learner_perspective = learner_perspective
        self.rotate_perspective = rotate_perspective
        self.my_deck = my_deck
        self.opponent_deck = opponent_deck
        self.opponent_model_path = opponent_model_path
        self.opponent_model = None
        self.opponent_model_cache = {}
        self.opponent_pool = list(opponent_pool or [])
        self.opponent_lstm_state = None
        self.opponent_episode_start = True
        self.pending_selection = []
        self.opponent_pending_selection = []
        self.sparse_rewards = sparse_rewards
        self.reward_config = DEFAULT_REWARD_CONFIG.copy()
        if reward_config:
            for key, value in reward_config.items():
                try:
                    self.reward_config[key] = float(value)
                except (TypeError, ValueError):
                    pass
                    
        if self.sparse_rewards:
            self.reward_config["STEP_PENALTY"] = 0.0
            self.reward_config["ATTACK_BONUS"] = 0.0
            self.reward_config["END_TURN"] = 0.0
            self.reward_config["SWITCH_PENALTY"] = 0.0
            self.reward_config["PRIZE_TAKEN"] = 0.0
            self.reward_config["PRIZE_LOST"] = 0.0
            self.reward_config["DAMAGE_DEALT"] = 0.0
            self.reward_config["DAMAGE_TAKEN"] = 0.0
            self.reward_config["ENERGY_ATTACHED"] = 0.0
            self.reward_config["ENERGY_ACTIVE_MULT"] = 0.0
            self.reward_config["ENERGY_USELESS_MULT"] = 0.0
            self.reward_config["DECK_SHRINK"] = 0.0
            self.reward_config["TIME_PENALTY"] = 0.0
        
        # Determine valid energy types from my_deck (Card ID 1..8 maps 1-to-1 to EnergyType 1..8)
        self.valid_energy_types = {0} # Colorless (0) is always valid
        for card_id in my_deck:
            if 1 <= card_id <= 8:
                self.valid_energy_types.add(card_id)
                
        # Build map of cardId -> exact attack costs dynamically
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
            self.pokemon_attack_costs = {}
        
        # Action space: a discrete selection of an option from the available ones.
        self.max_options = STOP_ACTION + 1
        self.action_space = spaces.Discrete(self.max_options)
        
        # Observation space: 1500-dim vector + action mask + aux target
        self.vector_dim = 1500
        self.aux_dim = 2000
        self.observation_space = spaces.Dict({
            "vector": spaces.Box(low=-1000, high=1000, shape=(self.vector_dim,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.max_options,), dtype=np.int8),
            "aux_target": spaces.Box(low=0, high=1, shape=(self.aux_dim,), dtype=np.float32)
        })
        
        self.reward_keys = [
            "step_penalty", "invalid_action", "attack_bonus", "end_turn",
            "prize_win", "bench_out_win", "deck_out_win", "loss", "deck_out_penalty",
            "prize_taken", "prize_lost", "deck_shrink", "damage_taken", "damage_dealt",
            "energy_active_useful", "energy_active_useless", "energy_bench",
            "switch_penalty", "time_penalty"
        ]
        self.current_obs_dict = None
        self.reward_stats = {k: 0.0 for k in self.reward_keys}

    def _reward(self, key):
        return self.reward_config.get(key, DEFAULT_REWARD_CONFIG[key])

    def _add_reward(self, name, value):
        self.reward_stats[name] += value
        return value

    def _deck_shrink_reward(self, old_count, new_count):
        cards_drawn = max(0, int(old_count) - int(new_count))
        if cards_drawn == 0:
            return 0.0

        base_penalty = -abs(self._reward("DECK_SHRINK"))
        low_count_mult = max(0.0, self._reward("DECK_LOW_COUNT_MULT"))
        danger_start = 30.0

        reward = 0.0
        for cards_left in range(int(new_count), int(old_count)):
            low_deck_pressure = max(0.0, (danger_start - cards_left) / danger_start)
            reward += base_penalty * (1.0 + low_deck_pressure * low_count_mult)
        return self._add_reward("deck_shrink", reward)

    def _switch_reward(self, old_obs, new_obs):
        if not old_obs.current or not new_obs.current:
            return 0.0

        old_active = old_obs.current.players[0].active[0] if old_obs.current.players[0].active else None
        new_active = new_obs.current.players[0].active[0] if new_obs.current.players[0].active else None
        if old_active is None or new_active is None or _same_pokemon(old_active, new_active):
            return 0.0
        if old_active.hp <= 0:
            return 0.0
        return self._add_reward("switch_penalty", self._reward("SWITCH_PENALTY"))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reward_stats = {k: 0.0 for k in self.reward_keys}

        if self.opponent_pool:
            weights = np.asarray(
                [max(0.0, float(entry.get("weight", 1.0))) for entry in self.opponent_pool],
                dtype=np.float64,
            )
            if weights.sum() <= 0:
                weights = np.ones(len(self.opponent_pool), dtype=np.float64)
            weights /= weights.sum()
            pool_index = int(self.np_random.choice(len(self.opponent_pool), p=weights))
            opponent = self.opponent_pool[pool_index]
            self.opponent_deck = list(opponent["deck"])
            self.opponent_model_path = opponent.get("model_path")
            self.opponent_model = None

        # Lazy load the model in the worker process to avoid pickling issues
        if self.opponent_model_path is not None:
            cached_model = self.opponent_model_cache.get(self.opponent_model_path)
            if cached_model is not None:
                self.opponent_model = cached_model
        if self.opponent_model_path is not None and self.opponent_model is None:
            # We import here to avoid circular imports if necessary
            from stable_baselines3 import PPO
            try:
                from src.custom_ppo import CustomPPO
                self.opponent_model = CustomPPO.load(self.opponent_model_path)
            except Exception as e:
                try:
                    self.opponent_model = PPO.load(self.opponent_model_path)
                except Exception as fallback_e:
                    raise RuntimeError(
                        f"Failed to load opponent model {self.opponent_model_path}: "
                        f"CustomPPO={e}; PPO={fallback_e}"
                    ) from fallback_e
            self.opponent_model_cache[self.opponent_model_path] = self.opponent_model
        
        if self.current_obs_dict is not None:
            battle_finish()

        # Perspective rotation: randomly swap who is Player 0 / Player 1
        if self.rotate_perspective:
            self.learner_perspective = int(self.np_random.integers(0, 2))

        if self.learner_perspective == 0:
            p0_deck, p1_deck = self.my_deck, self.opponent_deck
        else:
            p0_deck, p1_deck = self.opponent_deck, self.my_deck

        self.current_obs_dict, _ = battle_start(p0_deck, p1_deck)
        self.opponent_lstm_state = None
        self.opponent_episode_start = True
        self.pending_selection = []
        self.opponent_pending_selection = []
        self._process_opponent_turns()
        
        return self._get_obs(perspective=self.learner_perspective), self._get_info()

    def step(self, action):
        action = int(action) # Convert numpy scalar to python int
        old_obs = to_observation_class(self.current_obs_dict)

        valid_options = len(old_obs.select.option) if old_obs.select and old_obs.select.option else 0
        selected, committed, invalid = advance_selection(old_obs, action, self.pending_selection)
        self.pending_selection = selected

        if not committed:
            if invalid and len(self.pending_selection) > 0:
                # Auto-commit if minCount is already satisfied to avoid
                # infinite invalid-action loops (old models don't know STOP_ACTION).
                select = getattr(old_obs, "select", None)
                min_count = max(0, int(getattr(select, "minCount", 0) or 0))
                if len(self.pending_selection) >= min_count:
                    committed = True  # fall through to commit path below
                    invalid = True    # still penalise the invalid attempt
            if not committed:
                reward = self._add_reward("invalid_action", self._reward("INVALID_ACTION")) if invalid else 0.0
                return self._get_obs(perspective=self.learner_perspective), reward, False, False, self._get_info()

        step_reward = self._add_reward("step_penalty", self._reward("STEP_PENALTY"))
        if invalid:
            step_reward += self._add_reward("invalid_action", self._reward("INVALID_ACTION"))

        if valid_options > 0 and 0 <= action < valid_options:
            chosen_option = old_obs.select.option[action]
            if chosen_option.type == getattr(OptionType, 'ATTACK', 11):
                step_reward += self._add_reward("attack_bonus", self._reward("ATTACK_BONUS"))
            elif chosen_option.type == getattr(OptionType, 'END', 14):
                step_reward += self._add_reward("end_turn", self._reward("END_TURN"))
                
        self.current_obs_dict = battle_select(self.pending_selection)
        self.pending_selection = []
        after_action_obs = to_observation_class(self.current_obs_dict)
        step_reward += self._switch_reward(old_obs, after_action_obs)
        
        done = self._process_opponent_turns()
        
        new_obs = to_observation_class(self.current_obs_dict)
        step_reward += self._compute_dense_reward(old_obs, new_obs, done)
        
        return self._get_obs(perspective=self.learner_perspective), step_reward, done, False, self._get_info()
        
    def _compute_dense_reward(self, old_obs, new_obs, done):
        if done:
            if new_obs.current and new_obs.current.result != -1:
                result_reason = _terminal_result_reason(new_obs)
                # Ensure we properly evaluate if the learner (self.learner_perspective) won
                did_we_win = (new_obs.current.result == self.learner_perspective)
                did_we_lose = (new_obs.current.result == 1 - self.learner_perspective)
                
                if did_we_win:
                    if result_reason == RESULT_REASON_BENCH_OUT:
                        return self._add_reward("bench_out_win", self._reward("BENCH_OUT_WIN"))
                    elif result_reason == RESULT_REASON_DECK_OUT or getattr(new_obs.current.players[1 - self.learner_perspective], 'deckCount', 1) == 0:
                        return self._add_reward("deck_out_win", self._reward("DECK_OUT_WIN"))
                    else:
                        return self._add_reward("prize_win", self._reward("PRIZE_WIN"))
                elif did_we_lose:
                    loss_reward = self._add_reward("loss", self._reward("LOSS"))
                    if getattr(new_obs.current.players[self.learner_perspective], 'deckCount', 1) == 0:
                        loss_reward += self._add_reward("deck_out_penalty", self._reward("DECK_OUT_PENALTY"))
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
            reward += self._add_reward("prize_taken", delta_prize_0 * self._reward("PRIZE_TAKEN"))
            
        delta_prize_1 = len(old_p1.prize) - len(new_p1.prize)
        if delta_prize_1 > 0:
            reward += self._add_reward("prize_lost", delta_prize_1 * self._reward("PRIZE_LOST"))
            
        # Deck shrink penalty
        delta_deck_0 = old_p0.deckCount - new_p0.deckCount
        if delta_deck_0 > 0:
            reward += self._deck_shrink_reward(old_p0.deckCount, new_p0.deckCount)

        # Damage Deltas (BOOSTED for Aggro)
        def sum_hp(p):
            hp = 0
            if p.active and p.active[0]: hp += p.active[0].hp
            for b in p.bench:
                if b: hp += b.hp
            return hp
            
        delta_my_hp = sum_hp(old_p0) - sum_hp(new_p0)
        if delta_my_hp > 0:
            reward += self._add_reward("damage_taken", delta_my_hp * self._reward("DAMAGE_TAKEN"))
            
        delta_opp_hp = sum_hp(old_p1) - sum_hp(new_p1)
        if delta_opp_hp > 0:
            reward += self._add_reward("damage_dealt", delta_opp_hp * self._reward("DAMAGE_DEALT"))
            
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
            useful_energy_attached = False
            if old_p0.active and old_p0.active[0] and new_p0.active and new_p0.active[0]:
                if _same_pokemon(old_p0.active[0], new_p0.active[0]):
                    old_e_count = len(old_p0.active[0].energies)
                    new_e_count = len(new_p0.active[0].energies)
                    if new_e_count > old_e_count:
                        active_attached = True
                        old_e_list = [int(e) if not hasattr(e, 'value') else e.value for e in old_p0.active[0].energies]
                        new_e_list = [int(e) if not hasattr(e, 'value') else e.value for e in new_p0.active[0].energies]
                        
                        active_pokemon_id = new_p0.active[0].id
                        costs = self.pokemon_attack_costs.get(active_pokemon_id, [])
                        
                        def calc_deficit(attached, cost):
                            attached_counts = {}
                            for e in attached:
                                attached_counts[e] = attached_counts.get(e, 0) + 1
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
                reward += self._add_reward("energy_active_useful", delta_total_energy * self._reward("ENERGY_ATTACHED") * self._reward("ENERGY_ACTIVE_MULT"))
            elif active_attached and not useful_energy_attached:
                reward += self._add_reward("energy_active_useless", delta_total_energy * self._reward("ENERGY_ATTACHED") * self._reward("ENERGY_USELESS_MULT"))
            else:
                # User requested to remove bench attachment reward
                self._add_reward("energy_bench", 0.0)
            
        # Step Penalty (Time penalty)
        reward += self._add_reward("time_penalty", self._reward("TIME_PENALTY"))
                
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
            if obs.current is not None and obs.current.yourIndex == self.learner_perspective:
                break
                
            if self.opponent_model is not None:
                opponent_obs = self._get_obs(
                    perspective=1 - self.learner_perspective,
                    pending_selection=self.opponent_pending_selection,
                )
                
                # Expand dims to match what stable-baselines expects
                for k in opponent_obs:
                    opponent_obs[k] = np.expand_dims(opponent_obs[k], axis=0)

                opponent_space = getattr(self.opponent_model, "observation_space", None)
                if opponent_space is not None:
                    opponent_obs = _fit_observation_to_model_space(opponent_obs, opponent_space)
                
                episode_start = np.array([self.opponent_episode_start], dtype=bool)
                action, self.opponent_lstm_state = self.opponent_model.predict(
                    opponent_obs,
                    state=self.opponent_lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                self.opponent_episode_start = False
                action = int(np.asarray(action).item())

            else:
                fallback_obs = self._get_obs(
                    perspective=1 - self.learner_perspective,
                    pending_selection=self.opponent_pending_selection,
                )
                legal_actions = np.flatnonzero(fallback_obs["action_mask"])
                if len(legal_actions) == 0:
                    action = STOP_ACTION
                else:
                    action = int(self.np_random.choice(legal_actions))

            selected, committed, invalid = advance_selection(
                obs,
                action,
                self.opponent_pending_selection,
            )
            if invalid:
                legal_options = [
                    index for index in range(min(len(obs.select.option or []), STOP_ACTION))
                    if index not in self.opponent_pending_selection
                ]
                if legal_options:
                    import random
                    selected_fallback_action = random.choice(legal_options)
                    selected, committed, _ = advance_selection(
                        obs,
                        selected_fallback_action,
                        self.opponent_pending_selection,
                    )
                elif len(self.opponent_pending_selection) >= max(0, obs.select.minCount):
                    selected, committed = list(self.opponent_pending_selection), True

            self.opponent_pending_selection = selected
            if committed:
                self.current_obs_dict = battle_select(self.opponent_pending_selection)
                self.opponent_pending_selection = []
            
        return done

    def _get_obs(self, perspective=0, pending_selection=None):
        obs = to_observation_class(self.current_obs_dict)
        if pending_selection is None:
            pending_selection = self.pending_selection if perspective == 0 else self.opponent_pending_selection
        pending_selection = list(pending_selection or [])
        mask = np.zeros(self.max_options, dtype=np.int8)
        if obs.select and obs.select.option:
            num_opts = min(len(obs.select.option), STOP_ACTION)
            mask[:num_opts] = 1
            for selected_index in pending_selection:
                if 0 <= selected_index < num_opts:
                    mask[selected_index] = 0

            min_count = min(num_opts, max(0, int(obs.select.minCount or 0)))
            max_count = min(num_opts, max(0, int(obs.select.maxCount or 0)))
            if len(pending_selection) >= min_count and (min_count == 0 or max_count > 1):
                mask[STOP_ACTION] = 1
            
        vec = np.zeros((self.vector_dim,), dtype=np.float32)
        if obs.current is not None:
            state = obs.current
            # Continuous & Boolean Stats (0-299)
            vec[0] = state.turn
            vec[1] = float(abs(state.yourIndex - perspective))
            # 0 means "we go first", 1 means "the opponent goes first".
            # This preserves the old player-0 encoding while making player 1 equivalent.
            vec[2] = float(state.firstPlayer != perspective)
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
                    log_player = log.playerIndex if log.playerIndex is not None else perspective
                    vec[256 + i*4] = float(abs(int(log_player) - perspective))
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

            # Autoregressive selection state. Existing action/output dimensions stay unchanged.
            vec[1490] = float(len(pending_selection))
            vec[1491] = float(mask[STOP_ACTION])
            for i, selected_index in enumerate(pending_selection[:8]):
                vec[1492 + i] = float(selected_index + 1)

            # Options 0-49 use the legacy range. Options 50-64 use the previously
            # unused 650-799 range so existing checkpoints remain loadable.
            if obs.select and obs.select.option:
                opt_len = min(len(obs.select.option), MAX_ENCODED_OPTIONS)
                for i in range(opt_len):
                    opt = obs.select.option[i]
                    base = 800 + i*10 if i < 50 else 650 + (i - 50)*10
                    vec[base] = float(opt.type)
                    vec[base+1] = float(opt.cardId if opt.cardId is not None else 0)
                    vec[base+2] = float(opt.area if opt.area is not None else 0)
                    vec[base+3] = float(opt.index if opt.index is not None else 0)
                    vec[base+4] = float(opt.inPlayArea if opt.inPlayArea is not None else 0)
                    vec[base+5] = float(opt.inPlayIndex if opt.inPlayIndex is not None else 0)
                    vec[base+6] = float(opt.attackId if opt.attackId is not None else 0)
                    vec[base+7] = float(opt.specialConditionType if opt.specialConditionType is not None else 0)
                    option_player = opt.playerIndex if opt.playerIndex is not None else perspective
                    vec[base+8] = float(abs(int(option_player) - perspective))
                    vec[base+9] = float(opt.number if opt.number is not None else (opt.count if hasattr(opt, 'count') and opt.count is not None else 0))
                
        # Auxiliary Target: Predict the hidden cards of the opponent
        hidden_deck = self.opponent_deck if perspective == 0 else self.my_deck
        hidden_player_index = 1 - perspective
        hidden_counts = {}
        for card_id in hidden_deck:
            hidden_counts[card_id] = hidden_counts.get(card_id, 0) + 1
            
        if obs.current is not None:
            p1 = obs.current.players[hidden_player_index]
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
                info['winner'] = int(obs.current.result)
                result_reason = _terminal_result_reason(obs)
                if result_reason == RESULT_REASON_PRIZE:
                    info['win_reason'] = 'prize'
                elif result_reason == RESULT_REASON_DECK_OUT:
                    info['win_reason'] = 'deckout'
                elif result_reason == RESULT_REASON_BENCH_OUT:
                    info['win_reason'] = 'benchout'
                elif len(p0.prize) == 0 or len(p1.prize) == 0:
                    info['win_reason'] = 'prize'
                elif p0.deckCount == 0 or p1.deckCount == 0:
                    info['win_reason'] = 'deckout'
                else:
                    info['win_reason'] = 'other'
        except Exception:
            pass
        info['reward_breakdown'] = dict(self.reward_stats)
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
