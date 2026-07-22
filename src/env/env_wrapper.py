import gymnasium as gym
from gymnasium import spaces
import ctypes
import numpy as np
import os
import json
import random

from src.cg.game import battle_start, battle_select, battle_finish
from src.cg.sim import Battle, V6ObservationBuffer, lib
from src.cg.api import AreaType, OptionType, SelectContext, to_observation_class, Observation, all_card_data, all_attack, CardType, LogType

RESULT_REASON_PRIZE = 1
RESULT_REASON_DECK_OUT = 2
RESULT_REASON_BENCH_OUT = 3
MAX_ENCODED_OPTIONS = 65
V6_ACTION_SPACE_SIZE = MAX_ENCODED_OPTIONS + 1
V6_STOP_ACTION = MAX_ENCODED_OPTIONS
LEGACY_ACTION_SPACE_SIZE = 1000
LEGACY_STOP_ACTION = LEGACY_ACTION_SPACE_SIZE - 1
# Backwards-compatible import for legacy callers. New code should use the
# environment's ``stop_action`` because V6 uses 65 instead of 999.
STOP_ACTION = LEGACY_STOP_ACTION
MAX_CARD_ID = 1999
MAX_ATTACK_ID = 1999
ENTITY_SLOTS = 12
ENTITY_FEATURE_DIM = 36
MAX_ENTITY_PRE_EVOLUTIONS = 3
MAX_ENTITY_ENERGY_CARDS = 8
HAND_CARD_SLOTS = 24
DISCARD_CARD_SLOTS = 30
REVEALED_CARD_SLOTS = 120
CONTEXT_CARD_SLOTS = 2
LOG_CARD_SLOTS = 10
OPTION_FEATURE_DIM = 21
PRIZE_CARD_SLOTS = 6
DECK_LIST_SLOTS = 60
SEARCH_CARD_SLOTS = 60
LOOKING_CARD_SLOTS = 60
MAX_HIDDEN_CARD_COUNT = 60


def encode_hidden_card_count(count):
    """Map an absolute hidden-card count to [0, 1] without losing duplicates."""
    clipped = min(MAX_HIDDEN_CARD_COUNT, max(0, int(count)))
    return np.float32(np.log1p(clipped) / np.log1p(MAX_HIDDEN_CARD_COUNT))


def bound_entity_energy_features(entity_features):
    """Clip and bound entity energy features for observation parity tests."""
    out = np.copy(entity_features)
    out[..., 8:21] = np.clip(out[..., 8:21], 0.0, 10.0)
    return out


def encode_energy_count(count):
    """Map energy count to [0, 1]."""
    clipped = min(10, max(0, int(count)))
    return np.float32(clipped / 10.0)

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

def _enum_value(value, enum_class, default=0):
    """Convert API enum values and replay-friendly enum names to integers."""
    if value is None:
        return default
    raw_value = getattr(value, "value", value)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        pass

    if isinstance(raw_value, str):
        normalized = raw_value.replace("-", "_").replace(" ", "_").upper()
        for member in enum_class:
            if member.name == normalized:
                return int(member.value)
    return default

def _bounded_id(value, maximum):
    raw_value = getattr(value, "value", value)
    try:
        identifier = int(raw_value)
    except (TypeError, ValueError):
        return 0
    return identifier if 0 < identifier <= maximum else 0

def _cid(c):
    if c is None:
        return 0
    if isinstance(c, dict):
        return _bounded_id(c.get("id", c.get("cardId", 0)), MAX_CARD_ID)
    return _bounded_id(getattr(c, "id", getattr(c, "cardId", c)), MAX_CARD_ID)

def _energy_value(value):
    raw_value = getattr(value, "value", value)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return 0

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
            else:
                batch_shape = ()
                if "vector" in observation and hasattr(observation["vector"], "shape") and len(observation["vector"].shape) > 1:
                    batch_shape = observation["vector"].shape[:-1]
                fitted[key] = np.zeros((*batch_shape, *space.shape), dtype=space.dtype)
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
        action_space_size=V6_ACTION_SPACE_SIZE,
        structured_v2=True,
        zone_aux_targets=False,
        enable_lookahead_teacher=False,
        teacher_sample_rate=0.50,
    ):
        super().__init__()
        self.zone_aux_targets = zone_aux_targets
        self.structured_v2 = structured_v2
        self.enable_lookahead_teacher = enable_lookahead_teacher
        self.teacher_sample_rate = float(teacher_sample_rate)
        self.lookahead_teacher = None
        if self.enable_lookahead_teacher:
            try:
                from src.training.lookahead_teacher import LookaheadTeacher, LookaheadConfig
                self.lookahead_teacher = LookaheadTeacher(config=LookaheadConfig(node_budget=96, max_depth=5))
            except Exception:
                self.lookahead_teacher = None
        self.learner_perspective = learner_perspective
        self.rotate_perspective = rotate_perspective
        if action_space_size not in {LEGACY_ACTION_SPACE_SIZE, V6_ACTION_SPACE_SIZE}:
            raise ValueError(
                f"Unsupported action space size {action_space_size}; expected "
                f"{V6_ACTION_SPACE_SIZE} (V6) or {LEGACY_ACTION_SPACE_SIZE} (legacy)"
            )
        self.max_options = int(action_space_size)
        self.stop_action = self.max_options - 1
        self.policy_version = "v6" if self.max_options == V6_ACTION_SPACE_SIZE else "v5"
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
        self.max_option_count_seen = 0
        self.option_overflow_count = 0
        self.engine_error_count = 0
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
                
        # Build reusable card/attack lookups for reward logic and Observation V2.
        try:
            self.card_data_by_id = {card.cardId: card for card in all_card_data()}
            self.attack_data_by_id = {attack.attackId: attack for attack in all_attack()}
            attacks_map = {attack_id: attack.energies for attack_id, attack in self.attack_data_by_id.items()}
            self.pokemon_attack_costs = {}
            for card in self.card_data_by_id.values():
                if card.cardType == CardType.POKEMON:
                    costs = []
                    for attack_id in card.attacks:
                        energies = attacks_map.get(attack_id, [])
                        cost = [int(e) if not hasattr(e, 'value') else e.value for e in energies]
                        costs.append(cost)
                    self.pokemon_attack_costs[card.cardId] = costs
        except Exception as e:
            print("Failed to build dynamic pokemon energy mapping:", e)
            self.card_data_by_id = {}
            self.attack_data_by_id = {}
            self.pokemon_attack_costs = {}
        
        # Action space: a discrete selection of an option from the available ones.
        self.action_space = spaces.Discrete(self.max_options)
        
        # Observation V2 keeps the legacy vector for old opponent checkpoints,
        # while exposing categorical cards, structured field entities and legal
        # options to new policies.
        self.vector_dim = 1500
        self.aux_dim = 2000
        self.structured_v2 = structured_v2
        
        base_spaces = {
            "vector": spaces.Box(low=-2000, high=2000, shape=(self.vector_dim,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.max_options,), dtype=np.int8),
            "aux_target": spaces.Box(low=0, high=1, shape=(self.aux_dim,), dtype=np.float32),
            "teacher_action": spaces.Box(low=-1, high=self.max_options, shape=(1,), dtype=np.int32),
        }
        
        if self.structured_v2:
            base_spaces.update({
                "entity_ids": spaces.Box(low=0, high=MAX_CARD_ID, shape=(ENTITY_SLOTS,), dtype=np.int32),
                "entity_features": spaces.Box(
                    low=-1, high=10, shape=(ENTITY_SLOTS, ENTITY_FEATURE_DIM), dtype=np.float32
                ),
                "entity_tool_ids": spaces.Box(low=0, high=MAX_CARD_ID, shape=(ENTITY_SLOTS,), dtype=np.int32),
                "entity_pre_evolution_ids": spaces.Box(
                    low=0,
                    high=MAX_CARD_ID,
                    shape=(ENTITY_SLOTS, MAX_ENTITY_PRE_EVOLUTIONS),
                    dtype=np.int32,
                ),
                "entity_energy_card_ids": spaces.Box(
                    low=0,
                    high=MAX_CARD_ID,
                    shape=(ENTITY_SLOTS, MAX_ENTITY_ENERGY_CARDS),
                    dtype=np.int32,
                ),
                "hand_ids": spaces.Box(low=0, high=MAX_CARD_ID, shape=(HAND_CARD_SLOTS,), dtype=np.int32),
                "discard_ids": spaces.Box(
                    low=0, high=MAX_CARD_ID, shape=(2, DISCARD_CARD_SLOTS), dtype=np.int32
                ),
                "revealed_ids": spaces.Box(
                    low=0, high=MAX_CARD_ID, shape=(REVEALED_CARD_SLOTS,), dtype=np.int32
                ),
                "prize_ids": spaces.Box(
                    low=0, high=MAX_CARD_ID, shape=(2, PRIZE_CARD_SLOTS), dtype=np.int32
                ),
                "search_ids": spaces.Box(
                    low=0, high=MAX_CARD_ID, shape=(SEARCH_CARD_SLOTS,), dtype=np.int32
                ),
                "looking_ids": spaces.Box(
                    low=0, high=MAX_CARD_ID, shape=(LOOKING_CARD_SLOTS,), dtype=np.int32
                ),
                "own_deck_ids": spaces.Box(
                    low=0, high=MAX_CARD_ID, shape=(DECK_LIST_SLOTS,), dtype=np.int32
                ),
                "context_card_ids": spaces.Box(
                    low=0, high=MAX_CARD_ID, shape=(CONTEXT_CARD_SLOTS + 1,), dtype=np.int32
                ),
                "log_card_ids": spaces.Box(low=0, high=MAX_CARD_ID, shape=(LOG_CARD_SLOTS,), dtype=np.int32),
                "option_card_ids": spaces.Box(
                    low=0, high=MAX_CARD_ID, shape=(MAX_ENCODED_OPTIONS,), dtype=np.int32
                ),
                "option_attack_ids": spaces.Box(
                    low=0, high=MAX_ATTACK_ID, shape=(MAX_ENCODED_OPTIONS,), dtype=np.int32
                ),
                "option_types": spaces.Box(
                    low=0, high=len(OptionType), shape=(MAX_ENCODED_OPTIONS,), dtype=np.int32
                ),
                "option_areas": spaces.Box(
                    low=0, high=len(AreaType), shape=(MAX_ENCODED_OPTIONS,), dtype=np.int32
                ),
                "option_features": spaces.Box(
                    low=-1,
                    high=10,
                    shape=(MAX_ENCODED_OPTIONS, OPTION_FEATURE_DIM),
                    dtype=np.float32,
                ),
            })

        if self.zone_aux_targets:
            base_spaces.update({
                "aux_own_deck_ids": spaces.Box(low=0, high=MAX_CARD_ID, shape=(60,), dtype=np.int32),
                "aux_own_prize_ids": spaces.Box(low=0, high=MAX_CARD_ID, shape=(6,), dtype=np.int32),
                "aux_opponent_hand_ids": spaces.Box(low=0, high=MAX_CARD_ID, shape=(60,), dtype=np.int32),
                "aux_opponent_deck_ids": spaces.Box(low=0, high=MAX_CARD_ID, shape=(60,), dtype=np.int32),
                "aux_opponent_prize_ids": spaces.Box(low=0, high=MAX_CARD_ID, shape=(6,), dtype=np.int32),
            })
            
        self.observation_space = spaces.Dict(base_spaces)
        
        self.reward_keys = [
            "step_penalty", "invalid_action", "attack_bonus", "end_turn",
            "prize_win", "bench_out_win", "deck_out_win", "loss", "deck_out_penalty",
            "prize_taken", "prize_lost", "deck_shrink", "damage_taken", "damage_dealt",
            "energy_active_useful", "energy_active_useless",
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

        old_player = old_obs.current.players[self.learner_perspective]
        new_player = new_obs.current.players[self.learner_perspective]
        old_active = old_player.active[0] if old_player.active else None
        new_active = new_player.active[0] if new_player.active else None
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
            from src.agents.bot_loader import load_bot

            self.opponent_model = load_bot(self.opponent_model_path)
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

    @staticmethod
    def _is_native_engine_error(error):
        message = str(error)
        return (
            "inside the C++ engine" in message
            or "GetV6Observation failed" in message
        )

    def _truncate_native_engine_error(self, error):
        """Discard one corrupted native game without killing its vector worker."""
        self.engine_error_count += 1
        try:
            terminal_observation = self._get_obs_python(
                perspective=self.learner_perspective,
                pending_selection=self.pending_selection,
            )
        except Exception:
            terminal_observation = {
                key: np.zeros(space.shape, dtype=space.dtype)
                for key, space in self.observation_space.spaces.items()
            }

        info = self._get_info()
        info["engine_error"] = str(error)
        info["engine_error_count"] = self.engine_error_count
        try:
            if self.current_obs_dict is not None:
                battle_finish()
        finally:
            self.current_obs_dict = None
            self.pending_selection = []
            self.opponent_pending_selection = []
            self.opponent_lstm_state = None
            self.opponent_episode_start = True
        return terminal_observation, 0.0, False, True, info

    def step(self, action):
        try:
            return self._step_impl(action)
        except RuntimeError as error:
            if not self._is_native_engine_error(error):
                raise
            return self._truncate_native_engine_error(error)

    def _step_impl(self, action):
        action = int(action) # Convert numpy scalar to python int
        old_obs = to_observation_class(self.current_obs_dict)

        valid_options = len(old_obs.select.option) if old_obs.select and old_obs.select.option else 0
        selected, committed, invalid = advance_selection(
            old_obs, action, self.pending_selection, stop_action=self.stop_action
        )
        self.pending_selection = selected

        if not committed:
            if invalid:
                legal_options = [
                    index for index in range(min(valid_options, self.stop_action))
                    if index not in self.pending_selection
                ]
                if legal_options:
                    # Old policies can emit actions that are invalid for the
                    # mirrored player perspective. Advance with a legal option
                    # instead of returning the identical state forever.
                    action = legal_options[0]
                    selected, committed, _ = advance_selection(
                        old_obs,
                        action,
                        self.pending_selection,
                        stop_action=self.stop_action,
                    )
                    self.pending_selection = selected

            if invalid and not committed and len(self.pending_selection) > 0:
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

        if valid_options > 0 and 0 <= action < min(valid_options, self.stop_action):
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
        old_me = old_obs.current.players[self.learner_perspective]
        old_opponent = old_obs.current.players[1 - self.learner_perspective]
        new_me = new_obs.current.players[self.learner_perspective]
        new_opponent = new_obs.current.players[1 - self.learner_perspective]
        
        # Prize Card Taken/Lost
        delta_my_prize = len(old_me.prize) - len(new_me.prize)
        if delta_my_prize > 0:
            reward += self._add_reward("prize_taken", delta_my_prize * self._reward("PRIZE_TAKEN"))
            
        delta_opponent_prize = len(old_opponent.prize) - len(new_opponent.prize)
        if delta_opponent_prize > 0:
            reward += self._add_reward("prize_lost", delta_opponent_prize * self._reward("PRIZE_LOST"))
            
        # Deck shrink penalty
        delta_my_deck = old_me.deckCount - new_me.deckCount
        if delta_my_deck > 0:
            reward += self._deck_shrink_reward(old_me.deckCount, new_me.deckCount)

        # Damage Deltas (BOOSTED for Aggro)
        def sum_hp(p):
            hp = 0
            if p.active and p.active[0]: hp += p.active[0].hp
            for b in p.bench:
                if b: hp += b.hp
            return hp
            
        delta_my_hp = sum_hp(old_me) - sum_hp(new_me)
        if delta_my_hp > 0:
            reward += self._add_reward("damage_taken", delta_my_hp * self._reward("DAMAGE_TAKEN"))
            
        delta_opp_hp = sum_hp(old_opponent) - sum_hp(new_opponent)
        if delta_opp_hp > 0:
            reward += self._add_reward("damage_dealt", delta_opp_hp * self._reward("DAMAGE_DEALT"))
            
        # Energy Attached
        def sum_energies(p):
            e = 0
            if p.active and p.active[0]: e += len(p.active[0].energies)
            for b in p.bench:
                if b: e += len(b.energies)
            return e
            
        delta_total_energy = sum_energies(new_me) - sum_energies(old_me)
        if delta_total_energy > 0:
            active_attached = False
            useful_energy_attached = False
            if old_me.active and old_me.active[0] and new_me.active and new_me.active[0]:
                if _same_pokemon(old_me.active[0], new_me.active[0]):
                    old_e_count = len(old_me.active[0].energies)
                    new_e_count = len(new_me.active[0].energies)
                    if new_e_count > old_e_count:
                        active_attached = True
                        old_e_list = [int(e) if not hasattr(e, 'value') else e.value for e in old_me.active[0].energies]
                        new_e_list = [int(e) if not hasattr(e, 'value') else e.value for e in new_me.active[0].energies]
                        
                        active_pokemon_id = new_me.active[0].id
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
                opponent_action_space = getattr(self.opponent_model, "action_space", None)
                opponent_action_size = int(
                    getattr(opponent_action_space, "n", self.max_options)
                )
                if opponent_action_size not in {LEGACY_ACTION_SPACE_SIZE, V6_ACTION_SPACE_SIZE}:
                    opponent_action_size = self.max_options
                opponent_stop_action = opponent_action_size - 1
                opponent_space = getattr(self.opponent_model, "observation_space", None)
                opponent_uses_structured = getattr(self.opponent_model, "is_structured", False)
                if not opponent_uses_structured and opponent_space is not None and isinstance(opponent_space, spaces.Dict):
                    required_v2_keys = {"entity_ids", "entity_features", "option_card_ids", "option_attack_ids", "option_features"}
                    opponent_uses_structured = required_v2_keys.issubset(opponent_space.spaces)

                opponent_obs = self._get_obs(
                    perspective=1 - self.learner_perspective,
                    pending_selection=self.opponent_pending_selection,
                    action_space_size=opponent_action_size,
                    force_structured=opponent_uses_structured,
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
                opponent_stop_action = self.stop_action
                fallback_obs = self._get_obs(
                    perspective=1 - self.learner_perspective,
                    pending_selection=self.opponent_pending_selection,
                )
                legal_actions = np.flatnonzero(fallback_obs["action_mask"])
                if len(legal_actions) == 0:
                    action = opponent_stop_action
                else:
                    action = int(self.np_random.choice(legal_actions))

            selected, committed, invalid = advance_selection(
                obs,
                action,
                self.opponent_pending_selection,
                stop_action=opponent_stop_action,
            )
            if invalid:
                legal_options = [
                    index for index in range(min(len(obs.select.option or []), opponent_stop_action))
                    if index not in self.opponent_pending_selection
                ]
                if legal_options:
                    import random
                    selected_fallback_action = random.choice(legal_options)
                    selected, committed, _ = advance_selection(
                        obs,
                        selected_fallback_action,
                        self.opponent_pending_selection,
                        stop_action=opponent_stop_action,
                    )
                elif len(self.opponent_pending_selection) >= max(0, obs.select.minCount):
                    selected, committed = list(self.opponent_pending_selection), True

            self.opponent_pending_selection = selected
            if committed:
                self.current_obs_dict = battle_select(self.opponent_pending_selection)
                self.opponent_pending_selection = []
            
        return done

    @staticmethod
    def _attack_deficit(attached_energies, attack_cost):
        attached_counts = {}
        for energy in attached_energies:
            energy_type = _energy_value(energy)
            attached_counts[energy_type] = attached_counts.get(energy_type, 0) + 1

        missing_specific = 0
        colorless_cost = 0
        for required_energy in attack_cost:
            required_type = _energy_value(required_energy)
            if required_type == 0:
                colorless_cost += 1
            elif attached_counts.get(required_type, 0) > 0:
                attached_counts[required_type] -= 1
            else:
                missing_specific += 1

        remaining_energy = sum(attached_counts.values())
        return missing_specific + max(0, colorless_cost - remaining_energy)

    def _resolve_option_card_id(self, obs, option, perspective):
        """Resolve options that only reference ``area + index`` to a card ID."""
        direct_id = _bounded_id(getattr(option, "cardId", None), MAX_CARD_ID)
        if direct_id:
            return direct_id
        if not obs.current:
            return 0

        option_type = _enum_value(getattr(option, "type", None), OptionType)
        area = _enum_value(getattr(option, "area", None), AreaType)
        try:
            index = int(getattr(option, "index", 0) or 0)
        except (TypeError, ValueError):
            index = 0
        try:
            player_index = int(getattr(option, "playerIndex", perspective))
        except (TypeError, ValueError):
            player_index = perspective
        if player_index not in (0, 1):
            player_index = perspective
        player = obs.current.players[player_index]

        # Main-phase PLAY options omit ``area`` and point directly into the hand.
        if option_type == int(OptionType.PLAY) and area == 0:
            area = int(AreaType.HAND)

        collections = {
            int(AreaType.HAND): list(getattr(player, "hand", None) or []),
            int(AreaType.DISCARD): list(getattr(player, "discard", None) or []),
            int(AreaType.PRIZE): list(getattr(player, "prize", None) or []),
            int(AreaType.BENCH): list(getattr(player, "bench", None) or []),
            int(AreaType.ACTIVE): list(getattr(player, "active", None) or []),
            int(AreaType.STADIUM): list(getattr(obs.current, "stadium", None) or []),
            int(AreaType.LOOKING): list(getattr(obs.current, "looking", None) or []),
        }
        if area == int(AreaType.DECK):
            collections[area] = list(getattr(obs.select, "deck", None) or [])

        cards = collections.get(area, [])
        referenced_card = cards[index] if 0 <= index < len(cards) else None

        # Energy/tool selection options reference the owning in-play Pokémon.
        energy_index = getattr(option, "energyIndex", None)
        tool_index = getattr(option, "toolIndex", None)
        if referenced_card is not None and area in (int(AreaType.ACTIVE), int(AreaType.BENCH)):
            if energy_index is not None:
                energy_cards = list(getattr(referenced_card, "energyCards", None) or [])
                try:
                    energy_index = int(energy_index)
                except (TypeError, ValueError):
                    energy_index = -1
                if 0 <= energy_index < len(energy_cards):
                    return _bounded_id(energy_cards[energy_index], MAX_CARD_ID)
            if tool_index is not None:
                tools = list(getattr(referenced_card, "tools", None) or [])
                try:
                    tool_index = int(tool_index)
                except (TypeError, ValueError):
                    tool_index = -1
                if 0 <= tool_index < len(tools):
                    return _bounded_id(tools[tool_index], MAX_CARD_ID)

        if referenced_card is not None:
            return _bounded_id(referenced_card, MAX_CARD_ID)

        # ATTACK and RETREAT refer to the acting Active Pokémon without an area.
        if option_type in (int(OptionType.ATTACK), int(OptionType.RETREAT)):
            active = list(getattr(player, "active", None) or [])
            return _bounded_id(active[0], MAX_CARD_ID) if active else 0
        return 0

    def _immediate_option_features(self, obs, option, card_id, attack_id, perspective, options):
        """Factual option consequences and resource availability, never utility."""
        if obs.current is None:
            return [0.0] * 13
        our_player = obs.current.players[perspective]
        option_type = _enum_value(getattr(option, "type", None), OptionType)
        same_type_count = sum(
            _enum_value(getattr(candidate, "type", None), OptionType) == option_type
            for candidate in options
        )
        same_card_count = sum(
            self._resolve_option_card_id(obs, candidate, perspective) == card_id
            for candidate in options
        ) if card_id else 0
        bench_space = max(0, int(getattr(our_player, "benchMax", 5)) - len(getattr(our_player, "bench", None) or []))
        common = [
            min(1.0, len(options) / MAX_ENCODED_OPTIONS),
            min(1.0, same_type_count / MAX_ENCODED_OPTIONS),
            min(1.0, same_card_count / 4.0),
            min(1.0, len(getattr(our_player, "hand", None) or []) / 24.0),
            min(1.0, int(getattr(our_player, "deckCount", 0) or 0) / 60.0),
            min(1.0, bench_space / 5.0),
            float(option_type == int(OptionType.ABILITY)),
        ]
        if not attack_id:
            return [0.0] * 6 + common
        attack = self.attack_data_by_id.get(attack_id)
        opponents = list(getattr(obs.current.players[1 - perspective], "active", None) or [])
        target = opponents[0] if opponents else None
        if attack is None or target is None:
            return [0.0] * 6 + common

        printed_damage = max(0, int(getattr(attack, "damage", 0) or 0))
        target_hp = max(0, int(getattr(target, "hp", 0) or 0))
        target_card = self.card_data_by_id.get(_bounded_id(target, MAX_CARD_ID))
        attackers = list(getattr(our_player, "active", None) or [])
        attacker_card = self.card_data_by_id.get(_bounded_id(attackers[0], MAX_CARD_ID)) if attackers else None
        attacker_type = _energy_value(getattr(attacker_card, "energyType", -1)) if attacker_card else -1
        weakness = _energy_value(getattr(target_card, "weakness", -1)) if target_card else -1
        resistance = _energy_value(getattr(target_card, "resistance", -1)) if target_card else -1
        adjusted_damage = printed_damage
        if printed_damage and attacker_type == weakness:
            adjusted_damage *= 2
        if printed_damage and attacker_type == resistance:
            adjusted_damage = max(0, adjusted_damage - 30)
        is_ex = bool(getattr(target_card, "ex", False))
        is_mega_ex = bool(getattr(target_card, "megaEx", False))
        prizes = 3 if is_mega_ex else (2 if is_ex else 1)
        knockout = adjusted_damage > 0 and adjusted_damage >= target_hp
        return [
            min(1.0, printed_damage / 400.0),
            min(1.0, adjusted_damage / 400.0),
            min(1.0, target_hp / 400.0),
            float(knockout),
            (prizes / 3.0) if knockout else 0.0,
            float(is_ex),
        ] + common

    def _structured_observation(self, obs, perspective, pending_selection):
        entity_ids = np.zeros(ENTITY_SLOTS, dtype=np.int32)
        entity_features = np.zeros((ENTITY_SLOTS, ENTITY_FEATURE_DIM), dtype=np.float32)
        entity_tool_ids = np.zeros(ENTITY_SLOTS, dtype=np.int32)
        entity_pre_evolution_ids = np.zeros(
            (ENTITY_SLOTS, MAX_ENTITY_PRE_EVOLUTIONS), dtype=np.int32
        )
        entity_energy_card_ids = np.zeros(
            (ENTITY_SLOTS, MAX_ENTITY_ENERGY_CARDS), dtype=np.int32
        )
        hand_ids = np.zeros(HAND_CARD_SLOTS, dtype=np.int32)
        discard_ids = np.zeros((2, DISCARD_CARD_SLOTS), dtype=np.int32)
        revealed_ids = np.zeros(REVEALED_CARD_SLOTS, dtype=np.int32)
        prize_ids = np.zeros((2, PRIZE_CARD_SLOTS), dtype=np.int32)
        search_ids = np.zeros(SEARCH_CARD_SLOTS, dtype=np.int32)
        looking_ids = np.zeros(LOOKING_CARD_SLOTS, dtype=np.int32)
        own_deck_ids = np.zeros(DECK_LIST_SLOTS, dtype=np.int32)
        context_card_ids = np.zeros(CONTEXT_CARD_SLOTS + 1, dtype=np.int32)
        log_card_ids = np.zeros(LOG_CARD_SLOTS, dtype=np.int32)
        option_card_ids = np.zeros(MAX_ENCODED_OPTIONS, dtype=np.int32)
        option_attack_ids = np.zeros(MAX_ENCODED_OPTIONS, dtype=np.int32)
        option_types = np.zeros(MAX_ENCODED_OPTIONS, dtype=np.int32)
        option_areas = np.zeros(MAX_ENCODED_OPTIONS, dtype=np.int32)
        option_features = np.zeros((MAX_ENCODED_OPTIONS, OPTION_FEATURE_DIM), dtype=np.float32)

        if obs.current is None:
            return {
                "entity_ids": entity_ids,
                "entity_features": entity_features,
                "entity_tool_ids": entity_tool_ids,
                "entity_pre_evolution_ids": entity_pre_evolution_ids,
                "entity_energy_card_ids": entity_energy_card_ids,
                "hand_ids": hand_ids,
                "discard_ids": discard_ids,
                "revealed_ids": revealed_ids,
                "prize_ids": prize_ids,
                "search_ids": search_ids,
                "looking_ids": looking_ids,
                "own_deck_ids": own_deck_ids,
                "context_card_ids": context_card_ids,
                "log_card_ids": log_card_ids,
                "option_card_ids": option_card_ids,
                "option_attack_ids": option_attack_ids,
                "option_types": option_types,
                "option_areas": option_areas,
                "option_features": option_features,
            }

        players = [obs.current.players[perspective], obs.current.players[1 - perspective]]
        for relative_player, player in enumerate(players):
            pokemon = []
            active = list(getattr(player, "active", None) or [])
            pokemon.append(active[0] if active else None)
            pokemon.extend(list(getattr(player, "bench", None) or [])[:5])
            pokemon.extend([None] * (6 - len(pokemon)))

            for position, card in enumerate(pokemon[:6]):
                slot = relative_player * 6 + position
                if card is None:
                    continue
                card_id = _bounded_id(card, MAX_CARD_ID)
                entity_ids[slot] = card_id
                features = entity_features[slot]
                features[0] = 1.0
                features[1] = float(relative_player)
                features[2] = float(position == 0)
                features[3] = float(max(0, position - 1)) / 4.0
                hp = float(getattr(card, "hp", 0) or 0)
                max_hp = float(getattr(card, "maxHp", 0) or 0)
                features[4] = hp / 400.0
                features[5] = max_hp / 400.0
                features[6] = max(0.0, max_hp - hp) / 400.0
                features[7] = float(bool(getattr(card, "appearThisTurn", False)))

                attached_energies = list(getattr(card, "energies", None) or [])
                for energy in attached_energies:
                    energy_type = _energy_value(energy)
                    if 0 <= energy_type < 12:
                        features[8 + energy_type] = min(10.0, features[8 + energy_type] + 0.25)
                energy_count = len(attached_energies)
                features[20] = min(10.0, energy_count / 8.0)

                tools = list(getattr(card, "tools", None) or [])
                pre_evolutions = list(getattr(card, "preEvolution", None) or [])
                energy_cards = list(getattr(card, "energyCards", None) or [])
                features[21] = float(bool(tools))
                features[22] = min(1.0, len(pre_evolutions) / 3.0)
                if tools:
                    entity_tool_ids[slot] = _bounded_id(tools[0], MAX_CARD_ID)
                for index, pre_evolution in enumerate(pre_evolutions[:MAX_ENTITY_PRE_EVOLUTIONS]):
                    entity_pre_evolution_ids[slot, index] = _bounded_id(pre_evolution, MAX_CARD_ID)
                for index, energy_card in enumerate(energy_cards[:MAX_ENTITY_ENERGY_CARDS]):
                    entity_energy_card_ids[slot, index] = _bounded_id(energy_card, MAX_CARD_ID)

                static_card = self.card_data_by_id.get(card_id)
                retreat_cost = int(getattr(static_card, "retreatCost", 0) or 0)
                features[23] = retreat_cost / 5.0
                features[24] = max(0, retreat_cost - energy_count) / 5.0
                attack_costs = self.pokemon_attack_costs.get(card_id, [])
                for attack_index, cost in enumerate(attack_costs[:2]):
                    deficit = self._attack_deficit(attached_energies, cost)
                    features[25 + attack_index] = deficit / 5.0
                    features[27 + attack_index] = float(deficit == 0)

                if position == 0:
                    features[29] = float(bool(getattr(player, "poisoned", False)))
                    features[30] = float(bool(getattr(player, "burned", False)))
                    features[31] = float(bool(getattr(player, "asleep", False)))
                    features[32] = float(bool(getattr(player, "paralyzed", False)))
                    features[33] = float(bool(getattr(player, "confused", False)))
                    actual_retreat_available = any(
                        _enum_value(getattr(option, "type", None), OptionType)
                        == int(OptionType.RETREAT)
                        for option in list(getattr(obs.select, "option", None) or [])
                    )
                    features[34] = float(
                        relative_player == 0
                        and actual_retreat_available
                    )
                features[35] = min(1.0, len(attack_costs) / 2.0)

        our_player = players[0]
        for index, card in enumerate(list(getattr(our_player, "hand", None) or [])[:HAND_CARD_SLOTS]):
            hand_ids[index] = _bounded_id(card, MAX_CARD_ID)
        for relative_player, player in enumerate(players):
            discard = list(getattr(player, "discard", None) or [])[-DISCARD_CARD_SLOTS:]
            for index, card in enumerate(discard):
                discard_ids[relative_player, index] = _bounded_id(card, MAX_CARD_ID)
            for index, card in enumerate(list(getattr(player, "prize", None) or [])[:PRIZE_CARD_SLOTS]):
                prize_ids[relative_player, index] = _bounded_id(card, MAX_CARD_ID)

        searched = list(getattr(obs.select, "deck", None) or [])
        looking = list(getattr(obs.current, "looking", None) or [])
        revealed = searched + looking
        for index, card in enumerate(revealed[:REVEALED_CARD_SLOTS]):
            revealed_ids[index] = _bounded_id(card, MAX_CARD_ID)
        for index, card in enumerate(searched[:SEARCH_CARD_SLOTS]):
            search_ids[index] = _bounded_id(card, MAX_CARD_ID)
        for index, card in enumerate(looking[:LOOKING_CARD_SLOTS]):
            looking_ids[index] = _bounded_id(card, MAX_CARD_ID)
        own_source_deck = (
            self.my_deck
            if perspective == self.learner_perspective
            else self.opponent_deck
        )
        for index, card_id in enumerate(own_source_deck[:DECK_LIST_SLOTS]):
            own_deck_ids[index] = _bounded_id(card_id, MAX_CARD_ID)
        context_card_ids[0] = _bounded_id(getattr(obs.select, "contextCard", None), MAX_CARD_ID)
        context_card_ids[1] = _bounded_id(getattr(obs.select, "effect", None), MAX_CARD_ID)
        stadium = list(getattr(obs.current, "stadium", None) or [])
        if stadium:
            context_card_ids[2] = _bounded_id(stadium[0], MAX_CARD_ID)

        recent_logs = list(getattr(obs, "logs", None) or [])[-5:]
        for index, log in enumerate(reversed(recent_logs)):
            log_card_ids[index * 2] = _bounded_id(getattr(log, "cardId", None), MAX_CARD_ID)
            log_card_ids[index * 2 + 1] = _bounded_id(
                getattr(log, "cardIdTarget", None), MAX_CARD_ID
            )

        selected_indices = set(pending_selection)
        options = list(getattr(obs.select, "option", None) or [])[:MAX_ENCODED_OPTIONS]
        for index, option in enumerate(options):
            option_type = _enum_value(getattr(option, "type", None), OptionType)
            option_area = _enum_value(getattr(option, "area", None), AreaType)
            card_id = self._resolve_option_card_id(obs, option, perspective)
            attack_id = _bounded_id(getattr(option, "attackId", None), MAX_ATTACK_ID)
            # Reserve zero for padding; OptionType.NUMBER itself has enum value zero.
            option_types[index] = option_type + 1
            option_areas[index] = option_area
            option_card_ids[index] = card_id
            option_attack_ids[index] = attack_id

            raw_player = getattr(option, "playerIndex", perspective)
            try:
                raw_player = int(raw_player)
            except (TypeError, ValueError):
                raw_player = perspective
            raw_index = getattr(option, "index", 0)
            raw_in_play_index = getattr(option, "inPlayIndex", 0)
            raw_number = getattr(option, "number", 0)
            raw_count = getattr(option, "count", 0)
            numeric_values = []
            for value in (raw_index, raw_in_play_index, raw_number, raw_count):
                try:
                    numeric_values.append(float(value or 0))
                except (TypeError, ValueError):
                    numeric_values.append(0.0)
            option_features[index] = np.asarray(
                [
                    float(abs(raw_player - perspective)),
                    numeric_values[0] / 60.0,
                    numeric_values[1] / 5.0,
                    numeric_values[2] / 60.0,
                    numeric_values[3] / 60.0,
                    float(index in selected_indices),
                    float(card_id > 0),
                    float(attack_id > 0),
                    *self._immediate_option_features(
                        obs, option, card_id, attack_id, perspective, options
                    ),
                ],
                dtype=np.float32,
            )

        obs_dict = {
            "entity_ids": entity_ids,
            "entity_features": entity_features,
            "entity_tool_ids": entity_tool_ids,
            "entity_pre_evolution_ids": entity_pre_evolution_ids,
            "entity_energy_card_ids": entity_energy_card_ids,
            "hand_ids": hand_ids,
            "discard_ids": discard_ids,
            "revealed_ids": revealed_ids,
            "prize_ids": prize_ids,
            "search_ids": search_ids,
            "looking_ids": looking_ids,
            "own_deck_ids": own_deck_ids,
            "context_card_ids": context_card_ids,
            "log_card_ids": log_card_ids,
            "option_card_ids": option_card_ids,
            "option_attack_ids": option_attack_ids,
            "option_types": option_types,
            "option_areas": option_areas,
            "option_features": option_features,
        }
        if self.zone_aux_targets:
            aux_own_deck = np.zeros(60, dtype=np.int32)
            aux_own_prize = np.zeros(6, dtype=np.int32)
            aux_opp_hand = np.zeros(60, dtype=np.int32)
            aux_opp_deck = np.zeros(60, dtype=np.int32)
            aux_opp_prize = np.zeros(6, dtype=np.int32)

            if obs.current is not None:
                p_own = obs.current.players[perspective]
                p_opp = obs.current.players[1 - perspective]

                if getattr(p_own, "deck", None):
                    for i, c in enumerate(p_own.deck[:60]):
                        aux_own_deck[i] = _cid(c)
                if getattr(p_own, "prize", None):
                    for i, c in enumerate(p_own.prize[:6]):
                        aux_own_prize[i] = _cid(c)
                if getattr(p_opp, "hand", None):
                    for i, c in enumerate(p_opp.hand[:60]):
                        aux_opp_hand[i] = _cid(c)
                if getattr(p_opp, "deck", None):
                    for i, c in enumerate(p_opp.deck[:60]):
                        aux_opp_deck[i] = _cid(c)
                if getattr(p_opp, "prize", None):
                    for i, c in enumerate(p_opp.prize[:6]):
                        aux_opp_prize[i] = _cid(c)

            obs_dict.update({
                "aux_own_deck_ids": aux_own_deck,
                "aux_own_prize_ids": aux_own_prize,
                "aux_opponent_hand_ids": aux_opp_hand,
                "aux_opponent_deck_ids": aux_opp_deck,
                "aux_opponent_prize_ids": aux_opp_prize,
            })
        return obs_dict

    @staticmethod
    def _copy_cpp_array(value, dtype, shape=None):
        array = np.ctypeslib.as_array(value).astype(dtype, copy=True)
        return array.reshape(shape) if shape is not None else array

    def _pending_selection_for_perspective(self, perspective=0):
        return self.pending_selection if perspective == self.learner_perspective else self.opponent_pending_selection

    def _get_obs_cpp(self, perspective=0, pending_selection=None, action_space_size=None, force_structured=None):
        if pending_selection is None:
            pending_selection = self._pending_selection_for_perspective(perspective)
        pending_selection = list(pending_selection or [])
        output_size = int(action_space_size or self.max_options)
        if output_size not in {LEGACY_ACTION_SPACE_SIZE, V6_ACTION_SPACE_SIZE}:
            raise ValueError(f"Unsupported observation action-mask size: {output_size}")

        pending_array = None
        if pending_selection:
            pending_array = (ctypes.c_int * len(pending_selection))(*pending_selection)
        buffer = V6ObservationBuffer()
        error_code = lib.GetV6Observation(
            Battle.battle_ptr,
            int(perspective),
            pending_array,
            len(pending_selection),
            ctypes.byref(buffer),
        )
        if error_code != 0:
            raise RuntimeError(f"GetV6Observation failed with engine error {error_code}")

        cpp_mask = self._copy_cpp_array(buffer.action_mask, np.int8)
        if output_size == V6_ACTION_SPACE_SIZE:
            action_mask = cpp_mask
        else:
            action_mask = np.zeros(output_size, dtype=np.int8)
            action_mask[:MAX_ENCODED_OPTIONS] = cpp_mask[:MAX_ENCODED_OPTIONS]
            action_mask[output_size - 1] = cpp_mask[V6_STOP_ACTION]

        obs = to_observation_class(self.current_obs_dict)
        if obs.select and obs.select.option:
            raw_option_count = len(obs.select.option)
            self.max_option_count_seen = max(self.max_option_count_seen, raw_option_count)
            if raw_option_count > MAX_ENCODED_OPTIONS:
                self.option_overflow_count += 1

        result = {
            "vector": self._copy_cpp_array(buffer.vector, np.float32),
            "action_mask": action_mask,
            "aux_target": self._copy_cpp_array(buffer.aux_target, np.float32),
        }
        use_structured = self.structured_v2 if force_structured is None else force_structured
        if use_structured:
            result.update({
                "entity_ids": self._copy_cpp_array(buffer.entity_ids, np.int32),
                "entity_features": self._copy_cpp_array(
                    buffer.entity_features, np.float32, (ENTITY_SLOTS, ENTITY_FEATURE_DIM)
                ),
                "entity_tool_ids": self._copy_cpp_array(buffer.entity_tool_ids, np.int32),
                "entity_pre_evolution_ids": self._copy_cpp_array(
                    buffer.entity_pre_evolution_ids,
                    np.int32,
                    (ENTITY_SLOTS, MAX_ENTITY_PRE_EVOLUTIONS),
                ),
                "entity_energy_card_ids": self._copy_cpp_array(
                    buffer.entity_energy_card_ids,
                    np.int32,
                    (ENTITY_SLOTS, MAX_ENTITY_ENERGY_CARDS),
                ),
                "hand_ids": self._copy_cpp_array(buffer.hand_ids, np.int32),
                "discard_ids": self._copy_cpp_array(
                    buffer.discard_ids, np.int32, (2, DISCARD_CARD_SLOTS)
                ),
                "revealed_ids": self._copy_cpp_array(buffer.revealed_ids, np.int32),
                "prize_ids": self._copy_cpp_array(buffer.prize_ids, np.int32, (2, PRIZE_CARD_SLOTS)),
                "search_ids": self._copy_cpp_array(buffer.search_ids, np.int32),
                "looking_ids": self._copy_cpp_array(buffer.looking_ids, np.int32),
                "own_deck_ids": self._copy_cpp_array(buffer.own_deck_ids, np.int32),
                "context_card_ids": self._copy_cpp_array(buffer.context_card_ids, np.int32),
                "log_card_ids": self._copy_cpp_array(buffer.log_card_ids, np.int32),
                "option_card_ids": self._copy_cpp_array(buffer.option_card_ids, np.int32),
                "option_attack_ids": self._copy_cpp_array(buffer.option_attack_ids, np.int32),
                "option_types": self._copy_cpp_array(buffer.option_types, np.int32),
                "option_areas": self._copy_cpp_array(buffer.option_areas, np.int32),
                "option_features": self._copy_cpp_array(
                    buffer.option_features, np.float32, (MAX_ENCODED_OPTIONS, OPTION_FEATURE_DIM)
                ),
            })
        if self.zone_aux_targets:
            aux_own_deck = np.zeros(60, dtype=np.int32)
            aux_own_prize = np.zeros(6, dtype=np.int32)
            aux_opp_hand = np.zeros(60, dtype=np.int32)
            aux_opp_deck = np.zeros(60, dtype=np.int32)
            aux_opp_prize = np.zeros(6, dtype=np.int32)

            battle_ptr = getattr(self, "battle_ptr", None) or getattr(Battle, "battle_ptr", None)
            if battle_ptr:
                raw_json = lib.VisualizeData(battle_ptr)
                if raw_json:
                    parsed = json.loads(raw_json.decode("utf-8"))
                    if isinstance(parsed, dict) and "current" in parsed and isinstance(parsed["current"], dict):
                        players = parsed["current"].get("players", [{}, {}])
                    elif isinstance(parsed, dict):
                        players = parsed.get("players", [{}, {}])
                    else:
                        players = parsed if isinstance(parsed, list) else [{}, {}]

                    p_own = players[perspective] if isinstance(players, list) and perspective < len(players) and isinstance(players[perspective], dict) else {}
                    p_opp = players[1 - perspective] if isinstance(players, list) and (1 - perspective) < len(players) and isinstance(players[1 - perspective], dict) else {}

                    for i, c in enumerate((p_own.get("deck") or p_own.get("deckCards") or [])[:60]): aux_own_deck[i] = _cid(c)
                    for i, c in enumerate((p_own.get("prize") or p_own.get("prizeCards") or [])[:6]): aux_own_prize[i] = _cid(c)
                    for i, c in enumerate((p_opp.get("hand") or p_opp.get("handCards") or [])[:60]): aux_opp_hand[i] = _cid(c)
                    for i, c in enumerate((p_opp.get("deck") or p_opp.get("deckCards") or [])[:60]): aux_opp_deck[i] = _cid(c)
                    for i, c in enumerate((p_opp.get("prize") or p_opp.get("prizeCards") or [])[:6]): aux_opp_prize[i] = _cid(c)

            if np.count_nonzero(aux_own_prize) == 0 and getattr(self, "my_deck", None):
                own_d = self.my_deck if perspective == 0 else self.opponent_deck
                opp_d = self.opponent_deck if perspective == 0 else self.my_deck
                for i, c in enumerate(own_d[:60]): aux_own_deck[i] = _cid(c)
                for i, c in enumerate(own_d[:6]): aux_own_prize[i] = _cid(c)
                for i, c in enumerate(opp_d[:60]): aux_opp_deck[i] = _cid(c)
                for i, c in enumerate(opp_d[:6]): aux_opp_prize[i] = _cid(c)
                for i, c in enumerate(opp_d[6:66]): aux_opp_hand[i] = _cid(c)

            result.update({
                "aux_own_deck_ids": aux_own_deck,
                "aux_own_prize_ids": aux_own_prize,
                "aux_opponent_hand_ids": aux_opp_hand,
                "aux_opponent_deck_ids": aux_opp_deck,
                "aux_opponent_prize_ids": aux_opp_prize,
            })

        teacher_action = -1
        if self.enable_lookahead_teacher and self.lookahead_teacher is not None and perspective == self.learner_perspective:
            if random.random() < self.teacher_sample_rate:
                mask = result.get("action_mask", [])
                option_count = int(np.count_nonzero(mask))
                if option_count >= 2:
                    try:
                        from src.training.lookahead_teacher import build_search_hypotheses
                        your_d = self.my_deck if perspective == self.learner_perspective else self.opponent_deck
                        opp_d = self.opponent_deck if perspective == self.learner_perspective else self.my_deck
                        hypotheses = build_search_hypotheses(obs, your_deck=your_d, opponent_deck=opp_d, card_data_by_id=self.lookahead_teacher.card_data_by_id)
                        decision = self.lookahead_teacher.choose(obs, result, perspective=perspective, hypotheses=hypotheses)
                        if decision is not None:
                            teacher_action = int(decision.action)
                    except Exception:
                        teacher_action = -1

        result["teacher_action"] = np.array([teacher_action], dtype=np.int32)
        return result

    def _get_obs(self, perspective=0, pending_selection=None, action_space_size=None, force_structured=None):
        return self._get_obs_cpp(
            perspective=perspective,
            pending_selection=pending_selection,
            action_space_size=action_space_size,
            force_structured=force_structured,
        )

    def _get_obs_python(self, perspective=0, pending_selection=None, action_space_size=None, force_structured=None):
        obs = to_observation_class(self.current_obs_dict)
        if pending_selection is None:
            pending_selection = self._pending_selection_for_perspective(perspective)
        pending_selection = list(pending_selection or [])
        output_size = int(action_space_size or self.max_options)
        if output_size not in {LEGACY_ACTION_SPACE_SIZE, V6_ACTION_SPACE_SIZE}:
            raise ValueError(f"Unsupported observation action-mask size: {output_size}")
        stop_action = output_size - 1
        mask = np.zeros(output_size, dtype=np.int8)
        if obs.select and obs.select.option:
            raw_option_count = len(obs.select.option)
            self.max_option_count_seen = max(self.max_option_count_seen, raw_option_count)
            if raw_option_count > MAX_ENCODED_OPTIONS:
                self.option_overflow_count += 1
            num_opts = min(raw_option_count, MAX_ENCODED_OPTIONS, stop_action)
            mask[:num_opts] = 1
            for selected_index in pending_selection:
                if 0 <= selected_index < num_opts:
                    mask[selected_index] = 0

            min_count = min(num_opts, max(0, int(obs.select.minCount or 0)))
            max_count = min(num_opts, max(0, int(obs.select.maxCount or 0)))
            if len(pending_selection) >= min_count and (min_count == 0 or max_count > 1):
                mask[stop_action] = 1
            
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
                vec[250] = float(_enum_value(obs.select.context, SelectContext))
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
                    
                    vec[1300 + i*3] = float(_enum_value(log.type, LogType))
                    vec[1301 + i*3] = float(_enum_value(log.fromArea, AreaType))
                    vec[1302 + i*3] = float(_enum_value(log.toArea, AreaType))

            # Autoregressive selection state. Existing action/output dimensions stay unchanged.
            vec[1490] = float(len(pending_selection))
            vec[1491] = float(mask[stop_action])
            for i, selected_index in enumerate(pending_selection[:8]):
                vec[1492 + i] = float(selected_index + 1)

            # Options 0-49 use the legacy range. Options 50-64 use the previously
            # unused 650-799 range so existing checkpoints remain loadable.
            if obs.select and obs.select.option:
                opt_len = min(len(obs.select.option), MAX_ENCODED_OPTIONS)
                for i in range(opt_len):
                    opt = obs.select.option[i]
                    base = 800 + i*10 if i < 50 else 650 + (i - 50)*10
                    vec[base] = float(_enum_value(opt.type, OptionType))
                    vec[base+1] = float(opt.cardId if opt.cardId is not None else 0)
                    vec[base+2] = float(_enum_value(opt.area, AreaType))
                    vec[base+3] = float(opt.index if opt.index is not None else 0)
                    vec[base+4] = float(_enum_value(opt.inPlayArea, AreaType))
                    vec[base+5] = float(opt.inPlayIndex if opt.inPlayIndex is not None else 0)
                    vec[base+6] = float(opt.attackId if opt.attackId is not None else 0)
                    vec[base+7] = float(opt.specialConditionType if opt.specialConditionType is not None else 0)
                    option_player = opt.playerIndex if opt.playerIndex is not None else perspective
                    vec[base+8] = float(abs(int(option_player) - perspective))
                    vec[base+9] = float(opt.number if opt.number is not None else (opt.count if hasattr(opt, 'count') and opt.count is not None else 0))
                
        # Auxiliary target: predict how many copies of each card remain hidden.
        # Log scaling distinguishes ordinary 1-4 copy cards while still
        # representing decks with many copies of a basic Energy.
        hidden_deck = (
            self.opponent_deck
            if perspective == self.learner_perspective
            else self.my_deck
        )
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
                aux_target[card_id] = encode_hidden_card_count(count)
                
        use_structured = self.structured_v2 if force_structured is None else force_structured
        if use_structured:
            structured = self._structured_observation(obs, perspective, pending_selection)
            res = {"vector": vec, "action_mask": mask, "aux_target": aux_target, **structured}
        else:
            res = {"vector": vec, "action_mask": mask, "aux_target": aux_target}

        teacher_action = -1
        if self.enable_lookahead_teacher and self.lookahead_teacher is not None:
            if random.random() < self.teacher_sample_rate:
                option_count = int(np.count_nonzero(mask))
                if option_count >= 2:
                    try:
                        from src.training.lookahead_teacher import build_search_hypotheses
                        hypotheses = build_search_hypotheses(obs, perspective, card_data_by_id=self.lookahead_teacher.card_data_by_id)
                        decision = self.lookahead_teacher.choose(obs, res, perspective=perspective, hypotheses=hypotheses)
                        if decision is not None:
                            teacher_action = int(decision.action)
                    except Exception:
                        teacher_action = -1

        res["teacher_action"] = np.array([teacher_action], dtype=np.int32)

        if self.zone_aux_targets and not use_structured:
            res.update({
                "aux_own_deck_ids": np.zeros(60, dtype=np.int32),
                "aux_own_prize_ids": np.zeros(6, dtype=np.int32),
                "aux_opponent_hand_ids": np.zeros(60, dtype=np.int32),
                "aux_opponent_deck_ids": np.zeros(60, dtype=np.int32),
                "aux_opponent_prize_ids": np.zeros(6, dtype=np.int32),
            })
        return res
        
    def _get_info(self):
        info = {
            "policy_version": self.policy_version,
            "action_space_size": self.max_options,
            "max_option_count_seen": self.max_option_count_seen,
            "option_overflow_count": self.option_overflow_count,
            "engine_error_count": self.engine_error_count,
        }
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
