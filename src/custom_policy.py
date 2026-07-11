import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from src.cg.api import all_attack, all_card_data
from src.env_wrapper import (
    ENTITY_FEATURE_DIM,
    ENTITY_SLOTS,
    MAX_ATTACK_ID,
    MAX_CARD_ID,
    MAX_ENCODED_OPTIONS,
)

CARD_EMBED_DIM = 24
CARD_METADATA_DIM = 54
ATTACK_EMBED_DIM = 16
ATTACK_METADATA_DIM = 14
OPTION_EMBED_DIM = 64


def _integer_value(value, default=0):
    raw_value = getattr(value, "value", value)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def build_card_metadata():
    """Static, rule-relevant card attributes indexed by card ID."""
    metadata = np.zeros((MAX_CARD_ID + 1, CARD_METADATA_DIM), dtype=np.float32)
    for card in all_card_data():
        card_id = _integer_value(getattr(card, "cardId", 0))
        if not 0 < card_id <= MAX_CARD_ID:
            continue
        card_type = _integer_value(getattr(card, "cardType", 0))
        if 0 <= card_type < 7:
            metadata[card_id, card_type] = 1.0
        metadata[card_id, 7] = float(getattr(card, "retreatCost", 0) or 0) / 5.0
        metadata[card_id, 8] = float(getattr(card, "hp", 0) or 0) / 400.0

        energy_type = _integer_value(getattr(card, "energyType", 0))
        if 0 <= energy_type < 12:
            metadata[card_id, 9 + energy_type] = 1.0
        weakness = _integer_value(getattr(card, "weakness", -1), -1)
        if 0 <= weakness < 12:
            metadata[card_id, 21 + weakness] = 1.0
        resistance = _integer_value(getattr(card, "resistance", -1), -1)
        if 0 <= resistance < 12:
            metadata[card_id, 33 + resistance] = 1.0

        metadata[card_id, 45] = float(bool(getattr(card, "basic", False)))
        metadata[card_id, 46] = float(bool(getattr(card, "stage1", False)))
        metadata[card_id, 47] = float(bool(getattr(card, "stage2", False)))
        metadata[card_id, 48] = float(bool(getattr(card, "ex", False)))
        metadata[card_id, 49] = float(bool(getattr(card, "megaEx", False)))
        metadata[card_id, 50] = float(bool(getattr(card, "tera", False)))
        metadata[card_id, 51] = float(bool(getattr(card, "aceSpec", False)))
        metadata[card_id, 52] = min(1.0, len(getattr(card, "attacks", None) or []) / 3.0)
        metadata[card_id, 53] = min(1.0, len(getattr(card, "skills", None) or []) / 3.0)
    return metadata


def build_attack_metadata():
    """Damage and exact typed energy requirements indexed by attack ID."""
    metadata = np.zeros((MAX_ATTACK_ID + 1, ATTACK_METADATA_DIM), dtype=np.float32)
    for attack in all_attack():
        attack_id = _integer_value(getattr(attack, "attackId", 0))
        if not 0 < attack_id <= MAX_ATTACK_ID:
            continue
        energies = list(getattr(attack, "energies", None) or [])
        metadata[attack_id, 0] = float(getattr(attack, "damage", 0) or 0) / 400.0
        metadata[attack_id, 1] = min(1.0, len(energies) / 5.0)
        for energy in energies:
            energy_type = _integer_value(energy)
            if 0 <= energy_type < 12:
                metadata[attack_id, 2 + energy_type] += 0.2
    return metadata

class PokemonTCGFeatureExtractor(BaseFeaturesExtractor):
    """
    Legacy-compatible extractor with a structured Observation V2 path.

    Older checkpoints only contain ``vector/action_mask/aux_target`` and keep
    the exact original two-layer MLP. New checkpoints embed categorical IDs,
    preserve per-Pokemon attachments and encode each legal option with shared
    weights.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        vector_dim = observation_space.spaces['vector'].shape[0]
        required_v2_keys = {
            "entity_ids",
            "entity_features",
            "option_card_ids",
            "option_attack_ids",
            "option_features",
        }
        self.structured_v2 = required_v2_keys.issubset(observation_space.spaces)

        if not self.structured_v2:
            # Do not change module names or shapes: existing checkpoints rely on them.
            self.net = nn.Sequential(
                nn.Linear(vector_dim, 256),
                nn.ReLU(),
                nn.Linear(256, features_dim),
                nn.ReLU()
            )
            return

        self.card_embedding = nn.Embedding(MAX_CARD_ID + 1, CARD_EMBED_DIM, padding_idx=0)
        self.attack_embedding = nn.Embedding(MAX_ATTACK_ID + 1, ATTACK_EMBED_DIM, padding_idx=0)
        self.option_type_embedding = nn.Embedding(18, 8, padding_idx=0)
        self.option_area_embedding = nn.Embedding(14, 6, padding_idx=0)
        self.register_buffer("card_metadata", torch.as_tensor(build_card_metadata()))
        self.register_buffer("attack_metadata", torch.as_tensor(build_attack_metadata()))

        keep_mask = np.ones(vector_dim, dtype=np.float32)
        keep_mask[300:650] = 0.0
        for option_index in range(MAX_ENCODED_OPTIONS):
            base = 800 + option_index * 10 if option_index < 50 else 650 + (option_index - 50) * 10
            keep_mask[base + 1] = 0.0  # card ID
            keep_mask[base + 6] = 0.0  # attack ID
        self.register_buffer("vector_keep_mask", torch.as_tensor(keep_mask))

        self.vector_encoder = nn.Sequential(
            nn.LayerNorm(vector_dim),
            nn.Linear(vector_dim, 384),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU(),
        )
        card_repr_dim = CARD_EMBED_DIM + CARD_METADATA_DIM
        entity_input_dim = card_repr_dim * 4 + ENTITY_FEATURE_DIM
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        option_input_dim = (
            card_repr_dim + ATTACK_EMBED_DIM + ATTACK_METADATA_DIM + 8 + 6 + 8
        )
        self.option_encoder = nn.Sequential(
            nn.Linear(option_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, OPTION_EMBED_DIM),
            nn.ReLU(),
        )

        # vector + entities + pooled hand/discards/revealed/logs/context/options
        pooled_card_dim = card_repr_dim * 2
        combined_dim = (
            256
            + ENTITY_SLOTS * 64
            + pooled_card_dim
            + 2 * pooled_card_dim
            + pooled_card_dim
            + pooled_card_dim
            + 2 * card_repr_dim
            + 2 * OPTION_EMBED_DIM
        )
        self.net = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def _ids(self, values, maximum):
        return values.long().clamp_(0, maximum)

    def _card_repr(self, card_ids):
        card_ids = self._ids(card_ids, MAX_CARD_ID)
        return torch.cat(
            [self.card_embedding(card_ids), self.card_metadata[card_ids]], dim=-1
        )

    @staticmethod
    def _masked_mean(values, ids):
        mask = (ids > 0).unsqueeze(-1).to(values.dtype)
        return (values * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1.0)

    def _mean_card_set(self, card_ids):
        ids = self._ids(card_ids, MAX_CARD_ID)
        return self._masked_mean(self._card_repr(ids), ids)

    def _pool_card_set(self, card_ids):
        ids = self._ids(card_ids, MAX_CARD_ID)
        values = self._card_repr(ids)
        mean = self._masked_mean(values, ids)
        mask = (ids > 0).unsqueeze(-1)
        maximum = values.masked_fill(~mask, -1e9).max(dim=-2).values
        maximum = torch.where((ids > 0).any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum))
        return torch.cat([mean, maximum], dim=-1)

    def encode_options(self, observations):
        card_ids = self._ids(observations["option_card_ids"], MAX_CARD_ID)
        attack_ids = self._ids(observations["option_attack_ids"], MAX_ATTACK_ID)
        option_types = self._ids(observations["option_types"], 17)
        option_areas = self._ids(observations["option_areas"], 13)
        return self.option_encoder(
            torch.cat(
                [
                    self._card_repr(card_ids),
                    self.attack_embedding(attack_ids),
                    self.attack_metadata[attack_ids],
                    self.option_type_embedding(option_types),
                    self.option_area_embedding(option_areas),
                    observations["option_features"].float(),
                ],
                dim=-1,
            )
        )

    def forward(self, observations):
        if not self.structured_v2:
            return self.net(observations['vector'])

        vector = observations["vector"].float() * self.vector_keep_mask
        vector_features = self.vector_encoder(vector)

        entity_ids = self._ids(observations["entity_ids"], MAX_CARD_ID)
        entity_card_repr = self._card_repr(entity_ids)
        tool_repr = self._card_repr(observations["entity_tool_ids"])
        pre_evolution_repr = self._mean_card_set(observations["entity_pre_evolution_ids"])
        energy_card_repr = self._mean_card_set(observations["entity_energy_card_ids"])
        entities = self.entity_encoder(
            torch.cat(
                [
                    entity_card_repr,
                    tool_repr,
                    pre_evolution_repr,
                    energy_card_repr,
                    observations["entity_features"].float(),
                ],
                dim=-1,
            )
        ).flatten(start_dim=-2)

        hand = self._pool_card_set(observations["hand_ids"])
        discard_ids = observations["discard_ids"]
        our_discard = self._pool_card_set(discard_ids[..., 0, :])
        opponent_discard = self._pool_card_set(discard_ids[..., 1, :])
        revealed = self._pool_card_set(observations["revealed_ids"])
        logs = self._pool_card_set(observations["log_card_ids"])
        context = self._card_repr(observations["context_card_ids"]).flatten(start_dim=-2)

        options = self.encode_options(observations)
        option_present = observations["action_mask"][..., :MAX_ENCODED_OPTIONS] > 0
        option_mean = self._masked_mean(options, option_present.long())
        option_max = options.masked_fill(~option_present.unsqueeze(-1), -1e9).max(dim=-2).values
        option_max = torch.where(
            option_present.any(dim=-1, keepdim=True), option_max, torch.zeros_like(option_max)
        )

        return self.net(
            torch.cat(
                [
                    vector_features,
                    entities,
                    hand,
                    our_discard,
                    opponent_discard,
                    revealed,
                    logs,
                    context,
                    option_mean,
                    option_max,
                ],
                dim=-1,
            )
        )

class PokemonTCGNetwork(nn.Module):
    """
    Custom Network containing LSTM memory and the three heads:
    Actor, Critic, and Auxiliary (Hand/Deck prediction).
    """
    def __init__(self, feature_dim: int, action_dim: int, aux_dim: int = 2000, hidden_dim: int = 128):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Memory Layer
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Actor Head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Critic Head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Auxiliary Head
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, aux_dim)
        )

    def forward(self, features, hidden_state=None):
        """
        features shape: (batch_size, seq_len, feature_dim) or (batch_size, feature_dim)
        """
        # Ensure 3D for LSTM
        is_2d = False
        if features.dim() == 2:
            is_2d = True
            features = features.unsqueeze(1) # (batch, 1, feature_dim)
            
        lstm_out, hidden_state = self.lstm(features, hidden_state)
        
        if is_2d:
            lstm_out = lstm_out.squeeze(1) # (batch, hidden_dim)
            
        action_logits = self.actor_head(lstm_out)
        values = self.critic_head(lstm_out)
        aux_logits = self.aux_head(lstm_out)
        
        return action_logits, values, aux_logits, hidden_state
