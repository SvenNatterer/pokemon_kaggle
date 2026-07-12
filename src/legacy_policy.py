"""Inference-only feature extractors for historical PPO checkpoints."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.custom_policy import build_attack_metadata, build_card_metadata
from src.env_wrapper import (
    ENTITY_FEATURE_DIM,
    ENTITY_SLOTS,
    MAX_ATTACK_ID,
    MAX_CARD_ID,
    MAX_ENCODED_OPTIONS,
)


CARD_EMBED_DIM = 24
LEGACY_CARD_METADATA_DIM = 54
ATTACK_EMBED_DIM = 16
LEGACY_ATTACK_METADATA_DIM = 14
OPTION_EMBED_DIM = 64


class LegacyStructuredFeatureExtractor(BaseFeaturesExtractor):
    """Exact structured extractor layout used by pre-metadata-expansion V5."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        vector_dim = observation_space.spaces["vector"].shape[0]
        self.structured_v2 = True

        self.card_embedding = nn.Embedding(MAX_CARD_ID + 1, CARD_EMBED_DIM, padding_idx=0)
        self.attack_embedding = nn.Embedding(MAX_ATTACK_ID + 1, ATTACK_EMBED_DIM, padding_idx=0)
        self.option_type_embedding = nn.Embedding(18, 8, padding_idx=0)
        self.option_area_embedding = nn.Embedding(14, 6, padding_idx=0)
        self.register_buffer(
            "card_metadata",
            torch.as_tensor(build_card_metadata()[:, :LEGACY_CARD_METADATA_DIM]),
        )
        self.register_buffer(
            "attack_metadata",
            torch.as_tensor(build_attack_metadata()[:, :LEGACY_ATTACK_METADATA_DIM]),
        )

        keep_mask = np.ones(vector_dim, dtype=np.float32)
        keep_mask[300:650] = 0.0
        for option_index in range(MAX_ENCODED_OPTIONS):
            base = 800 + option_index * 10 if option_index < 50 else 650 + (option_index - 50) * 10
            keep_mask[base + 1] = 0.0
            keep_mask[base + 6] = 0.0
        self.register_buffer("vector_keep_mask", torch.as_tensor(keep_mask))

        self.vector_encoder = nn.Sequential(
            nn.LayerNorm(vector_dim), nn.Linear(vector_dim, 384), nn.ReLU(),
            nn.Linear(384, 256), nn.ReLU(),
        )
        card_repr_dim = CARD_EMBED_DIM + LEGACY_CARD_METADATA_DIM
        self.entity_encoder = nn.Sequential(
            nn.Linear(card_repr_dim * 4 + ENTITY_FEATURE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.option_encoder = nn.Sequential(
            nn.Linear(
                card_repr_dim + ATTACK_EMBED_DIM + LEGACY_ATTACK_METADATA_DIM + 8 + 6 + 8,
                128,
            ),
            nn.ReLU(), nn.Linear(128, OPTION_EMBED_DIM), nn.ReLU(),
        )
        pooled_card_dim = card_repr_dim * 2
        combined_dim = (
            256 + ENTITY_SLOTS * 64 + pooled_card_dim + 2 * pooled_card_dim
            + pooled_card_dim + pooled_card_dim + 2 * card_repr_dim + 2 * OPTION_EMBED_DIM
        )
        self.net = nn.Sequential(
            nn.Linear(combined_dim, 512), nn.ReLU(),
            nn.Linear(512, features_dim), nn.ReLU(),
        )

    @staticmethod
    def _ids(values, maximum):
        return values.long().clamp_(0, maximum)

    def _card_repr(self, card_ids):
        card_ids = self._ids(card_ids, MAX_CARD_ID)
        return torch.cat([self.card_embedding(card_ids), self.card_metadata[card_ids]], dim=-1)

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
        maximum = torch.where(
            (ids > 0).any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum)
        )
        return torch.cat([mean, maximum], dim=-1)

    def encode_options(self, observations):
        card_ids = self._ids(observations["option_card_ids"], MAX_CARD_ID)
        attack_ids = self._ids(observations["option_attack_ids"], MAX_ATTACK_ID)
        option_types = self._ids(observations["option_types"], 17)
        option_areas = self._ids(observations["option_areas"], 13)
        return self.option_encoder(torch.cat([
            self._card_repr(card_ids), self.attack_embedding(attack_ids),
            self.attack_metadata[attack_ids], self.option_type_embedding(option_types),
            self.option_area_embedding(option_areas), observations["option_features"].float(),
        ], dim=-1))

    def forward(self, observations):
        vector_features = self.vector_encoder(
            observations["vector"].float() * self.vector_keep_mask
        )
        entity_ids = self._ids(observations["entity_ids"], MAX_CARD_ID)
        entities = self.entity_encoder(torch.cat([
            self._card_repr(entity_ids),
            self._card_repr(observations["entity_tool_ids"]),
            self._mean_card_set(observations["entity_pre_evolution_ids"]),
            self._mean_card_set(observations["entity_energy_card_ids"]),
            observations["entity_features"].float(),
        ], dim=-1)).flatten(start_dim=-2)

        discard_ids = observations["discard_ids"]
        context = self._card_repr(observations["context_card_ids"]).flatten(start_dim=-2)
        options = self.encode_options(observations)
        option_present = observations["action_mask"][..., :MAX_ENCODED_OPTIONS] > 0
        option_mean = self._masked_mean(options, option_present.long())
        option_max = options.masked_fill(~option_present.unsqueeze(-1), -1e9).max(dim=-2).values
        option_max = torch.where(
            option_present.any(dim=-1, keepdim=True), option_max, torch.zeros_like(option_max)
        )
        return self.net(torch.cat([
            vector_features, entities,
            self._pool_card_set(observations["hand_ids"]),
            self._pool_card_set(discard_ids[..., 0, :]),
            self._pool_card_set(discard_ids[..., 1, :]),
            self._pool_card_set(observations["revealed_ids"]),
            self._pool_card_set(observations["log_card_ids"]),
            context, option_mean, option_max,
        ], dim=-1))
