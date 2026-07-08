import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class PokemonTCGFeatureExtractor(BaseFeaturesExtractor):
    """
    Shared Feature Extractor that combines vector obs and (optional) action masks.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        vector_dim = observation_space.spaces['vector'].shape[0]
        
        self.net = nn.Sequential(
            nn.Linear(vector_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # We only pass the vector part through the feature extractor
        return self.net(observations['vector'])

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
