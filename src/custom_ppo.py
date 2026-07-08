import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import RecurrentMultiInputActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.utils import explained_variance

class PokemonTCGRecurrentPolicy(RecurrentMultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add the auxiliary head
        # In RecurrentMultiInputActorCriticPolicy, we use lstm_actor
        lstm_hidden_dim = self.lstm_actor.hidden_size
        self.aux_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2000) # aux_dim = 2000
        )
        
        # Add aux head parameters to the existing optimizer
        self.optimizer.add_param_group({'params': self.aux_head.parameters()})

    def forward(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ):
        """
        Forward pass in all the networks (actor and critic) with Action Masking.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alias
        else:
            pi_features, vf_features = features
            
        latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        
        # APPLY ACTION MASKING HERE
        mean_actions = self.action_net(latent_pi)
        if isinstance(obs, dict) and 'action_mask' in obs:
            mask = obs['action_mask']
            mean_actions = mean_actions + (1.0 - mask) * -1e8
            
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def get_distribution(
        self,
        obs,
        lstm_states,
        episode_starts,
    ):
        """
        Get the current policy distribution given the observations.
        """
        features = self.extract_features(obs)
        latent_pi, lstm_states_pi = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
        latent_pi_mlp = self.mlp_extractor.forward_actor(latent_pi)
        
        mean_actions = self.action_net(latent_pi_mlp)
        
        if isinstance(obs, dict) and 'action_mask' in obs:
            mask = obs['action_mask']
            mean_actions = mean_actions + (1.0 - mask) * -1e8
            
        return self.action_dist.proba_distribution(action_logits=mean_actions), lstm_states_pi

    def evaluate_actions_with_aux(self, obs, actions, lstm_states, episode_starts):
        """
        Evaluate actions and also return auxiliary logits.
        We have to re-implement parts of evaluate_actions to get the shared features.
        """
        features = self.extract_features(obs)
        
        # Process sequence for actor and critic separately
        latent_pi, lstm_states_pi = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)
        latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        
        # Standard PPO heads
        latent_pi_mlp = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf_mlp = self.mlp_extractor.forward_critic(latent_vf)
        
        mean_actions = self.action_net(latent_pi_mlp)
        
        if isinstance(obs, dict) and 'action_mask' in obs:
            mask = obs['action_mask']
            mean_actions = mean_actions + (1.0 - mask) * -1e8
            
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf_mlp)
        entropy = distribution.entropy()
        
        # Auxiliary Head (we predict from latent_pi to have access to actor context, or just from lstm output)
        # We use latent_pi as it contains the processed memory state
        aux_logits = self.aux_head(latent_pi)
        
        return values, log_prob, entropy, aux_logits

class CustomPPO(RecurrentPPO):
    def __init__(self, *args, c_aux=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_aux = c_aux
        
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        entropy_losses = []
        pg_losses, value_losses, aux_losses = [], [], []
        clip_fractions = []

        continue_training = True
        
        # Evaluate clip_range schedules
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, aux_logits = self.policy.evaluate_actions_with_aux(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                
                values = values.flatten()
                
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                
                # Value loss using the TD(lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # Auxiliary Loss (BCEWithLogitsLoss)
                aux_target = rollout_data.observations['aux_target']
                bce_loss = F.binary_cross_entropy_with_logits(aux_logits, aux_target)
                aux_losses.append(bce_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.c_aux * bce_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/aux_loss", np.mean(aux_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", self.clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", self.clip_range_vf)

