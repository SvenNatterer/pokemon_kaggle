import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import math

from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import RecurrentMultiInputActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.utils import explained_variance

class PokemonTCGRecurrentPolicy(RecurrentMultiInputActorCriticPolicy):
    def __init__(
        self,
        *args,
        use_belief_actor=True,
        belief_dim=64,
        detach_belief_actor=True,
        **kwargs,
    ):
        kwargs.pop("use_zone_aux", None)
        kwargs.pop("full", None)
        kwargs.pop("balanced", None)
        kwargs.pop("compact_no_legacy", None)
        super().__init__(*args, **kwargs)

        self.use_belief_actor = bool(use_belief_actor)
        self.belief_dim = int(belief_dim)
        self.detach_belief_actor = bool(detach_belief_actor)

        lstm_hidden_dim = self.lstm_actor.hidden_size
        if self.use_belief_actor:
            self.belief_encoder = nn.Sequential(
                nn.Linear(lstm_hidden_dim, self.belief_dim),
                nn.ReLU(),
            )
            self.aux_head = nn.Sequential(
                nn.Linear(self.belief_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 2000), # aux_dim = 2000
            )

            old_action_net = self.action_net
            self.action_net = nn.Linear(old_action_net.in_features + self.belief_dim, old_action_net.out_features)
            nn.init.orthogonal_(self.action_net.weight, gain=0.01)
            nn.init.constant_(self.action_net.bias, 0.0)
        else:
            # Legacy architecture: keep parameter names/shapes compatible with existing checkpoints.
            self.aux_head = nn.Sequential(
                nn.Linear(lstm_hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 2000), # aux_dim = 2000
            )

        self.structured_options = bool(
            getattr(self.features_extractor, "structured_v2", False)
            and hasattr(self.features_extractor, "encode_options")
        )
        if self.structured_options:
            option_dim = int(getattr(self.features_extractor, "option_encoder")[-2].out_features)
            self.option_scorer = nn.Sequential(
                nn.Linear(self.action_net.in_features + option_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        if self.use_belief_actor:
            # The actor input dimension changed; rebuild once after all V2 heads exist.
            initial_lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer = self.optimizer_class(self.parameters(), lr=initial_lr, **self.optimizer_kwargs)
        else:
            new_parameters = list(self.aux_head.parameters())
            if self.structured_options:
                new_parameters.extend(self.option_scorer.parameters())
            self.optimizer.add_param_group({'params': new_parameters})

    def load_state_dict(self, state_dict, strict=True):
        if "action_net.weight" in state_dict:
            weight_in = state_dict["action_net.weight"].shape[1]
            if weight_in != self.action_net.in_features:
                self.use_belief_actor = (weight_in == 128)
                self.belief_dim = 64
                old_action_net = self.action_net
                device = old_action_net.weight.device
                self.action_net = nn.Linear(weight_in, old_action_net.out_features).to(device)
                if hasattr(self, "option_scorer") and hasattr(self.features_extractor, "option_encoder"):
                    option_dim = int(getattr(self.features_extractor, "option_encoder")[-2].out_features)
                    self.option_scorer = nn.Sequential(
                        nn.Linear(weight_in + option_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                    ).to(device)
        return super().load_state_dict(state_dict, strict=False)

    def _actor_latent_with_belief(self, latent_memory, latent_actor):
        if self.action_net.in_features == 128:
            self.use_belief_actor = True

        if not getattr(self, "use_belief_actor", False):
            return latent_actor, None

        if not hasattr(self, "belief_encoder"):
            self.belief_encoder = nn.Sequential(
                nn.Linear(latent_memory.shape[-1], getattr(self, "belief_dim", 64)),
                nn.ReLU(),
            ).to(latent_memory.device)

        belief_embedding = self.belief_encoder(latent_memory)
        actor_belief = belief_embedding.detach() if getattr(self, "detach_belief_actor", True) else belief_embedding
        return torch.cat([latent_actor, actor_belief], dim=-1), belief_embedding

    def _action_logits(self, obs, actor_latent):
        """Score encoded legal options with shared weights, then apply legality."""
        logits = self.action_net(actor_latent)
        if getattr(self, "structured_options", False) and hasattr(self, "option_scorer"):
            option_embeddings = self.features_extractor.take_option_embedding_cache()
            if option_embeddings is None:
                option_embeddings = self.features_extractor.encode_options(obs)
            option_count = option_embeddings.shape[-2]
            expanded_state = actor_latent.unsqueeze(-2).expand(
                *actor_latent.shape[:-1], option_count, actor_latent.shape[-1]
            )
            option_scores = self.option_scorer(
                torch.cat([expanded_state, option_embeddings], dim=-1)
            ).squeeze(-1)
            logits = logits.clone()
            logits[..., :option_count] = option_scores

        if isinstance(obs, dict) and 'action_mask' in obs:
            mask = obs['action_mask']
            logits = logits + (1.0 - mask) * -1e8
        return logits

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

        latent_pi_memory = latent_pi
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        latent_pi, _ = self._actor_latent_with_belief(latent_pi_memory, latent_pi)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        
        # APPLY ACTION MASKING HERE
        mean_actions = self._action_logits(obs, latent_pi)
            
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
        latent_pi_mlp, _ = self._actor_latent_with_belief(latent_pi, latent_pi_mlp)
        
        mean_actions = self._action_logits(obs, latent_pi_mlp)
            
        return self.action_dist.proba_distribution(action_logits=mean_actions), lstm_states_pi

    def evaluate_actions_with_aux(
        self, obs, actions, lstm_states, episode_starts, compute_aux=True
    ):
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
        latent_pi_mlp, belief_embedding = self._actor_latent_with_belief(latent_pi, latent_pi_mlp)
        
        mean_actions = self._action_logits(obs, latent_pi_mlp)
            
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf_mlp)
        entropy = distribution.entropy()
        
        # Auxiliary Head: with belief actor enabled, the supervised belief embedding is also fed to the actor.
        aux_logits = None
        if compute_aux:
            aux_input = belief_embedding if self.use_belief_actor else latent_pi
            aux_logits = self.aux_head(aux_input)
        
        return values, log_prob, entropy, aux_logits, mean_actions

class CustomPPO(RecurrentPPO):
    def __init__(self, *args, c_aux=0.5, distill_coef=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_aux = c_aux
        self.distill_coef = distill_coef

    def set_parameters(self, load_path_or_dict, exact_match=True, device="auto"):
        if isinstance(load_path_or_dict, dict):
            params = dict(load_path_or_dict)
            params.pop("policy.optimizer", None)
            params.pop("optimizer", None)
            params.pop("optimizer_state_dict", None)
            return super().set_parameters(params, exact_match=False, device=device)
        return super().set_parameters(load_path_or_dict, exact_match=exact_match, device=device)

    @staticmethod
    def _sparse_card_distribution_loss(logits, targets):
        """Cross entropy loss normalized over non-zero target slots."""
        if targets.numel() == 0 or logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)
        mask = targets > 0
        if not mask.any():
            return torch.tensor(0.0, device=logits.device), torch.tensor(1.0, device=logits.device)
        num_classes = logits.size(-1)
        if logits.dim() == 2 and targets.dim() == 2:
            num_slots = targets.size(1)
            logits = logits.unsqueeze(1).expand(-1, num_slots, -1)
        flat_logits = logits.reshape(-1, num_classes)
        flat_targets = targets.reshape(-1)
        valid_indices = flat_targets > 0
        valid_logits = flat_logits[valid_indices]
        valid_targets = flat_targets[valid_indices]
        loss = F.cross_entropy(valid_logits, valid_targets) / math.log(num_classes)
        preds = valid_logits.argmax(dim=-1)
        acc = (preds == valid_targets).float().mean()
        return loss, acc
        
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        entropy_losses = []
        pg_losses, value_losses, aux_losses, distill_losses = [], [], [], []
        aux_precision_at_20, aux_recall_at_20, aux_count_scaled_mae = [], [], []
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

                # Recurrent rollout batches contain padded sequence steps.
                # They must be excluded from every PPO statistic and loss term.
                mask = rollout_data.mask > 1e-8

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, aux_logits, action_logits = self.policy.evaluate_actions_with_aux(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                    compute_aux=self.c_aux != 0.0,
                )
                
                values = values.flatten()
                
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2)[mask].mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()[mask]).item()
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
                value_loss = F.mse_loss(rollout_data.returns[mask], values_pred[mask])
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob[mask])
                else:
                    entropy_loss = -torch.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                aux_loss = None
                if aux_logits is not None:
                    # Count-aware hidden-card loss. Targets are log-scaled to [0, 1].
                    aux_target = rollout_data.observations['aux_target']
                    valid_aux_logits = aux_logits[mask]
                    valid_aux_target = aux_target[mask]
                    present_mask = valid_aux_target > 0
                    positive_count = present_mask.sum()
                    negative_count = valid_aux_target.numel() - positive_count
                    positive_weight = torch.clamp(
                        negative_count / torch.clamp(positive_count, min=1.0),
                        min=1.0,
                        max=20.0,
                    )
                    aux_prediction = torch.sigmoid(valid_aux_logits)
                    aux_element_loss = F.smooth_l1_loss(
                        aux_prediction, valid_aux_target, reduction="none", beta=0.1
                    )
                    aux_weights = torch.where(
                        present_mask,
                        positive_weight,
                        torch.ones_like(valid_aux_target),
                    )
                    aux_loss = (
                        (aux_element_loss * aux_weights).sum()
                        / aux_weights.sum().clamp_min(1.0)
                    )
                    aux_losses.append(aux_loss.item())

                    with torch.no_grad():
                        top_k = min(20, valid_aux_logits.shape[-1])
                        top_indices = torch.topk(valid_aux_logits, k=top_k, dim=-1).indices
                        top_hits = torch.gather(present_mask, 1, top_indices).sum(dim=1)
                        aux_precision_at_20.append((top_hits / top_k).mean().item())
                        positives_per_step = present_mask.sum(dim=1).clamp_min(1.0)
                        aux_recall_at_20.append((top_hits / positives_per_step).mean().item())
                        if present_mask.any():
                            aux_count_scaled_mae.append(
                                torch.abs(
                                    aux_prediction[present_mask] - valid_aux_target[present_mask]
                                ).mean().item()
                            )

                distill_loss = None
                if self.distill_coef > 0.0 and "teacher_action" in rollout_data.observations:
                    teacher_actions = rollout_data.observations["teacher_action"].squeeze(-1)
                    valid_teacher_mask = (teacher_actions >= 0) & mask
                    if valid_teacher_mask.any():
                        distill_loss = F.cross_entropy(action_logits[valid_teacher_mask], teacher_actions[valid_teacher_mask].long())
                        distill_losses.append(distill_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if aux_loss is not None:
                    loss = loss + self.c_aux * aux_loss
                if distill_loss is not None:
                    loss = loss + self.distill_coef * distill_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean(((torch.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
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

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/aux_loss", np.mean(aux_losses) if aux_losses else 0.0)
        self.logger.record("train/distill_loss", np.mean(distill_losses) if distill_losses else 0.0)
        self.logger.record(
            "train/aux_precision_at_20",
            np.mean(aux_precision_at_20) if aux_precision_at_20 else 0.0,
        )
        self.logger.record(
            "train/aux_recall_at_20",
            np.mean(aux_recall_at_20) if aux_recall_at_20 else 0.0,
        )
        self.logger.record(
            "train/aux_count_scaled_mae",
            np.mean(aux_count_scaled_mae) if aux_count_scaled_mae else 0.0,
        )
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", self.clip_range_vf)
