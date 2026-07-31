"""Shared construction helpers for fresh training models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.models.custom_policy import PokemonTCGFeatureExtractor
from src.training.custom_ppo import CustomPPO, PokemonTCGRecurrentPolicy


def build_fresh_custom_ppo(env: Any, args: Any) -> CustomPPO:
    """Build the canonical fresh CustomPPO model for the supplied training args."""
    features_dim = getattr(args, "features_dim", 256)
    hidden_dim = getattr(args, "hidden_dim", 128)
    policy_kwargs = dict(
        features_extractor_class=PokemonTCGFeatureExtractor,
        features_extractor_kwargs={
            "features_dim": features_dim,
            "feature_variant": args.feature_variant,
            "use_card_table": args.card_table,
            "entity_relation_mode": args.entity_relation_mode,
        },
        use_belief_actor=args.belief_actor,
        belief_dim=args.belief_dim,
        detach_belief_actor=args.belief_detach,
    )
    if features_dim != 256 or hidden_dim != 128:
        policy_kwargs["net_arch"] = dict(
            pi=[hidden_dim, hidden_dim],
            vf=[hidden_dim, hidden_dim],
        )

    return CustomPPO(
        PokemonTCGRecurrentPolicy,
        env,
        verbose=1,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.999,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        target_kl=args.target_kl,
        c_aux=args.aux_coef,
        distill_coef=args.distill_coef,
        value_distill_coef=getattr(args, "value_distill_coef", 0.0),
        seed=args.seed,
        device="cpu",
        tensorboard_log="logs/",
        policy_kwargs=policy_kwargs,
    )


def reset_policy_optimizer(policy: Any, learning_rate: float):
    """Replace a policy optimizer, discarding state from a prior training phase."""
    policy.optimizer = policy.optimizer_class(
        policy.parameters(),
        lr=learning_rate,
        **policy.optimizer_kwargs,
    )
    return policy.optimizer


def save_model_atomically(model: Any, path: str | Path) -> Path:
    """Save a Stable-Baselines archive without exposing a partial target file."""
    target = Path(path)
    if target.suffix != ".zip":
        target = target.with_name(f"{target.name}.zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}.zip")
    try:
        model.save(str(temporary))
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target
