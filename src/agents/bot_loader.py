"""Single runtime loader used by arena, evaluation, and replay code."""

from __future__ import annotations

import os

from src.agents.rule_based_agent import (
    RuleBasedPokemonAgent,
    is_rule_based_model_spec,
    rule_based_profile_from_spec,
)
from src.league.model_paths import strip_zip_suffix


def _resolve_load_path(model_path: str) -> str:
    """Prefer an explicit archive over a same-named extracted directory."""
    if os.path.isfile(model_path):
        return model_path
    return strip_zip_suffix(model_path)


def load_bot(model_path: str | None, env=None):
    if is_rule_based_model_spec(model_path):
        return RuleBasedPokemonAgent(spec=model_path)
    if not model_path:
        raise ValueError("a PPO bot requires a model path")

    from stable_baselines3 import PPO
    from src.training.custom_ppo import CustomPPO

    from stable_baselines3.common.save_util import load_from_zip_file

    path = _resolve_load_path(model_path)
    errors = []
    for loader in (CustomPPO, PPO):
        attempts = ({"env": env}, {}) if env is not None else ({},)
        for kwargs in attempts:
            try:
                bot = loader.load(path, custom_objects={"optimizer": None}, **kwargs)
                if hasattr(bot, "policy") and hasattr(bot.policy, "features_extractor"):
                    extractor = bot.policy.features_extractor
                    if hasattr(extractor, "use_card_table"):
                        extractor.use_card_table = True
                return bot
            except Exception as exc:
                suffix = " with env" if kwargs else ""
                errors.append(f"{loader.__name__}{suffix}: {exc}")

    # Fallback for legacy checkpoints with optimizer group mismatches
    try:
        data, params, pytorch_variables = load_from_zip_file(path)
        if params:
            params.pop("policy.optimizer", None)
            params.pop("optimizer", None)
            params.pop("optimizer_state_dict", None)
        bot = CustomPPO(policy=data["policy_class"], env=env, _init_setup_model=False)
        bot.__dict__.update(data)
        bot._setup_model()
        bot.optimizer = None
        bot.set_parameters(params, exact_match=False)
        if hasattr(bot, "policy") and hasattr(bot.policy, "features_extractor"):
            extractor = bot.policy.features_extractor
            if hasattr(extractor, "use_card_table"):
                extractor.use_card_table = True
        return bot
    except Exception as exc:
        errors.append(f"Fallback: {exc}")

    raise RuntimeError(f"Could not load bot from {path}: {'; '.join(errors)}")
