"""Single runtime loader used by arena, evaluation, and replay code."""

from __future__ import annotations

from src.agents.rule_based_agent import (
    RuleBasedPokemonAgent,
    is_rule_based_model_spec,
    rule_based_profile_from_spec,
)
from src.model_paths import strip_zip_suffix


def _load_legacy_structured_model(loader, path):
    """Rebuild historical structured checkpoints with their original extractor."""
    from stable_baselines3.common.save_util import load_from_zip_file
    from src.legacy_policy import LegacyStructuredFeatureExtractor

    data, _, _ = load_from_zip_file(path, device="auto")
    policy_kwargs = dict(data.get("policy_kwargs") or {})
    policy_kwargs["features_extractor_class"] = LegacyStructuredFeatureExtractor
    policy_kwargs["features_extractor_kwargs"] = {}
    return loader.load(path, custom_objects={"policy_kwargs": policy_kwargs})


def load_bot(model_path: str | None, env=None):
    if is_rule_based_model_spec(model_path):
        return RuleBasedPokemonAgent(profile=rule_based_profile_from_spec(model_path))
    if not model_path:
        raise ValueError("a PPO bot requires a model path")

    from stable_baselines3 import PPO
    from src.custom_ppo import CustomPPO

    path = strip_zip_suffix(model_path)
    errors = []
    for loader in (CustomPPO, PPO):
        attempts = ({"env": env}, {}) if env is not None else ({},)
        for kwargs in attempts:
            try:
                return loader.load(path, **kwargs)
            except Exception as exc:
                suffix = " with env" if kwargs else ""
                errors.append(f"{loader.__name__}{suffix}: {exc}")
        try:
            return _load_legacy_structured_model(loader, path)
        except Exception as exc:
            errors.append(f"{loader.__name__} legacy structured: {exc}")
    raise RuntimeError(f"could not load bot {model_path}: {'; '.join(errors)}")
