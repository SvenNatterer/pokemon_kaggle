"""Single runtime loader used by arena, evaluation, and replay code."""

from __future__ import annotations

from src.agents.rule_based_agent import RuleBasedPokemonAgent, is_rule_based_model_spec
from src.model_paths import strip_zip_suffix


def load_bot(model_path: str | None, env=None):
    if is_rule_based_model_spec(model_path):
        return RuleBasedPokemonAgent()
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
    raise RuntimeError(f"could not load bot {model_path}: {'; '.join(errors)}")
