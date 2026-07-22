"""Wrapper for executing Kaggle Python heuristic agents within the environment and league."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

# Ensure `cg` resolves to `src.cg` in sys.modules to avoid double-loading libcg
try:
    import src.cg as _src_cg
    import src.cg.api as _src_cg_api
    if "cg" not in sys.modules:
        sys.modules["cg"] = _src_cg
    if "cg.api" not in sys.modules:
        sys.modules["cg.api"] = _src_cg_api
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[3]


def resolve_script_path(spec_or_path: str | Path) -> Path:
    s = str(spec_or_path).strip()
    if s.startswith("python_script:"):
        s = s[len("python_script:") :]
    p = Path(s)
    if p.is_file():
        return p
    if (ROOT / p).is_file():
        return ROOT / p
    if (ROOT / "src" / "agents" / "kaggle_bots" / p.name).is_file():
        return ROOT / "src" / "agents" / "kaggle_bots" / p.name
    return p


def is_python_script_agent_spec(value: Any) -> bool:
    if not value:
        return False
    s = str(value).strip()
    if s.startswith("python_script:"):
        return True
    if s.endswith(".py"):
        p = resolve_script_path(s)
        return p.is_file()
    return False


class KagglePythonAgentWrapper:
    """Wraps a standalone Kaggle python script `agent(obs_dict, config)` for compatibility with load_bot."""

    def __init__(self, script_path: str | Path):
        self.script_path = resolve_script_path(script_path)
        if not self.script_path.is_file():
            raise FileNotFoundError(f"Kaggle agent script not found: {self.script_path}")

        module_name = f"kaggle_agent_{self.script_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, self.script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec for {self.script_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "agent") or not callable(module.agent):
            raise AttributeError(f"Script {self.script_path} does not export an `agent` function")

        self._agent_fn: Callable[[dict[str, Any]], Any] = module.agent

    def predict(
        self,
        observation: dict[str, Any] | np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray | int, Any]:
        """Invoke Kaggle Python agent with observation dictionary and return selected action index."""
        if isinstance(observation, dict):
            obs_payload = observation
        else:
            obs_payload = {"vector": observation}

        try:
            raw_action = self._agent_fn(obs_payload)
        except Exception:
            raw_action = [0]

        if isinstance(raw_action, list) and raw_action:
            action = int(raw_action[0])
        elif isinstance(raw_action, (int, np.integer)):
            action = int(raw_action)
        else:
            action = 0

        return np.array(action, dtype=np.int64), state


def load_python_script_agent(script_path: str | Path) -> KagglePythonAgentWrapper:
    return KagglePythonAgentWrapper(script_path)
