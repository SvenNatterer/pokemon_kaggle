"""Run-level PPO stopping based on low KL and entropy stagnation."""

from __future__ import annotations

import math
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback


class LowKLAndEntropyStagnationCallback(BaseCallback):
    """Stop after several low-KL updates whose entropy loss has gone flat.

    The callback observes each completed PPO training call exactly once. SB3
    exposes the training metrics during the following rollout, so returning
    ``False`` stops collection before another policy update is performed.
    """

    def __init__(
        self,
        *,
        kl_threshold: float,
        entropy_trend_threshold: float,
        min_steps: int,
        patience: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if kl_threshold <= 0.0:
            raise ValueError("kl_threshold must be positive")
        if entropy_trend_threshold < 0.0:
            raise ValueError("entropy_trend_threshold must be non-negative")
        if min_steps < 0:
            raise ValueError("min_steps must be non-negative")
        if patience < 2:
            raise ValueError("patience must be at least 2")

        self.kl_threshold = float(kl_threshold)
        self.entropy_trend_threshold = float(entropy_trend_threshold)
        self.min_steps = int(min_steps)
        self.patience = int(patience)

        self.start_timesteps = 0
        self.last_seen_updates = 0
        self.observed_updates = 0
        self.patience_counter = 0
        self.entropy_window: deque[float] = deque(maxlen=self.patience)
        self.last_approx_kl: float | None = None
        self.last_entropy_loss: float | None = None
        self.last_entropy_trend: float | None = None
        self.triggered = False
        self.stop_reason: str | None = None
        self.stop_timesteps: int | None = None
        self.stop_run_steps: int | None = None

    def _init_callback(self) -> None:
        self.start_timesteps = int(getattr(self.model, "num_timesteps", 0))
        self.last_seen_updates = int(getattr(self.model, "_n_updates", 0))
        self._record_configuration()

    @property
    def run_steps(self) -> int:
        return max(0, int(self.num_timesteps) - self.start_timesteps)

    def _record_configuration(self) -> None:
        self.logger.record("adaptive_stop/kl_threshold", self.kl_threshold)
        self.logger.record(
            "adaptive_stop/entropy_trend_threshold", self.entropy_trend_threshold
        )
        self.logger.record("adaptive_stop/min_steps", self.min_steps)
        self.logger.record("adaptive_stop/patience", self.patience)

    def _on_step(self) -> bool:
        current_updates = int(getattr(self.model, "_n_updates", 0))
        if current_updates == self.last_seen_updates:
            return True
        self.last_seen_updates = current_updates

        values = getattr(self.logger, "name_to_value", {})
        approx_kl = values.get("train/approx_kl")
        entropy_loss = values.get("train/entropy_loss")
        try:
            approx_kl = float(approx_kl)
            entropy_loss = float(entropy_loss)
        except (TypeError, ValueError):
            self._reset_window()
            self._record_state(condition_met=False)
            return True
        if not (math.isfinite(approx_kl) and math.isfinite(entropy_loss)):
            self._reset_window()
            self._record_state(condition_met=False)
            return True

        self.observed_updates += 1
        self.last_approx_kl = approx_kl
        self.last_entropy_loss = entropy_loss

        eligible = self.run_steps >= self.min_steps
        low_kl = approx_kl < self.kl_threshold
        if not eligible or not low_kl:
            self._reset_window()
        else:
            self.entropy_window.append(entropy_loss)
            self.patience_counter = len(self.entropy_window)

        entropy_trend = self._entropy_trend()
        self.last_entropy_trend = entropy_trend
        entropy_stagnant = (
            entropy_trend is not None
            and abs(entropy_trend) <= self.entropy_trend_threshold
        )
        condition_met = (
            eligible
            and low_kl
            and self.patience_counter == self.patience
            and entropy_stagnant
        )

        self._record_state(condition_met=condition_met)
        if not condition_met:
            return True

        self.triggered = True
        self.stop_reason = "low_kl_and_entropy_loss_stagnation"
        self.stop_timesteps = int(self.num_timesteps)
        self.stop_run_steps = self.run_steps
        if self.verbose:
            print(
                "Adaptive stop: "
                f"KL={approx_kl:.6g} < {self.kl_threshold:.6g}, "
                f"absolute entropy-loss trend={abs(entropy_trend):.6g} <= "
                f"{self.entropy_trend_threshold:.6g} across "
                f"{self.patience_counter} consecutive low-KL updates after "
                f"{self.run_steps} run steps.",
                flush=True,
            )
        return False

    def _reset_window(self) -> None:
        self.entropy_window.clear()
        self.patience_counter = 0
        self.last_entropy_trend = None

    def _entropy_trend(self) -> float | None:
        """Least-squares entropy-loss slope per PPO update."""
        if len(self.entropy_window) < self.patience:
            return None
        values = tuple(self.entropy_window)
        x_mean = (len(values) - 1) / 2.0
        y_mean = sum(values) / len(values)
        denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
        if denominator == 0.0:
            return 0.0
        return sum(
            (index - x_mean) * (value - y_mean)
            for index, value in enumerate(values)
        ) / denominator

    def _record_state(self, *, condition_met: bool) -> None:
        self._record_configuration()
        self.logger.record("adaptive_stop/run_steps", self.run_steps)
        self.logger.record("adaptive_stop/observed_updates", self.observed_updates)
        self.logger.record("adaptive_stop/patience_counter", self.patience_counter)
        self.logger.record("adaptive_stop/condition_met", float(condition_met))
        if self.last_entropy_trend is not None:
            self.logger.record(
                "adaptive_stop/entropy_loss_trend", self.last_entropy_trend
            )

    def summary(self) -> dict[str, object]:
        return {
            "enabled": True,
            "kl_threshold": self.kl_threshold,
            "entropy_trend_threshold": self.entropy_trend_threshold,
            "min_steps": self.min_steps,
            "patience": self.patience,
            "patience_counter": self.patience_counter,
            "observed_updates": self.observed_updates,
            "last_approx_kl": self.last_approx_kl,
            "last_entropy_loss": self.last_entropy_loss,
            "last_entropy_trend": self.last_entropy_trend,
            "triggered": self.triggered,
            "stop_timesteps": self.stop_timesteps,
            "stop_run_steps": self.stop_run_steps,
            "stop_reason": self.stop_reason,
        }
