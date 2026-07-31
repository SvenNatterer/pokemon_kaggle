import os
import sys
import argparse
import signal
import time
import pandas as pd
from pathlib import Path
from typing import Any

# Add src to pythonpath so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
import json
if "WANDB_MODE" not in os.environ:
    os.environ["WANDB_MODE"] = "online"
import wandb
from wandb.integration.sb3 import WandbCallback
from src.agents.rule_based_agent import is_rule_based_model_spec
from src.utils import deck_display_name_for_path, model_display_name_for_path, resolve_deck_path, resolve_pool_path
from src.league.experiment_registry import git_revision, registry_path, write_experiment
from src.training.training_health import FeatureMetricsCallback, GuardrailMetricsCallback
from src.league.pfsp import PFSPLite, labels_and_weights
from src.evaluation.checkpoint_evaluation import evaluate_checkpoint

TRAINING_USES_POTENTIAL_REWARDS = True
DEFAULT_TRAINING_POOL = "decks/pools/default_training.json"

class LiveStatusCallback(BaseCallback):
    def __init__(self, action_text, total_timesteps, status_freq=10000, verbose=0):
        super(LiveStatusCallback, self).__init__(verbose)
        self.action_text = action_text
        self.total_timesteps = max(0, total_timesteps)
        self.status_freq = max(1, self.total_timesteps // 100) if self.total_timesteps > 0 else max(1, status_freq)
        self.next_status = self.status_freq

    def _init_callback(self) -> None:
        completed = int(getattr(self.model, "num_timesteps", 0))
        self.next_status = ((completed // self.status_freq) + 1) * self.status_freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_status or (self.total_timesteps > 0 and self.num_timesteps >= self.total_timesteps):
            status_data = {
                "action": self.action_text,
                "completed": self.num_timesteps,
                "total": self.total_timesteps,
                "endless": self.total_timesteps == 0,
            }
            try:
                with open("decks/status.json", "w") as f:
                    json.dump(status_data, f)
            except Exception:
                pass
            self.next_status = ((self.num_timesteps // self.status_freq) + 1) * self.status_freq
        return True

class RewardBreakdownCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        from collections import deque, defaultdict
        self.episode_rewards = defaultdict(lambda: deque(maxlen=100))
        self.episode_outcomes = deque(maxlen=100)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, done in enumerate(dones):
            if done and i < len(infos):
                info = infos[i]
                # SB3 sometimes puts terminal info in "terminal_info" dict when using wrappers
                terminal_info = info.get("terminal_info", info)
                if "reward_breakdown" in terminal_info:
                    for key, val in terminal_info["reward_breakdown"].items():
                        self.episode_rewards[key].append(val)
                try:
                    winner = int(terminal_info.get("winner", -1))
                    learner_perspective = int(terminal_info.get("learner_perspective", -1))
                except (TypeError, ValueError):
                    winner = -1
                    learner_perspective = -1
                if winner in (0, 1) and learner_perspective in (0, 1):
                    self.episode_outcomes.append(float(winner == learner_perspective))
        return True

    def _on_rollout_end(self) -> None:
        for key, recent_values in self.episode_rewards.items():
            if key in ("potential", "gameplan_bonus", "gameplan_violation_penalty"):
                continue
            if len(recent_values) > 0:
                if any(v != 0.0 for v in recent_values):
                    mean_val = sum(recent_values) / len(recent_values)
                    self.logger.record(f"rewards/{key}", mean_val)
        for k in ("gameplan_bonus", "gameplan_violation_penalty"):
            vals = self.episode_rewards.get(k, [])
            mean_val = float(sum(vals) / len(vals)) if len(vals) > 0 else 0.0
            self.logger.record(f"rewards/{k}", mean_val)
        if self.episode_outcomes:
            self.logger.record(
                "rollout/win_rate",
                sum(self.episode_outcomes) / len(self.episode_outcomes),
            )


class PFSPCallback(BaseCallback):
    """Reweight opponents from recent outcomes and log each probability."""

    def __init__(self, opponent_pool, update_steps=250_000, window_games=150, verbose=0):
        super().__init__(verbose)
        labels, weights = labels_and_weights(opponent_pool)
        self.opponent_pool = opponent_pool
        self.controller = PFSPLite(
            labels,
            weights,
            random_fraction=0.20,
            max_probability=0.18,
            window_games=window_games,
            minimum_games=25,
        )
        self.update_steps = max(1, int(update_steps))
        self.next_update = self.update_steps

    def _record_pool_metrics(self) -> None:
        probability_by_label = dict(
            zip(self.controller.labels, self.controller.current_probabilities)
        )
        for label, probability in probability_by_label.items():
            self.logger.record(f"pfsp/probability/{label}", probability)

        macro_win_rate = self.controller.recent_macro_win_rate()
        if macro_win_rate is not None:
            self.logger.record("train/pfsp_macro_win_rate", macro_win_rate)
        worst_win_rate = self.controller.recent_worst_win_rate()
        if worst_win_rate is not None:
            self.logger.record("train/pfsp_worst_win_rate", worst_win_rate)
            self.logger.record("rollout/min_win_rate", worst_win_rate)
        self.logger.record(
            "train/pfsp_effective_opponent_count",
            self.controller.effective_opponent_count(),
        )

    def _on_training_start(self) -> None:
        # Make the complete initial pool visible in W&B before the first 250k-step
        # PFSP reweighting event.
        self._record_pool_metrics()

    def _on_rollout_end(self) -> None:
        # Keep the multi-line pool plot and current weakest matchup live between
        # reweighting events, not just at PFSP boundaries.
        self._record_pool_metrics()

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if not done:
                continue
            terminal = info.get("terminal_info", info)
            label = terminal.get("opponent_label")
            try:
                winner = int(terminal.get("winner", -1))
                learner = int(terminal.get("learner_perspective", -1))
            except (TypeError, ValueError):
                continue
            outcome = 1 if winner == learner else (-1 if winner in (0, 1) else 0)
            self.controller.observe(str(label), outcome)

        if self.num_timesteps < self.next_update or self.controller.segment_games == 0:
            return True
        segment_score = sum(
            record.effective_wins for record in self.controller.segment_records.values()
        ) / self.controller.segment_games
        probabilities, segment = self.controller.finish_segment()
        probability_by_label = dict(zip(self.controller.labels, probabilities))
        self.training_env.env_method("set_opponent_probabilities", probability_by_label)
        for entry in self.opponent_pool:
            entry["weight"] = probability_by_label[str(entry["label"])]
        self.logger.record("pfsp/segment_games", segment["games"])
        self.logger.record("train/pfsp_weighted_win_rate", segment_score)
        self._record_pool_metrics()
        self.next_update = ((self.num_timesteps // self.update_steps) + 1) * self.update_steps
        return True


class CheckpointEvaluationCallback(BaseCallback):
    """Save immutable checkpoints and evaluate only validation and holdout pools."""

    def __init__(self, *, model_path, deck_path, opponent_pool, interval_steps, games_per_opponent, verbose=0):
        super().__init__(verbose)
        self.model_path = Path(model_path)
        self.deck_path = str(deck_path)
        self.opponent_pool = opponent_pool
        self.interval_steps = max(1, int(interval_steps))
        self.games_per_opponent = max(1, int(games_per_opponent))
        self.next_checkpoint = self.interval_steps

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_checkpoint:
            return True
        checkpoint_dir = self.model_path.parent / "checkpoints"
        checkpoint_path = checkpoint_dir / f"{self.model_path.stem}_{self.num_timesteps}.zip"
        save_model_atomically(self.model, checkpoint_path)
        report_path = Path("evaluation_results") / "checkpoints" / f"{checkpoint_path.stem}.json"
        report = evaluate_checkpoint(
            candidate_model=str(checkpoint_path),
            candidate_deck=self.deck_path,
            validation_manifest="decks/pools/validation_opponents.json",
            holdout_manifest="decks/pools/holdout_opponents.json",
            games_per_opponent=self.games_per_opponent,
            output_path=report_path,
            global_steps=self.num_timesteps,
            training_pool=self.opponent_pool,
        )
        for pool_name in ("validation", "holdout"):
            summary = report[pool_name]
            self.logger.record(f"checkpoint/{pool_name}/macro_win_rate", summary["macro_win_rate"])
            self.logger.record(f"checkpoint/{pool_name}/micro_win_rate", summary["micro_win_rate"])
            self.logger.record(f"checkpoint/{pool_name}/worst_win_rate", summary["worst_win_rate"])
            self.logger.record(
                f"checkpoint/{pool_name}/worst_wilson_lower_bound_95",
                summary["worst_wilson_lower_bound_95"],
            )
            for opponent in summary["opponents"]:
                self.logger.record(
                    f"checkpoint/{pool_name}/win_rate/{opponent['label']}", opponent["win_rate"]
                )
        self.logger.record("checkpoint/global_steps", self.num_timesteps)
        self.next_checkpoint = ((self.num_timesteps // self.interval_steps) + 1) * self.interval_steps
        return True

from stable_baselines3.common.monitor import Monitor
from src.env.env_wrapper import LEGACY_ACTION_SPACE_SIZE, V6_ACTION_SPACE_SIZE, PokemonTCGEnv
from src.training.training_health import TrainingHealthCallback, summarize_health, health_gate
from src.training.custom_ppo import CustomPPO
from src.training.model_factory import build_fresh_custom_ppo, save_model_atomically

def read_deck(deck_path):
    resolved = resolve_deck_path(deck_path)
    df = pd.read_csv(resolved, header=None)
    return df[0].tolist()

def endless_learn_budget(current_timesteps: int) -> int:
    return max(0, sys.maxsize - int(current_timesteps))


def resolve_model_path(model_name):
    model_path = model_name if os.path.dirname(model_name) else os.path.join("models", model_name)
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    return model_path


def save_final_model_atomically(model, model_path):
    """Save exactly one final target model without exposing a partial ZIP."""
    save_model_atomically(model, f"{model_path}.zip")


def validate_policy_action_space(model, expected_size, policy_version):
    loaded_size = int(getattr(getattr(model, "action_space", None), "n", 0))
    if loaded_size != expected_size:
        raise RuntimeError(
            f"Policy/action-space mismatch: checkpoint has {loaded_size} actions, "
            f"but --policy-version={policy_version} requires {expected_size}. "
            "V5 and V6 checkpoints are intentionally incompatible."
        )

def load_opponent_pool(pool_path):
    if not pool_path:
        return None
    with open(pool_path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Opponent pool must be a non-empty JSON list: {pool_path}")

    pool = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("deck"):
            raise ValueError(f"Opponent pool entry {index} needs a deck path")
        deck_path = entry["deck"]
        model_path = entry.get("model") or entry.get("model_path") or entry.get("policy")
        if not os.path.exists(deck_path):
            raise FileNotFoundError(f"Opponent deck not found: {deck_path}")
        if model_path and not is_rule_based_model_spec(model_path) and not os.path.exists(model_path):
            raise FileNotFoundError(f"Opponent model not found: {model_path}")
        pool.append({
            "deck": read_deck(deck_path),
            "model_path": model_path,
            "weight": float(entry.get("weight", 1.0)),
            "label": entry.get("label", os.path.basename(deck_path)),
        })
    return pool

def make_env(
    deck_path,
    opp_deck_path,
    opp_model_path,
    sparse_rewards=False,
    reward_config=None,
    opponent_pool=None,
    rotate_perspective=False,
    action_space_size=V6_ACTION_SPACE_SIZE,
    structured_v2=True,
    feature_variant="compact",
    enable_lookahead_teacher=False,
    teacher_sample_rate=0.50,
    inference_guardrails=False,
    inference_guardrail_mode="active",
    search_guardrail_rate=0.0,
    lookahead_config=None,
    enable_archetype_prediction=True,
):
    def _init():
        import torch
        torch.set_num_threads(1)
        deck = read_deck(deck_path)
        opp_deck = read_deck(opp_deck_path)
        env = PokemonTCGEnv(
            deck,
            opp_deck,
            opponent_model_path=opp_model_path,
            reward_config=reward_config,
            sparse_rewards=sparse_rewards,
            opponent_pool=opponent_pool,
            rotate_perspective=rotate_perspective,
            action_space_size=action_space_size,
            structured_v2=structured_v2,
            feature_variant=feature_variant,
            enable_lookahead_teacher=enable_lookahead_teacher,
            teacher_sample_rate=teacher_sample_rate,
            inference_guardrails=inference_guardrails,
            inference_guardrail_mode=inference_guardrail_mode,
            search_guardrail_rate=search_guardrail_rate,
            lookahead_config=lookahead_config,
            enable_archetype_prediction=enable_archetype_prediction,
        )
        return Monitor(env)
    return _init

DEFAULT_QUEUE_PATH = "configs/training_queue.json"


def load_yaml_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}
    return data


def add_to_queue(queue_path: str, item: str) -> None:
    target_path = queue_path or DEFAULT_QUEUE_PATH
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    queue_data = []
    if os.path.exists(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                queue_data = json.load(f)
                if not isinstance(queue_data, list):
                    queue_data = []
        except Exception:
            queue_data = []

    entry: Any = item
    try:
        entry = json.loads(item)
    except Exception:
        entry = item.strip()

    queue_data.append(entry)

    temp_path = f"{target_path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(queue_data, f, indent=2)
    os.replace(temp_path, target_path)
    print(f"Added item to training queue ({target_path}): {entry}")


def pop_next_queue_item(queue_path: str) -> Any:
    target_path = queue_path or DEFAULT_QUEUE_PATH
    if not os.path.exists(target_path):
        return None
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            queue_data = json.load(f)
            if not isinstance(queue_data, list) or len(queue_data) == 0:
                return None
    except Exception:
        return None

    item = queue_data.pop(0)

    temp_path = f"{target_path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(queue_data, f, indent=2)
    os.replace(temp_path, target_path)

    return item


CONFIG_ARG_MAPPING = {
    "learning_rate": "lr",
    "lr": "lr",
    "deck": "deck",
    "model_name": "model_name",
    "model": "model_name",
    "timesteps": "timesteps",
    "total_timesteps": "timesteps",
    "n_steps": "n_steps",
    "batch_size": "batch_size",
    "n_epochs": "n_epochs",
    "ent_coef": "ent_coef",
    "clip_range": "clip_range",
    "target_kl": "target_kl",
    "aux_coef": "aux_coef",
    "distill_coef": "distill_coef",
    "value_distill_coef": "value_distill_coef",
    "teacher_sample_rate": "teacher_sample_rate",
    "base_model": "base_model",
    "opp_pool": "opp_pool",
    "opp_deck": "opp_deck",
    "opp_model": "opp_model",
    "seed": "seed",
    "sparse_rewards": "sparse_rewards",
    "endless": "endless",
    "features_dim": "features_dim",
    "hidden_dim": "hidden_dim",
    "num_envs": "num_envs",
    "entity_relation_mode": "entity_relation_mode",
    "wandb_mode": "wandb_mode",
    "inference_guardrail_mode": "inference_guardrail_mode",
    "lookahead": "lookahead_config",
    "archetype_prediction": "enable_archetype_prediction",
}


def apply_config_dict(args: argparse.Namespace, config_dict: dict) -> None:
    for k, v in config_dict.items():
        if k == "rewards":
            setattr(args, "reward_config", v)
        elif k == "features" and isinstance(v, dict):
            if "archetype_prediction" in v:
                setattr(args, "enable_archetype_prediction", bool(v["archetype_prediction"]))
        elif k in CONFIG_ARG_MAPPING:
            setattr(args, CONFIG_ARG_MAPPING[k], v)
        elif hasattr(args, k):
            setattr(args, k, v)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pokemon TCG RL Training Script")
    parser.add_argument("--deck", type=str, default=None, help="Path to deck.csv")
    parser.add_argument("--model-name", type=str, default=None, help="Name of the model to save")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML experiment config file.")
    parser.add_argument("--queue", "--cue", nargs="?", const=DEFAULT_QUEUE_PATH, default=DEFAULT_QUEUE_PATH, help="Path to training queue file.")
    parser.add_argument("--add-to-queue", "--add-to-cue", type=str, default=None, help="Add a training spec to the queue file and exit.")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Number of training timesteps. Use 0 for endless training.")
    parser.add_argument("--endless", action="store_true", help="Train forever until interrupted.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--continue-existing", action="store_true", help="Explicitly continue the exact target model if it exists.")
    mode.add_argument("--overwrite", action="store_true", help="Train a new model and replace an existing target only after success.")
    parser.add_argument("--base-model", type=str, default=None, help="Path to base model checkpoint to load weights from before training.")
    parser.add_argument("--opp-deck", type=str, help="Path to opponent deck.csv", default=None)
    parser.add_argument("--opp-model", type=str, help="Path to opponent model .zip", default=None)
    parser.add_argument("--opp-pool", type=str, default=DEFAULT_TRAINING_POOL, help="JSON list of weighted opponent deck/model entries sampled per episode")
    parser.add_argument("--sparse-rewards", action="store_true", help="Use sparse rewards (+1 for win, -1 for loss)")
    parser.add_argument("--num-envs", type=int, default=7, help="Number of parallel environments (default: 7)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.008, help="Entropy coefficient")
    parser.add_argument("--n-epochs", type=int, default=2, help="PPO epochs per rollout")
    parser.add_argument("--clip-range", type=float, default=0.12, help="PPO clipping range")
    parser.add_argument("--target-kl", type=float, default=0.03, help="Stop PPO update early above this KL")
    parser.add_argument("--batch-size", type=int, default=1024, help="Minibatch size")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per env per rollout")
    parser.add_argument("--aux-coef", type=float, default=0.1, help="Weight for hidden-card count auxiliary loss")
    parser.add_argument("--distill-coef", type=float, default=0.1, help="Weight for policy distillation loss")
    parser.add_argument("--value-distill-coef", type=float, default=0.0, help="Weight for value distillation loss")
    parser.add_argument("--enable-lookahead-teacher", action="store_true", default=True, help="Enable lookahead teacher sampling")
    parser.add_argument("--no-lookahead-teacher", dest="enable_lookahead_teacher", action="store_false")
    parser.add_argument("--teacher-sample-rate", type=float, default=0.50, help="Sampling rate for lookahead teacher on complex decisions")
    parser.add_argument("--no-belief-actor", dest="belief_actor", action="store_false", help="Disable hidden-card belief actor")
    parser.add_argument("--belief-dim", type=int, default=64, help="Size of the learned belief embedding used by --belief-actor")
    parser.add_argument("--no-belief-detach", dest="belief_detach", action="store_false", help="Allow PPO loss gradients into the belief encoder")
    parser.add_argument("--no-rotate-perspective", dest="rotate_perspective", action="store_false", help="Disable random perspective rotation")
    parser.add_argument("--features-dim", type=int, default=256, help="Dimension of feature extractor output")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for policy MLP/LSTM layers")
    parser.add_argument(
        "--entity-relation-mode",
        choices=("baseline", "masked", "relational", "two_step", "python_object_relational"),
        default="baseline",
        help=(
            "Entity encoder mode: baseline, masked, relational, two-step relational, "
            "or the EXP-023 Python-object relational ablation."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases tracking mode (default: online).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Policy and vector-environment seed for reproducible experiment families")
    parser.add_argument(
        "--policy-version",
        choices=("v6",),
        default="v6",
        help="V6 uses a compact 66-action head.",
    )
    parser.add_argument(
        "--feature-variant",
        choices=("compact", "strategic_vector_v1", "strategic_vector_v2", "strategic_vector_v3"),
        default="compact",
        help="Observation encoder family: structured compact V2 or strategic scalar vector v1/v2.",
    )
    parser.add_argument(
        "--no-card-table",
        dest="card_table",
        action="store_false",
        help="Disable card-table lookup optimization.",
    )
    parser.add_argument(
        "--scalar-obs",
        action="store_true",
        help="Use the fast 1D scalar observation space instead of the structured V2 dict space.",
    )
    parser.add_argument(
        "--scalar-embeddings",
        action="store_true",
        help="Enable scalar embeddings.",
    )
    parser.add_argument(
        "--no-inference-guardrails",
        dest="inference_guardrails",
        action="store_false",
        help="Disable inference guardrails.",
    )
    parser.add_argument(
        "--inference-guardrail-mode",
        choices=("off", "shadow", "active"),
        default="active",
        help="Guardrail execution mode: off, shadow telemetry, or active masking.",
    )
    parser.add_argument(
        "--adaptive-stop",
        action="store_true",
        help="Enable adaptive stopping.",
    )
    parser.add_argument(
        "--no-pfsp-lite",
        dest="pfsp_lite",
        action="store_false",
        help="Disable PFSP-Lite.",
    )
    parser.add_argument("--pfsp-update-steps", type=int, default=250000, help="Training steps between PFSP opponent reweights.")
    parser.add_argument("--pfsp-window-games", type=int, default=150, help="Recent games retained per opponent for PFSP.")
    parser.add_argument("--checkpoint-eval-steps", type=int, default=250000, help="Training steps between saved checkpoint evaluations.")
    parser.add_argument("--checkpoint-eval-games-per-opponent", type=int, default=50, help="Validation and holdout games per opponent for each checkpoint.")
    parser.add_argument(
        "--search-guardrail-rate",
        type=float,
        default=0.0,
        help="Search guardrail rate.",
    )
    parser.add_argument(
        "--no-health-gate",
        dest="health_gate",
        action="store_false",
        help="Disable health gate.",
    )
    parser.add_argument(
        "--reserved-opponents", action="append", default=[
            "decks/holdout_opponents.json",
            "decks/validation_opponents.json",
        ],
        help="Opponent manifest reserved for validation/final evaluation; training overlap is rejected.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=int(os.getenv("TORCH_THREADS", "2")),
        help="Number of PyTorch CPU threads for main process (default: 2 for Daytime mode, 0 for Nighttime/full speed).",
    )
    parser.add_argument(
        "--no-archetype-prediction",
        dest="enable_archetype_prediction",
        action="store_false",
        help="Disable opponent archetype prediction feature metric updates.",
    )
    parser.add_argument(
        "--no-extra-metrics",
        dest="extra_metrics",
        action="store_false",
        help="Disable extra guardrail/feature telemetry callbacks.",
    )
    parser.set_defaults(
        belief_actor=True,
        belief_detach=True,
        card_table=True,
        extra_metrics=True,
        inference_guardrails=True,
        rotate_perspective=True,
        pfsp_lite=True,
        health_gate=True,
        enable_archetype_prediction=True,
    )
    return parser


def parse_args_with_config(argv: list[str] | None = None, parser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    if parser is None:
        parser = build_arg_parser()
    args = parser.parse_args() if argv is None else parser.parse_args(argv)
    if args.config:
        yaml_dict = load_yaml_config(args.config)
        cli_supplied = set()
        if argv is not None:
            for tok in argv:
                if tok.startswith("--"):
                    flag = tok.lstrip("-").split("=")[0].replace("-", "_")
                    cli_supplied.add(flag)
        for k, v in yaml_dict.items():
            mapped_key = CONFIG_ARG_MAPPING.get(k, k)
            if mapped_key not in cli_supplied and (
                hasattr(args, mapped_key) or k in CONFIG_ARG_MAPPING
            ):
                setattr(args, mapped_key, v)
        if "rewards" in yaml_dict and not getattr(args, "reward_config", None):
            setattr(args, "reward_config", yaml_dict["rewards"])
        if "features" in yaml_dict and isinstance(yaml_dict["features"], dict):
            if "archetype_prediction" in yaml_dict["features"]:
                setattr(args, "enable_archetype_prediction", bool(yaml_dict["features"]["archetype_prediction"]))

    # Defaults for deck and model_name if missing
    if not args.deck:
        if os.path.exists("decks/deck_bank/bank_18.csv"):
            args.deck = "decks/deck_bank/bank_18.csv"
        elif os.path.exists("decks/deck_18.csv"):
            args.deck = "decks/deck_18.csv"
    if args.deck and not args.model_name:
        deck_stem = Path(args.deck).stem
        args.model_name = f"models/ppo_{deck_stem}.zip"
    return args


def parse_queue_item(item: Any, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if isinstance(item, dict):
        args = parser.parse_args([])
        if "config" in item:
            yaml_dict = load_yaml_config(item["config"])
            apply_config_dict(args, yaml_dict)
        apply_config_dict(args, item)
        return args
    elif isinstance(item, str):
        item_str = item.strip()
        if item_str.endswith(".yaml") or item_str.endswith(".yml") or item_str.startswith("configs/"):
            return parse_args_with_config(["--config", item_str], parser)
        elif item_str.startswith("{"):
            try:
                dict_val = json.loads(item_str)
                return parse_queue_item(dict_val, parser)
            except Exception:
                pass
        import shlex
        tokens = shlex.split(item_str)
        return parse_args_with_config(tokens, parser)
    else:
        raise ValueError(f"Unrecognized queue item: {item}")


def run_single_training(args: argparse.Namespace) -> None:
    # Keep online experiment tracking the default even when a parent shell has
    # a stale WANDB_MODE setting. Offline/disabled operation remains explicit.
    os.environ["WANDB_MODE"] = args.wandb_mode
    torch_threads = getattr(args, "torch_threads", 2)
    if torch_threads is not None and torch_threads > 0:
        import torch
        torch.set_num_threads(torch_threads)
        print(f"[Daytime Mode] PyTorch main process restricted to {torch_threads} CPU threads for lag-free operation.")
    else:
        print("[Nighttime Mode] PyTorch main process using default full-capacity CPU threads.")

    opp_deck_path = args.opp_deck if args.opp_deck else args.deck
    from scripts.check_holdout_safe import check_paths
    reserved_files = ["decks/holdout_opponents.json", *args.reserved_opponents]
    for holdout_file in dict.fromkeys(reserved_files):
        resolved_path = resolve_pool_path(holdout_file)
        if not resolved_path.is_file():
            continue
        check_paths(
            str(resolved_path),
            [opp_deck_path],
            [args.opp_model] if args.opp_model else [],
            [args.opp_pool] if args.opp_pool else [],
        )
    opponent_pool = load_opponent_pool(args.opp_pool)
    action_space_size = V6_ACTION_SPACE_SIZE
    reward_config = getattr(args, "reward_config", None)
    
    opponent_description = f"pool {args.opp_pool}" if opponent_pool else opp_deck_path
    print(f"Initializing environment with {args.num_envs} workers for deck {args.deck} against {opponent_description}...")
    env = SubprocVecEnv([
        make_env(
            args.deck,
            opp_deck_path,
            args.opp_model,
            args.sparse_rewards,
            reward_config=reward_config,
            opponent_pool=opponent_pool,
            rotate_perspective=args.rotate_perspective,
            action_space_size=action_space_size,
            structured_v2=not args.scalar_obs,
            feature_variant=args.feature_variant,
            enable_lookahead_teacher=args.enable_lookahead_teacher,
            teacher_sample_rate=args.teacher_sample_rate,
            inference_guardrails=args.inference_guardrails,
            inference_guardrail_mode=args.inference_guardrail_mode,
            search_guardrail_rate=args.search_guardrail_rate,
            lookahead_config=getattr(args, "lookahead_config", None),
            enable_archetype_prediction=getattr(args, "enable_archetype_prediction", True),
        )
        for _ in range(args.num_envs)
    ])
    if args.seed is not None:
        env.seed(args.seed)
    
    model_path = resolve_model_path(args.model_name)
    experiment_file = registry_path(model_path)
    experiment = {
        "schema_version": 1,
        "status": "running",
        "model_path": f"{model_path}.zip",
        "git_revision": git_revision(),
        "arguments": vars(args),
        "reserved_opponent_manifests": [path for path in reserved_files if os.path.exists(path)],
    }
    write_experiment(experiment_file, experiment)

    base_model_path = getattr(args, "base_model", None)
    should_load_existing = args.continue_existing or (base_model_path is not None)
    source_model_path = resolve_model_path(base_model_path) if base_model_path else model_path

    target_exists = os.path.exists(f"{model_path}.zip")
    if target_exists and not args.continue_existing and not args.overwrite:
        env.close()
        raise FileExistsError(
            f"Target model already exists: {model_path}.zip. Use --continue-existing to continue that exact "
            "final model or --overwrite to deliberately train a new replacement."
        )
    source_exists = os.path.exists(f"{source_model_path}.zip")
    if should_load_existing and not source_exists:
        env.close()
        raise FileNotFoundError(f"Cannot continue missing source model: {source_model_path}.zip")

    if should_load_existing:
        print(f"Loading base checkpoint {source_model_path}.zip to train target {model_path}.zip...")
        model = CustomPPO.load(source_model_path, env=env, device="cpu")
        try:
            validate_policy_action_space(model, action_space_size, args.policy_version)
        except RuntimeError:
            env.close()
            raise
        loaded_structured_v2 = bool(
            getattr(model.policy.features_extractor, "structured_v2", False)
        )
        if not loaded_structured_v2 and args.feature_variant not in {"strategic_vector_v1", "strategic_vector_v2", "strategic_vector_v3"}:
            env.close()
            raise RuntimeError(
                f"Model {model_path}.zip uses the legacy scalar-card observation and "
                "cannot be resumed as the requested feature variant. Keep it as an --opp-model and choose a "
                "fresh --model-name such as models/ppo_v5_deck_<id>.zip."
            )
        loaded_feature_variant = str(
            getattr(model.policy.features_extractor, "feature_variant", "full")
        )
        if loaded_feature_variant != args.feature_variant:
            env.close()
            raise RuntimeError(
                f"Feature variant mismatch: checkpoint uses {loaded_feature_variant}, "
                f"but --feature-variant={args.feature_variant}. Start a fresh model."
            )
        loaded_entity_relation_mode = str(
            getattr(model.policy.features_extractor, "entity_relation_mode", "baseline")
        )
        if loaded_entity_relation_mode != args.entity_relation_mode:
            env.close()
            raise RuntimeError(
                "Entity-relation mode mismatch: checkpoint uses "
                f"{loaded_entity_relation_mode}, but --entity-relation-mode="
                f"{args.entity_relation_mode}. Start a fresh model."
            )
        loaded_card_table = bool(
            getattr(model.policy.features_extractor, "use_card_table", False)
        )
        if loaded_card_table != args.card_table:
            if args.card_table and not loaded_card_table:
                print("Enabling the output-equivalent card table on the loaded checkpoint...")
                model.policy.features_extractor.use_card_table = True
                policy_kwargs = dict(getattr(model, "policy_kwargs", {}) or {})
                extractor_kwargs = dict(
                    policy_kwargs.get("features_extractor_kwargs", {}) or {}
                )
                extractor_kwargs["use_card_table"] = True
                policy_kwargs["features_extractor_kwargs"] = extractor_kwargs
                model.policy_kwargs = policy_kwargs
            else:
                env.close()
                raise RuntimeError(
                    "Card-table mismatch: checkpoint uses card_table=True, but "
                    "--card-table was omitted. Keep the saved setting when continuing."
                )
        loaded_n_steps = int(getattr(model, "n_steps", args.n_steps))
        if loaded_n_steps != args.n_steps:
            raise RuntimeError(
                f"Cannot continue with --n-steps={args.n_steps}: saved rollout buffer "
                f"uses n_steps={loaded_n_steps}. Keep the saved value or start a fresh model."
            )
        loaded_belief_actor = bool(getattr(model.policy, "use_belief_actor", False))
        if args.belief_actor and not loaded_belief_actor:
            raise RuntimeError(
                "--belief-actor was requested, but the existing model uses the legacy actor. "
                "Use a fresh --model-name for the belief-actor experiment."
            )
        if loaded_belief_actor and not args.belief_actor:
            print("Loaded a belief-actor model; continuing with its saved architecture.")
        model.c_aux = args.aux_coef
        model.distill_coef = args.distill_coef
        model.value_distill_coef = getattr(args, "value_distill_coef", 0.0)
        model.ent_coef = args.ent_coef
        model.learning_rate = args.lr
        from stable_baselines3.common.utils import get_schedule_fn
        model.lr_schedule = get_schedule_fn(args.lr)
        model.clip_range = get_schedule_fn(args.clip_range)
        model.target_kl = args.target_kl
        model.n_epochs = args.n_epochs
        model.batch_size = args.batch_size
        if hasattr(model, 'policy') and hasattr(model.policy, 'optimizer'):
            for param_group in model.policy.optimizer.param_groups:
                param_group['lr'] = args.lr
    else:
        model = build_fresh_custom_ppo(env, args)
    
    endless_training = args.endless or args.timesteps <= 0
    if endless_training:
        print("Starting endless training without periodic saves; interrupt gracefully to save the target model.")
    else:
        print(f"Starting training for {args.timesteps} timesteps...")

    deck_id = args.deck.split('_')[-1].split('.')[0]
    opp_id = "pool" if opponent_pool else opp_deck_path.split('_')[-1].split('.')[0]
    deck_name = model_display_name_for_path(f"{model_path}.zip", args.deck)
    opp_name = "Opponent League" if opponent_pool else deck_display_name_for_path(opp_deck_path)

    action_text = f"🧠 Training: {deck_name} vs {opp_name}"
    
    run_suffix = "endless" if endless_training else str(args.timesteps)
    if "WANDB_NAME" in os.environ:
        run_name = os.environ["WANDB_NAME"]
    elif getattr(args, "config", None):
        config_stem = Path(args.config).stem
        run_name = f"{config_stem}_{run_suffix}"
    elif getattr(args, "model_name", None):
        model_stem = Path(args.model_name).stem
        run_name = f"{model_stem}_{run_suffix}"
    else:
        run_name = f"D{deck_id}_vs_D{opp_id}_{run_suffix}"
    
    run = None
    if args.wandb_mode != "disabled":
        run = wandb.init(
            project="pokemon_kaggle",
            name=run_name,
            group=os.environ.get("WANDB_RUN_GROUP", f"deck_{deck_id}"),
            config=vars(args),
            sync_tensorboard=True,
            monitor_gym=True,
            # Dirty replay files can make W&B's Git diff snapshot several gigabytes.
            save_code=False,
            dir="logs/",
            mode=args.wandb_mode,
        )
    tb_run_id = getattr(run, "id", None) or str(int(time.time()))
    tb_log_name = os.environ.get("TB_LOG_NAME", f"Deck_{deck_id}_{tb_run_id}")
    if run is not None:
        run.config.update({"tb_log_name": tb_log_name}, allow_val_change=True)
    
    status_total = 0 if endless_training else args.timesteps
    live_status_callback = LiveStatusCallback(action_text=action_text, total_timesteps=status_total)
    reward_callback = RewardBreakdownCallback()
    callbacks_list = [live_status_callback, reward_callback]
    if opponent_pool and args.pfsp_lite:
        callbacks_list.append(
            PFSPCallback(
                opponent_pool,
                update_steps=args.pfsp_update_steps,
                window_games=args.pfsp_window_games,
            )
        )
    if opponent_pool:
        callbacks_list.append(
            CheckpointEvaluationCallback(
                model_path=model_path,
                deck_path=args.deck,
                opponent_pool=opponent_pool,
                interval_steps=args.checkpoint_eval_steps,
                games_per_opponent=args.checkpoint_eval_games_per_opponent,
            )
        )
    if run is not None:
        callbacks_list.append(
            WandbCallback(
                gradient_save_freq=0,
                verbose=2,
            )
        )
    if args.extra_metrics:
        callbacks_list.append(GuardrailMetricsCallback())
        callbacks_list.append(FeatureMetricsCallback())
    callbacks = CallbackList(callbacks_list)

    def handle_stop_signal(signum, frame):
        raise KeyboardInterrupt

    old_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, handle_stop_signal)

    save_model = False
    try:
        if endless_training:
            while True:
                model.learn(
                    total_timesteps=max(1, args.n_steps * args.num_envs),
                    callback=callbacks,
                    tb_log_name=tb_log_name,
                    reset_num_timesteps=False,
                )
        else:
            model.learn(
                total_timesteps=args.timesteps,
                callback=callbacks,
                tb_log_name=tb_log_name,
                reset_num_timesteps=False,
            )
        print("Training finished! Saving model...")
        save_model = True
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model before shutdown...")
        save_model = True
    finally:
        if save_model:
            save_final_model_atomically(model, model_path)
            print(f"Model saved to {model_path}.zip")
            experiment.update({"status": "completed", "num_timesteps": int(model.num_timesteps)})
        else:
            experiment["status"] = "failed"
        write_experiment(experiment_file, experiment)
        try:
            env.close()
        except Exception:
            pass
        signal.signal(signal.SIGTERM, old_sigterm_handler)
        if run is not None:
            run.finish()


def train(argv: list[str] | None = None) -> None:
    if os.path.exists("stop_factory"):
        print("Stop file 'stop_factory' detected. Deleting 'stop_factory' and exiting with code 1 to terminate opponent factory...")
        try:
            os.remove("stop_factory")
        except Exception:
            pass
        sys.exit(1)

    parser = build_arg_parser()
    args = parse_args_with_config(argv, parser)

    queue_path = args.queue if isinstance(args.queue, str) and args.queue else (DEFAULT_QUEUE_PATH if args.queue is not None else None)

    if args.add_to_queue is not None:
        target_queue = queue_path or DEFAULT_QUEUE_PATH
        add_to_queue(target_queue, args.add_to_queue)
        return

    cli_args = argv if argv is not None else sys.argv[1:]
    has_explicit_cli_job = any(tok.startswith("--deck") or tok.startswith("--config") or tok.startswith("--opp-") for tok in cli_args)
    has_initial_job = has_explicit_cli_job or (queue_path is None and bool(args.deck and args.model_name))

    if has_initial_job:
        run_single_training(args)

    if queue_path is not None or not has_initial_job:
        effective_queue = queue_path or DEFAULT_QUEUE_PATH
        print(f"Monitoring training queue: {effective_queue}")
        while True:
            item = pop_next_queue_item(effective_queue)
            if item is None:
                print(f"Training queue '{effective_queue}' is empty. Processing complete.")
                break
            print(f"\n========================================================")
            print(f"Processing queued item: {item}")
            print(f"========================================================\n")
            try:
                item_args = parse_queue_item(item, parser)
                run_single_training(item_args)
            except Exception as err:
                print(f"Error executing queued item {item}: {err}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train()
