import os
import sys
import argparse
import signal
import time
import pandas as pd

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

TRAINING_USES_POTENTIAL_REWARDS = True

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
        return True

    def _on_rollout_end(self) -> None:
        for key, recent_values in self.episode_rewards.items():
            if key == "potential":
                continue
            if len(recent_values) > 0:
                if any(v != 0.0 for v in recent_values):
                    mean_val = sum(recent_values) / len(recent_values)
                    self.logger.record(f"rewards/{key}", mean_val)
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            self.logger.record("rollout/ep_rew_max", max(ep_rewards))
            self.logger.record("rollout/ep_rew_min", min(ep_rewards))
            wins = sum(1 for ep in self.model.ep_info_buffer if ep.get("r", 0) > 0)
            self.logger.record("rollout/win_rate", wins / len(self.model.ep_info_buffer))
        elif "prize_win" in self.episode_rewards and len(self.episode_rewards["prize_win"]) > 0:
            wins = sum(1 for v in self.episode_rewards["prize_win"] if v > 0)
            self.logger.record("rollout/win_rate", wins / len(self.episode_rewards["prize_win"]))

from stable_baselines3.common.monitor import Monitor
from src.env.env_wrapper import LEGACY_ACTION_SPACE_SIZE, V6_ACTION_SPACE_SIZE, PokemonTCGEnv
from src.training.training_health import TrainingHealthCallback, summarize_health, health_gate
from src.training.custom_ppo import CustomPPO, PokemonTCGRecurrentPolicy

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
    directory = os.path.dirname(model_path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary_base = os.path.join(directory, f".{os.path.basename(model_path)}.training-{os.getpid()}")
    temporary_zip = f"{temporary_base}.zip"
    save_model = False
    try:
        model.save(temporary_zip)
        os.replace(temporary_zip, f"{model_path}.zip")
    finally:
        if os.path.exists(temporary_zip):
            os.unlink(temporary_zip)


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
    opponent_pool=None,
    rotate_perspective=False,
    action_space_size=V6_ACTION_SPACE_SIZE,
    structured_v2=True,
    enable_lookahead_teacher=False,
    teacher_sample_rate=0.50,
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
            sparse_rewards=sparse_rewards,
            opponent_pool=opponent_pool,
            rotate_perspective=rotate_perspective,
            action_space_size=action_space_size,
            structured_v2=structured_v2,
            enable_lookahead_teacher=enable_lookahead_teacher,
            teacher_sample_rate=teacher_sample_rate,
        )
        return Monitor(env)
    return _init

def train():
    if os.path.exists("stop_factory"):
        print("Stop file 'stop_factory' detected. Deleting 'stop_factory' and exiting with code 1 to terminate opponent factory...")
        try:
            os.remove("stop_factory")
        except Exception:
            pass
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=str, required=True, help="Path to deck.csv")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model to save")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Number of training timesteps. Use 0 for endless training.")
    parser.add_argument("--endless", action="store_true", help="Train forever until interrupted.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--continue-existing", action="store_true", help="Explicitly continue the exact target model if it exists.")
    mode.add_argument("--overwrite", action="store_true", help="Train a new model and replace an existing target only after success.")
    parser.add_argument("--opp-deck", type=str, help="Path to opponent deck.csv", default=None)
    parser.add_argument("--opp-model", type=str, help="Path to opponent model .zip", default=None)
    parser.add_argument("--opp-pool", type=str, default=None, help="JSON list of weighted opponent deck/model entries sampled per episode")
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
    parser.add_argument("--enable-lookahead-teacher", action="store_true", default=True, help="Enable lookahead teacher sampling")
    parser.add_argument("--no-lookahead-teacher", dest="enable_lookahead_teacher", action="store_false")
    parser.add_argument("--teacher-sample-rate", type=float, default=0.50, help="Sampling rate for lookahead teacher on complex decisions")
    parser.add_argument("--no-belief-actor", dest="belief_actor", action="store_false", help="Disable hidden-card belief actor")
    parser.add_argument("--belief-dim", type=int, default=64, help="Size of the learned belief embedding used by --belief-actor")
    parser.add_argument("--no-belief-detach", dest="belief_detach", action="store_false", help="Allow PPO loss gradients into the belief encoder")
    parser.add_argument("--no-rotate-perspective", dest="rotate_perspective", action="store_false", help="Disable random perspective rotation")
    parser.add_argument("--seed", type=int, default=None, help="Policy and vector-environment seed for reproducible experiment families")
    parser.add_argument(
        "--policy-version",
        choices=("v6",),
        default="v6",
        help="V6 uses a compact 66-action head.",
    )
    parser.add_argument(
        "--feature-variant",
        choices=("compact",),
        default="compact",
        help="Structured V6 Compact feature width.",
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
    parser.set_defaults(
        belief_actor=True,
        belief_detach=True,
        card_table=True,
        inference_guardrails=True,
        rotate_perspective=True,
        pfsp_lite=True,
        health_gate=True,
    )
    args = parser.parse_args()

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
    
    opponent_description = f"pool {args.opp_pool}" if opponent_pool else opp_deck_path
    print(f"Initializing environment with {args.num_envs} workers for deck {args.deck} against {opponent_description}...")
    # Vectorized environment - must use SubprocVecEnv because cg library uses a global singleton
    env = SubprocVecEnv([
        make_env(
            args.deck,
            opp_deck_path,
            args.opp_model,
            args.sparse_rewards,
            opponent_pool=opponent_pool,
            rotate_perspective=args.rotate_perspective,
            action_space_size=action_space_size,
            structured_v2=not args.scalar_obs,
            enable_lookahead_teacher=args.enable_lookahead_teacher,
            teacher_sample_rate=args.teacher_sample_rate,
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

    target_exists = os.path.exists(f"{model_path}.zip")
    if target_exists and not args.continue_existing and not args.overwrite:
        env.close()
        raise FileExistsError(
            f"Target model already exists: {model_path}.zip. Use --continue-existing to continue that exact "
            "final model or --overwrite to deliberately train a new replacement."
        )
    if args.continue_existing and not target_exists:
        env.close()
        raise FileNotFoundError(f"Cannot continue missing target model: {model_path}.zip")

    if args.continue_existing:
        print(f"Explicitly continuing target model {model_path}.zip...")
        model = CustomPPO.load(model_path, env=env, device="cpu")
        try:
            validate_policy_action_space(model, action_space_size, args.policy_version)
        except RuntimeError:
            env.close()
            raise
        if not bool(getattr(model.policy.features_extractor, "structured_v2", False)):
            env.close()
            raise RuntimeError(
                f"Model {model_path}.zip uses the legacy scalar-card observation and "
                "cannot be resumed as Observation V2. Keep it as an --opp-model and choose a "
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
        # Update parameters on loaded model
        model.c_aux = args.aux_coef
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
        print("Creating new Custom PPO model...")
        
        from src.models.custom_policy import PokemonTCGFeatureExtractor
        policy_kwargs = dict(
            features_extractor_class=PokemonTCGFeatureExtractor,
            features_extractor_kwargs={
                "feature_variant": args.feature_variant,
                "use_card_table": args.card_table,
            },
            use_belief_actor=args.belief_actor,
            belief_dim=args.belief_dim,
            detach_belief_actor=args.belief_detach,
        )
        
        model = CustomPPO(
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
            seed=args.seed,
            device="cpu",
            tensorboard_log="logs/",
            policy_kwargs=policy_kwargs
        )
    
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
    run_name = os.environ.get("WANDB_NAME", f"D{deck_id}_vs_D{opp_id}_{run_suffix}")
    
    # Initialize wandb
    run = wandb.init(
        project="pokemon_kaggle",
        name=run_name,
        group=os.environ.get("WANDB_RUN_GROUP", f"deck_{deck_id}"),
        config=vars(args),
        sync_tensorboard=True, # auto-upload sb3's tensorboard metrics
        monitor_gym=True,
        save_code=True,
        dir="/tmp",
        mode=os.environ.get("WANDB_MODE", "online"),
    )
    tb_run_id = getattr(run, "id", None) or str(int(time.time()))
    tb_log_name = os.environ.get("TB_LOG_NAME", f"Deck_{deck_id}_{tb_run_id}")
    run.config.update({"tb_log_name": tb_log_name}, allow_val_change=True)
    
    status_total = 0 if endless_training else args.timesteps
    live_status_callback = LiveStatusCallback(action_text=action_text, total_timesteps=status_total)
    wandb_callback = WandbCallback(
        gradient_save_freq=0, # disable saving gradients to save space
        verbose=2,
    )
    reward_callback = RewardBreakdownCallback()
    callbacks = CallbackList([live_status_callback, wandb_callback, reward_callback])

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
        run.finish()

if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    train()
