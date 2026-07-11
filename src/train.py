import os
import sys
import argparse
import signal
import time
import glob
import pandas as pd
import zipfile

# Add src to pythonpath so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
import json
if "WANDB_MODE" not in os.environ:
    os.environ["WANDB_MODE"] = "online"
import wandb
from wandb.integration.sb3 import WandbCallback

CHECKPOINT_JSON_DIRNAME = "json"
MODEL_BACKUP_DIRNAME = "backup"

def checkpoint_metadata_path(base_path):
    normalized = os.path.normpath(base_path)
    parts = normalized.split(os.sep)
    if "models" in parts:
        model_index = parts.index("models")
        model_root = os.sep.join(parts[:model_index + 1])
        directory = os.path.join(model_root, CHECKPOINT_JSON_DIRNAME)
    else:
        directory = os.path.join(os.path.dirname(base_path) or ".", CHECKPOINT_JSON_DIRNAME)
    filename = f"{os.path.basename(base_path)}_checkpoints.json"
    return os.path.join(directory, filename)

def adjacent_checkpoint_metadata_path(base_path):
    directory = os.path.dirname(base_path) or "."
    filename = f"{os.path.basename(base_path)}_checkpoints.json"
    return os.path.join(directory, CHECKPOINT_JSON_DIRNAME, filename)

def legacy_checkpoint_metadata_path(base_path):
    return f"{base_path}_checkpoints.json"

def checkpoint_metadata_candidates(base_path):
    paths = [
        checkpoint_metadata_path(base_path),
        adjacent_checkpoint_metadata_path(base_path),
        legacy_checkpoint_metadata_path(base_path),
    ]
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique

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
        for key, queue in self.episode_rewards.items():
            if len(queue) > 0:
                mean_val = sum(queue) / len(queue)
                self.logger.record(f"rewards/{key}", mean_val)

class RotatingCheckpointCallback(BaseCallback):
    def __init__(self, model_path, save_freq=250000, keep=2, verbose=0):
        super().__init__(verbose)
        self.model_path = model_path
        self.save_freq = int(save_freq)
        self.keep = max(1, int(keep))
        self.next_checkpoint = self.save_freq

    def _init_callback(self) -> None:
        if self.save_freq <= 0:
            return
        completed = int(getattr(self.model, "num_timesteps", 0))
        self.next_checkpoint = ((completed // self.save_freq) + 1) * self.save_freq
        directory = os.path.dirname(self.model_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _checkpoint_path(self, slot):
        return f"{self.model_path}_checkpoint_{slot}"

    def _metadata_path(self):
        return checkpoint_metadata_path(self.model_path)

    def _write_metadata(self, slot, checkpoint_file):
        metadata_path = self._metadata_path()
        metadata = {}
        for existing_path in checkpoint_metadata_candidates(self.model_path):
            if not os.path.exists(existing_path):
                continue
            try:
                with open(existing_path, "r") as f:
                    metadata = json.load(f)
                break
            except Exception:
                metadata = {}

        slots = metadata.get("slots", {})
        slots[str(slot)] = {
            "file": checkpoint_file,
            "step": int(self.num_timesteps),
            "saved_at": int(time.time()),
        }

        metadata.update({
            "latest": checkpoint_file,
            "latest_slot": slot,
            "latest_step": int(self.num_timesteps),
            "checkpoint_interval": self.save_freq,
            "keep_checkpoints": self.keep,
            "slots": slots,
        })
        metadata_dir = os.path.dirname(metadata_path)
        if metadata_dir:
            os.makedirs(metadata_dir, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _on_step(self) -> bool:
        if self.save_freq <= 0:
            return True

        while self.num_timesteps >= self.next_checkpoint:
            checkpoint_number = self.next_checkpoint // self.save_freq
            slot = ((checkpoint_number - 1) % self.keep) + 1
            checkpoint_path = self._checkpoint_path(slot)
            self.model.save(checkpoint_path)
            checkpoint_file = f"{checkpoint_path}.zip"
            self._write_metadata(slot, checkpoint_file)
            print(f"Checkpoint saved at {self.num_timesteps} steps: {checkpoint_file}")
            self.next_checkpoint += self.save_freq
        return True

from stable_baselines3.common.monitor import Monitor
from src.env_wrapper import PokemonTCGEnv
from src.custom_ppo import CustomPPO, PokemonTCGRecurrentPolicy

def read_deck(deck_path):
    df = pd.read_csv(deck_path, header=None)
    return df[0].tolist()

def resolve_model_path(model_name):
    model_path = model_name if os.path.dirname(model_name) else os.path.join("models", model_name)
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    return model_path

def resolve_legacy_model_path(model_name):
    if not os.path.dirname(model_name):
        return None
    legacy_path = os.path.join("models", model_name)
    if legacy_path.endswith(".zip"):
        legacy_path = legacy_path[:-4]
    return legacy_path

def normalize_zip_path(path):
    return path if path.endswith(".zip") else f"{path}.zip"

def model_file_candidates(zip_path):
    zip_path = normalize_zip_path(zip_path)
    paths = [zip_path]

    normalized = os.path.normpath(zip_path)
    parts = normalized.split(os.sep)
    if "models" in parts:
        model_index = parts.index("models")
        model_root = os.sep.join(parts[:model_index + 1])
        paths.append(os.path.join(model_root, MODEL_BACKUP_DIRNAME, os.path.basename(zip_path)))

    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique

def read_model_timesteps(zip_path):
    try:
        with zipfile.ZipFile(zip_path) as archive:
            if "data" not in archive.namelist():
                return None
            data = json.loads(archive.read("data").decode("utf-8"))
        value = data.get("num_timesteps")
        return int(value) if value is not None else None
    except Exception:
        return None

def add_model_candidate(candidates, zip_path, label, step_hint=None):
    zip_path = next((path for path in model_file_candidates(zip_path) if os.path.exists(path)), normalize_zip_path(zip_path))
    if zip_path in candidates or not os.path.exists(zip_path):
        return

    steps = read_model_timesteps(zip_path)
    if steps is None and step_hint is not None:
        try:
            steps = int(step_hint)
        except Exception:
            steps = None

    candidates[zip_path] = {
        "path": zip_path[:-4] if zip_path.endswith(".zip") else zip_path,
        "zip_path": zip_path,
        "label": label,
        "steps": steps,
        "mtime": os.path.getmtime(zip_path),
    }

def add_checkpoint_candidates(candidates, base_path, label_prefix):
    for metadata_path in checkpoint_metadata_candidates(base_path):
        if not os.path.exists(metadata_path):
            continue
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            latest = metadata.get("latest")
            if latest:
                add_model_candidate(candidates, latest, f"{label_prefix} latest checkpoint", metadata.get("latest_step"))

            for slot, slot_data in metadata.get("slots", {}).items():
                if isinstance(slot_data, dict) and slot_data.get("file"):
                    add_model_candidate(
                        candidates,
                        slot_data["file"],
                        f"{label_prefix} checkpoint slot {slot}",
                        slot_data.get("step"),
                    )
        except Exception as e:
            print(f"Warning: Failed to read checkpoint metadata {metadata_path}: {e}")

    for checkpoint_path in glob.glob(f"{base_path}_checkpoint_*.zip"):
        add_model_candidate(candidates, checkpoint_path, f"{label_prefix} checkpoint file")

def choose_model_to_load(model_path, legacy_model_path=None):
    candidates = {}

    add_model_candidate(candidates, f"{model_path}.zip", "base model")
    add_checkpoint_candidates(candidates, model_path, "base")

    if legacy_model_path:
        add_model_candidate(candidates, f"{legacy_model_path}.zip", "legacy model")
        add_checkpoint_candidates(candidates, legacy_model_path, "legacy")

    if not candidates:
        return model_path

    best = max(candidates.values(), key=lambda item: (-1 if item["steps"] is None else item["steps"], item["mtime"]))

    print("Model candidates:")
    for candidate in sorted(candidates.values(), key=lambda item: (-1 if item["steps"] is None else item["steps"], item["mtime"])):
        step_text = "unknown" if candidate["steps"] is None else f"{candidate['steps']:,}"
        print(f"  - {candidate['label']}: {candidate['zip_path']} ({step_text} steps)")
    print(f"Loading best model candidate: {best['zip_path']}")

    return best["path"]

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
        model_path = entry.get("model")
        if not os.path.exists(deck_path):
            raise FileNotFoundError(f"Opponent deck not found: {deck_path}")
        if model_path and not os.path.exists(model_path):
            raise FileNotFoundError(f"Opponent model not found: {model_path}")
        pool.append({
            "deck": read_deck(deck_path),
            "model_path": model_path,
            "weight": float(entry.get("weight", 1.0)),
            "label": entry.get("label", os.path.basename(deck_path)),
        })
    return pool

def make_env(deck_path, opp_deck_path, opp_model_path, sparse_rewards=False, opponent_pool=None, rotate_perspective=False):
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
        )
        return Monitor(env)
    return _init

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=str, required=True, help="Path to deck.csv")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model to save")
    parser.add_argument("--timesteps", type=int, default=0, help="Number of training timesteps. Use 0 for endless training.")
    parser.add_argument("--endless", action="store_true", help="Train forever until interrupted.")
    parser.add_argument("--checkpoint-interval", type=int, default=250000, help="Save a rotating checkpoint every N timesteps")
    parser.add_argument("--keep-checkpoints", type=int, default=2, help="Number of rotating checkpoint files to keep")
    parser.add_argument("--opp-deck", type=str, help="Path to opponent deck.csv", default=None)
    parser.add_argument("--opp-model", type=str, help="Path to opponent model .zip", default=None)
    parser.add_argument("--opp-pool", type=str, default=None, help="JSON list of weighted opponent deck/model entries sampled per episode")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments (default: 8)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.02, help="Entropy coefficient")
    parser.add_argument("--n-epochs", type=int, default=2, help="PPO epochs per rollout")
    parser.add_argument("--clip-range", type=float, default=0.1, help="PPO clipping range")
    parser.add_argument("--target-kl", type=float, default=0.05, help="Stop PPO update early above this KL")
    parser.add_argument("--batch-size", type=int, default=512, help="Minibatch size")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per env per rollout")
    parser.add_argument("--sparse-rewards", action="store_true", help="Disable aggressive reward shaping")
    parser.add_argument("--aux-coef", type=float, default=0.5, help="Weight for hidden-card auxiliary loss")
    parser.add_argument("--belief-actor", action="store_true", help="Feed the learned hidden-card belief embedding into the actor")
    parser.add_argument("--belief-dim", type=int, default=64, help="Size of the learned belief embedding used by --belief-actor")
    parser.add_argument("--no-belief-detach", dest="belief_detach", action="store_false", help="Allow PPO loss gradients into the belief encoder")
    parser.add_argument("--rotate-perspective", action="store_true", help="Randomly play as Player 0 or Player 1 each episode")
    parser.set_defaults(belief_detach=True)
    args = parser.parse_args()

    opp_deck_path = args.opp_deck if args.opp_deck else args.deck
    opponent_pool = load_opponent_pool(args.opp_pool)
    
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
        )
        for _ in range(args.num_envs)
    ])
    
    model_path = resolve_model_path(args.model_name)

    legacy_model_path = resolve_legacy_model_path(args.model_name)
    load_model_path = choose_model_to_load(model_path, legacy_model_path)

    if os.path.exists(f"{load_model_path}.zip"):
        print(f"Loading existing model from {load_model_path}.zip...")
        model = CustomPPO.load(load_model_path, env=env)
        loaded_n_steps = int(getattr(model, "n_steps", args.n_steps))
        if loaded_n_steps != args.n_steps:
            raise RuntimeError(
                f"Cannot resume with --n-steps={args.n_steps}: checkpoint rollout buffer "
                f"uses n_steps={loaded_n_steps}. Keep the saved value or start a fresh model."
            )
        loaded_belief_actor = bool(getattr(model.policy, "use_belief_actor", False))
        if args.belief_actor and not loaded_belief_actor:
            raise RuntimeError(
                "--belief-actor was requested, but the existing checkpoint uses the legacy actor. "
                "Use a fresh --model-name for the belief-actor experiment."
            )
        if loaded_belief_actor and not args.belief_actor:
            print("Loaded a belief-actor checkpoint; continuing with its saved architecture.")
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
        
        from src.custom_policy import PokemonTCGFeatureExtractor
        policy_kwargs = dict(
            features_extractor_class=PokemonTCGFeatureExtractor,
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
            device="cpu",
            tensorboard_log="logs/",
            policy_kwargs=policy_kwargs
        )
    
    endless_training = args.endless or args.timesteps <= 0
    if endless_training:
        print(f"Starting endless training. Rotating checkpoints every {args.checkpoint_interval} steps...")
    else:
        print(f"Starting training for {args.timesteps} timesteps...")

    deck_id = args.deck.split('_')[-1].split('.')[0]
    opp_id = "pool" if opponent_pool else opp_deck_path.split('_')[-1].split('.')[0]
    
    deck_name = "Unknown"
    opp_name = "Unknown"
    if os.path.exists("decks/deck_names.json"):
        try:
            with open("decks/deck_names.json", "r") as f:
                names = json.load(f)
                deck_name = names.get(str(deck_id), "Unknown")
                opp_name = "Opponent League" if opponent_pool else names.get(str(opp_id), "Unknown")
        except: pass
        
    action_text = f"🧠 Training: {deck_name} (D{deck_id}) vs {opp_name} (D{opp_id})"
    
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
        mode="online",
    )
    tb_run_id = getattr(run, "id", None) or str(int(time.time()))
    tb_log_name = os.environ.get("TB_LOG_NAME", f"Deck_{deck_id}_{tb_run_id}")
    run.config.update({"tb_log_name": tb_log_name}, allow_val_change=True)
    
    status_total = 0 if endless_training else args.timesteps
    live_status_callback = LiveStatusCallback(action_text=action_text, total_timesteps=status_total)
    checkpoint_callback = RotatingCheckpointCallback(
        model_path=model_path,
        save_freq=args.checkpoint_interval,
        keep=args.keep_checkpoints,
    )
    wandb_callback = WandbCallback(
        gradient_save_freq=0, # disable saving gradients to save space
        verbose=2,
    )
    reward_callback = RewardBreakdownCallback()
    callbacks = CallbackList([live_status_callback, checkpoint_callback, wandb_callback, reward_callback])

    def handle_stop_signal(signum, frame):
        raise KeyboardInterrupt

    old_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, handle_stop_signal)

    try:
        if endless_training:
            learn_chunk = max(1, args.checkpoint_interval)
            while True:
                model.learn(
                    total_timesteps=learn_chunk,
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
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model before shutdown...")
    finally:
        model.save(model_path)
        print(f"Model saved to {model_path}.zip")
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
