"""Offline behavioural-cloning utilities for recurrent V6 policies.

The dataset format deliberately keeps training targets separate from actor-visible
observations.  Episodes are the unit of splitting and recurrent training so an
LSTM state is reset only at a true episode boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from src.training.custom_ppo import hidden_card_auxiliary_loss
from src.training.model_factory import reset_policy_optimizer


DATASET_SCHEMA_VERSION = 1
TRAINING_ONLY_OBSERVATIONS = {"aux_target", "teacher_action", "teacher_value"}
REQUIRED_ARRAYS = {
    "target__aux_target",
    "action",
    "loss_mask",
    "episode_id",
    "step",
    "episode_start",
    "done",
    "perspective",
    "outcome",
    "legal_count",
    "label_source",
    "teacher_confidence",
}


@dataclass(frozen=True)
class _LoadedDemonstrationEpisode:
    """Arrays for one episode, resident only while that episode is processed."""

    episode_id: int
    observations: dict[str, np.ndarray]
    aux_target: np.ndarray
    action: np.ndarray
    loss_mask: np.ndarray
    step: np.ndarray
    episode_start: np.ndarray
    done: np.ndarray
    perspective: np.ndarray
    outcome: np.ndarray
    legal_count: np.ndarray
    label_source: np.ndarray
    teacher_confidence: np.ndarray
    teacher_q: np.ndarray | None

    def __len__(self) -> int:
        return int(self.action.shape[0])


@dataclass(frozen=True)
class DemonstrationEpisode:
    """Validated lazy reference to one canonical episode NPZ shard."""

    episode_id: int
    path: Path
    sample_count: int
    perspective: int
    outcome: int
    observation_keys: tuple[str, ...]
    field_schema: dict[str, tuple[tuple[int, ...], np.dtype]]
    has_teacher_q: bool
    file_size: int
    modified_ns: int

    def __len__(self) -> int:
        return self.sample_count

    def load(self) -> _LoadedDemonstrationEpisode:
        """Load and revalidate this episode without retaining it on the dataset."""

        self._require_unchanged_file()
        arrays = _load_shard(self.path)
        _, observation_keys = _validate_shard(
            self.path,
            arrays,
            {f"obs__{key}" for key in self.observation_keys},
        )
        field_schema = _field_schema(arrays)
        if field_schema != self.field_schema:
            raise ValueError(f"Shard {self.path.name} field shapes or dtypes changed")
        loaded = _episode_from_arrays(
            self.path,
            arrays,
            tuple(sorted(key.removeprefix("obs__") for key in observation_keys)),
        )
        if (
            loaded.episode_id != self.episode_id
            or len(loaded) != self.sample_count
            or int(loaded.perspective[0]) != self.perspective
            or int(loaded.outcome[0]) != self.outcome
            or (loaded.teacher_q is not None) != self.has_teacher_q
        ):
            raise ValueError(f"Shard {self.path.name} trajectory metadata changed")
        self._require_unchanged_file()
        return loaded

    def _require_unchanged_file(self) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Dataset shard disappeared: {self.path}") from error
        if stat.st_size != self.file_size or stat.st_mtime_ns != self.modified_ns:
            raise ValueError(f"Dataset shard changed after validation: {self.path.name}")


@dataclass(frozen=True)
class DemonstrationDataset:
    """Validated demonstration episodes and their immutable manifest."""

    root: Path
    manifest: dict[str, Any]
    observation_keys: tuple[str, ...]
    episodes: tuple[DemonstrationEpisode, ...]

    @property
    def transition_count(self) -> int:
        return sum(len(episode) for episode in self.episodes)


@dataclass(frozen=True)
class EpochMetrics:
    nll: float
    agreement: float
    entropy: float
    aux_loss: float
    labelled_steps: int


@dataclass(frozen=True)
class BehavioralCloningResult:
    best_epoch: int
    best_validation_nll: float
    train_metrics: tuple[EpochMetrics, ...]
    validation_metrics: tuple[EpochMetrics, ...]
    train_episode_ids: tuple[int, ...]
    validation_episode_ids: tuple[int, ...]


def _require_one_dimensional(name: str, values: np.ndarray, count: int) -> np.ndarray:
    if values.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},), got {values.shape}")
    return values


def _require_integral(name: str, values: np.ndarray) -> None:
    if not np.issubdtype(values.dtype, np.integer) and values.dtype != np.bool_:
        raise ValueError(f"{name} must have an integer or boolean dtype, got {values.dtype}")


def _require_finite(name: str, values: np.ndarray) -> None:
    if values.dtype == np.bool_:
        return
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got {values.dtype}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity")


def _manifest_shard_paths(root: Path, manifest: dict[str, Any]) -> list[Path]:
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported behavioural-cloning dataset schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if int(manifest.get("action_space_size", 0)) != 66:
        raise ValueError("Behavioural-cloning datasets must use the V6 66-action space")

    entries = manifest.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest.json must contain a non-empty 'episodes' list")

    paths: list[Path] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Episode entry {index} must contain path and SHA-256 metadata")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"Shard entry {index} has no valid relative path")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Shard path escapes the dataset directory: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"Dataset shard does not exist: {path}")
        expected_digest = entry.get("sha256")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError(f"Shard entry {index} has an invalid SHA-256 digest")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_digest.casefold():
            raise ValueError(f"SHA-256 mismatch for dataset shard: {path.name}")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("manifest.json contains duplicate shard paths")
    return paths


def _load_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        missing = REQUIRED_ARRAYS - keys
        if missing:
            raise ValueError(f"Shard {path.name} is missing arrays: {sorted(missing)}")
        observation_keys = {key for key in keys if key.startswith("obs__")}
        if not observation_keys:
            raise ValueError(f"Shard {path.name} contains no obs__* arrays")
        # NPZ members are materialized by NumPy already. Avoid an additional
        # copy; the returned ndarrays remain valid after the archive is closed.
        return {key: np.asarray(archive[key]) for key in archive.files}


def _field_schema(
    arrays: dict[str, np.ndarray],
) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    return {
        key: (tuple(values.shape[1:]), np.dtype(values.dtype))
        for key, values in arrays.items()
        if key.startswith("obs__") or key == "target__aux_target"
    }


def _validate_shard(
    path: Path,
    arrays: dict[str, np.ndarray],
    expected_observation_keys: set[str] | None,
) -> tuple[int, set[str]]:
    observation_keys = {key for key in arrays if key.startswith("obs__")}
    actor_keys = {key.removeprefix("obs__") for key in observation_keys}
    leaked = actor_keys & TRAINING_ONLY_OBSERVATIONS
    if leaked:
        raise ValueError(
            f"Shard {path.name} stores training-only values as observations: {sorted(leaked)}"
        )
    if expected_observation_keys is not None and observation_keys != expected_observation_keys:
        missing = expected_observation_keys - observation_keys
        extra = observation_keys - expected_observation_keys
        raise ValueError(
            f"Shard {path.name} observation schema differs; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    action = arrays["action"]
    if action.ndim != 1:
        raise ValueError(f"{path.name}: action must be one-dimensional, got {action.shape}")
    count = int(action.shape[0])
    if count == 0:
        raise ValueError(f"{path.name}: empty shards are not allowed")

    for name in REQUIRED_ARRAYS - {"label_source"}:
        values = arrays[name]
        if values.shape[0] != count:
            raise ValueError(
                f"{path.name}: {name} has {values.shape[0]} rows, expected {count}"
            )
        _require_finite(f"{path.name}:{name}", values)

    label_source = arrays["label_source"]
    if label_source.shape != (count,):
        raise ValueError(
            f"{path.name}: label_source must have shape ({count},), got {label_source.shape}"
        )
    if label_source.dtype.kind not in {"S", "U"}:
        raise ValueError(f"{path.name}: label_source must use a string dtype")

    for name in (
        "action",
        "loss_mask",
        "episode_id",
        "step",
        "episode_start",
        "done",
        "perspective",
        "outcome",
        "legal_count",
    ):
        _require_one_dimensional(name, arrays[name], count)
        _require_integral(name, arrays[name])
    _require_one_dimensional("teacher_confidence", arrays["teacher_confidence"], count)

    for key in observation_keys:
        values = arrays[key]
        if values.shape[0] != count:
            raise ValueError(
                f"{path.name}: {key} has {values.shape[0]} rows, expected {count}"
            )
        _require_finite(f"{path.name}:{key}", values)

    loss_mask = arrays["loss_mask"]
    episode_start = arrays["episode_start"]
    done = arrays["done"]
    perspective = arrays["perspective"]
    for name, values in (
        ("loss_mask", loss_mask),
        ("episode_start", episode_start),
        ("done", done),
    ):
        if not np.isin(values, (0, 1)).all():
            raise ValueError(f"{path.name}: {name} must contain only 0/1 values")
    if not np.isin(perspective, (0, 1)).all():
        raise ValueError(f"{path.name}: perspective must contain only 0/1 values")
    if not np.isin(arrays["outcome"], (-1, 0, 1)).all():
        raise ValueError(f"{path.name}: outcome must contain only -1/0/1 values")

    mask = arrays.get("obs__action_mask")
    if mask is None or mask.shape != (count, 66):
        actual = None if mask is None else mask.shape
        raise ValueError(f"{path.name}: obs__action_mask must have shape ({count}, 66), got {actual}")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError(f"{path.name}: obs__action_mask must be binary")
    counted_legal = np.count_nonzero(mask, axis=1)
    if np.any(counted_legal == 0):
        raise ValueError(f"{path.name}: every observation must have at least one legal action")
    if not np.array_equal(counted_legal, arrays["legal_count"]):
        raise ValueError(f"{path.name}: legal_count does not match obs__action_mask")
    expected_loss_mask = counted_legal >= 2
    if not np.array_equal(loss_mask.astype(bool), expected_loss_mask):
        raise ValueError(f"{path.name}: loss_mask must equal legal_count >= 2")

    if np.any(np.char.str_len(np.char.strip(label_source.astype(str))) == 0):
        raise ValueError(f"{path.name}: every row requires a non-empty label_source")
    if np.any(arrays["teacher_confidence"] < 0.0):
        raise ValueError(f"{path.name}: teacher_confidence must be non-negative")
    if np.any((action < 0) | (action >= 66)):
        raise ValueError(f"{path.name}: expert actions must be in [0, 65]")
    rows = np.arange(count)
    if not np.all(mask[rows, action] == 1):
        raise ValueError(f"{path.name}: an expert action is illegal")

    if "teacher_q" in arrays:
        teacher_q = arrays["teacher_q"]
        if teacher_q.shape != (count, 66):
            raise ValueError(
                f"{path.name}: teacher_q must have shape ({count}, 66), got {teacher_q.shape}"
            )
        if np.isinf(teacher_q).any():
            raise ValueError(f"{path.name}: teacher_q contains infinity")

    return count, observation_keys


def _episode_from_arrays(
    path: Path,
    arrays: dict[str, np.ndarray],
    observation_names: tuple[str, ...],
) -> _LoadedDemonstrationEpisode:
    """Build one loaded episode and reject non-canonical multi-episode shards."""

    episode_ids = arrays["episode_id"].astype(np.int64, copy=False)
    unique_episode_ids = np.unique(episode_ids)
    if unique_episode_ids.size != 1:
        raise ValueError(
            f"Shard {path.name} must contain exactly one complete episode, got "
            f"{unique_episode_ids.tolist()}"
        )
    episode_id = int(unique_episode_ids[0])
    count = int(episode_ids.shape[0])
    steps = arrays["step"].astype(np.int64, copy=False)
    expected_steps = np.arange(count, dtype=np.int64)
    if not np.array_equal(steps, expected_steps):
        raise ValueError(
            f"Episode {episode_id} steps must be contiguous from zero; got {steps.tolist()}"
        )
    starts = arrays["episode_start"].astype(bool, copy=False)
    dones = arrays["done"].astype(bool, copy=False)
    if not starts[0] or starts[1:].any():
        raise ValueError(f"Episode {episode_id} must mark only its first row as episode_start")
    if not dones[-1] or dones[:-1].any():
        raise ValueError(f"Episode {episode_id} must mark only its final row as done")
    perspectives = arrays["perspective"]
    if np.unique(perspectives).size != 1:
        raise ValueError(f"Episode {episode_id} changes learner perspective")
    outcomes = arrays["outcome"]
    if np.unique(outcomes).size != 1:
        raise ValueError(f"Episode {episode_id} contains inconsistent outcomes")

    return _LoadedDemonstrationEpisode(
        episode_id=episode_id,
        observations={name: arrays[f"obs__{name}"] for name in observation_names},
        aux_target=arrays["target__aux_target"],
        action=arrays["action"].astype(np.int64, copy=False),
        loss_mask=arrays["loss_mask"].astype(bool, copy=False),
        step=steps,
        episode_start=starts,
        done=dones,
        perspective=perspectives.astype(np.int8, copy=False),
        outcome=outcomes,
        legal_count=arrays["legal_count"].astype(np.int64, copy=False),
        label_source=arrays["label_source"],
        teacher_confidence=arrays["teacher_confidence"].astype(np.float32, copy=False),
        teacher_q=arrays.get("teacher_q"),
    )


def load_demonstration_dataset(dataset_directory: str | Path) -> DemonstrationDataset:
    """Validate shards one episode at a time and retain only lazy references."""

    root = Path(dataset_directory).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")

    shard_paths = _manifest_shard_paths(root, manifest)
    expected_observation_keys: set[str] | None = None
    expected_field_schema: dict[str, tuple[tuple[int, ...], np.dtype]] | None = None
    observation_names: tuple[str, ...] | None = None
    episodes: list[DemonstrationEpisode] = []
    seen_episode_ids: set[int] = set()
    for path in shard_paths:
        stat_before = path.stat()
        arrays = _load_shard(path)
        _, observation_keys = _validate_shard(path, arrays, expected_observation_keys)
        field_schema = _field_schema(arrays)
        if expected_observation_keys is None:
            expected_observation_keys = observation_keys
            expected_field_schema = field_schema
            observation_names = tuple(
                sorted(key.removeprefix("obs__") for key in observation_keys)
            )
        elif field_schema != expected_field_schema:
            raise ValueError(f"Shard {path.name} field shapes or dtypes differ from prior shards")
        assert observation_names is not None
        loaded = _episode_from_arrays(path, arrays, observation_names)
        if loaded.episode_id in seen_episode_ids:
            raise ValueError(f"Dataset contains duplicate episode_id {loaded.episode_id}")
        seen_episode_ids.add(loaded.episode_id)
        stat_after = path.stat()
        if (
            stat_after.st_size != stat_before.st_size
            or stat_after.st_mtime_ns != stat_before.st_mtime_ns
        ):
            raise ValueError(f"Dataset shard changed while validating: {path.name}")
        episodes.append(
            DemonstrationEpisode(
                episode_id=loaded.episode_id,
                path=path,
                sample_count=len(loaded),
                perspective=int(loaded.perspective[0]),
                outcome=int(loaded.outcome[0]),
                observation_keys=observation_names,
                field_schema=dict(field_schema),
                has_teacher_q=loaded.teacher_q is not None,
                file_size=stat_after.st_size,
                modified_ns=stat_after.st_mtime_ns,
            )
        )
        # `arrays` and `loaded` are intentionally not retained. At the next
        # iteration the previous episode becomes eligible for reclamation.
        del loaded, arrays

    if len(episodes) < 2:
        raise ValueError("At least two complete episodes are required for a train/validation split")
    declared_episode_count = manifest.get("num_episodes")
    if declared_episode_count is not None and int(declared_episode_count) != len(episodes):
        raise ValueError(
            f"Manifest declares {declared_episode_count} episodes, loaded {len(episodes)}"
        )
    declared_sample_count = manifest.get("num_samples")
    loaded_sample_count = sum(len(episode) for episode in episodes)
    if declared_sample_count is not None and int(declared_sample_count) != loaded_sample_count:
        raise ValueError(
            f"Manifest declares {declared_sample_count} samples, loaded {loaded_sample_count}"
        )

    episode_entries = manifest.get("episodes")
    if isinstance(episode_entries, list) and episode_entries and all(
        isinstance(entry, dict) and "episode_id" in entry for entry in episode_entries
    ):
        entries_by_id = {int(entry["episode_id"]): entry for entry in episode_entries}
        if len(entries_by_id) != len(episode_entries):
            raise ValueError("Manifest contains duplicate episode_id entries")
        loaded_ids = {episode.episode_id for episode in episodes}
        if set(entries_by_id) != loaded_ids:
            raise ValueError("Manifest episode_id entries do not match loaded trajectories")
        for episode in episodes:
            entry = entries_by_id.get(episode.episode_id)
            if entry is None:
                raise ValueError(f"Manifest has no metadata for episode {episode.episode_id}")
            checks = {
                "samples": len(episode),
                "perspective": episode.perspective,
                "outcome": episode.outcome,
            }
            for key, actual in checks.items():
                if key in entry and int(entry[key]) != actual:
                    raise ValueError(
                        f"Manifest {key} mismatch for episode {episode.episode_id}: "
                        f"{entry[key]} != {actual}"
                    )
    return DemonstrationDataset(
        root=root,
        manifest=copy.deepcopy(manifest),
        observation_keys=observation_names or (),
        episodes=tuple(episodes),
    )


def split_episodes(
    episodes: Iterable[DemonstrationEpisode],
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[DemonstrationEpisode, ...], tuple[DemonstrationEpisode, ...]]:
    """Split whole episodes deterministically without trajectory leakage."""

    episode_list = sorted(episodes, key=lambda episode: episode.episode_id)
    if len(episode_list) < 2:
        raise ValueError("At least two episodes are required")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(episode_list))
    validation_count = min(
        len(episode_list) - 1,
        max(1, int(round(len(episode_list) * validation_fraction))),
    )
    validation_indices = set(int(index) for index in shuffled_indices[:validation_count])
    train = tuple(
        episode for index, episode in enumerate(episode_list) if index not in validation_indices
    )
    validation = tuple(
        episode for index, episode in enumerate(episode_list) if index in validation_indices
    )
    return train, validation


def validate_dataset_for_model(dataset: DemonstrationDataset, model: Any) -> None:
    """Require an exact actor-observation and action-space match."""

    action_space = getattr(model, "action_space", None)
    if not isinstance(action_space, gym.spaces.Discrete) or int(action_space.n) != 66:
        raise ValueError("Behavioural cloning requires a model with Discrete(66) actions")
    observation_space = getattr(model, "observation_space", None)
    if not isinstance(observation_space, gym.spaces.Dict):
        raise ValueError("Behavioural cloning requires a Dict observation space")

    expected_keys = set(observation_space.spaces) - TRAINING_ONLY_OBSERVATIONS
    actual_keys = set(dataset.observation_keys)
    if actual_keys != expected_keys:
        raise ValueError(
            "Dataset/model observation mismatch; "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    first = dataset.episodes[0].load()
    for key in dataset.observation_keys:
        values = first.observations[key]
        space = observation_space.spaces[key]
        if values.shape[1:] != space.shape:
            raise ValueError(
                f"Observation {key} has sample shape {values.shape[1:]}, expected {space.shape}"
            )
        if np.dtype(values.dtype) != np.dtype(space.dtype):
            raise ValueError(
                f"Observation {key} has dtype {values.dtype}, expected {space.dtype}"
            )
    aux_space = observation_space.spaces.get("aux_target")
    if aux_space is None:
        raise ValueError("Model observation space does not declare aux_target")
    if first.aux_target.shape[1:] != aux_space.shape:
        raise ValueError(
            "Auxiliary target has sample shape "
            f"{first.aux_target.shape[1:]}, expected {aux_space.shape}"
        )
    if np.dtype(first.aux_target.dtype) != np.dtype(aux_space.dtype):
        raise ValueError(
            f"Auxiliary target has dtype {first.aux_target.dtype}, expected {aux_space.dtype}"
        )


def _initial_lstm_state(policy: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lstm = policy.lstm_actor
    shape = (int(lstm.num_layers), 1, int(lstm.hidden_size))
    return (
        torch.zeros(shape, dtype=torch.float32, device=device),
        torch.zeros(shape, dtype=torch.float32, device=device),
    )


def _detach_lstm_state(state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return state[0].detach(), state[1].detach()


def _tensor_observations(
    episode: _LoadedDemonstrationEpisode,
    begin: int,
    end: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(values[begin:end], device=device)
        for key, values in episode.observations.items()
    }


def _set_policy_training_mode(policy: Any, training: bool) -> None:
    setter = getattr(policy, "set_training_mode", None)
    if callable(setter):
        setter(training)
    else:
        policy.train(training)


def _run_episodes(
    model: Any,
    episodes: Iterable[DemonstrationEpisode],
    sequence_length: int,
    aux_coef: float,
    optimizer: torch.optim.Optimizer | None,
) -> EpochMetrics:
    training = optimizer is not None
    policy = model.policy
    device = torch.device(model.device)
    _set_policy_training_mode(policy, training)

    nll_sum = 0.0
    agreement_sum = 0
    entropy_sum = 0.0
    aux_sum = 0.0
    aux_steps = 0
    labelled_steps = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for episode_reference in episodes:
            episode = episode_reference.load()
            lstm_state = _initial_lstm_state(policy, device)
            for begin in range(0, len(episode), sequence_length):
                end = min(len(episode), begin + sequence_length)
                observations = _tensor_observations(episode, begin, end, device)
                episode_starts = torch.as_tensor(
                    episode.episode_start[begin:end], dtype=torch.float32, device=device
                )
                distribution, logits, aux_logits, next_lstm_state = policy.evaluate_behavior_cloning(
                    observations,
                    lstm_state,
                    episode_starts,
                    compute_aux=aux_coef > 0.0,
                )
                lstm_state = _detach_lstm_state(next_lstm_state)

                label_mask = torch.as_tensor(
                    episode.loss_mask[begin:end], dtype=torch.bool, device=device
                )
                actions = torch.as_tensor(
                    episode.action[begin:end], dtype=torch.long, device=device
                )
                bc_loss: torch.Tensor | None = None
                if label_mask.any():
                    bc_loss = F.cross_entropy(logits[label_mask], actions[label_mask])

                aux_loss: torch.Tensor | None = None
                if aux_coef > 0.0:
                    if aux_logits is None:
                        raise RuntimeError("Policy did not return auxiliary logits")
                    aux_target = torch.as_tensor(
                        episode.aux_target[begin:end], dtype=torch.float32, device=device
                    )
                    aux_loss = hidden_card_auxiliary_loss(aux_logits, aux_target)

                if training and (bc_loss is not None or aux_loss is not None):
                    loss = torch.zeros((), dtype=torch.float32, device=device)
                    if bc_loss is not None:
                        loss = loss + bc_loss
                    if aux_loss is not None:
                        loss = loss + aux_coef * aux_loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
                    optimizer.step()

                with torch.no_grad():
                    if label_mask.any():
                        count = int(label_mask.sum().item())
                        selected_logits = logits[label_mask]
                        selected_actions = actions[label_mask]
                        chunk_nll = F.cross_entropy(
                            selected_logits, selected_actions, reduction="sum"
                        )
                        nll_sum += float(chunk_nll.item())
                        agreement_sum += int(
                            (selected_logits.argmax(dim=-1) == selected_actions).sum().item()
                        )
                        entropy = distribution.entropy()[label_mask]
                        entropy_sum += float(entropy.sum().item())
                        labelled_steps += count
                    if aux_loss is not None:
                        chunk_steps = end - begin
                        aux_sum += float(aux_loss.item()) * chunk_steps
                        aux_steps += chunk_steps
            del episode

    if labelled_steps == 0:
        raise ValueError("The selected episodes contain no labelled steps")
    return EpochMetrics(
        nll=nll_sum / labelled_steps,
        agreement=agreement_sum / labelled_steps,
        entropy=entropy_sum / labelled_steps,
        aux_loss=aux_sum / aux_steps if aux_steps else 0.0,
        labelled_steps=labelled_steps,
    )


def train_behavioral_cloning(
    model: Any,
    dataset: DemonstrationDataset,
    *,
    epochs: int,
    sequence_length: int,
    learning_rate: float,
    aux_coef: float,
    validation_fraction: float,
    patience: int,
    seed: int,
) -> BehavioralCloningResult:
    """Train, early-stop on sequence validation NLL, and prepare PPO handoff."""

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if aux_coef < 0.0:
        raise ValueError("aux_coef must be non-negative")
    if patience < 1:
        raise ValueError("patience must be positive")

    validate_dataset_for_model(dataset, model)
    train_episodes, validation_episodes = split_episodes(
        dataset.episodes, validation_fraction, seed
    )
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_nll = float("inf")
    epochs_without_improvement = 0
    train_history: list[EpochMetrics] = []
    validation_history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_episodes))
        ordered_train = tuple(train_episodes[int(index)] for index in order)
        train_metrics = _run_episodes(
            model, ordered_train, sequence_length, aux_coef, optimizer
        )
        validation_metrics = _run_episodes(
            model, validation_episodes, sequence_length, aux_coef, optimizer=None
        )
        train_history.append(train_metrics)
        validation_history.append(validation_metrics)

        if validation_metrics.nll < best_validation_nll:
            best_validation_nll = validation_metrics.nll
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.policy.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Behavioural cloning produced no validation checkpoint")
    model.policy.load_state_dict(best_state)
    reset_policy_optimizer(model.policy, float(model.lr_schedule(1.0)))
    model.num_timesteps = 0
    if hasattr(model, "_n_updates"):
        model._n_updates = 0
    return BehavioralCloningResult(
        best_epoch=best_epoch,
        best_validation_nll=best_validation_nll,
        train_metrics=tuple(train_history),
        validation_metrics=tuple(validation_history),
        train_episode_ids=tuple(episode.episode_id for episode in train_episodes),
        validation_episode_ids=tuple(episode.episode_id for episode in validation_episodes),
    )
