"""Versioned, episode-sharded datasets for behavioural cloning.

The writer deliberately separates actor-visible observations from auxiliary
training targets.  A shard is published only after the complete episode and
its outcome are known, so interrupted collection cannot expose partial
trajectories as training data.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from src.utils import atomic_write_json, resolve_deck_path


SCHEMA_VERSION = 1
ACTION_SPACE_SIZE = 66
EXCLUDED_OBSERVATION_KEYS = frozenset(
    {"aux_target", "teacher_action", "teacher_value"}
)
REQUIRED_SCALAR_OBSERVATION_KEYS = frozenset({"vector", "action_mask"})
STRUCTURED_ONLY_OBSERVATION_KEYS = frozenset(
    {
        "entity_ids",
        "entity_features",
        "option_card_ids",
        "option_attack_ids",
        "option_types",
        "option_areas",
        "option_features",
    }
)
REQUIRED_STRUCTURED_OBSERVATION_KEYS = frozenset(
    {
        *REQUIRED_SCALAR_OBSERVATION_KEYS,
        *STRUCTURED_ONLY_OBSERVATION_KEYS,
    }
)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one immutable dataset input."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_deck(path: str | Path) -> list[int]:
    """Read and validate a 60-card CSV/plain-text deck."""

    resolved = resolve_deck_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Deck does not exist: {path}")
    cards: list[int] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip().split(",", 1)[0]
            if not value:
                continue
            try:
                cards.append(int(value))
            except ValueError as error:
                raise ValueError(f"Invalid card ID at {resolved}:{line_number}") from error
    if len(cards) != 60:
        raise ValueError(f"Deck must contain 60 cards, got {len(cards)}: {resolved}")
    return cards


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _array_schema(array: np.ndarray) -> dict[str, Any]:
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def actor_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Copy actor-visible arrays and enforce one complete V6 observation contract."""

    missing_scalar = sorted(REQUIRED_SCALAR_OBSERVATION_KEYS - set(observation))
    if missing_scalar:
        raise ValueError(
            "Demonstration observation is missing required actor fields: "
            + ", ".join(missing_scalar)
        )
    present_structured = set(observation) & STRUCTURED_ONLY_OBSERVATION_KEYS
    missing = sorted(REQUIRED_STRUCTURED_OBSERVATION_KEYS - set(observation))
    if present_structured and missing:
        raise ValueError(
            "Demonstration observation has incomplete structured V6 fields: "
            + ", ".join(missing)
        )
    result: dict[str, np.ndarray] = {}
    for key, value in observation.items():
        if key in EXCLUDED_OBSERVATION_KEYS:
            continue
        array = np.asarray(value)
        if array.dtype == object:
            raise TypeError(f"Observation field {key!r} has object dtype")
        result[key] = array.copy()

    mask = np.asarray(result["action_mask"])
    if mask.shape != (ACTION_SPACE_SIZE,):
        raise ValueError(
            f"V6 action_mask must have shape ({ACTION_SPACE_SIZE},), got {mask.shape}"
        )
    return result


def validate_legal_action(observation: Mapping[str, Any], action: int) -> int:
    """Return the number of legal actions after validating one V6 label."""

    mask = np.asarray(observation["action_mask"]).reshape(-1)
    if mask.shape != (ACTION_SPACE_SIZE,):
        raise ValueError(
            f"V6 action_mask must have shape ({ACTION_SPACE_SIZE},), got {mask.shape}"
        )
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("V6 action_mask must contain only binary 0/1 values")
    legal_count = int(np.count_nonzero(mask))
    if legal_count == 0:
        raise ValueError("V6 action_mask contains no legal action")
    try:
        action = int(action)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Demonstration action is not an integer: {action!r}") from error
    if not 0 <= action < ACTION_SPACE_SIZE:
        raise ValueError(f"Demonstration action {action} is outside V6 action space")
    if not bool(mask[action]):
        raise ValueError(f"Demonstration action {action} is masked/illegal")
    return legal_count


class DemonstrationDatasetWriter:
    """Write one compressed NPZ shard per completed episode."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.shard_dir = self.output_dir / "shards"
        self.manifest_path = self.output_dir / "manifest.json"
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"Dataset output directory is not empty: {self.output_dir}; "
                "choose a new output directory"
            )

        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self._episode: dict[str, Any] | None = None
        self._observation_schema: dict[str, dict[str, Any]] | None = None
        self._target_schema: dict[str, dict[str, Any]] | None = None
        self.manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "action_space_size": ACTION_SPACE_SIZE,
            "format": "episode_npz",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "num_episodes": 0,
            "num_samples": 0,
            "shards": [],
            "episodes": [],
            "observation_schema": {},
            "target_schema": {},
            "metadata": dict(metadata or {}),
        }
        self._write_manifest()

    @property
    def episode_active(self) -> bool:
        return self._episode is not None

    @property
    def next_episode_id(self) -> int:
        return int(self.manifest["num_episodes"])

    def start_episode(self, *, perspective: int) -> int:
        if self._episode is not None:
            raise RuntimeError("Cannot start a new episode before committing or discarding the current one")
        if int(perspective) not in (0, 1):
            raise ValueError(f"Invalid player perspective: {perspective}")
        episode_id = self.next_episode_id
        self._episode = {
            "episode_id": episode_id,
            "perspective": int(perspective),
            "observations": {},
            "aux_targets": [],
            "actions": [],
            "legal_counts": [],
            "label_sources": [],
            "teacher_confidences": [],
            "teacher_q": [],
        }
        return episode_id

    def append(
        self,
        observation: Mapping[str, Any],
        *,
        action: int,
        label_source: str,
        teacher_confidence: float = 0.0,
        teacher_q: Any | None = None,
    ) -> None:
        if self._episode is None:
            raise RuntimeError("start_episode() must be called before append()")
        if not str(label_source).strip():
            raise ValueError("label_source must be non-empty")
        if "aux_target" not in observation:
            raise ValueError("Demonstration observation is missing aux_target")

        visible = actor_observation(observation)
        legal_count = validate_legal_action(visible, action)
        self._validate_schema(visible, np.asarray(observation["aux_target"]))

        observations = self._episode["observations"]
        for key, value in visible.items():
            observations.setdefault(key, []).append(value)
        self._episode["aux_targets"].append(np.asarray(observation["aux_target"]).copy())
        self._episode["actions"].append(int(action))
        self._episode["legal_counts"].append(legal_count)
        self._episode["label_sources"].append(str(label_source))
        self._episode["teacher_confidences"].append(float(teacher_confidence))

        if teacher_q is None:
            self._episode["teacher_q"].append(None)
        else:
            q_values = np.asarray(teacher_q, dtype=np.float32)
            if q_values.shape != (ACTION_SPACE_SIZE,):
                raise ValueError(
                    f"teacher_q must have shape ({ACTION_SPACE_SIZE},), got {q_values.shape}"
                )
            self._episode["teacher_q"].append(q_values.copy())

    def commit_episode(self, *, outcome: int) -> Path:
        if self._episode is None:
            raise RuntimeError("No active episode to commit")
        if int(outcome) not in (-1, 0, 1):
            raise ValueError(f"Outcome must be -1, 0, or 1, got {outcome}")

        episode = self._episode
        sample_count = len(episode["actions"])
        if sample_count == 0:
            raise ValueError("Cannot commit an episode without decision samples")

        episode_id = int(episode["episode_id"])
        step = np.arange(sample_count, dtype=np.int32)
        episode_start = np.zeros(sample_count, dtype=np.bool_)
        episode_start[0] = True
        done = np.zeros(sample_count, dtype=np.bool_)
        done[-1] = True
        legal_count = np.asarray(episode["legal_counts"], dtype=np.int16)
        arrays: dict[str, np.ndarray] = {
            **{
                f"obs__{key}": np.stack(values)
                for key, values in episode["observations"].items()
            },
            "target__aux_target": np.stack(episode["aux_targets"]),
            "action": np.asarray(episode["actions"], dtype=np.int64),
            "loss_mask": (legal_count >= 2).astype(np.bool_),
            "episode_id": np.full(sample_count, episode_id, dtype=np.int64),
            "step": step,
            "episode_start": episode_start,
            "done": done,
            "perspective": np.full(sample_count, episode["perspective"], dtype=np.int8),
            "outcome": np.full(sample_count, int(outcome), dtype=np.int8),
            "legal_count": legal_count,
            "label_source": np.asarray(episode["label_sources"], dtype=np.str_),
            "teacher_confidence": np.asarray(
                episode["teacher_confidences"], dtype=np.float32
            ),
        }
        if any(value is not None for value in episode["teacher_q"]):
            arrays["teacher_q"] = np.stack(
                [
                    value
                    if value is not None
                    else np.full(ACTION_SPACE_SIZE, np.nan, dtype=np.float32)
                    for value in episode["teacher_q"]
                ]
            )

        relative_path = Path("shards") / f"episode-{episode_id:06d}.npz"
        target = self.output_dir / relative_path
        self._atomic_save_npz(target, arrays)
        checksum = self._sha256(target)

        self.manifest["shards"].append(relative_path.as_posix())
        self.manifest["episodes"].append(
            {
                "episode_id": episode_id,
                "path": relative_path.as_posix(),
                "samples": sample_count,
                "perspective": int(episode["perspective"]),
                "outcome": int(outcome),
                "sha256": checksum,
            }
        )
        self.manifest["num_episodes"] += 1
        self.manifest["num_samples"] += sample_count
        self.manifest["updated_at"] = _utc_now()
        self._write_manifest()
        self._episode = None
        return target

    def discard_episode(self) -> None:
        self._episode = None

    def update_metadata(self, values: Mapping[str, Any]) -> None:
        """Merge JSON-serializable collection results into the manifest."""

        self.manifest["metadata"].update(dict(values))
        self.manifest["updated_at"] = _utc_now()
        self._write_manifest()

    def _validate_schema(
        self,
        observation: Mapping[str, np.ndarray],
        aux_target: np.ndarray,
    ) -> None:
        current = {key: _array_schema(value) for key, value in observation.items()}
        target = {"aux_target": _array_schema(np.asarray(aux_target))}
        if self._observation_schema is None:
            self._observation_schema = current
            self._target_schema = target
            self.manifest["observation_schema"] = current
            self.manifest["target_schema"] = target
            return
        if current != self._observation_schema:
            raise ValueError("Observation keys, shapes, or dtypes changed within the dataset")
        if target != self._target_schema:
            raise ValueError("Auxiliary-target shape or dtype changed within the dataset")

    def _write_manifest(self) -> None:
        atomic_write_json(self.manifest_path, self.manifest)

    @staticmethod
    def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _sha256(path: Path) -> str:
        return sha256_file(path)
