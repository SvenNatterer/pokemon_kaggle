"""Queue-free arena domain model, persistence, matchmaking, and ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import tempfile
import uuid
import zipfile
from typing import Any, Iterable

from src.model_paths import parse_deck_model_path


ROOT = Path(__file__).resolve().parents[1]
ARENA_DIR = ROOT / "arena_data"
STATE_FILE = ARENA_DIR / "state.json"
MATCHES_FILE = ARENA_DIR / "matches.json"
LEADERBOARD_FILE = ARENA_DIR / "leaderboard.json"
EVALUATION_FILE = ARENA_DIR / "evaluation.json"
BOT_HEALTH_FILE = ARENA_DIR / "bot_health.json"
PARTICIPANT_MANIFEST = ROOT / "decks" / "arena_agents.json"
SCHEMA_VERSION = 1
DEFAULT_ELO = 1200.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Durably write JSON by flushing a sibling temp file before os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: str | Path, default: Any) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


@dataclass
class Participant:
    bot_id: str
    display_name: str
    bot_type: str
    deck_path: str
    model_path: str | None = None
    version: str = "legacy"
    enabled: bool = True
    load_status: str = "unknown"
    load_error: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def deck_path_for_id(deck_id: str) -> Path:
    if str(deck_id).startswith("bank_"):
        return ROOT / "decks" / "deck_bank" / f"{deck_id}.csv"
    return ROOT / "decks" / f"deck_{deck_id}.csv"


def deck_id_for_path(deck_path: str) -> str:
    """Return the deck-name key for a regular or bank deck path."""
    stem = Path(deck_path).stem
    return stem[5:] if stem.startswith("deck_") else stem


def deck_name_for_path(deck_path: str) -> str:
    """Return the archetype name generated from the strongest Pokemon."""
    deck_id = deck_id_for_path(deck_path)
    names = read_json(ROOT / "decks" / "deck_names.json", {})
    name = str(names.get(deck_id) or "").strip()
    if name == "Hydrapple ex":
        return "Ogerpon"
    return name


def deck_display_name_for_path(deck_path: str) -> str:
    """Return a friendly label like `Ogerpon` for UI and logs."""
    archetype = deck_name_for_path(deck_path) or "Unknown"
    return archetype


def model_display_name_for_path(model_path: str, deck_path: str) -> str:
    """Return a display label like `V5 Mega Lucario ex` for PPO bots."""
    parsed = parse_deck_model_path(model_path) or {}
    prefix = str(parsed.get("prefix") or "").strip()
    version = "PPO"
    if prefix.startswith("ppo_v"):
        version = f"V{prefix.split('_v', 1)[1].split('_', 1)[0]}"
    elif prefix == "ppo_deck":
        version = "PPO"
    archetype = deck_name_for_path(deck_path) or "Unknown"
    return f"{version} {archetype}"


def _model_bot_id(path: Path) -> str:
    relative_parent = path.parent.resolve().relative_to((ROOT / "models").resolve()).as_posix()
    prefix = "" if relative_parent == "." else relative_parent.replace("/", "__") + "__"
    return prefix + path.stem


def _model_tags(path: Path) -> list[str]:
    rel = _relative(path)
    tags = ["current"] if path.parent == ROOT / "models" else ["legacy"]
    if "/backup/" in f"/{rel}":
        tags.append("backup")
    if "checkpoint" in path.name.lower() or "_ckpt" in path.name.lower():
        tags.append("historical_checkpoint")
    if "curriculum_snapshots" in rel or "stage_snapshots" in rel:
        tags.append("snapshot")
    return tags


def _validate_archive(path: Path) -> tuple[str, str]:
    try:
        if not path.is_file():
            return "unloadable", "model file does not exist"
        if not zipfile.is_zipfile(path):
            return "unloadable", "model is not a valid ZIP archive"
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member:
                return "unloadable", f"corrupt ZIP member: {bad_member}"
        return "loadable", ""
    except OSError as exc:
        return "unloadable", str(exc)


def _is_holdout_model_path(model_path: str | Path | None) -> bool:
    """Return whether a model is reserved solely for frozen holdout evaluation."""
    if not model_path:
        return False
    try:
        return (ROOT / model_path).resolve().is_relative_to((ROOT / "models" / "holdout").resolve())
    except (OSError, ValueError):
        return False


def _manifest_participants() -> list[Participant]:
    data = read_json(PARTICIPANT_MANIFEST, {"agents": []})
    participants = []
    for entry in data.get("agents", []):
        if not isinstance(entry, dict):
            continue
        bot_id = str(entry.get("bot_id") or entry.get("id") or "").strip()
        deck = str(entry.get("deck_path") or entry.get("deck") or "").strip()
        if not bot_id or not deck:
            continue
        if _is_holdout_model_path(entry.get("model_path")):
            continue
        participant = Participant(
            bot_id=bot_id,
            display_name="",
            bot_type=str(entry.get("bot_type") or entry.get("agent_type") or "rule_based"),
            model_path=entry.get("model_path"),
            deck_path=deck,
            version=str(entry.get("version") or "rule-v1"),
            enabled=bool(entry.get("enabled", True)),
            tags=list(entry.get("tags") or ["current"]),
        )
        if participant.bot_type == "ppo" and participant.model_path:
            participant.display_name = model_display_name_for_path(str(participant.model_path), participant.deck_path)
        else:
            participant.display_name = str(entry.get("display_name") or entry.get("name") or bot_id)
        deck_abs = ROOT / participant.deck_path
        if not deck_abs.is_file():
            participant.load_status = "unloadable"
            participant.load_error = f"deck not found: {participant.deck_path}"
        elif participant.bot_type == "rule_based":
            participant.load_status = "loadable"
        else:
            model_abs = ROOT / str(participant.model_path or "")
            participant.load_status, participant.load_error = _validate_archive(model_abs)
        participants.append(participant)
    return participants


def discover_participants() -> list[Participant]:
    """Discover current and legacy bots, excluding the evaluation-only holdout."""
    participants = _manifest_participants()
    seen_ids = {participant.bot_id for participant in participants}
    roots = [
        ROOT / "models",
        ROOT / "models" / "backup",
        ROOT / "models" / "curriculum_snapshots",
        ROOT / "models" / "stage_snapshots",
    ]
    seen_paths: set[Path] = set()
    for model_root in roots:
        if not model_root.is_dir():
            continue
        for model_path in sorted(model_root.glob("*.zip")):
            resolved = model_path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            parsed = parse_deck_model_path(model_path.as_posix())
            if not parsed:
                continue
            deck_path = deck_path_for_id(parsed["deck_id"])
            bot_id = _model_bot_id(model_path)
            if bot_id in seen_ids:
                continue
            status, error = _validate_archive(model_path)
            if not deck_path.is_file():
                status, error = "unloadable", f"deck not found: {_relative(deck_path)}"
            version = parsed["prefix"].replace("_deck", "").replace("ppo_", "") or "ppo"
            display_name = model_display_name_for_path(_relative(model_path), _relative(deck_path))
            participants.append(Participant(
                bot_id=bot_id,
                display_name=display_name,
                bot_type="ppo",
                model_path=_relative(model_path),
                deck_path=_relative(deck_path),
                version=version,
                enabled=True,
                load_status=status,
                load_error=error,
                tags=_model_tags(model_path),
            ))
            seen_ids.add(bot_id)
    health = read_json(BOT_HEALTH_FILE, {})
    now = datetime.now(timezone.utc).timestamp()
    for participant in participants:
        issue = health.get(participant.bot_id) or {}
        if participant.load_status == "loadable" and float(issue.get("retry_after", 0)) > now:
            participant.load_status = "cooldown"
            participant.load_error = str(issue.get("error") or "temporarily skipped after match failure")
    return sorted(participants, key=lambda item: item.bot_id)


def mark_bot_failure(bot_ids: Iterable[str], error: str, cooldown_seconds: int = 300) -> None:
    health = read_json(BOT_HEALTH_FILE, {})
    retry_after = datetime.now(timezone.utc).timestamp() + cooldown_seconds
    for bot_id in bot_ids:
        health[str(bot_id)] = {"error": error, "retry_after": retry_after, "updated_at": utc_now()}
    atomic_write_json(BOT_HEALTH_FILE, health)


def enabled_participants(participants: Iterable[Participant]) -> list[Participant]:
    return [p for p in participants if p.enabled and p.load_status == "loadable"]


def wilson_lower_bound(wins: int, losses: int, draws: int, z: float = 1.96) -> float:
    """Wilson 95% lower bound; each draw counts as half a success."""
    total = wins + losses + draws
    if total <= 0:
        return 0.0
    successes = wins + 0.5 * draws
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    spread = z * math.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total)
    return max(0.0, min(1.0, (centre - spread) / denominator))


def _normalise_elos(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Map Elo to expected score versus the fixed 1200 baseline.

    Unlike min-max scaling, this value does not change when an unrelated bot is
    added to or removed from the leaderboard.
    """
    return {
        row["bot_id"]: 1.0 / (1.0 + 10.0 ** ((DEFAULT_ELO - float(row.get("elo", DEFAULT_ELO))) / 400.0))
        for row in rows
    }


def rank_participants(
    participants: Iterable[Participant],
    matches: Iterable[dict[str, Any]],
    holdout: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    holdout = holdout or {}
    stats: dict[str, dict[str, Any]] = {}
    for participant in participants:
        stats[participant.bot_id] = {
            **participant.to_dict(), "elo": DEFAULT_ELO, "wins": 0, "losses": 0, "draws": 0,
        }
    for match in matches:
        a, b = match.get("bot_a"), match.get("bot_b")
        if a not in stats or b not in stats or match.get("error_status"):
            continue
        stats[a]["elo"] = float(match.get("elo_a_after", stats[a]["elo"]))
        stats[b]["elo"] = float(match.get("elo_b_after", stats[b]["elo"]))
        for bot_id, wins_key, losses_key in ((a, "wins_a", "wins_b"), (b, "wins_b", "wins_a")):
            stats[bot_id]["wins"] += int(match.get(wins_key, 0))
            stats[bot_id]["losses"] += int(match.get(losses_key, 0))
            stats[bot_id]["draws"] += int(match.get("draws", 0))
    rows = list(stats.values())
    normalized_elos = _normalise_elos(rows)
    for row in rows:
        games = row["wins"] + row["losses"] + row["draws"]
        row["matches"] = games
        row["arena_winrate"] = (row["wins"] + 0.5 * row["draws"]) / games if games else 0.0
        row["arena_wilson"] = wilson_lower_bound(row["wins"], row["losses"], row["draws"])
        row["normalized_elo"] = normalized_elos[row["bot_id"]]
        holdout_row = holdout.get(row["bot_id"]) or holdout.get(Path(str(row.get("model_path") or "")).stem)
        row["holdout_missing"] = holdout_row is None
        row["holdout_games"] = int((holdout_row or {}).get("games", 0))
        row["holdout_winrate"] = (holdout_row or {}).get("score_rate")
        row["holdout_wilson"] = (holdout_row or {}).get("wilson95_score_lb")
        row["ranking_components"] = {
            "arena_wilson": row["arena_wilson"],
            "normalized_elo": row["normalized_elo"],
            "arena_winrate": row["arena_winrate"],
        }
        row["ranking_score"] = (
            0.50 * row["arena_wilson"] + 0.35 * row["normalized_elo"]
            + 0.15 * row["arena_winrate"]
        )
    rows.sort(key=lambda row: (row["ranking_score"], row["arena_wilson"], row["elo"]), reverse=True)
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return rows


def pair_counts(matches: Iterable[dict[str, Any]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for match in matches:
        pair = tuple(sorted((str(match.get("bot_a")), str(match.get("bot_b")))))
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def select_matchup(
    participants: Iterable[Participant], matches: Iterable[dict[str, Any]], rng: random.Random | None = None,
) -> tuple[Participant, Participant, int]:
    """Prefer underplayed bots, then close Elo/rare pairs with 15% exploration."""
    rng = rng or random.Random()
    roster = enabled_participants(participants)
    if len(roster) < 2:
        raise RuntimeError("at least two enabled, loadable participants are required")
    match_list = list(matches)
    board = {row["bot_id"]: row for row in rank_participants(roster, match_list)}
    min_games = min(board[p.bot_id]["matches"] for p in roster)
    underplayed = [p for p in roster if board[p.bot_id]["matches"] <= min_games + 5]
    first = rng.choice(underplayed)
    counts = pair_counts(match_list)
    candidates = [p for p in roster if p.bot_id != first.bot_id]
    if rng.random() < 0.15:
        second = rng.choice(candidates)
    else:
        def score(candidate: Participant) -> float:
            elo_gap = abs(board[first.bot_id]["elo"] - board[candidate.bot_id]["elo"]) / 400.0
            games = board[candidate.bot_id]["matches"]
            paired = counts.get(tuple(sorted((first.bot_id, candidate.bot_id))), 0)
            return elo_gap + 0.04 * games + 0.25 * paired + rng.random() * 0.05
        second = min(candidates, key=score)
    prior_pair_matches = counts.get(tuple(sorted((first.bot_id, second.bot_id))), 0)
    start_player = prior_pair_matches % 2
    return first, second, start_player


def new_match_record(a: Participant, b: Participant, start_player: int) -> dict[str, Any]:
    return {
        "match_id": str(uuid.uuid4()), "timestamp": utc_now(), "bot_a": a.bot_id, "bot_b": b.bot_id,
        "deck_a": a.deck_path, "deck_b": b.deck_path, "bot_type_a": a.bot_type, "bot_type_b": b.bot_type,
        "winner": None, "result": "pending", "result_reason": "", "turns": None,
        "start_player": start_player, "perspective": start_player, "elo_a_before": DEFAULT_ELO,
        "elo_b_before": DEFAULT_ELO, "elo_a_after": DEFAULT_ELO, "elo_b_after": DEFAULT_ELO,
        "wins_a": 0, "wins_b": 0, "draws": 0, "error_status": "", "replay_ref": None,
        "arena_version": SCHEMA_VERSION,
    }


class ArenaStore:
    def __init__(self, arena_dir: str | Path = ARENA_DIR):
        self.arena_dir = Path(arena_dir)
        self.state_file = self.arena_dir / "state.json"
        self.matches_file = self.arena_dir / "matches.json"
        self.leaderboard_file = self.arena_dir / "leaderboard.json"
        self.evaluation_file = self.arena_dir / "evaluation.json"

    def state(self) -> dict[str, Any]:
        return read_json(self.state_file, {"state": "stopped", "updated_at": utc_now(), "error": ""})

    def set_state(self, state: str, **extra: Any) -> dict[str, Any]:
        if state not in {"stopped", "running", "paused", "evaluating", "error"}:
            raise ValueError(f"invalid arena state: {state}")
        value = {**self.state(), "state": state, "updated_at": utc_now(), **extra}
        atomic_write_json(self.state_file, value)
        return value

    def matches(self) -> list[dict[str, Any]]:
        value = read_json(self.matches_file, [])
        return value if isinstance(value, list) else []

    def append_match(self, match: dict[str, Any]) -> None:
        matches = self.matches()
        matches.append(match)
        atomic_write_json(self.matches_file, matches)

    def save_leaderboard(self, rows: list[dict[str, Any]]) -> None:
        atomic_write_json(self.leaderboard_file, {"schema_version": SCHEMA_VERSION, "updated_at": utc_now(), "rows": rows})

    def reset(self, include_replays: bool = False) -> None:
        # Replay deletion is deliberately handled by the controller after explicit confirmation.
        atomic_write_json(self.matches_file, [])
        atomic_write_json(self.leaderboard_file, {"schema_version": SCHEMA_VERSION, "updated_at": utc_now(), "rows": []})
        atomic_write_json(self.arena_dir / "bot_health.json", {})
        self.set_state("stopped", error="", current_match=None)
