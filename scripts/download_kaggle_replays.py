#!/usr/bin/env python3
"""Batch-download replay JSON files for your Kaggle simulation bots."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_COMPETITION = "pokemon-tcg-ai-battle"
DEFAULT_OUTPUT = Path("replays/kaggle")
SUBMISSION_METADATA_FILE = "submissions.json"
TERMINAL_EPISODE_STATES = {"COMPLETED", "ERRORED"}
DEFAULT_REQUEST_DELAY = 2.0
DEFAULT_MAX_RETRIES = 5
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}


@dataclass
class DownloadStats:
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download all Kaggle replay JSONs belonging to your own simulation "
            "competition submissions. Existing files are skipped."
        )
    )
    parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help=f"Kaggle competition slug (default: {DEFAULT_COMPETITION})",
    )
    parser.add_argument(
        "--submission-id",
        action="append",
        type=int,
        default=[],
        help="Only download this submission/bot; may be specified repeatedly",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="Maximum number of newest episodes per submission",
    )
    parser.add_argument(
        "--include-logs",
        action="store_true",
        help="Also download the JSON log for your agent in each episode",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite replay and log JSONs that already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without writing files",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only refresh submission descriptions; do not list or download episodes",
    )
    parser.add_argument(
        "--page-size",
        type=positive_int,
        default=100,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--request-delay",
        type=non_negative_float,
        default=DEFAULT_REQUEST_DELAY,
        help=(
            "Minimum seconds between Kaggle API calls "
            f"(default: {DEFAULT_REQUEST_DELAY:g})"
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=non_negative_int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Retries for rate limits and temporary server errors "
            f"(default: {DEFAULT_MAX_RETRIES})"
        ),
    )
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return number


def non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return number


class ApiPacer:
    """Space API calls out and retry only transient failures."""

    def __init__(self, delay: float, max_retries: int) -> None:
        self.delay = delay
        self.max_retries = max_retries
        self.last_request_at: float | None = None

    def _wait_for_slot(self) -> None:
        if self.last_request_at is not None:
            remaining = self.delay - (time.monotonic() - self.last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_at = time.monotonic()

    def call(self, label: str, function: Any, *args: Any, **kwargs: Any) -> Any:
        for attempt in range(self.max_retries + 1):
            self._wait_for_slot()
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                status = exception_status(exc)
                if status not in RETRYABLE_HTTP_STATUSES or attempt >= self.max_retries:
                    raise

                retry_after = retry_after_seconds(exc)
                backoff = min(300.0, 10.0 * (2**attempt))
                wait = retry_after if retry_after is not None else backoff + random.uniform(0, 2)
                print(
                    f"RATE  {label}: HTTP {status}; waiting {wait:.1f}s "
                    f"before retry {attempt + 1}/{self.max_retries}",
                    file=sys.stderr,
                )
                time.sleep(wait)


def exception_status(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    candidates = (
        getattr(response, "status_code", None),
        getattr(response, "status", None),
        getattr(exc, "status", None),
        getattr(exc, "status_code", None),
    )
    for value in candidates:
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def create_api() -> Any:
    # Keep this import lazy so --help and unit tests never require credentials.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value).rsplit(".", 1)[-1].upper()


def discover_submissions(
    api: Any,
    competition: str,
    page_size: int,
    pacer: ApiPacer | None = None,
) -> dict[int, Any]:
    """Return the authenticated user's submissions keyed by submission ID."""
    found: dict[int, Any] = {}
    page = 1

    while True:
        call = pacer.call if pacer is not None else lambda _label, function, *args, **kwargs: function(*args, **kwargs)
        batch = call(
            f"submission page {page}",
            api.competition_submissions,
            competition,
            page_number=page,
            page_size=page_size,
        ) or []
        for submission in batch:
            submission_id = int(getattr(submission, "ref", 0) or 0)
            if submission_id:
                found[submission_id] = submission

        if len(batch) < page_size:
            break
        page += 1

    return found


def discover_submission_ids(api: Any, competition: str, page_size: int) -> list[int]:
    """Return all unique IDs for the authenticated user's submissions."""
    return list(discover_submissions(api, competition, page_size))


def submission_metadata(submission: Any) -> dict[str, str]:
    date = getattr(submission, "date", None)
    return {
        "description": str(getattr(submission, "description", "") or "").strip(),
        "date": date.isoformat() if hasattr(date, "isoformat") else str(date or ""),
        "status": enum_name(getattr(submission, "status", "")),
        "file_name": str(getattr(submission, "file_name", "") or ""),
    }


def write_submission_metadata(
    output: Path,
    competition: str,
    submissions: dict[int, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / SUBMISSION_METADATA_FILE
    payload = {
        "competition": competition,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "submissions": {
            str(submission_id): submission_metadata(submission)
            for submission_id, submission in submissions.items()
        },
    }
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)
    print(f"Updated submission descriptions in {path}")


def terminal_episodes(episodes: Iterable[Any], limit: int | None) -> list[Any]:
    result = [
        episode
        for episode in episodes
        if int(getattr(episode, "id", 0) or 0)
        and enum_name(getattr(episode, "state", "")) in TERMINAL_EPISODE_STATES
    ]
    result.sort(
        key=lambda episode: (
            getattr(episode, "end_time", None) is not None,
            getattr(episode, "end_time", None),
            int(getattr(episode, "id", 0)),
        ),
        reverse=True,
    )
    return result[:limit] if limit is not None else result


def own_agent_indexes(episode: Any, submission_id: int) -> list[int]:
    return [
        int(getattr(agent, "index", 0))
        for agent in (getattr(episode, "agents", None) or [])
        if int(getattr(agent, "submission_id", 0) or 0) == submission_id
    ]


def download_one(
    api: Any,
    *,
    submission_id: int,
    episode: Any,
    output: Path,
    include_logs: bool,
    force: bool,
    dry_run: bool,
    pacer: ApiPacer | None = None,
) -> DownloadStats:
    stats = DownloadStats()
    episode_id = int(episode.id)
    submission_dir = output / str(submission_id)
    replay_path = submission_dir / f"episode-{episode_id}-replay.json"

    if replay_path.exists() and not force:
        print(f"SKIP  {replay_path}")
        stats.skipped += 1
    elif dry_run:
        print(f"WOULD {replay_path}")
        stats.downloaded += 1
    else:
        try:
            submission_dir.mkdir(parents=True, exist_ok=True)
            call = pacer.call if pacer is not None else lambda _label, function, *args, **kwargs: function(*args, **kwargs)
            call(
                f"replay {episode_id}",
                api.competition_episode_replay,
                episode_id,
                path=str(submission_dir),
                quiet=True,
            )
            print(f"OK    {replay_path}")
            stats.downloaded += 1
        except Exception as exc:  # Continue so one bad episode does not stop the batch.
            print(f"ERROR episode {episode_id}: {exc}", file=sys.stderr)
            stats.failed += 1

    if not include_logs:
        return stats

    indexes = own_agent_indexes(episode, submission_id)
    if not indexes:
        print(
            f"WARN  episode {episode_id}: no agent index for submission {submission_id}",
            file=sys.stderr,
        )
        return stats

    for agent_index in indexes:
        log_path = submission_dir / f"episode-{episode_id}-agent-{agent_index}-logs.json"
        if log_path.exists() and not force:
            print(f"SKIP  {log_path}")
            stats.skipped += 1
        elif dry_run:
            print(f"WOULD {log_path}")
            stats.downloaded += 1
        else:
            try:
                call = pacer.call if pacer is not None else lambda _label, function, *args, **kwargs: function(*args, **kwargs)
                call(
                    f"logs {episode_id}/{agent_index}",
                    api.competition_episode_agent_logs,
                    episode_id,
                    agent_index,
                    path=str(submission_dir),
                    quiet=True,
                )
                print(f"OK    {log_path}")
                stats.downloaded += 1
            except Exception as exc:
                print(
                    f"ERROR episode {episode_id}, agent {agent_index}: {exc}",
                    file=sys.stderr,
                )
                stats.failed += 1

    return stats


def run(args: argparse.Namespace, api: Any) -> DownloadStats:
    pacer = ApiPacer(
        delay=getattr(args, "request_delay", DEFAULT_REQUEST_DELAY),
        max_retries=getattr(args, "max_retries", DEFAULT_MAX_RETRIES),
    )
    print(f"Discovering your submissions for {args.competition} ...")
    submissions = discover_submissions(api, args.competition, args.page_size, pacer)
    submission_ids = list(dict.fromkeys(args.submission_id))
    if not submission_ids:
        submission_ids = list(submissions)

    if not submission_ids:
        raise RuntimeError(
            f"No submissions found for competition '{args.competition}'. "
            "Check the competition slug and your Kaggle login."
        )

    if not args.dry_run:
        write_submission_metadata(args.output, args.competition, submissions)

    if getattr(args, "metadata_only", False):
        return DownloadStats()

    total = DownloadStats()
    print(f"Found {len(submission_ids)} submission(s): {', '.join(map(str, submission_ids))}")

    for submission_id in submission_ids:
        episodes = terminal_episodes(
            pacer.call(
                f"episodes for submission {submission_id}",
                api.competition_list_episodes,
                submission_id,
            ) or [],
            args.limit,
        )
        print(f"\nSubmission {submission_id}: {len(episodes)} terminal episode(s)")
        for episode in episodes:
            current = download_one(
                api,
                submission_id=submission_id,
                episode=episode,
                output=args.output,
                include_logs=args.include_logs,
                force=args.force,
                dry_run=args.dry_run,
                pacer=pacer,
            )
            total.downloaded += current.downloaded
            total.skipped += current.skipped
            total.failed += current.failed

    return total


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stats = run(args, create_api())
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    action = "would download" if args.dry_run else "downloaded"
    print(
        f"\nDone: {stats.downloaded} {action}, "
        f"{stats.skipped} skipped, {stats.failed} failed."
    )
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
