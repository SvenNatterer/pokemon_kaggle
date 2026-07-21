#!/usr/bin/env python3
"""Repair Compact V6 target models trained with incorrectly scraped card prints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_opponent_factory import (  # noqa: E402
    DEFAULT_CONFIG,
    marker_for,
    read_json,
    relative,
    repo_path,
    run_command,
    target_deck_path,
    validate_config,
    validate_static_inputs,
)
from scripts.run_compact_opponent_factory import (  # noqa: E402
    BASE_RESULTS,
    BASE_SELECTION,
    COMPACT_BASES,
    DEFAULT_OUTPUT_DIR,
    compact_finetune_command,
    compact_target_path,
    current_selection,
    evaluate_bases,
    file_fingerprint,
    file_sha256,
    selection_input_fingerprints,
)
from src.arena_core import atomic_write_json  # noqa: E402
from src.experiment_registry import registry_path  # noqa: E402


REPAIR_TARGET_IDS = (
    "bank_33",
    "bank_49",
    "bank_61",
    "bank_25",
    "bank_36",
    "bank_84",
    "bank_2",
    "bank_99",
)
DEFAULT_WAIT_FOR = (
    ROOT
    / "models"
    / "opponent_factory_v6_compact_potential"
    / "ppo_v6_deck_bank_70_compact_c.complete"
)
DEFAULT_ARCHIVE_DIR = ROOT / "models" / "archive_scrape_bug_20260716"
LOG_DIR = ROOT / "logs" / "opponent_factory_v6_compact_potential"
TARGET_RESULTS = LOG_DIR / "post_scrape_target_evaluation.json"
TARGET_PROGRESS = LOG_DIR / "post_scrape_target_evaluation_progress.json"
REPAIR_SUMMARY = LOG_DIR / "scrape_bug_repair.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wait_for_file(path: Path, poll_seconds: int) -> None:
    if path.is_file():
        return
    print(f"Waiting for the current factory to finish: {relative(path)}", flush=True)
    last_report = time.monotonic()
    while not path.is_file():
        time.sleep(poll_seconds)
        if time.monotonic() - last_report >= 600:
            print(f"Still waiting for {relative(path)}", flush=True)
            last_report = time.monotonic()
    print(f"Detected completed final factory target: {relative(path)}", flush=True)


def target_assignments(config: dict, selected_bases: list[dict]) -> list[tuple[dict, dict]]:
    return [
        (target, selected_bases[index % len(selected_bases)])
        for index, target in enumerate(config["targets"])
    ]


def archive_path(path: Path, archive_root: Path) -> Path:
    return archive_root / relative(path)


def archive_artifact(path: Path, archive_root: Path) -> None:
    if not path.exists():
        return
    destination = archive_path(path, archive_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if path.is_file() and file_sha256(path) == file_sha256(destination):
            path.unlink()
            return
        retry_index = 1
        retry_destination = destination.with_name(
            f"{destination.name}.retry_{retry_index}"
        )
        while retry_destination.exists():
            retry_index += 1
            retry_destination = destination.with_name(
                f"{destination.name}.retry_{retry_index}"
            )
        destination = retry_destination
    shutil.move(path, destination)
    print(f"Archived {relative(path)} -> {relative(destination)}", flush=True)


def repair_marker_for(model_path: Path) -> Path:
    return model_path.with_suffix(".scrape_fix.json")


def target_provenance(
    config: dict,
    target: dict,
    base: dict,
    source: Path,
    output: Path,
) -> dict:
    return {
        "repair_version": 1,
        "reason": "Limitless scraper selected duplicate-name card prints without set/number matching.",
        "completed_at": utc_now(),
        "target_deck_id": target["deck_id"],
        "target_deck": relative(target_deck_path(target)),
        "target_deck_fingerprint": file_fingerprint(target_deck_path(target)),
        "development_pool": config["development_pool"],
        "development_pool_fingerprint": file_fingerprint(
            repo_path(config["development_pool"])
        ),
        "base_id": base["id"],
        "base_model": relative(source),
        "base_model_fingerprint": file_fingerprint(source),
        "output_model": relative(output),
        "output_model_fingerprint": file_fingerprint(output),
    }


def repair_targets(
    config: dict,
    selected_bases: list[dict],
    output_dir: Path,
    archive_root: Path,
    python: str,
    num_envs: int,
    wandb_mode: str,
    dry_run: bool,
) -> list[str]:
    repaired = []
    assignments = {
        target["deck_id"]: (target, base)
        for target, base in target_assignments(config, selected_bases)
    }
    missing = sorted(set(REPAIR_TARGET_IDS) - set(assignments))
    if missing:
        raise RuntimeError(f"Repair targets are missing from the config: {missing}")

    for deck_id in REPAIR_TARGET_IDS:
        target, base = assignments[deck_id]
        source = repo_path(base["model_path"])
        output = compact_target_path(target, base["id"], output_dir)
        complete_marker = marker_for(output)
        repair_marker = repair_marker_for(output)

        if repair_marker.is_file() and output.is_file() and complete_marker.is_file():
            provenance = read_json(repair_marker)
            current_deck = file_fingerprint(target_deck_path(target))
            if provenance.get("target_deck_fingerprint") == current_deck:
                print(f"Reusing repaired target: {relative(output)}", flush=True)
                repaired.append(deck_id)
                continue

        command = compact_finetune_command(
            config, target, base, output, python, num_envs
        )
        if dry_run:
            print(f"archive {relative(output)} under {relative(archive_root)}")
            print(f"cp {relative(source)} {relative(output)}")
            print(" ".join(command))
            repaired.append(deck_id)
            continue

        if not source.is_file() or not marker_for(source).is_file():
            raise RuntimeError(f"Compact base is not complete: {relative(source)}")

        for artifact in (
            output,
            complete_marker,
            repair_marker,
            registry_path(str(output)),
        ):
            archive_artifact(artifact, archive_root)

        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
        run_command(command, wandb_mode)
        complete_marker.touch()
        atomic_write_json(
            repair_marker,
            target_provenance(config, target, base, source, output),
        )
        repaired.append(deck_id)

    return repaired


def evaluate_targets(
    config: dict,
    selected_bases: list[dict],
    output_dir: Path,
    python: str,
    games: int,
    dry_run: bool,
) -> None:
    command = [
        python,
        "scripts/evaluate_submission.py",
        "--holdout-file",
        config["base_evaluation"]["manifest"],
        "--games",
        str(games),
        "--results-file",
        relative(TARGET_RESULTS),
        "--progress-file",
        relative(TARGET_PROGRESS),
    ]
    for target, base in target_assignments(config, selected_bases):
        candidate = compact_target_path(target, base["id"], output_dir)
        if not dry_run and (
            not candidate.is_file() or not marker_for(candidate).is_file()
        ):
            raise RuntimeError(f"Target is not complete: {relative(candidate)}")
        command.extend(["--candidate", relative(candidate)])

    if dry_run:
        print(" ".join(command))
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_command(command, "disabled")


def snapshot_evaluation_logs(archive_root: Path, dry_run: bool) -> None:
    for path in (BASE_RESULTS, BASE_SELECTION):
        if not path.is_file():
            continue
        destination = archive_path(path, archive_root)
        if dry_run:
            print(f"snapshot {relative(path)} -> {relative(destination)}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)


def factory_summary(
    config: dict,
    selected_bases: list[dict],
    repaired: list[str],
    output_dir: Path,
    archive_root: Path,
) -> dict:
    assignments = target_assignments(config, selected_bases)
    return {
        "status": "completed",
        "completed_at": utc_now(),
        "repair_version": 1,
        "repaired_target_ids": repaired,
        "selected_base_ids": [base["id"] for base in selected_bases],
        "selection_input_fingerprints": selection_input_fingerprints(config),
        "target_deck_fingerprints": {
            target["deck_id"]: file_fingerprint(target_deck_path(target))
            for target, _ in assignments
        },
        "target_model_fingerprints": {
            target["deck_id"]: file_fingerprint(
                compact_target_path(target, base["id"], output_dir)
            )
            for target, base in assignments
        },
        "archive_dir": relative(archive_root),
        "base_evaluation": relative(BASE_RESULTS),
        "base_selection": relative(BASE_SELECTION),
        "target_evaluation": relative(TARGET_RESULTS),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=relative(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--archive-dir", default=relative(DEFAULT_ARCHIVE_DIR))
    parser.add_argument("--wait-for", default=relative(DEFAULT_WAIT_FOR))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-envs", type=int, default=7)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_seconds <= 0 or args.num_envs <= 0 or args.games <= 0:
        raise ValueError("poll-seconds, num-envs and games must be positive")

    config = read_json(repo_path(args.config))
    validate_config(config)
    validate_static_inputs(config)
    wait_path = repo_path(args.wait_for)
    output_dir = repo_path(args.output_dir)
    archive_root = repo_path(args.archive_dir)

    if not args.no_wait and not args.dry_run:
        wait_for_file(wait_path, args.poll_seconds)

    snapshot_evaluation_logs(archive_root, args.dry_run)
    selected = evaluate_bases(
        config, args.python, args.wandb_mode, args.dry_run
    )
    if args.dry_run:
        if BASE_SELECTION.is_file():
            by_id = {base["id"]: base for base in COMPACT_BASES}
            selected_ids = read_json(BASE_SELECTION).get("selected_base_ids", [])
            if len(selected_ids) == len(COMPACT_BASES) and all(
                base_id in by_id for base_id in selected_ids
            ):
                selected = [by_id[base_id] for base_id in selected_ids]
            else:
                selected = list(selected)
        else:
            selected = list(selected)
    elif current_selection(config) is None:
        raise RuntimeError("Base selection is stale immediately after evaluation")

    repaired = repair_targets(
        config,
        selected,
        output_dir,
        archive_root,
        args.python,
        args.num_envs,
        args.wandb_mode,
        args.dry_run,
    )
    evaluate_targets(
        config,
        selected,
        output_dir,
        args.python,
        args.games,
        args.dry_run,
    )
    if not args.dry_run:
        atomic_write_json(
            REPAIR_SUMMARY,
            factory_summary(
                config, selected, repaired, output_dir, archive_root
            ),
        )
    print("Compact V6 scraper repair pipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
