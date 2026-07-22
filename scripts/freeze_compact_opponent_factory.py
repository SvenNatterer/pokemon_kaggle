#!/usr/bin/env python3
"""Freeze the validated Compact/Potential V6 factory into disjoint leagues."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_opponent_factory import (  # noqa: E402
    DEFAULT_CONFIG,
    marker_for,
    read_json,
    relative,
    repo_path,
    target_deck_path,
    validate_config,
)
from scripts.run_compact_opponent_factory import (  # noqa: E402
    COMPACT_BASES,
    DEFAULT_OUTPUT_DIR,
    compact_target_path,
    file_fingerprint,
    file_sha256,
)
from src.arena.arena_core import atomic_write_json  # noqa: E402


DEFAULT_REPAIR_SUMMARY = (
    ROOT
    / "logs"
    / "opponent_factory_v6_compact_potential"
    / "scrape_bug_repair.json"
)
DEFAULT_TARGET_EVALUATION = (
    ROOT
    / "logs"
    / "opponent_factory_v6_compact_potential"
    / "post_scrape_target_evaluation.json"
)
DEFAULT_GENERATED_DIR = (
    ROOT / "decks" / "generated" / "opponent_factory_v6_compact_potential"
)
ACTIVE_VALIDATION = ROOT / "decks" / "validation_opponents.json"
ACTIVE_HOLDOUT = ROOT / "decks" / "holdout_opponents.json"
HISTORICAL_VALIDATION = ROOT / "decks" / "historical" / "validation_opponents_v1.json"
HISTORICAL_HOLDOUT = ROOT / "decks" / "historical" / "holdout_opponents_v1.json"

FROZEN_DIRECTORIES = {
    "training": ROOT / "models" / "training_v6",
    "validation": ROOT / "models" / "validation" / "v6",
    "holdout": ROOT / "models" / "holdout" / "v6",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selected_bases_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {base["id"]: base for base in COMPACT_BASES}
    selected_ids = summary.get("selected_base_ids", [])
    if len(selected_ids) != len(COMPACT_BASES) or any(
        base_id not in by_id for base_id in selected_ids
    ):
        raise RuntimeError("Repair summary has an invalid Compact base selection")
    return [by_id[base_id] for base_id in selected_ids]


def target_assignments(
    config: dict[str, Any], selected_bases: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (target, selected_bases[index % len(selected_bases)])
        for index, target in enumerate(config["targets"])
    ]


def frozen_model_path(source: Path, split: str) -> Path:
    return FROZEN_DIRECTORIES[split] / source.name


def verify_factory(
    config: dict[str, Any],
    output_dir: Path,
    repair_summary: dict[str, Any],
    target_evaluation: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]]:
    if repair_summary.get("status") != "completed":
        raise RuntimeError("Compact factory repair is not marked completed")

    selected_bases = selected_bases_from_summary(repair_summary)
    expected_fingerprints = repair_summary.get("target_model_fingerprints", {})
    evaluation_by_candidate = {
        row.get("candidate"): row for row in target_evaluation.get("summary", [])
    }
    verified = []
    for target, base in target_assignments(config, selected_bases):
        source = compact_target_path(target, base["id"], output_dir)
        if not source.is_file() or not marker_for(source).is_file():
            raise RuntimeError(f"Compact target is incomplete: {relative(source)}")

        expected = expected_fingerprints.get(target["deck_id"])
        actual = file_fingerprint(source)
        if not expected or actual.get("sha256") != expected.get("sha256"):
            raise RuntimeError(
                f"Compact target no longer matches its repair summary: {relative(source)}"
            )

        evaluation = evaluation_by_candidate.get(source.stem)
        if not evaluation or int(evaluation.get("games", 0)) <= 0:
            raise RuntimeError(f"Target evaluation is missing: {source.stem}")
        if int(evaluation.get("crashes", 0)):
            raise RuntimeError(f"Target evaluation contains crashes: {source.stem}")
        verified.append((target, base, source, evaluation))
    return verified


def opponent_entry(
    target: dict[str, Any], base: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return {
        "label": f"ppo_v6_{base['id']}_{target['deck_id']}",
        "deck_id": target["deck_id"],
        "archetype": target["archetype"],
        "policy_family": base["id"],
        "policy_version": "v6",
        "feature_variant": "compact",
        "reward_variant": "potential",
        "action_space_size": 66,
        "model_path": relative(destination),
        "deck_path": relative(target_deck_path(target)),
        "sha256": file_sha256(destination),
    }


def build_manifests(
    entries: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    return {
        "training_pool_v6.json": [
            {
                "label": row["label"],
                "deck": row["deck_path"],
                "model": row["model_path"],
                "weight": 1.0,
            }
            for row in entries["training"]
        ],
        "validation_opponents_v6.json": {
            "version": 6,
            "purpose": (
                "Active V6 model selection only; never train against these exact opponents."
            ),
            "factory": "compact_potential_v6",
            "opponents": entries["validation"],
        },
        "final_holdout_opponents_v6.json": {
            "version": 6,
            "purpose": (
                "Untouched final V6 evaluation only; do not inspect during model selection."
            ),
            "factory": "compact_potential_v6",
            "opponents": entries["holdout"],
        },
    }


def preserve_historical_manifest(active: Path, historical: Path) -> None:
    if historical.exists():
        return
    if not active.is_file():
        raise FileNotFoundError(f"Active historical manifest is missing: {relative(active)}")
    historical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(active, historical)


def copy_frozen_model(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(destination) != file_sha256(source):
            raise RuntimeError(f"Frozen model already differs: {relative(destination)}")
        return
    shutil.copy2(source, destination)


def freeze(
    config: dict[str, Any],
    output_dir: Path,
    generated_dir: Path,
    repair_summary: dict[str, Any],
    target_evaluation: dict[str, Any],
    activate_validation: bool,
    activate_holdout: bool,
    dry_run: bool,
) -> dict[str, Any]:
    verified = verify_factory(
        config, output_dir, repair_summary, target_evaluation
    )
    entries: dict[str, list[dict[str, Any]]] = {
        "training": [],
        "validation": [],
        "holdout": [],
    }

    if dry_run:
        for target, _base, source, _evaluation in verified:
            print(
                f"freeze {relative(source)} -> "
                f"{relative(frozen_model_path(source, target['split']))}"
            )
        print(f"write manifests under {relative(generated_dir)}")
        if activate_validation:
            print(f"activate {relative(ACTIVE_VALIDATION)}")
        if activate_holdout:
            print(f"activate {relative(ACTIVE_HOLDOUT)}")
        return {}

    for target, base, source, _evaluation in verified:
        destination = frozen_model_path(source, target["split"])
        copy_frozen_model(source, destination)
        entries[target["split"]].append(opponent_entry(target, base, destination))

    manifests = build_manifests(entries)
    generated_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in manifests.items():
        atomic_write_json(generated_dir / name, payload)

    preserve_historical_manifest(ACTIVE_VALIDATION, HISTORICAL_VALIDATION)
    preserve_historical_manifest(ACTIVE_HOLDOUT, HISTORICAL_HOLDOUT)
    if activate_validation:
        atomic_write_json(
            ACTIVE_VALIDATION, manifests["validation_opponents_v6.json"]
        )
    if activate_holdout:
        atomic_write_json(
            ACTIVE_HOLDOUT, manifests["final_holdout_opponents_v6.json"]
        )

    report = {
        "status": "completed",
        "completed_at": utc_now(),
        "factory": "compact_potential_v6",
        "repair_summary": relative(DEFAULT_REPAIR_SUMMARY),
        "target_evaluation": relative(DEFAULT_TARGET_EVALUATION),
        "counts": {split: len(rows) for split, rows in entries.items()},
        "active_validation": activate_validation,
        "active_holdout": activate_holdout,
        "historical_validation": relative(HISTORICAL_VALIDATION),
        "historical_holdout": relative(HISTORICAL_HOLDOUT),
    }
    atomic_write_json(generated_dir / "freeze_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=relative(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--repair-summary", default=relative(DEFAULT_REPAIR_SUMMARY))
    parser.add_argument(
        "--target-evaluation", default=relative(DEFAULT_TARGET_EVALUATION)
    )
    parser.add_argument("--generated-dir", default=relative(DEFAULT_GENERATED_DIR))
    parser.add_argument(
        "--staged-only",
        action="store_true",
        help="Write frozen artifacts without activating the V6 validation manifest.",
    )
    parser.add_argument(
        "--activate-holdout",
        action="store_true",
        help="Explicitly replace the active historical holdout with the sealed V6 holdout.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(repo_path(args.config))
    validate_config(config)
    report = freeze(
        config=config,
        output_dir=repo_path(args.output_dir),
        generated_dir=repo_path(args.generated_dir),
        repair_summary=read_json(repo_path(args.repair_summary)),
        target_evaluation=read_json(repo_path(args.target_evaluation)),
        activate_validation=not args.staged_only,
        activate_holdout=args.activate_holdout,
        dry_run=args.dry_run,
    )
    if report:
        print(
            "Frozen Compact/Potential V6 league: "
            f"{report['counts']['training']} training, "
            f"{report['counts']['validation']} validation, "
            f"{report['counts']['holdout']} sealed holdout.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
