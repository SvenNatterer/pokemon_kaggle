#!/usr/bin/env python3
"""Train, select, and fine-tune a Compact V6 factory with potential rewards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
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


COMPACT_BASES = (
    {
        "id": "compact_a",
        "seed": 20260721,
        "deck_path": "decks/deck_bank/bank_54.csv",
        "model_path": "models/foundation/compact_potential/ppo_v6_deck_bank_54_compact_a.zip",
    },
    {
        "id": "compact_b",
        "seed": 20260722,
        "deck_path": "decks/deck_bank/bank_55.csv",
        "model_path": "models/foundation/compact_potential/ppo_v6_deck_bank_55_compact_b.zip",
    },
    {
        "id": "compact_c",
        "seed": 20260723,
        "deck_path": "decks/deck_bank/bank_56.csv",
        "model_path": "models/foundation/compact_potential/ppo_v6_deck_bank_56_compact_c.zip",
    },
)
DEFAULT_OUTPUT_DIR = ROOT / "models" / "opponent_factory_v6_compact_potential"
BASE_RESULTS = ROOT / "logs" / "opponent_factory_v6_compact_potential" / "base_evaluation.json"
BASE_SELECTION = ROOT / "logs" / "opponent_factory_v6_compact_potential" / "base_selection.json"
STOP_MARKER = ROOT / "stop_factory"


def compact_target_path(target: dict, base_id: str, output_dir: Path) -> Path:
    return output_dir / f"ppo_v6_deck_{target['deck_id']}_{base_id}.zip"


def common_compact_args(config: dict, num_envs: int) -> list[str]:
    defaults = config["defaults"]
    return [
        "--policy-version",
        "v6",
        "--feature-variant",
        "compact",
        "--card-table",
        "--opp-pool",
        config["development_pool"],
        "--num-envs",
        str(num_envs),
        "--n-steps",
        str(defaults["n_steps"]),
        "--lr",
        str(defaults["learning_rate"]),
        "--ent-coef",
        str(defaults["entropy_coefficient"]),
        "--clip-range",
        str(defaults["clip_range"]),
        "--target-kl",
        str(defaults["target_kl"]),
        "--belief-actor",
        "--belief-dim",
        str(defaults["belief_dim"]),
        "--rotate-perspective",
    ]


def compact_base_command(
    config: dict, base: dict, python: str, num_envs: int
) -> list[str]:
    defaults = config["defaults"]
    return [
        python,
        "src/train.py",
        "--deck",
        base["deck_path"],
        "--model-name",
        base["model_path"],
        "--timesteps",
        str(defaults["base_steps"]),
        "--seed",
        str(base["seed"]),
        "--aux-coef",
        str(defaults["base_aux_coefficient"]),
        "--batch-size",
        "1024",
        "--n-epochs",
        "2",
        *common_compact_args(config, num_envs),
    ]


def compact_finetune_command(
    config: dict,
    target: dict,
    base: dict,
    output: Path,
    python: str,
    num_envs: int,
) -> list[str]:
    defaults = config["defaults"]
    target_seed = int(base["seed"]) + int(str(target["deck_id"]).split("_")[-1])
    return [
        python,
        "src/train.py",
        "--deck",
        relative(target_deck_path(target)),
        "--model-name",
        relative(output),
        "--continue-existing",
        "--timesteps",
        str(defaults["finetune_steps"]),
        "--seed",
        str(target_seed),
        "--aux-coef",
        str(defaults["finetune_aux_coefficient"]),
        "--batch-size",
        str(defaults["batch_size"]),
        "--n-epochs",
        str(defaults["n_epochs"]),
        *common_compact_args(config, num_envs),
    ]


def base_evaluation_command(config: dict, python: str) -> list[str]:
    evaluation = config["base_evaluation"]
    command = [
        python,
        "scripts/evaluate_submission.py",
        "--holdout-file",
        evaluation["manifest"],
        "--games",
        str(evaluation["games_per_opponent"]),
        "--results-file",
        relative(BASE_RESULTS),
    ]
    for base in COMPACT_BASES:
        command.extend(["--candidate", base["model_path"]])
    return command


def consume_stop_marker() -> bool:
    if not STOP_MARKER.exists():
        return False
    STOP_MARKER.unlink()
    print(
        "Stop marker detected. The current unit is complete; stopping before the next unit.",
        flush=True,
    )
    return True


def train_bases(
    config: dict, python: str, num_envs: int, wandb_mode: str, dry_run: bool
) -> bool:
    for base in COMPACT_BASES:
        output = repo_path(base["model_path"])
        marker = marker_for(output)
        if output.is_file() and marker.is_file():
            print(f"Reusing completed Compact base: {relative(output)}", flush=True)
            continue
        if output.exists():
            raise RuntimeError(
                f"Incomplete Compact base exists; inspect it before restarting: {relative(output)}"
            )
        if not dry_run and consume_stop_marker():
            return False
        command = compact_base_command(config, base, python, num_envs)
        if dry_run:
            print(" ".join(command))
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        run_command(command, wandb_mode)
        marker.touch()
    return True


def model_fingerprints() -> dict[str, dict[str, int | str]]:
    result = {}
    for base in COMPACT_BASES:
        path = repo_path(base["model_path"])
        stat = path.stat()
        result[base["id"]] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": file_sha256(path),
        }
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(path),
    }


def _referenced_decks(manifest_path: Path) -> list[Path]:
    payload = read_json(manifest_path)
    entries = payload if isinstance(payload, list) else payload.get("opponents", [])
    paths = []
    for entry in entries:
        value = entry.get("deck") or entry.get("deck_path")
        if value:
            paths.append(repo_path(value))
    return paths


def selection_input_fingerprints(config: dict) -> dict[str, dict[str, int | str]]:
    manifest_paths = [
        repo_path(config["development_pool"]),
        repo_path(config["base_evaluation"]["manifest"]),
    ]
    paths = [
        *(repo_path(base["deck_path"]) for base in COMPACT_BASES),
        *manifest_paths,
    ]
    for manifest_path in manifest_paths:
        paths.extend(_referenced_decks(manifest_path))

    unique_paths = sorted({path.resolve() for path in paths})
    return {
        relative(path): file_fingerprint(path)
        for path in unique_paths
    }


def current_selection(config: dict | None = None) -> list[dict] | None:
    if not BASE_SELECTION.is_file():
        return None
    config = config or read_json(repo_path(DEFAULT_CONFIG))
    payload = read_json(BASE_SELECTION)
    if payload.get("base_fingerprints") != model_fingerprints():
        return None
    if payload.get("input_fingerprints") != selection_input_fingerprints(config):
        return None
    by_id = {base["id"]: base for base in COMPACT_BASES}
    selected_ids = payload.get("selected_base_ids", [])
    if len(selected_ids) != len(COMPACT_BASES) or any(item not in by_id for item in selected_ids):
        return None
    return [by_id[item] for item in selected_ids]


def evaluate_bases(
    config: dict, python: str, wandb_mode: str, dry_run: bool
) -> list[dict]:
    if not dry_run:
        selected = current_selection(config)
        if selected is not None:
            print(
                "Reusing current Compact base selection: "
                + ", ".join(base["id"] for base in selected),
                flush=True,
            )
            return selected
        for base in COMPACT_BASES:
            path = repo_path(base["model_path"])
            if not path.is_file() or not marker_for(path).is_file():
                raise RuntimeError(f"Compact base is not complete: {relative(path)}")

    command = base_evaluation_command(config, python)
    if dry_run:
        print(" ".join(command))
        return list(COMPACT_BASES)

    BASE_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    run_command(command, wandb_mode)
    summaries = read_json(BASE_RESULTS).get("summary", [])
    by_candidate = {row["candidate"]: row for row in summaries}
    evaluation = config["base_evaluation"]
    ranked = []
    for base in COMPACT_BASES:
        summary = by_candidate.get(Path(base["model_path"]).stem)
        if summary is None:
            raise RuntimeError(f"Missing Compact evaluation result for {base['id']}")
        if int(summary.get("crashes", 0)):
            raise RuntimeError(f"Compact base crashed during evaluation: {base['id']}")
        if float(summary["score_rate"]) < float(evaluation["minimum_score_rate"]):
            raise RuntimeError(f"Compact base failed score gate: {base['id']}")
        if float(summary["perspective_score_gap"]) > float(
            evaluation["maximum_perspective_gap"]
        ):
            raise RuntimeError(f"Compact base failed perspective gate: {base['id']}")
        ranked.append((base, summary))

    ranked.sort(
        key=lambda item: (
            item[1]["wilson95_score_lb"],
            item[1]["worst_score_rate"],
            item[1]["score_rate"],
        ),
        reverse=True,
    )
    selected = [item[0] for item in ranked]
    payload = {
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "selected_base_ids": [base["id"] for base in selected],
        "base_fingerprints": model_fingerprints(),
        "input_fingerprints": selection_input_fingerprints(config),
        "ranking": [
            {"base_id": base["id"], "summary": summary}
            for base, summary in ranked
        ],
    }
    BASE_SELECTION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "Selected Compact base order: " + ", ".join(payload["selected_base_ids"]),
        flush=True,
    )
    return selected


def train_targets(
    config: dict,
    selected_bases: list[dict],
    output_dir: Path,
    python: str,
    num_envs: int,
    wandb_mode: str,
    split: str | None,
    dry_run: bool,
) -> bool:
    for index, target in enumerate(config["targets"]):
        if split and target["split"] != split:
            continue
        base = selected_bases[index % len(selected_bases)]
        source = repo_path(base["model_path"])
        output = compact_target_path(target, base["id"], output_dir)
        marker = marker_for(output)
        if output.is_file() and marker.is_file():
            print(f"Reusing completed Compact target: {relative(output)}", flush=True)
            continue
        if output.exists():
            raise RuntimeError(
                f"Incomplete Compact target exists; inspect it before restarting: {relative(output)}"
            )
        if not dry_run and consume_stop_marker():
            return False

        command = compact_finetune_command(
            config, target, base, output, python, num_envs
        )
        if dry_run:
            print(f"cp {relative(source)} {relative(output)}")
            print(" ".join(command))
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
        run_command(command, wandb_mode)
        marker.touch()
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=relative(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-envs", type=int, default=7)
    parser.add_argument(
        "--stage",
        choices=("bases", "evaluate-bases", "targets", "all"),
        default="all",
    )
    parser.add_argument("--split", choices=("training", "validation", "holdout"))
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")

    config = read_json(repo_path(args.config))
    validate_config(config)
    validate_static_inputs(config)
    selected = None

    if args.stage in {"bases", "all"}:
        if not train_bases(
            config, args.python, args.num_envs, args.wandb_mode, args.dry_run
        ):
            return 0
    if args.stage in {"evaluate-bases", "all"}:
        if not args.dry_run and consume_stop_marker():
            return 0
        selected = evaluate_bases(config, args.python, args.wandb_mode, args.dry_run)
    if args.stage in {"targets", "all"}:
        if selected is None:
            selected = list(COMPACT_BASES) if args.dry_run else current_selection(config)
        if selected is None:
            raise RuntimeError("Compact base selection is missing; run --stage evaluate-bases first")
        if not train_targets(
            config,
            selected,
            repo_path(args.output_dir),
            args.python,
            args.num_envs,
            args.wandb_mode,
            args.split,
            args.dry_run,
        ):
            return 0

    print("Compact opponent factory stage completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
