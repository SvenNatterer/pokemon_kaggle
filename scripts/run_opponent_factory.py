#!/usr/bin/env python3
"""Train, gate, and freeze independent compact-action V6 opponent families."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "decks" / "opponent_factory_v6.json"
VALID_SPLITS = {"training", "validation", "holdout"}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def marker_for(model_path: Path) -> Path:
    return model_path.with_suffix(".complete")


def target_deck_path(target: dict[str, Any]) -> Path:
    return ROOT / "decks" / "deck_bank" / f"{target['deck_id']}.csv"


def target_candidate_path(target: dict[str, Any], base_id: str) -> Path:
    return ROOT / "models" / "opponent_factory_v6" / (
        f"ppo_v6_deck_{target['deck_id']}_{base_id}.zip"
    )


def frozen_model_path(target: dict[str, Any], base_id: str) -> Path:
    directory = {
        "training": ROOT / "models" / "training_v6",
        "validation": ROOT / "models" / "validation" / "v6",
        "holdout": ROOT / "models" / "holdout" / "v6",
    }[target["split"]]
    return directory / target_candidate_path(target, base_id).name


def all_base_definitions(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [*config["bases"], *config.get("fallback_bases", [])]


def validate_config(config: dict[str, Any]) -> None:
    if config.get("version") != 3 or config.get("policy_version") != "v6":
        raise ValueError("Opponent factory requires config version 3 and policy_version v6")
    bases = config.get("bases")
    targets = config.get("targets")
    if not isinstance(bases, list) or len(bases) < 3:
        raise ValueError("At least three independent V6 base definitions are required")
    if not isinstance(targets, list) or not targets:
        raise ValueError("At least one target definition is required")

    fallback_bases = config.get("fallback_bases", [])
    if not isinstance(fallback_bases, list) or len(fallback_bases) != 2:
        raise ValueError("Exactly two fallback base definitions are required")

    all_bases = all_base_definitions(config)
    base_ids = [item.get("id") for item in all_bases]
    if len(base_ids) != len(set(base_ids)) or any(not item for item in base_ids):
        raise ValueError("Base IDs must be present and unique")
    seeds = [item.get("seed") for item in all_bases]
    if len(seeds) != len(set(seeds)) or any(seed is None for seed in seeds):
        raise ValueError("Base seeds must be present and unique")

    evaluation = config.get("base_evaluation") or {}
    selected_count = int(evaluation.get("selected_base_count", 0))
    minimum_passing = int(evaluation.get("minimum_passing_bases", 0))
    if not 1 <= selected_count <= len(bases):
        raise ValueError("selected_base_count must be between one and the base count")
    if not selected_count <= minimum_passing <= len(bases):
        raise ValueError("minimum_passing_bases must cover every selected base")

    deck_ids: set[str] = set()
    for target in targets:
        deck_id = target.get("deck_id")
        if not deck_id or deck_id in deck_ids:
            raise ValueError(f"Target deck IDs must be present and unique: {deck_id}")
        deck_ids.add(deck_id)
        if target.get("split") not in VALID_SPLITS:
            raise ValueError(f"Invalid split for {deck_id}: {target.get('split')}")

    base_decks = {relative(repo_path(item["deck_path"])) for item in all_bases}
    target_decks = {relative(target_deck_path(item)) for item in targets}
    overlap = base_decks & target_decks
    if overlap:
        raise ValueError(f"Base decks must not also be benchmark targets: {sorted(overlap)}")


def validate_static_inputs(config: dict[str, Any]) -> None:
    required = [
        repo_path(config["development_pool"]),
        repo_path(config["base_evaluation"]["manifest"]),
        ROOT / "src" / "train.py",
        ROOT / "scripts" / "evaluate_submission.py",
    ]
    required.extend(repo_path(base["deck_path"]) for base in all_base_definitions(config))
    required.extend(target_deck_path(target) for target in config["targets"])

    pool = read_json(repo_path(config["development_pool"]))
    for entry in pool:
        required.append(repo_path(entry["deck"]))
        model = entry.get("model")
        if model and not str(model).startswith(("rule_based:", "heuristic")):
            required.append(repo_path(model))
    missing = sorted({relative(path) for path in required if not path.is_file()})
    if missing:
        raise FileNotFoundError("Missing opponent-factory inputs:\n  - " + "\n  - ".join(missing))


def common_train_args(config: dict[str, Any], scalar_obs: bool = False) -> list[str]:
    defaults = config["defaults"]
    args = [
        "--policy-version", "v6",
        "--feature-variant", "full",
        "--card-table",
        "--opp-pool", config["development_pool"],
        "--num-envs", str(defaults["num_envs"]),
        "--n-steps", str(defaults["n_steps"]),
        "--batch-size", str(defaults["batch_size"]),
        "--n-epochs", str(defaults["n_epochs"]),
        "--lr", str(defaults["learning_rate"]),
        "--ent-coef", str(defaults["entropy_coefficient"]),
        "--clip-range", str(defaults["clip_range"]),
        "--target-kl", str(defaults["target_kl"]),
        "--belief-actor",
        "--belief-dim", str(defaults["belief_dim"]),
        "--rotate-perspective",
    ]
    if scalar_obs:
        args.append("--scalar-obs")
    return args


def base_command(config: dict[str, Any], base: dict[str, Any], python: str, scalar_obs: bool = False) -> list[str]:
    defaults = config["defaults"]
    return [
        python, "src/train.py",
        "--deck", base["deck_path"],
        "--model-name", base["model_path"],
        "--timesteps", str(defaults["base_steps"]),
        "--seed", str(base["seed"]),
        "--aux-coef", str(defaults["base_aux_coefficient"]),
        *common_train_args(config, scalar_obs),
    ]


def finetune_command(
    config: dict[str, Any], target: dict[str, Any], base: dict[str, Any], python: str, scalar_obs: bool = False
) -> list[str]:
    defaults = config["defaults"]
    target_seed = int(base["seed"]) + int(str(target["deck_id"]).split("_")[-1])
    return [
        python, "src/train.py",
        "--deck", relative(target_deck_path(target)),
        "--model-name", relative(target_candidate_path(target, base["id"])),
        "--continue-existing",
        "--timesteps", str(defaults["finetune_steps"]),
        "--seed", str(target_seed),
        "--aux-coef", str(defaults["finetune_aux_coefficient"]),
        *common_train_args(config, scalar_obs),
    ]


def base_evaluation_command(
    config: dict[str, Any], python: str, bases: list[dict[str, Any]] | None = None
) -> list[str]:
    evaluation = config["base_evaluation"]
    command = [
        python, "scripts/evaluate_submission.py",
        "--holdout-file", evaluation["manifest"],
        "--games", str(evaluation["games_per_opponent"]),
        "--results-file", evaluation["results_file"],
    ]
    for base in bases or config["bases"]:
        command.extend(["--candidate", base["model_path"]])
    return command


def print_command(command: list[str]) -> None:
    print(" ".join(shlex.quote(part) for part in command), flush=True)


def run_command(command: list[str], wandb_mode: str) -> None:
    print_command(command)
    environment = os.environ.copy()
    environment.setdefault("PYTHONPATH", "src")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment["WANDB_MODE"] = wandb_mode
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def run_bases(
    config: dict[str, Any], python: str, dry_run: bool, force: bool, wandb_mode: str,
    bases: list[dict[str, Any]] | None = None, scalar_obs: bool = False,
) -> None:
    for base in bases or config["bases"]:
        output = repo_path(base["model_path"])
        marker = marker_for(output)
        if output.exists() and marker.exists() and not force:
            print(f"Reusing completed base: {relative(output)}")
            continue
        if output.exists() and not force:
            raise RuntimeError(f"Incomplete base exists; inspect it or rerun with --force: {relative(output)}")
        command = base_command(config, base, python, scalar_obs)
        if dry_run:
            print_command(command)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        if force and output.exists():
            output.unlink()
        run_command(command, wandb_mode)
        marker.touch()


def evaluate_bases(
    config: dict[str, Any], python: str, dry_run: bool, wandb_mode: str, scalar_obs: bool = False
) -> None:
    bases = list(config["bases"])
    if dry_run:
        print_command(base_evaluation_command(config, python, bases))
        fallback_bases = list(config.get("fallback_bases", []))
        print("If fewer than the required bases pass, train two fallback bases:")
        run_bases(config, python, True, False, wandb_mode, fallback_bases, scalar_obs)
        print_command(base_evaluation_command(config, python, bases + fallback_bases))
        return

    results_path = repo_path(config["base_evaluation"]["results_file"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation = config["base_evaluation"]

    def run_evaluation(
        candidate_bases: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        for base in candidate_bases:
            model = repo_path(base["model_path"])
            if not model.is_file() or not marker_for(model).is_file():
                raise RuntimeError(f"Base is not complete: {relative(model)}")
        run_command(base_evaluation_command(config, python, candidate_bases), wandb_mode)

        summaries = read_json(results_path).get("summary", [])
        by_candidate = {row["candidate"]: row for row in summaries}
        passing: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for base in candidate_bases:
            candidate = Path(base["model_path"]).stem
            row = by_candidate.get(candidate)
            reasons = []
            if row is None:
                reasons.append("missing result")
            else:
                if float(row.get("score_rate", 0)) < float(evaluation["minimum_score_rate"]):
                    reasons.append("score below gate")
                if float(row.get("perspective_score_gap", 1)) > float(evaluation["maximum_perspective_gap"]):
                    reasons.append("perspective gap above gate")
                if int(row.get("crashes", 0)):
                    reasons.append("evaluation crash")
            item = {
                "base_id": base["id"],
                "candidate": candidate,
                "summary": row,
                "reasons": reasons,
            }
            (rejected if reasons else passing).append(item)
        return passing, rejected

    passing, rejected = run_evaluation(bases)
    minimum = int(evaluation["minimum_passing_bases"])
    fallback_used = False
    if len(passing) < minimum:
        fallback_bases = list(config["fallback_bases"])
        print(
            f"Only {len(passing)} bases passed; training two fallback bases.",
            flush=True,
        )
        run_bases(config, python, False, False, wandb_mode, fallback_bases, scalar_obs)
        bases.extend(fallback_bases)
        passing, rejected = run_evaluation(bases)
        fallback_used = True

    passing.sort(
        key=lambda item: (
            item["summary"]["wilson95_score_lb"],
            item["summary"]["worst_score_rate"],
            item["summary"]["score_rate"],
        ),
        reverse=True,
    )
    if len(passing) < minimum:
        raise RuntimeError(
            f"Only {len(passing)} bases passed after fallback training; "
            f"{minimum} are required. Fine-tuning is blocked."
        )
    selected = passing[: int(evaluation["selected_base_count"])]
    selection = {
        "policy_version": "v6",
        "selected_base_ids": [item["base_id"] for item in selected],
        "fallback_used": fallback_used,
        "passing": passing,
        "rejected": rejected,
        "gates": {
            "minimum_score_rate": evaluation["minimum_score_rate"],
            "maximum_perspective_gap": evaluation["maximum_perspective_gap"],
        },
    }
    selection_path = repo_path(evaluation["selection_file"])
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(f"Selected V6 bases: {', '.join(selection['selected_base_ids'])}")


def selected_bases(config: dict[str, Any], dry_run: bool) -> list[dict[str, Any]]:
    bases = {base["id"]: base for base in all_base_definitions(config)}
    count = int(config["base_evaluation"]["selected_base_count"])
    if dry_run:
        return list(config["bases"])[:count]
    selection_path = repo_path(config["base_evaluation"]["selection_file"])
    if not selection_path.is_file():
        raise RuntimeError("Base evaluation selection is missing; run --stage evaluate-bases first")
    ids = read_json(selection_path).get("selected_base_ids", [])
    if len(ids) != count or any(base_id not in bases for base_id in ids):
        raise RuntimeError("Base selection is invalid or stale")
    return [bases[base_id] for base_id in ids]


def target_assignments(config: dict[str, Any], dry_run: bool) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    bases = selected_bases(config, dry_run)
    return [(target, bases[index % len(bases)]) for index, target in enumerate(config["targets"])]


def run_targets(
    config: dict[str, Any], python: str, dry_run: bool, force: bool,
    split: str | None, wandb_mode: str, scalar_obs: bool = False,
) -> None:
    for target, base in target_assignments(config, dry_run):
        if split and target["split"] != split:
            continue
        source = repo_path(base["model_path"])
        output = target_candidate_path(target, base["id"])
        marker = marker_for(output)
        if output.exists() and marker.exists() and not force:
            print(f"Reusing completed target: {relative(output)}")
            continue
        if not dry_run and not marker_for(source).exists():
            raise RuntimeError(f"Base is not marked complete: {relative(source)}")
        if output.exists() and not force:
            raise RuntimeError(f"Incomplete target exists; inspect it or rerun with --force: {relative(output)}")
        command = finetune_command(config, target, base, python, scalar_obs)
        if dry_run:
            print(f"cp {shlex.quote(relative(source))} {shlex.quote(relative(output))}")
            print_command(command)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        if force:
            marker.unlink(missing_ok=True)
        shutil.copy2(source, output)
        run_command(command, wandb_mode)
        marker.touch()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opponent_entry(
    target: dict[str, Any], base: dict[str, Any], destination: Path
) -> dict[str, Any]:
    return {
        "label": f"ppo_v6_{base['id']}_{target['deck_id']}",
        "deck_id": target["deck_id"],
        "archetype": target["archetype"],
        "policy_family": base["id"],
        "policy_version": "v6",
        "action_space_size": 66,
        "model_path": relative(destination),
        "deck_path": relative(target_deck_path(target)),
        "sha256": sha256(destination),
    }


def freeze(config: dict[str, Any], dry_run: bool) -> None:
    generated = ROOT / "decks" / "generated" / "opponent_factory_v6"
    entries: dict[str, list[dict[str, Any]]] = {split: [] for split in VALID_SPLITS}
    for target, base in target_assignments(config, dry_run):
        source = target_candidate_path(target, base["id"])
        destination = frozen_model_path(target, base["id"])
        if dry_run:
            print(f"freeze {relative(source)} -> {relative(destination)}")
            continue
        if not source.is_file() or not marker_for(source).is_file():
            raise RuntimeError(f"Target is not complete and cannot be frozen: {relative(source)}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != sha256(source):
            raise RuntimeError(f"Frozen destination already differs: {relative(destination)}")
        if not destination.exists():
            shutil.copy2(source, destination)
        entries[target["split"]].append(opponent_entry(target, base, destination))

    if dry_run:
        print(f"would write staged manifests under {relative(generated)}")
        return
    generated.mkdir(parents=True, exist_ok=True)
    payloads = {
        "training_pool_v6.json": [
            {"label": row["label"], "deck": row["deck_path"], "model": row["model_path"], "weight": 1.0}
            for row in entries["training"]
        ],
        "validation_opponents_v6.json": {
            "version": 6,
            "purpose": "Repeatable V6 model selection only; never train against these exact opponents.",
            "opponents": entries["validation"],
        },
        "final_holdout_opponents_v6.json": {
            "version": 6,
            "purpose": "Untouched final V6 evaluation only; do not inspect during model selection.",
            "opponents": entries["holdout"],
        },
    }
    for name, payload in payloads.items():
        (generated / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote staged V6 manifests under {relative(generated)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument(
        "--stage", choices=("bases", "evaluate-bases", "targets", "freeze", "all"), default="all"
    )
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), help="Limit target training to one split")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Explicitly replace training outputs")
    parser.add_argument("--scalar-obs", action="store_true", help="Train with flat vector observation space (V5b hybrid)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(repo_path(args.config))
    validate_config(config)
    validate_static_inputs(config)
    if args.stage in {"bases", "all"}:
        run_bases(config, args.python, args.dry_run, args.force, args.wandb_mode, scalar_obs=args.scalar_obs)
    if args.stage in {"evaluate-bases", "all"}:
        evaluate_bases(config, args.python, args.dry_run, args.wandb_mode, scalar_obs=args.scalar_obs)
    if args.stage in {"targets", "all"}:
        run_targets(config, args.python, args.dry_run, args.force, args.split, args.wandb_mode, scalar_obs=args.scalar_obs)
    if args.stage in {"freeze", "all"}:
        freeze(config, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
