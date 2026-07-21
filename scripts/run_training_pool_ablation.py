#!/usr/bin/env python3
"""Compare static and PFSP-lite sampling from one frozen training parent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONTROLLED_OPTIONS = {
    "--deck",
    "--model-name",
    "--timesteps",
    "--seed",
    "--opp-pool",
    "--continue-existing",
    "--overwrite",
    "--pfsp-lite",
    "--pfsp-segment-episodes",
    "--pfsp-prior-games",
    "--pfsp-random-fraction",
    "--pfsp-max-probability",
    "--policy-version",
    "--feature-variant",
    "--card-table",
    "--no-card-table",
}
COMPACT_V6_OPTIONS = (
    "--policy-version",
    "v6",
    "--feature-variant",
    "compact",
    "--card-table",
)
PFSP_OPTIONS = (
    "--pfsp-lite",
    "--pfsp-segment-episodes",
    "200",
    "--pfsp-prior-games",
    "4.0",
    "--pfsp-random-fraction",
    "0.20",
    "--pfsp-max-probability",
    "0.35",
)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def model_outputs(parent: Path, output_dir: Path, steps: int) -> tuple[Path, Path]:
    stem = parent.stem
    budget_label = f"{steps // 1_000_000}m" if steps % 1_000_000 == 0 else str(steps)
    return (
        output_dir / f"{stem}_pool_static_{budget_label}.zip",
        output_dir / f"{stem}_pool_pfsp_{budget_label}.zip",
    )


def validate_extra_train_args(extra: list[str]) -> None:
    for token in extra:
        option = token.split("=", 1)[0]
        if option in CONTROLLED_OPTIONS:
            raise ValueError(
                f"{option} is controlled by the ablation runner and cannot appear after --"
            )


def build_train_command(
    args: argparse.Namespace,
    output: Path,
    *,
    pfsp: bool,
) -> list[str]:
    command = [
        args.python,
        "src/train.py",
        "--deck",
        relative(args.deck),
        "--model-name",
        relative(output),
        "--continue-existing",
        "--timesteps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--opp-pool",
        relative(args.opp_pool),
        *COMPACT_V6_OPTIONS,
        *args.train_args,
    ]
    if pfsp:
        command.extend(PFSP_OPTIONS)
    else:
        command.append("--no-pfsp-lite")
    return command


def print_command(command: list[str]) -> None:
    print(" ".join(shlex.quote(part) for part in command), flush=True)


def run(command: list[str], args: argparse.Namespace, arm: str) -> None:
    print_command(command)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["WANDB_MODE"] = args.wandb_mode
    environment["WANDB_RUN_GROUP"] = f"training_pool_ablation_{args.parent.stem}"
    environment["WANDB_NAME"] = f"training_pool_{arm}_{args.parent.stem}"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def experiment_for(model: Path) -> dict[str, Any]:
    path = ROOT / "models" / "experiments" / f"{model.stem}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def summary_for(results: dict[str, Any], model: Path) -> dict[str, Any]:
    for summary in results.get("summary", []):
        if summary.get("candidate") == model.stem:
            return summary
    raise KeyError(f"Validation summary is missing {model.stem}")


def health_passed(experiment: dict[str, Any]) -> bool:
    return experiment.get("training_health", {}).get("gate", {}).get("passed") is True


def comparison_report(
    parent: Path,
    static_model: Path,
    pfsp_model: Path,
    results: dict[str, Any],
) -> dict[str, Any]:
    static_experiment = experiment_for(static_model)
    pfsp_experiment = experiment_for(pfsp_model)
    static = summary_for(results, static_model)
    pfsp = summary_for(results, pfsp_model)
    excluded_arguments = {
        "model_name",
        "pfsp_lite",
        "pfsp_segment_episodes",
        "pfsp_prior_games",
        "pfsp_random_fraction",
        "pfsp_max_probability",
    }
    static_arguments = {
        key: value
        for key, value in static_experiment.get("arguments", {}).items()
        if key not in excluded_arguments
    }
    pfsp_arguments = {
        key: value
        for key, value in pfsp_experiment.get("arguments", {}).items()
        if key not in excluded_arguments
    }
    static_health = health_passed(static_experiment)
    pfsp_health = health_passed(pfsp_experiment)
    evaluation_health = results.get("health_gate", {}).get("passed") is True
    no_crashes = static.get("crashes", 0) == 0 and pfsp.get("crashes", 0) == 0
    controlled_arguments_match = static_arguments == pfsp_arguments
    training_completed = (
        static_experiment.get("status") == "completed"
        and pfsp_experiment.get("status") == "completed"
    )
    return {
        "schema_version": 1,
        "controlled_variables": {
            "parent": relative(parent),
            "seed": static_experiment.get("arguments", {}).get("seed"),
            "opponent_pool": static_experiment.get("arguments", {}).get("opp_pool"),
            "steps": static_experiment.get("arguments", {}).get("timesteps"),
        },
        "static": {"model": relative(static_model), **static},
        "pfsp": {
            "model": relative(pfsp_model),
            "settings": pfsp_experiment.get("pfsp_lite"),
            **pfsp,
        },
        "gates": {
            "controlled_arguments_match": controlled_arguments_match,
            "training_completed": training_completed,
            "static_training_health": static_health,
            "pfsp_training_health": pfsp_health,
            "evaluation_health": evaluation_health,
            "no_crashes": no_crashes,
        },
        "ready_for_model_selection": (
            controlled_arguments_match
            and training_completed
            and static_health
            and pfsp_health
            and evaluation_health
            and no_crashes
        ),
    }


def resolve(path: Path) -> Path:
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Defaults define the Deck 38, 1M-step static-vs-PFSP experiment. "
            "Pass architecture or optimizer settings after --."
        ),
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("models/training_v6/ppo_v6_deck_bank_38_compact_a.zip"),
    )
    parser.add_argument("--deck", type=Path, default=Path("decks/deck_bank/bank_38.csv"))
    parser.add_argument(
        "--opp-pool",
        type=Path,
        default=Path("experiments/2026-07/deck38_static_pfsp_pool_20260718.json"),
    )
    parser.add_argument(
        "--validation-file", type=Path, default=Path("decks/validation_opponents.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/training_pool_ablation/deck38_v6_compact_20260718")
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("logs/training_pool_ablation/deck38_v6_compact_20260718")
    )
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.train_args[:1] == ["--"]:
        args.train_args = args.train_args[1:]
    for name in ("parent", "deck", "opp_pool", "validation_file", "output_dir", "run_dir"):
        setattr(args, name, resolve(getattr(args, name)))
    return args


def check_reserved_opponents(args: argparse.Namespace) -> None:
    from scripts.check_holdout_safe import check_paths

    for reserved in (
        ROOT / "decks/holdout_opponents.json",
        ROOT / "decks/validation_opponents.json",
    ):
        if reserved.is_file():
            check_paths(str(reserved), [str(args.deck)], [], [str(args.opp_pool)])


def validate_pool_files(pool_path: Path) -> None:
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Opponent pool must be a non-empty JSON list: {relative(pool_path)}")
    missing = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid opponent-pool entry {index}: expected an object")
        for key in ("deck", "model"):
            raw_path = entry.get(key)
            if key == "model" and str(raw_path or "").startswith(("rule_based:", "heuristic")):
                continue
            path = resolve(Path(str(raw_path))) if raw_path else None
            if path is None or not path.is_file():
                missing.append(f"entry {index} {key}: {raw_path}")
    if missing:
        raise FileNotFoundError("Missing opponent-pool inputs: " + "; ".join(missing))


def main() -> int:
    args = parse_args()
    validate_extra_train_args(args.train_args)
    if args.steps <= 0 or args.games <= 0:
        raise ValueError("steps and games must be positive")
    required = (args.parent, args.deck, args.opp_pool, args.validation_file)
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ablation inputs: " + ", ".join(missing))
    validate_pool_files(args.opp_pool)
    check_reserved_opponents(args)

    static_model, pfsp_model = model_outputs(args.parent, args.output_dir, args.steps)
    commands = [
        build_train_command(args, static_model, pfsp=False),
        build_train_command(args, pfsp_model, pfsp=True),
    ]
    results_file = args.run_dir / "validation_results.json"
    evaluation = [
        args.python,
        "scripts/evaluate_submission.py",
        "--holdout-file",
        relative(args.validation_file),
        "--games",
        str(args.games),
        "--results-file",
        relative(results_file),
        "--candidate",
        relative(static_model),
        "--candidate",
        relative(pfsp_model),
    ]
    if args.dry_run:
        for command in (*commands, evaluation):
            print_command(command)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for arm, output, command in zip(("static", "pfsp"), (static_model, pfsp_model), commands):
        marker = output.with_suffix(".complete")
        if marker.exists():
            raise RuntimeError(f"Refusing to reuse marked ablation arm: {relative(marker)}")
        if output.exists():
            raise RuntimeError(f"Incomplete ablation output exists: {relative(output)}")
        shutil.copy2(args.parent, output)
        run(command, args, arm)

    print_command(evaluation)
    subprocess.run(evaluation, cwd=ROOT, check=True)
    results = json.loads(results_file.read_text(encoding="utf-8"))
    report = comparison_report(args.parent, static_model, pfsp_model, results)
    report_file = args.run_dir / "comparison_report.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["ready_for_model_selection"]:
        raise RuntimeError("Training provenance, health gate, or validation failed")

    static_model.with_suffix(".complete").touch()
    pfsp_model.with_suffix(".complete").touch()
    print(f"PFSP comparison is ready: {relative(report_file)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
