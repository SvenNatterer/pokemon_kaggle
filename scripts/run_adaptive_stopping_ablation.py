#!/usr/bin/env python3
"""Run a fixed-budget/adaptive-stop comparison from one frozen parent."""

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
    "--adaptive-stop",
    "--adaptive-kl-threshold",
    "--adaptive-entropy-delta",
    "--adaptive-entropy-trend",
    "--adaptive-min-steps",
    "--adaptive-patience",
    "--pfsp-lite",
    "--no-pfsp-lite",
    "--pfsp-segment-episodes",
    "--pfsp-prior-games",
    "--pfsp-random-fraction",
    "--pfsp-max-probability",
}

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


def model_outputs(parent: Path, output_dir: Path) -> tuple[Path, Path]:
    stem = parent.stem
    return (
        output_dir / f"{stem}_adaptive_ablation_fixed.zip",
        output_dir / f"{stem}_adaptive_ablation_enabled.zip",
    )


def validate_extra_train_args(extra: list[str]) -> None:
    for token in extra:
        option = token.split("=", 1)[0]
        if option in CONTROLLED_OPTIONS:
            raise ValueError(
                f"{option} is controlled by the ablation runner and cannot appear after --"
            )


def resolve(path: Path) -> Path:
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


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


def check_reserved_opponents(args: argparse.Namespace) -> None:
    from scripts.check_holdout_safe import check_paths

    for reserved in (
        ROOT / "decks/holdout_opponents.json",
        ROOT / "decks/validation_opponents.json",
    ):
        if reserved.is_file():
            check_paths(str(reserved), [str(args.deck)], [], [str(args.opp_pool)])


def build_train_command(
    args: argparse.Namespace,
    output: Path,
    *,
    adaptive: bool,
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
        str(args.max_steps),
        "--seed",
        str(args.seed),
        "--opp-pool",
        relative(args.opp_pool),
        *PFSP_OPTIONS,
        *args.train_args,
    ]
    if adaptive:
        command.extend(
            [
                "--adaptive-stop",
                "--adaptive-kl-threshold",
                str(args.kl_threshold),
                "--adaptive-entropy-trend",
                str(args.entropy_trend),
                "--adaptive-min-steps",
                str(args.min_steps),
                "--adaptive-patience",
                str(args.patience),
            ]
        )
    return command


def print_command(command: list[str]) -> None:
    print(" ".join(shlex.quote(part) for part in command), flush=True)


def run(command: list[str], args: argparse.Namespace, arm: str) -> None:
    print_command(command)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["WANDB_MODE"] = args.wandb_mode
    environment["WANDB_RUN_GROUP"] = f"adaptive_stopping_ablation_{args.parent.stem}"
    environment["WANDB_NAME"] = f"adaptive_stopping_{arm}_{args.parent.stem}"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def experiment_for(model: Path) -> dict[str, Any]:
    path = ROOT / "models" / "experiments" / f"{model.stem}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def summary_for(results: dict[str, Any], model: Path) -> dict[str, Any]:
    for summary in results.get("summary", []):
        if summary.get("candidate") == model.stem:
            return summary
    raise KeyError(f"Validation summary is missing {model.stem}")


def comparison_report(
    parent: Path,
    fixed_model: Path,
    adaptive_model: Path,
    results: dict[str, Any],
    *,
    strength_tolerance: float,
    perspective_tolerance: float,
) -> dict[str, Any]:
    fixed_experiment = experiment_for(fixed_model)
    adaptive_experiment = experiment_for(adaptive_model)
    fixed = summary_for(results, fixed_model)
    adaptive = summary_for(results, adaptive_model)
    fixed_steps = int(fixed_experiment.get("actual_run_steps", 0))
    adaptive_steps = int(adaptive_experiment.get("actual_run_steps", 0))

    excluded_arguments = {
        "model_name",
        "adaptive_stop",
        "adaptive_kl_threshold",
        "adaptive_entropy_trend",
        "adaptive_min_steps",
        "adaptive_patience",
    }
    fixed_arguments = {
        key: value
        for key, value in fixed_experiment.get("arguments", {}).items()
        if key not in excluded_arguments
    }
    adaptive_arguments = {
        key: value
        for key, value in adaptive_experiment.get("arguments", {}).items()
        if key not in excluded_arguments
    }
    controlled_arguments_match = fixed_arguments == adaptive_arguments
    training_completed = (
        fixed_experiment.get("status") == "completed"
        and adaptive_experiment.get("status") == "completed"
    )
    no_crashes = fixed.get("crashes", 0) == 0 and adaptive.get("crashes", 0) == 0
    saves_compute = adaptive_steps < fixed_steps
    strength_not_reduced = all(
        adaptive[key] + strength_tolerance >= fixed[key]
        for key in ("wilson95_score_lb", "worst_score_rate", "score_rate")
    )
    perspective_not_increased = (
        adaptive["perspective_score_gap"]
        <= fixed["perspective_score_gap"] + perspective_tolerance
    )
    return {
        "schema_version": 1,
        "controlled_variables": {
            "parent": relative(parent),
            "seed": fixed_experiment.get("arguments", {}).get("seed"),
            "opponent_pool": fixed_experiment.get("arguments", {}).get("opp_pool"),
            "maximum_steps": fixed_experiment.get("arguments", {}).get("timesteps"),
        },
        "fixed": {"model": relative(fixed_model), "run_steps": fixed_steps, **fixed},
        "adaptive": {
            "model": relative(adaptive_model),
            "run_steps": adaptive_steps,
            "stop_reason": adaptive_experiment.get("stop_reason"),
            **adaptive,
        },
        "gates": {
            "controlled_arguments_match": controlled_arguments_match,
            "training_completed": training_completed,
            "no_crashes": no_crashes,
            "saves_compute": saves_compute,
            "strength_not_reduced": strength_not_reduced,
            "perspective_bias_not_increased": perspective_not_increased,
            "strength_tolerance": strength_tolerance,
            "perspective_tolerance": perspective_tolerance,
        },
        "adopt_adaptive_stopping": (
            controlled_arguments_match
            and training_completed
            and no_crashes
            and saves_compute
            and strength_not_reduced
            and perspective_not_increased
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Pass architecture and optimizer settings after --. Example: "
            "-- --policy-version v6 --feature-variant compact --card-table "
            "--belief-actor --rotate-perspective"
        ),
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path(
            "models/training_v6/ppo_v6_deck_bank_54_compact_a_newpool_pfsp_2m_20260718.zip"
        ),
    )
    parser.add_argument("--deck", type=Path, default=Path("decks/deck_bank/bank_54.csv"))
    parser.add_argument(
        "--opp-pool",
        type=Path,
        default=Path("decks/opponent_factory_v6_development_pool.json"),
    )
    parser.add_argument("--validation-file", type=Path, default=Path("decks/validation_opponents.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/adaptive_stopping_ablation"))
    parser.add_argument("--run-dir", type=Path, default=Path("logs/adaptive_stopping_ablation"))
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--kl-threshold", type=float, default=0.001)
    parser.add_argument(
        "--entropy-trend",
        "--entropy-delta",
        dest="entropy_trend",
        type=float,
        default=0.002,
    )
    parser.add_argument("--min-steps", type=int, default=250_000)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--strength-tolerance", type=float, default=0.0)
    parser.add_argument("--perspective-tolerance", type=float, default=0.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.train_args[:1] == ["--"]:
        args.train_args = args.train_args[1:]
    args.parent = (ROOT / args.parent).resolve() if not args.parent.is_absolute() else args.parent.resolve()
    args.deck = (ROOT / args.deck).resolve() if not args.deck.is_absolute() else args.deck.resolve()
    args.opp_pool = (ROOT / args.opp_pool).resolve() if not args.opp_pool.is_absolute() else args.opp_pool.resolve()
    args.validation_file = (
        (ROOT / args.validation_file).resolve()
        if not args.validation_file.is_absolute()
        else args.validation_file.resolve()
    )
    args.output_dir = (
        (ROOT / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    args.run_dir = (
        (ROOT / args.run_dir).resolve()
        if not args.run_dir.is_absolute()
        else args.run_dir.resolve()
    )
    return args


def main() -> int:
    args = parse_args()
    validate_extra_train_args(args.train_args)
    if args.max_steps <= 0 or args.games <= 0 or args.patience < 2:
        raise ValueError("max-steps and games must be positive; patience must be at least 2")
    if args.min_steps < 0 or args.min_steps >= args.max_steps:
        raise ValueError("min-steps must be non-negative and below max-steps")
    required = (args.parent, args.deck, args.opp_pool, args.validation_file)
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ablation inputs: " + ", ".join(missing))
    validate_pool_files(args.opp_pool)
    check_reserved_opponents(args)

    fixed_model, adaptive_model = model_outputs(args.parent, args.output_dir)
    commands = [
        build_train_command(args, fixed_model, adaptive=False),
        build_train_command(args, adaptive_model, adaptive=True),
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
        relative(fixed_model),
        "--candidate",
        relative(adaptive_model),
    ]
    if args.dry_run:
        for command in (*commands, evaluation):
            print_command(command)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for arm, output, command in zip(("fixed", "adaptive"), (fixed_model, adaptive_model), commands):
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
    report = comparison_report(
        args.parent,
        fixed_model,
        adaptive_model,
        results,
        strength_tolerance=args.strength_tolerance,
        perspective_tolerance=args.perspective_tolerance,
    )
    report_file = args.run_dir / "adoption_decision.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not all(
        report["gates"][key]
        for key in ("controlled_arguments_match", "training_completed", "no_crashes")
    ):
        raise RuntimeError(
            "Training provenance or validation failed; outputs remain incomplete"
        )

    fixed_model.with_suffix(".complete").touch()
    adaptive_model.with_suffix(".complete").touch()
    decision = "ADOPT" if report["adopt_adaptive_stopping"] else "REJECT"
    print(f"Adaptive stopping decision: {decision}", flush=True)
    print(f"Report: {relative(report_file)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
