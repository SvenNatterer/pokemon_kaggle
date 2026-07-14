#!/usr/bin/env python3
"""Compare Full V6 with balanced and compact feature extractors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FULL_MODEL = ROOT / "models" / "foundation" / "ppo_v6_deck_bank_54_base_a.zip"
FULL_MARKER = FULL_MODEL.with_suffix(".complete")
BASE_B_MODEL = ROOT / "models" / "foundation" / "ppo_v6_deck_bank_55_base_b.zip"
BASE_B_MARKER = BASE_B_MODEL.with_suffix(".complete")
DECK = "decks/deck_bank/bank_54.csv"
POOL = "decks/opponent_factory_v6_development_pool.json"
VALIDATION = "decks/validation_opponents.json"
OUTPUT_DIR = ROOT / "models" / "architecture_ablation"
LOG_DIR = ROOT / "logs" / "v6_architecture_ablation"
VARIANTS = ("compact", "compact_no_legacy", "balanced")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def output_for(variant: str) -> Path:
    return OUTPUT_DIR / f"ppo_v6_deck_bank_54_{variant}.zip"


def marker_for(path: Path) -> Path:
    return path.with_suffix(".complete")


def train_command(variant: str, steps: int, python: str) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported ablation variant: {variant}")
    command = [
        python, "src/train.py",
        "--deck", DECK,
        "--model-name", relative(output_for(variant)),
        "--timesteps", str(steps),
        "--seed", "20260721",
        "--policy-version", "v6",
        "--feature-variant", variant,
        "--aux-coef", "0.1",
        "--opp-pool", POOL,
        "--num-envs", "8",
        "--n-steps", "2048",
        "--batch-size", "1024",
        "--n-epochs", "2",
        "--lr", "0.0001",
        "--ent-coef", "0.008",
        "--clip-range", "0.12",
        "--target-kl", "0.03",
        "--belief-actor",
        "--belief-dim", "64",
        "--rotate-perspective",
    ]
    if variant == "balanced":
        command.append("--card-table")
    return command


def evaluation_command(games: int, python: str) -> list[str]:
    command = [
        python, "scripts/evaluate_submission.py",
        "--holdout-file", VALIDATION,
        "--games", str(games),
        "--results-file", relative(LOG_DIR / "results.json"),
        "--best-candidate-file", relative(LOG_DIR / "selection.json"),
        "--candidate", relative(FULL_MODEL),
    ]
    for variant in VARIANTS:
        command.extend(["--candidate", relative(output_for(variant))])
    return command


def print_command(command: list[str]) -> None:
    print(" ".join(shlex.quote(item) for item in command), flush=True)


def wait_for_full_base(poll_seconds: int) -> None:
    print(f"Waiting for Full V6 Base A: {relative(FULL_MARKER)}", flush=True)
    while not (FULL_MODEL.is_file() and FULL_MARKER.is_file()):
        time.sleep(poll_seconds)
    print("Full V6 Base A is complete.", flush=True)


def base_b_is_complete() -> bool:
    return BASE_B_MODEL.is_file() and BASE_B_MARKER.is_file()


def wait_for_base_b(poll_seconds: int) -> None:
    print(f"Waiting for Full V6 Base B: {relative(BASE_B_MARKER)}", flush=True)
    while not base_b_is_complete():
        time.sleep(poll_seconds)
    print("Full V6 Base B is complete.", flush=True)


def matching_pids(pattern: str) -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", pattern], capture_output=True, text=True, check=False
    )
    pids = []
    for value in result.stdout.split():
        try:
            pid = int(value)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def stop_full_factory(screen_name: str) -> None:
    """Stop the obsolete full-architecture queue before Base B consumes compute."""
    for pid in matching_pids("scripts/run_opponent_factory.py --stage bases"):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # A Base-B subprocess can be spawned in the small marker-detection race.
    for base_id, deck_id in (("base_b", "55"), ("base_c", "56"), ("base_d", "65")):
        pattern = f"ppo_v6_deck_bank_{deck_id}_{base_id}.zip"
        for pid in matching_pids(pattern):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    subprocess.run(["screen", "-S", screen_name, "-X", "quit"], check=False)
    print("Stopped the queued Full-V6 Base B-D run.", flush=True)


def run(command: list[str], wandb_mode: str, variant: str | None = None) -> None:
    print_command(command)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["WANDB_MODE"] = wandb_mode
    if variant:
        environment["WANDB_NAME"] = f"V6_{variant}_D54"
        environment["WANDB_RUN_GROUP"] = "v6_architecture_ablation"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def parameter_profile(model_path: Path, python: str) -> dict[str, Any]:
    code = (
        "import json; from src.custom_ppo import CustomPPO; "
        f"m=CustomPPO.load({str(model_path)!r},device='cpu'); "
        "f=m.policy.features_extractor; "
        "print(json.dumps({'actions':int(m.action_space.n),'feature_variant':getattr(f,'feature_variant','full'),"
        "'policy_parameters':sum(p.numel() for p in m.policy.parameters()),"
        "'extractor_parameters':sum(p.numel() for p in f.parameters()),"
        "'card_table':bool(getattr(f,'use_card_table',False)),"
        "'combined_features':f.net[0].in_features}))"
    )
    result = subprocess.run(
        [python, "-c", code], cwd=ROOT, env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout.splitlines()[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wait-for-base-a", action="store_true")
    parser.add_argument(
        "--wait-for-base-b",
        action="store_true",
        help="Wait for Base B to finish before stopping the full-architecture factory.",
    )
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--stop-factory-screen", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.games <= 0 or args.poll_seconds <= 0:
        raise ValueError("steps, games, and poll-seconds must be positive")
    required = [ROOT / DECK, ROOT / POOL, ROOT / VALIDATION]
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ablation inputs: " + ", ".join(missing))

    commands = [train_command(variant, args.steps, args.python) for variant in VARIANTS]
    commands.append(evaluation_command(args.games, args.python))
    if args.dry_run:
        for command in commands:
            print_command(command)
        return 0

    if args.wait_for_base_b:
        if not args.stop_factory_screen:
            raise ValueError("--wait-for-base-b requires --stop-factory-screen")
        wait_for_base_b(args.poll_seconds)
    elif args.wait_for_base_a:
        wait_for_full_base(args.poll_seconds)
    elif not (FULL_MODEL.is_file() and FULL_MARKER.is_file()):
        raise RuntimeError("Full V6 Base A is not complete; use --wait-for-base-a")

    if args.stop_factory_screen:
        stop_full_factory(args.stop_factory_screen)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        output = output_for(variant)
        marker = marker_for(output)
        if output.is_file() and marker.is_file():
            print(f"Reusing completed {variant}: {relative(output)}", flush=True)
            continue
        if output.exists():
            raise RuntimeError(f"Incomplete ablation output exists: {relative(output)}")
        run(train_command(variant, args.steps, args.python), args.wandb_mode, variant)
        marker.touch()

    profiles = {
        "full": parameter_profile(FULL_MODEL, args.python),
        **{variant: parameter_profile(output_for(variant), args.python) for variant in VARIANTS},
    }
    (LOG_DIR / "parameter_profiles.json").write_text(
        json.dumps(profiles, indent=2) + "\n", encoding="utf-8"
    )
    run(evaluation_command(args.games, args.python), args.wandb_mode)
    print(f"Ablation complete: {relative(LOG_DIR / 'results.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
