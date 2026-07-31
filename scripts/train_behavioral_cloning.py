#!/usr/bin/env python3
"""Pretrain the recurrent V6 PPO policy from an offline expert dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.env_wrapper import PokemonTCGEnv, V6_ACTION_SPACE_SIZE
from src.training.behavioral_cloning import (
    load_demonstration_dataset,
    train_behavioral_cloning,
)
from src.training.model_factory import build_fresh_custom_ppo, save_model_atomically
from src.data.demonstrations import read_deck, sha256_file
from src.league.experiment_registry import git_revision
from src.utils import atomic_write_json, resolve_deck_path


REQUIRED_RESERVED_MANIFEST_NAMES = {
    "holdout_opponents.json",
    "validation_opponents.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Warm-start the recurrent V6 PPO policy with behavioural cloning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="YAML behavioural-cloning experiment config")
    parser.add_argument("--dataset", help="Directory containing manifest.json")
    parser.add_argument("--deck", help="Learner deck CSV")
    parser.add_argument("--opp-deck", help="Opponent deck CSV")
    parser.add_argument("--output-model", help="Output CustomPPO .zip checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--aux-coef", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260721)

    # Architecture and saved PPO settings must remain compatible with the
    # downstream `src/training/train.py --base-model ...` invocation.
    parser.add_argument("--policy-version", choices=("v6",), default="v6")
    parser.add_argument(
        "--feature-variant",
        choices=("compact", "strategic_vector_v1", "strategic_vector_v2", "strategic_vector_v3"),
        default="compact",
    )
    parser.add_argument("--scalar-obs", action="store_true")
    parser.add_argument("--features-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--entity-relation-mode",
        choices=("baseline", "masked", "relational", "two_step", "python_object_relational"),
        default="baseline",
    )
    parser.add_argument("--no-card-table", dest="card_table", action="store_false")
    parser.add_argument("--no-belief-actor", dest="belief_actor", action="store_false")
    parser.add_argument("--belief-dim", type=int, default=64)
    parser.add_argument("--no-belief-detach", dest="belief_detach", action="store_false")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--ppo-epochs", dest="n_epochs", type=int, default=2)
    parser.add_argument("--ent-coef", type=float, default=0.008)
    parser.add_argument("--clip-range", type=float, default=0.12)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--distill-coef", type=float, default=0.1)
    parser.add_argument("--value-distill-coef", type=float, default=0.0)
    parser.set_defaults(card_table=True, belief_actor=True, belief_detach=True)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Apply YAML defaults first so explicit CLI flags always win."""

    argv = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    preliminary, _ = config_parser.parse_known_args(argv)
    parser = build_parser()

    if preliminary.config:
        config_path = Path(preliminary.config)
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError(f"BC config must contain a YAML mapping: {config_path}")
        aliases = {"lr": "learning_rate", "ppo_epochs": "n_epochs"}
        valid_destinations = {action.dest for action in parser._actions}
        defaults: dict[str, Any] = {}
        for key, value in config.items():
            destination = aliases.get(str(key), str(key))
            if destination not in valid_destinations:
                raise ValueError(f"Unknown behavioural-cloning config key: {key}")
            defaults[destination] = value
        parser.set_defaults(**defaults)

    args = parser.parse_args(argv)
    missing = [
        flag
        for flag, value in (
            ("dataset", args.dataset),
            ("deck", args.deck),
            ("opp_deck", args.opp_deck),
            ("output_model", args.output_model),
        )
        if not value
    ]
    if missing:
        parser.error(
            "the following values are required via CLI or --config: " + ", ".join(missing)
        )
    if args.policy_version != "v6":
        raise ValueError("Behavioural cloning currently supports only V6 policies")
    scalar_variant = args.feature_variant in {"strategic_vector_v1", "strategic_vector_v2", "strategic_vector_v3"}
    if args.scalar_obs != scalar_variant:
        raise ValueError(
            "scalar_obs and feature_variant must be paired: strategic_vector_v1/v2/v3 "
            "require scalar_obs, while compact V6 must not enable it"
        )
    if args.entity_relation_mode not in {
        "baseline", "masked", "relational", "two_step", "python_object_relational"
    }:
        raise ValueError(f"Unsupported entity_relation_mode: {args.entity_relation_mode}")
    return args


def _resolved_output(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.suffix == ".zip" else path.with_name(f"{path.name}.zip")


def validate_dataset_provenance(
    dataset,
    *,
    deck: str,
    opponent_deck: str,
    scalar_obs: bool,
    feature_variant: str,
) -> None:
    """Bind model-space construction to the canonical holdout-safe collection."""

    metadata = dataset.manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Dataset manifest is missing collection metadata")
    if metadata.get("collector") != "scripts/collect_lookahead_teacher.py":
        raise ValueError("Dataset was not produced by the canonical demonstration collector")
    if metadata.get("scalar_obs") is not scalar_obs:
        raise ValueError("Dataset scalar_obs setting does not match the requested policy")
    if metadata.get("feature_variant") != feature_variant:
        raise ValueError("Dataset feature_variant does not match the requested policy")

    expected_hashes = {
        "deck_sha256": sha256_file(resolve_deck_path(deck)),
        "opponent_deck_sha256": sha256_file(resolve_deck_path(opponent_deck)),
    }
    for field, actual_hash in expected_hashes.items():
        recorded_hash = metadata.get(field)
        if recorded_hash != actual_hash:
            raise ValueError(
                f"Dataset {field} does not match the deck supplied for model construction"
            )

    reserved = metadata.get("reserved_opponent_manifests")
    if not isinstance(reserved, list):
        raise ValueError("Dataset does not record reserved-opponent safety checks")
    recorded_names = {Path(value).name for value in reserved if isinstance(value, str)}
    missing = REQUIRED_RESERVED_MANIFEST_NAMES - recorded_names
    if missing:
        raise ValueError(
            "Dataset is missing required reserved-opponent checks: " + ", ".join(sorted(missing))
        )


def _source_config(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"BC config must contain a YAML mapping: {path}")
    return value


def build_sidecar_payload(
    *,
    args: argparse.Namespace,
    dataset,
    result,
    output_path: Path,
    checkpoint_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Build complete, JSON-serializable provenance for one BC checkpoint."""

    return {
        "schema_version": 1,
        "kind": "behavioural_cloning",
        "git_revision": git_revision(),
        "checkpoint": {
            "path": str(output_path),
            "sha256": checkpoint_sha256,
        },
        "resolved_arguments": dict(vars(args)),
        "source_config": _source_config(args.config),
        "dataset": {
            "root": str(dataset.root),
            "manifest_path": str(dataset.root / "manifest.json"),
            "manifest_sha256": manifest_sha256,
            "manifest": dataset.manifest,
        },
        "split": {
            "train_episode_ids": list(result.train_episode_ids),
            "validation_episode_ids": list(result.validation_episode_ids),
        },
        "training": {
            "best_epoch": result.best_epoch,
            "best_validation_nll": result.best_validation_nll,
            "train_history": [asdict(metrics) for metrics in result.train_metrics],
            "validation_history": [
                asdict(metrics) for metrics in result.validation_metrics
            ],
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # The shared model factory follows the PPO training argument names.
    args.lr = args.learning_rate

    output_path = _resolved_output(args.output_model).resolve()
    sidecar_path = output_path.with_suffix(".bc.json")
    existing_outputs = [path for path in (output_path, sidecar_path) if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing BC output: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = load_demonstration_dataset(args.dataset)
    validate_dataset_provenance(
        dataset,
        deck=args.deck,
        opponent_deck=args.opp_deck,
        scalar_obs=args.scalar_obs,
        feature_variant=args.feature_variant,
    )
    manifest_path = dataset.root / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    env = PokemonTCGEnv(
        my_deck=read_deck(args.deck),
        opponent_deck=read_deck(args.opp_deck),
        action_space_size=V6_ACTION_SPACE_SIZE,
        structured_v2=not args.scalar_obs,
        feature_variant=args.feature_variant,
        enable_lookahead_teacher=False,
        inference_guardrails=False,
    )
    try:
        model = build_fresh_custom_ppo(env, args)
        result = train_behavioral_cloning(
            model,
            dataset,
            epochs=args.epochs,
            sequence_length=args.sequence_length,
            learning_rate=args.learning_rate,
            aux_coef=args.aux_coef,
            validation_fraction=args.validation_fraction,
            patience=args.patience,
            seed=args.seed,
        )
        if sha256_file(manifest_path) != manifest_sha256:
            raise RuntimeError("Dataset manifest changed during behavioural-cloning training")
        save_model_atomically(model, output_path)
    finally:
        env.close()

    payload = build_sidecar_payload(
        args=args,
        dataset=dataset,
        result=result,
        output_path=output_path,
        checkpoint_sha256=sha256_file(output_path),
        manifest_sha256=manifest_sha256,
    )
    atomic_write_json(sidecar_path, payload)

    summary = {
        "output_model": str(output_path),
        "sidecar": str(sidecar_path),
        "dataset": str(dataset.root),
        "episodes": len(dataset.episodes),
        "transitions": dataset.transition_count,
        "best_epoch": result.best_epoch,
        "best_validation_nll": result.best_validation_nll,
        "train_episode_ids": result.train_episode_ids,
        "validation_episode_ids": result.validation_episode_ids,
        "final_train": asdict(result.train_metrics[-1]),
        "final_validation": asdict(result.validation_metrics[-1]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
