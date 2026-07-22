#!/usr/bin/env python3
"""Measure repeated card encoding against one lookup table per forward pass."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.custom_policy import PokemonTCGFeatureExtractor
from src.env.env_wrapper import V6_ACTION_SPACE_SIZE, PokemonTCGEnv


def observation_batch(observation_space, batch_size: int) -> dict[str, torch.Tensor]:
    sample = observation_space.sample()
    sample["action_mask"][:] = 0
    sample["action_mask"][:8] = 1
    result = {}
    for key, value in sample.items():
        tensor = torch.as_tensor(np.asarray(value)).unsqueeze(0)
        repeats = (batch_size,) + (1,) * (tensor.ndim - 1)
        result[key] = tensor.repeat(repeats)
    return result


def inference_times(
    model: PokemonTCGFeatureExtractor,
    observations: dict[str, torch.Tensor],
    warmup: int,
    repeats: int,
) -> list[float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(observations)
        times = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(observations)
            times.append(time.perf_counter() - started)
    return times


def training_times(
    model: PokemonTCGFeatureExtractor,
    observations: dict[str, torch.Tensor],
    warmup: int,
    repeats: int,
) -> list[float]:
    model.train()

    def step() -> None:
        model.zero_grad(set_to_none=True)
        features = model(observations)
        features.square().mean().backward()

    for _ in range(warmup):
        step()
    times = []
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        features = model(observations)
        features.square().mean().backward()
        times.append(time.perf_counter() - started)
    return times


def timing_summary(baseline: list[float], card_table: list[float], batch_size: int) -> dict:
    baseline_median = statistics.median(baseline)
    table_median = statistics.median(card_table)
    return {
        "batch_size": batch_size,
        "baseline_ms": baseline_median * 1000.0,
        "card_table_ms": table_median * 1000.0,
        "speedup": baseline_median / table_median,
        "baseline_samples_per_second": batch_size / baseline_median,
        "card_table_samples_per_second": batch_size / table_median,
    }


def verify_equivalence(
    baseline: PokemonTCGFeatureExtractor,
    card_table: PokemonTCGFeatureExtractor,
    observation_space,
) -> dict[str, float]:
    observations = observation_batch(observation_space, 2)
    baseline.zero_grad(set_to_none=True)
    card_table.zero_grad(set_to_none=True)
    baseline_features = baseline(observations)
    table_features = card_table(observations)
    if not torch.allclose(baseline_features, table_features, rtol=1e-5, atol=1e-6):
        difference = (baseline_features - table_features).abs().max().item()
        raise RuntimeError(f"Card-table output mismatch: max_abs_difference={difference}")

    baseline_features.square().mean().backward()
    table_features.square().mean().backward()
    max_gradient_difference = 0.0
    table_parameters = dict(card_table.named_parameters())
    for name, parameter in baseline.named_parameters():
        other = table_parameters[name]
        if parameter.grad is None or other.grad is None:
            if parameter.grad is not None or other.grad is not None:
                raise RuntimeError(f"Gradient presence mismatch for {name}")
            continue
        difference = (parameter.grad - other.grad).abs().max().item()
        max_gradient_difference = max(max_gradient_difference, difference)
        if not torch.allclose(parameter.grad, other.grad, rtol=2e-4, atol=2e-6):
            raise RuntimeError(
                f"Card-table gradient mismatch for {name}: max_abs_difference={difference}"
            )
    return {
        "max_output_difference": (baseline_features - table_features).abs().max().item(),
        "max_gradient_difference": max_gradient_difference,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("full", "balanced", "compact", "compact_no_legacy"),
        default="compact",
    )
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rollout-repeats", type=int, default=10)
    parser.add_argument("--train-repeats", type=int, default=3)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Set PyTorch CPU threads; 0 preserves the current runtime default.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    positive = (
        args.rollout_batch_size,
        args.train_batch_size,
        args.rollout_repeats,
        args.train_repeats,
    )
    if any(value <= 0 for value in positive) or args.warmup < 0 or args.torch_threads < 0:
        raise ValueError("Batch sizes/repeats must be positive; warmup/threads cannot be negative")
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    torch.manual_seed(20260713)
    env = PokemonTCGEnv([6] * 60, [5] * 60, action_space_size=V6_ACTION_SPACE_SIZE)
    baseline = PokemonTCGFeatureExtractor(
        env.observation_space,
        feature_variant=args.variant,
        use_card_table=False,
    )
    card_table = copy.deepcopy(baseline)
    card_table.use_card_table = True

    equivalence = verify_equivalence(baseline, card_table, env.observation_space)
    rollout_observations = observation_batch(env.observation_space, args.rollout_batch_size)
    train_observations = observation_batch(env.observation_space, args.train_batch_size)
    results = {
        "variant": args.variant,
        "torch_threads": torch.get_num_threads(),
        "equivalence": equivalence,
        "rollout_forward": timing_summary(
            inference_times(baseline, rollout_observations, args.warmup, args.rollout_repeats),
            inference_times(card_table, rollout_observations, args.warmup, args.rollout_repeats),
            args.rollout_batch_size,
        ),
        "ppo_minibatch_forward_backward": timing_summary(
            training_times(baseline, train_observations, args.warmup, args.train_repeats),
            training_times(card_table, train_observations, args.warmup, args.train_repeats),
            args.train_batch_size,
        ),
    }

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Variant: {args.variant} | PyTorch CPU threads: {results['torch_threads']}")
        print(
            "Equivalence: "
            f"output max diff={equivalence['max_output_difference']:.3g}, "
            f"gradient max diff={equivalence['max_gradient_difference']:.3g}"
        )
        print("\nPath                              Baseline     Card table   Speedup")
        for label, key in (
            ("Rollout forward", "rollout_forward"),
            ("PPO minibatch forward+backward", "ppo_minibatch_forward_backward"),
        ):
            row = results[key]
            print(
                f"{label:<34} {row['baseline_ms']:>8.2f} ms "
                f"{row['card_table_ms']:>10.2f} ms {row['speedup']:>8.2f}x"
            )
        print("\nThis isolates the feature extractor; it does not include game simulation or IPC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
