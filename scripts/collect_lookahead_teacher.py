#!/usr/bin/env python3
"""Collect complete V6 expert trajectories, optionally relabelled by lookahead."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.bot_loader import load_bot
from src.agents.rule_based_agent import RuleBasedPokemonAgent
from src.cg.api import to_observation_class
from src.data.demonstrations import (
    ACTION_SPACE_SIZE,
    DemonstrationDatasetWriter,
    read_deck,
    sha256_file,
    validate_legal_action,
)
from src.env.env_wrapper import (
    V6_ACTION_SPACE_SIZE,
    PokemonTCGEnv,
    _fit_observation_to_model_space,
)
from src.training.lookahead_teacher import (
    LookaheadConfig,
    LookaheadTeacher,
    TeacherDecision,
    build_search_hypotheses,
)
from src.league.experiment_registry import git_revision
from src.utils import resolve_deck_path, resolve_pool_path


DEFAULT_RESERVED_MANIFESTS = (
    "decks/pools/holdout_opponents.json",
    "decks/pools/validation_opponents.json",
)


def _scalar_action(action: Any) -> int:
    return int(np.asarray(action).reshape(-1)[0])


def _teacher_candidate(
    raw_observation: Any,
    encoded_observation: dict[str, np.ndarray],
    *,
    all_decisions: bool,
) -> bool:
    """Keep lookahead's supported root restriction without dropping trajectory steps."""

    select = getattr(raw_observation, "select", None)
    current = getattr(raw_observation, "current", None)
    if select is None or current is None:
        return False
    if int(getattr(select, "minCount", -1)) != 1 or int(getattr(select, "maxCount", -1)) != 1:
        return False
    option_count = len(getattr(select, "option", None) or [])
    legal_count = int(
        np.count_nonzero(np.asarray(encoded_observation["action_mask"])[:option_count])
    )
    if legal_count < 2:
        return False
    if all_decisions:
        return True
    players = list(getattr(current, "players", None) or [])
    return len(players) == 2 and min(len(players[0].prize), len(players[1].prize)) <= 3


def _teacher_q(decision: TeacherDecision) -> np.ndarray:
    values = np.full(ACTION_SPACE_SIZE, np.nan, dtype=np.float32)
    for action, score in decision.scores.items():
        if 0 <= int(action) < ACTION_SPACE_SIZE:
            values[int(action)] = float(score)
    return values


def _validate_expert(expert: Any, model_spec: str) -> None:
    """Rule bots and Kaggle script agents are V6 adapters; neural experts must declare 66 actions."""

    from src.agents.kaggle_bots.wrapper import KagglePythonAgentWrapper

    if isinstance(expert, (RuleBasedPokemonAgent, KagglePythonAgentWrapper)):
        return
    action_space = getattr(expert, "action_space", None)
    action_count = getattr(action_space, "n", None)
    if action_count is None:
        raise ValueError(
            f"Expert {model_spec!r} does not declare a V6 action space; "
            "use a V6 PPO checkpoint, rule_based:* expert, or python_script:* expert"
        )
    if int(action_count) != V6_ACTION_SPACE_SIZE:
        raise ValueError(
            f"Non-V6 neural expert has {action_count} actions; expected {V6_ACTION_SPACE_SIZE}"
        )


def _check_reserved_data(args: argparse.Namespace) -> list[str]:
    """Fail closed if either deck/model overlaps validation or final holdout."""

    from scripts.check_holdout_safe import check_paths

    checked: list[str] = []
    for manifest in dict.fromkeys(args.reserved_opponents):
        resolved = resolve_pool_path(manifest)
        if not resolved.is_file():
            raise FileNotFoundError(f"Reserved-opponent manifest does not exist: {manifest}")
        check_paths(
            str(resolved),
            [str(resolve_deck_path(args.deck)), str(resolve_deck_path(args.opp_deck))],
            [value for value in (args.model, args.opp_model) if value],
            [],
        )
        checked.append(str(resolved))
    return checked


def _outcome(info: dict[str, Any], perspective: int) -> int:
    winner = info.get("winner", -1)
    try:
        winner = int(winner)
    except (TypeError, ValueError):
        return 0
    if winner == int(perspective):
        return 1
    if winner in (0, 1):
        return -1
    return 0


def _resolved_file(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path.resolve() if path.is_file() else None


def _teacher_decision(
    args: argparse.Namespace,
    *,
    teacher: LookaheadTeacher | None,
    rng: np.random.Generator,
    env: PokemonTCGEnv,
    observation: dict[str, np.ndarray],
    deck: list[int],
    opponent_deck: list[int],
) -> tuple[TeacherDecision | None, bool]:
    if teacher is None or float(rng.random()) >= args.lookahead_sample_rate:
        return None, False
    raw_observation = to_observation_class(env.current_obs_dict)
    if not _teacher_candidate(
        raw_observation,
        observation,
        all_decisions=args.all_decisions,
    ):
        return None, False
    hypotheses = build_search_hypotheses(
        raw_observation,
        your_deck=deck,
        opponent_deck=opponent_deck,
        count=args.hypotheses,
        rng=rng,
        card_data_by_id=env.card_data_by_id,
    )
    decision = teacher.choose(
        raw_observation,
        observation,
        perspective=env.learner_perspective,
        hypotheses=hypotheses,
    )
    if decision is None or decision.confidence < args.min_confidence:
        return None, True
    validate_legal_action(observation, decision.action)
    return decision, True


def collect(args: argparse.Namespace) -> dict[str, Any]:
    if V6_ACTION_SPACE_SIZE != ACTION_SPACE_SIZE:
        raise RuntimeError("Demonstration schema and environment V6 action spaces disagree")

    reserved = _check_reserved_data(args)
    deck_path = resolve_deck_path(args.deck).resolve()
    opponent_deck_path = resolve_deck_path(args.opp_deck).resolve()
    expert_path = _resolved_file(args.model)
    opponent_model_path = _resolved_file(args.opp_model)
    deck = read_deck(deck_path)
    opponent_deck = read_deck(opponent_deck_path)
    expert = load_bot(args.model)
    _validate_expert(expert, args.model)
    expert_space = getattr(expert, "observation_space", None)
    rng = np.random.default_rng(args.seed)

    teacher = None
    if args.lookahead_sample_rate > 0.0:
        teacher = LookaheadTeacher(
            LookaheadConfig(
                max_depth=args.depth,
                beam_width=args.beam_width,
                node_budget=args.node_budget,
                max_combinations=args.max_combinations,
            )
        )

    collection_metadata = {
        "collector": "scripts/collect_lookahead_teacher.py",
        "git_revision": git_revision(),
        "expert_spec": args.model,
        "expert_path": str(expert_path) if expert_path is not None else None,
        "expert_sha256": sha256_file(expert_path) if expert_path is not None else None,
        "deck_path": str(deck_path),
        "deck_sha256": sha256_file(deck_path),
        "opponent_model_spec": args.opp_model,
        "opponent_model_path": (
            str(opponent_model_path) if opponent_model_path is not None else None
        ),
        "opponent_model_sha256": (
            sha256_file(opponent_model_path) if opponent_model_path is not None else None
        ),
        "opponent_deck_path": str(opponent_deck_path),
        "opponent_deck_sha256": sha256_file(opponent_deck_path),
        "seed": args.seed,
        "rotate_perspective": bool(args.rotate_perspective),
        "scalar_obs": bool(args.scalar_obs),
        "feature_variant": args.feature_variant,
        "inference_guardrails": bool(args.inference_guardrails),
        "inference_guardrail_mode": args.inference_guardrail_mode,
        "reserved_opponent_manifests": reserved,
        "lookahead_sample_rate": args.lookahead_sample_rate,
        "lookahead_relabel": bool(args.lookahead_relabel),
        "teacher_control": bool(args.teacher_control),
    }
    counters = {
        "requested_games": int(args.games),
        "completed_games": 0,
        "discarded_games": 0,
        "decision_samples": 0,
        "branching_decisions": 0,
        "stop_available_states": 0,
        "stop_labels": 0,
        "lookahead_queries": 0,
        "lookahead_labels": 0,
        "lookahead_counterfactual_labels": 0,
        "teacher_overrides": 0,
        "search_failures": 0,
    }
    env = PokemonTCGEnv(
        my_deck=deck,
        opponent_deck=opponent_deck,
        opponent_model_path=args.opp_model,
        rotate_perspective=args.rotate_perspective,
        action_space_size=V6_ACTION_SPACE_SIZE,
        structured_v2=not args.scalar_obs,
        feature_variant=args.feature_variant,
        zone_aux_targets=False,
        enable_lookahead_teacher=False,
        inference_guardrails=args.inference_guardrails,
        inference_guardrail_mode=args.inference_guardrail_mode,
        search_guardrail_rate=args.search_guardrail_rate,
    )
    writer = DemonstrationDatasetWriter(args.out, metadata=collection_metadata)
    try:
        for source_episode in range(args.games):
            observation, _ = env.reset(seed=args.seed + source_episode)
            writer.start_episode(perspective=env.learner_perspective)
            lstm_state = None
            episode_start = np.ones((1,), dtype=bool)
            terminated = truncated = False
            info: dict[str, Any] = {}

            episode_samples = 0
            episode_counters = {
                key: 0
                for key in (
                    "branching_decisions",
                    "stop_available_states",
                    "stop_labels",
                    "lookahead_queries",
                    "lookahead_labels",
                    "lookahead_counterfactual_labels",
                    "teacher_overrides",
                    "search_failures",
                )
            }
            try:
                for _ in range(args.max_steps):
                    expert_observation = (
                        _fit_observation_to_model_space(observation, expert_space)
                        if expert_space is not None
                        else observation
                    )
                    if hasattr(env, "current_obs_dict") and env.current_obs_dict is not None:
                        if isinstance(expert_observation, dict):
                            expert_observation = dict(expert_observation)
                            expert_observation["raw_obs"] = env.current_obs_dict

                    proposed_action, lstm_state = expert.predict(
                        expert_observation,
                        state=lstm_state,
                        episode_start=episode_start,
                        deterministic=not args.stochastic_expert,
                    )
                    episode_start[:] = False
                    expert_action = _scalar_action(proposed_action)
                    validate_legal_action(observation, expert_action)

                    decision = None
                    if teacher is not None:
                        teacher.last_error = None
                        decision, queried = _teacher_decision(
                            args,
                            teacher=teacher,
                            rng=rng,
                            env=env,
                            observation=observation,
                            deck=deck,
                            opponent_deck=opponent_deck,
                        )
                        if queried:
                            episode_counters["lookahead_queries"] += 1
                        if teacher.last_error is not None:
                            episode_counters["search_failures"] += 1

                    use_teacher_label = decision is not None and (
                        args.lookahead_relabel or args.teacher_control
                    )
                    label_action = decision.action if use_teacher_label else expert_action
                    action_to_play = (
                        decision.action
                        if decision is not None and args.teacher_control
                        else expert_action
                    )
                    if use_teacher_label:
                        episode_counters["lookahead_labels"] += 1
                        if label_action != action_to_play:
                            episode_counters["lookahead_counterfactual_labels"] += 1
                    if action_to_play != expert_action:
                        episode_counters["teacher_overrides"] += 1

                    legal_count = validate_legal_action(observation, label_action)
                    if legal_count >= 2:
                        episode_counters["branching_decisions"] += 1
                    if bool(np.asarray(observation["action_mask"])[V6_ACTION_SPACE_SIZE - 1]):
                        episode_counters["stop_available_states"] += 1
                    if label_action == V6_ACTION_SPACE_SIZE - 1:
                        episode_counters["stop_labels"] += 1

                    writer.append(
                        observation,
                        action=label_action,
                        label_source=(
                            "lookahead_counterfactual"
                            if use_teacher_label and label_action != action_to_play
                            else "lookahead"
                            if use_teacher_label
                            else "expert"
                        ),
                        teacher_confidence=(decision.confidence if decision is not None else 0.0),
                        teacher_q=(_teacher_q(decision) if decision is not None else None),
                    )
                    episode_samples += 1
                    observation, _, terminated, truncated, info = env.step(action_to_play)
                    if terminated or truncated:
                        break

                if terminated and not truncated:
                    writer.commit_episode(
                        outcome=_outcome(info, env.learner_perspective)
                    )
                    counters["completed_games"] += 1
                    counters["decision_samples"] += episode_samples
                    for key, value in episode_counters.items():
                        counters[key] += value
                else:
                    writer.discard_episode()
                    counters["discarded_games"] += 1
            except Exception:
                writer.discard_episode()
                raise
    finally:
        env.close()

    writer.update_metadata(counters)
    return {**counters, "output": str(Path(args.out) / "manifest.json")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect complete structured-V6 expert trajectories with optional "
            "lookahead labels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "--expert",
        dest="model",
        required=True,
        help="V6 PPO or rule_based:* expert",
    )
    parser.add_argument("--deck", required=True)
    parser.add_argument("--opp-deck", required=True)
    parser.add_argument("--opp-model", default=None)
    parser.add_argument("--scalar-obs", action="store_true")
    parser.add_argument(
        "--feature-variant",
        choices=("compact", "strategic_vector_v1", "strategic_vector_v2", "strategic_vector_v3"),
        default="compact",
    )
    parser.add_argument("--out", required=True, help="New dataset output directory")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--sample-rate",
        "--lookahead-sample-rate",
        dest="lookahead_sample_rate",
        type=float,
        default=0.0,
        help="Fraction of supported states queried from lookahead",
    )
    parser.add_argument("--no-lookahead", dest="lookahead_sample_rate", action="store_const", const=0.0)
    parser.add_argument("--hypotheses", type=int, default=4)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--node-budget", type=int, default=96)
    parser.add_argument("--max-combinations", type=int, default=16)
    parser.add_argument("--min-confidence", type=float, default=1.0)
    parser.add_argument("--all-decisions", action="store_true")
    parser.add_argument("--lookahead-relabel", action="store_true")
    parser.add_argument("--teacher-control", action="store_true")
    parser.add_argument("--stochastic-expert", action="store_true")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--rotate-perspective", dest="rotate_perspective", action="store_true")
    parser.add_argument(
        "--no-rotate-perspective",
        "--fixed-perspective",
        dest="rotate_perspective",
        action="store_false",
    )
    parser.add_argument("--inference-guardrails", dest="inference_guardrails", action="store_true")
    parser.add_argument("--no-inference-guardrails", dest="inference_guardrails", action="store_false")
    parser.add_argument(
        "--inference-guardrail-mode",
        choices=("active", "shadow"),
        default="active",
    )
    parser.add_argument("--search-guardrail-rate", type=float, default=0.0)
    parser.add_argument(
        "--reserved-opponents",
        action="append",
        default=list(DEFAULT_RESERVED_MANIFESTS),
        help="Validation/final-holdout manifest that collection must not overlap",
    )
    parser.set_defaults(rotate_perspective=True, inference_guardrails=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.games < 1 or args.max_steps < 1:
        raise ValueError("games and max-steps must be positive")
    if not 0.0 <= args.lookahead_sample_rate <= 1.0:
        raise ValueError("lookahead sample rate must be between 0 and 1")
    if args.hypotheses < 1:
        raise ValueError("hypotheses must be positive")
    scalar_variant = args.feature_variant in {"strategic_vector_v1", "strategic_vector_v2", "strategic_vector_v3"}
    if args.scalar_obs != scalar_variant:
        raise ValueError(
            "scalar_obs and feature_variant must be paired: use --scalar-obs "
            "with strategic_vector_v1/v2/v3, or neither for compact V6"
        )
    summary = collect(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
