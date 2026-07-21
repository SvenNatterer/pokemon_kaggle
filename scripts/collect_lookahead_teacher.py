#!/usr/bin/env python3
"""Collect bounded look-ahead labels on states visited by an existing bot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot_loader import load_bot
from src.cg.api import to_observation_class
from src.env_wrapper import (
    LEGACY_ACTION_SPACE_SIZE,
    V6_ACTION_SPACE_SIZE,
    PokemonTCGEnv,
    _fit_observation_to_model_space,
)
from src.lookahead_teacher import (
    LookaheadConfig,
    LookaheadTeacher,
    build_search_hypotheses,
)


def read_deck(path: str) -> list[int]:
    cards: list[int] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip().split(",", 1)[0]
            if not value:
                continue
            try:
                cards.append(int(value))
            except ValueError as error:
                raise ValueError(f"Invalid card ID at {path}:{line_number}") from error
    if len(cards) != 60:
        raise ValueError(f"Deck must contain 60 cards, got {len(cards)}: {path}")
    return cards


def _scalar_action(action) -> int:
    return int(np.asarray(action).reshape(-1)[0])


def _is_interesting(raw_observation, encoded_observation, *, all_decisions: bool) -> bool:
    select = getattr(raw_observation, "select", None)
    current = getattr(raw_observation, "current", None)
    if select is None or current is None:
        return False
    if int(getattr(select, "minCount", -1)) != 1 or int(getattr(select, "maxCount", -1)) != 1:
        return False
    option_count = len(getattr(select, "option", None) or [])
    legal_count = int(np.count_nonzero(np.asarray(encoded_observation["action_mask"])[:option_count]))
    if legal_count < 2:
        return False
    if all_decisions:
        return True
    players = list(getattr(current, "players", None) or [])
    return len(players) == 2 and min(len(players[0].prize), len(players[1].prize)) <= 3


def _append_sample(
    samples: dict[str, list[np.ndarray]],
    observation: dict[str, np.ndarray],
    *,
    teacher_action: int,
    teacher_scores: dict[int, float],
    confidence: float,
    student_action: int,
    episode: int,
    step: int,
    perspective: int,
) -> None:
    for key, value in observation.items():
        # aux_target is a training target for a different task and is not an
        # actor-visible feature needed by behaviour cloning.
        if key != "aux_target":
            samples.setdefault(f"obs__{key}", []).append(np.asarray(value).copy())

    action_count = len(np.asarray(observation["action_mask"]))
    q_values = np.full(action_count, np.nan, dtype=np.float32)
    for action, score in teacher_scores.items():
        if 0 <= action < action_count:
            q_values[action] = float(score)
    samples.setdefault("teacher_q", []).append(q_values)
    samples.setdefault("teacher_action", []).append(np.asarray(teacher_action, dtype=np.int64))
    samples.setdefault("teacher_confidence", []).append(np.asarray(confidence, dtype=np.float32))
    samples.setdefault("student_action", []).append(np.asarray(student_action, dtype=np.int64))
    samples.setdefault("episode", []).append(np.asarray(episode, dtype=np.int32))
    samples.setdefault("step", []).append(np.asarray(step, dtype=np.int32))
    samples.setdefault("perspective", []).append(np.asarray(perspective, dtype=np.int8))


def _save_dataset(path: str, samples: dict[str, list[np.ndarray]], metadata: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: np.stack(values)
        for key, values in samples.items()
        if values
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)


def collect(args) -> dict:
    rng = np.random.default_rng(args.seed)
    deck = read_deck(args.deck)
    opponent_deck = read_deck(args.opp_deck)
    model = load_bot(args.model)
    model_space = getattr(model, "observation_space", None)
    action_space_size = int(
        getattr(getattr(model, "action_space", None), "n", LEGACY_ACTION_SPACE_SIZE)
    )
    if action_space_size not in {LEGACY_ACTION_SPACE_SIZE, V6_ACTION_SPACE_SIZE}:
        raise ValueError(f"Unsupported model action space: {action_space_size}")
    structured_v2 = bool(
        model_space is not None
        and hasattr(model_space, "spaces")
        and "entity_ids" in model_space.spaces
    )

    teacher = LookaheadTeacher(
        LookaheadConfig(
            max_depth=args.depth,
            beam_width=args.beam_width,
            node_budget=args.node_budget,
            max_combinations=args.max_combinations,
        )
    )
    samples: dict[str, list[np.ndarray]] = {}
    queried = 0
    labelled = 0
    overridden = 0
    search_failures = 0
    completed_games = 0

    env = PokemonTCGEnv(
        my_deck=deck,
        opponent_deck=opponent_deck,
        opponent_model_path=args.opp_model,
        rotate_perspective=args.rotate_perspective,
        action_space_size=action_space_size,
        structured_v2=structured_v2,
        inference_guardrails=False,
    )
    try:
        for episode in range(args.games):
            observation, _ = env.reset(seed=args.seed + episode)
            lstm_state = None
            episode_start = np.ones((1,), dtype=bool)
            terminated = truncated = False

            for step in range(args.max_steps):
                observation_for_model = (
                    _fit_observation_to_model_space(observation, model_space)
                    if model_space is not None
                    else observation
                )
                action, lstm_state = model.predict(
                    observation_for_model,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start[:] = False
                student_action = _scalar_action(action)
                action_to_play = student_action

                raw_observation = to_observation_class(env.current_obs_dict)
                if (
                    _is_interesting(
                        raw_observation,
                        observation,
                        all_decisions=args.all_decisions,
                    )
                    and float(rng.random()) < args.sample_rate
                ):
                    queried += 1
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
                    if teacher.last_error is not None:
                        search_failures += 1
                    if decision is not None and decision.confidence >= args.min_confidence:
                        labelled += 1
                        _append_sample(
                            samples,
                            observation_for_model,
                            teacher_action=decision.action,
                            teacher_scores=decision.scores,
                            confidence=decision.confidence,
                            student_action=student_action,
                            episode=episode,
                            step=step,
                            perspective=env.learner_perspective,
                        )
                        if args.teacher_control and decision.action != student_action:
                            action_to_play = decision.action
                            overridden += 1

                observation, _, terminated, truncated, _ = env.step(action_to_play)
                if terminated or truncated:
                    completed_games += int(terminated)
                    break
    finally:
        env.close()

    metadata = {
        "schema_version": 1,
        "model": args.model,
        "deck": args.deck,
        "opponent_model": args.opp_model,
        "opponent_deck": args.opp_deck,
        "games": args.games,
        "completed_games": completed_games,
        "queried_states": queried,
        "labelled_states": labelled,
        "teacher_overrides": overridden,
        "search_failures": search_failures,
        "teacher_control": bool(args.teacher_control),
        "depth": args.depth,
        "beam_width": args.beam_width,
        "node_budget": args.node_budget,
        "hypotheses": args.hypotheses,
        "seed": args.seed,
    }
    _save_dataset(args.out, samples, metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect look-ahead teacher labels from states visited by a PPO bot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--deck", required=True)
    parser.add_argument("--opp-deck", required=True)
    parser.add_argument("--opp-model", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--sample-rate", type=float, default=0.10)
    parser.add_argument("--hypotheses", type=int, default=4)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--node-budget", type=int, default=96)
    parser.add_argument("--max-combinations", type=int, default=16)
    parser.add_argument("--min-confidence", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--all-decisions", action="store_true")
    parser.add_argument("--teacher-control", action="store_true")
    parser.add_argument("--rotate-perspective", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.games < 1 or args.max_steps < 1:
        raise ValueError("games and max-steps must be positive")
    if not 0.0 <= args.sample_rate <= 1.0:
        raise ValueError("sample-rate must be between 0 and 1")
    metadata = collect(args)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
