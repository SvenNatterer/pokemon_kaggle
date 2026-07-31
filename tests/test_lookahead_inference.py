from unittest.mock import patch

import numpy as np

from src.models.lookahead_inference import LookaheadInferenceAgent
from src.training.lookahead_teacher import TeacherDecision


class _BaseBot:
    def predict(self, *_args, **_kwargs):
        return np.asarray([1]), "lstm-state"


class _Teacher:
    def choose(self, *_args, **_kwargs):
        return TeacherDecision(
            action=2,
            scores={1: 1.0, 2: 2.0},
            confidence=1.0,
            successful_hypotheses=1,
            searched_nodes=1,
        )


def test_lookahead_diagnostics_record_ppo_override():
    agent = LookaheadInferenceAgent(_BaseBot(), my_deck=[1], opponent_deck=[2])
    agent.teacher = _Teacher()

    with patch(
        "src.models.lookahead_inference.build_search_hypotheses",
        return_value=[{"my_hand": [], "opponent_hand": [], "my_prizes": [], "opponent_prizes": []}],
    ):
        action, state = agent.predict(
            {"action_mask": np.asarray([1, 1, 1])},
            raw_observation=object(),
        )

    assert action == 2
    assert state == "lstm-state"
    assert agent.diagnostics_snapshot() == {
        "base_predictions": 1,
        "search_attempts": 1,
        "search_decisions": 1,
        "overrides": 1,
        "search_errors": 0,
        "confidence_sum": 1.0,
        "override_rate": 1.0,
        "mean_confidence": 1.0,
    }
