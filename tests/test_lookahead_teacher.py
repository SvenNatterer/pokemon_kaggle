from types import SimpleNamespace

import numpy as np

from src.lookahead_teacher import (
    LookaheadConfig,
    LookaheadTeacher,
    build_search_hypotheses,
)


def _pokemon(card_id=1, hp=100, energies=()):
    return SimpleNamespace(
        id=card_id,
        hp=hp,
        energies=list(energies),
        preEvolution=[],
        tools=[],
        energyCards=[],
    )


def _player(*, prizes=2, deck=20, hand=(), hand_count=None, active=None, bench=()):
    return SimpleNamespace(
        prize=[None] * prizes,
        deckCount=deck,
        hand=list(hand),
        handCount=len(hand) if hand_count is None else hand_count,
        discard=[],
        active=[active] if active is not None else [],
        bench=list(bench),
    )


def _observation(
    *,
    actor=0,
    result=-1,
    option_count=2,
    minimum=1,
    maximum=1,
    players=None,
):
    current = SimpleNamespace(
        yourIndex=actor,
        result=result,
        players=players
        or [
            _player(active=_pokemon(1)),
            _player(active=_pokemon(1)),
        ],
    )
    select = None
    if result == -1:
        select = SimpleNamespace(
            option=[SimpleNamespace() for _ in range(option_count)],
            minCount=minimum,
            maxCount=maximum,
        )
    return SimpleNamespace(current=current, select=select)


class _FakeSearch:
    def __init__(self, transitions, observations):
        self.transitions = transitions
        self.observations = observations
        self.released = []
        self.end_count = 0

    def begin(self, observation, **_hypothesis):
        return SimpleNamespace(searchId=0, observation=observation)

    def step(self, search_id, selection):
        child_id = self.transitions[(search_id, tuple(selection))]
        return SimpleNamespace(searchId=child_id, observation=self.observations[child_id])

    def release(self, search_id):
        self.released.append(search_id)

    def end(self):
        self.end_count += 1


def _teacher(fake, **config):
    return LookaheadTeacher(
        LookaheadConfig(**config),
        card_data_by_id={1: SimpleNamespace(ex=False, megaEx=False)},
        search_begin_fn=fake.begin,
        search_step_fn=fake.step,
        search_release_fn=fake.release,
        search_end_fn=fake.end,
    )


def test_teacher_selects_immediate_winning_action():
    root = _observation()
    observations = {
        1: _observation(result=1),
        2: _observation(result=0),
    }
    fake = _FakeSearch(
        transitions={(0, (0,)): 1, (0, (1,)): 2},
        observations=observations,
    )
    teacher = _teacher(fake, max_depth=2, beam_width=2, node_budget=10)

    decision = teacher.choose(
        root,
        {"action_mask": np.array([1, 1, 0], dtype=np.int8)},
        perspective=0,
        hypotheses={"unused": []},
    )

    assert decision is not None
    assert decision.action == 1
    assert decision.scores[1] > decision.scores[0]
    assert decision.confidence > 0
    assert fake.end_count == 1
    assert set(fake.released) == {1, 2}


def test_teacher_uses_minimax_for_opponent_reply():
    root = _observation(actor=0)
    observations = {
        1: _observation(actor=1),
        2: _observation(actor=1),
        3: _observation(result=0),
        4: _observation(result=1),
        5: _observation(result=0),
        6: _observation(result=2),
    }
    fake = _FakeSearch(
        transitions={
            (0, (0,)): 1,
            (0, (1,)): 2,
            (1, (0,)): 3,
            (1, (1,)): 4,
            (2, (0,)): 5,
            (2, (1,)): 6,
        },
        observations=observations,
    )
    teacher = _teacher(fake, max_depth=3, beam_width=2, node_budget=20)

    decision = teacher.choose(
        root,
        {"action_mask": np.array([1, 1, 0], dtype=np.int8)},
        perspective=0,
        hypotheses={"unused": []},
    )

    assert decision is not None
    # Action 0 lets the opponent force our loss.  Action 1 lets the opponent
    # force only a draw, so minimax must prefer action 1.
    assert decision.action == 1
    assert decision.scores[0] < decision.scores[1]


def test_teacher_skips_multi_selection_root():
    root = _observation(minimum=1, maximum=2)
    fake = _FakeSearch(transitions={}, observations={})
    teacher = _teacher(fake, max_depth=2, beam_width=2, node_budget=10)

    decision = teacher.choose(
        root,
        {"action_mask": np.array([1, 1, 0], dtype=np.int8)},
        perspective=0,
        hypotheses={"unused": []},
    )

    assert decision is None
    assert fake.end_count == 0


def test_hidden_hypotheses_have_exact_zone_counts():
    players = [
        _player(
            prizes=1,
            deck=2,
            hand=[SimpleNamespace(id=2)],
            active=_pokemon(1),
        ),
        _player(
            prizes=1,
            deck=2,
            hand_count=1,
            active=_pokemon(3),
        ),
    ]
    observation = _observation(actor=0, players=players)
    card_data = {
        1: SimpleNamespace(basic=True),
        2: SimpleNamespace(basic=False),
        3: SimpleNamespace(basic=True),
        4: SimpleNamespace(basic=False),
        5: SimpleNamespace(basic=False),
    }

    hypotheses = build_search_hypotheses(
        observation,
        your_deck=[1, 2, 3, 4, 5],
        opponent_deck=[1, 2, 3, 4, 5],
        count=3,
        rng=np.random.default_rng(7),
        card_data_by_id=card_data,
    )

    assert len(hypotheses) == 3
    for hypothesis in hypotheses:
        assert len(hypothesis["your_deck"]) == 2
        assert len(hypothesis["your_prize"]) == 1
        assert len(hypothesis["opponent_deck"]) == 2
        assert len(hypothesis["opponent_prize"]) == 1
        assert len(hypothesis["opponent_hand"]) == 1
        assert hypothesis["opponent_active"] == []
