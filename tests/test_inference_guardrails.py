from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.cg.api import Option, OptionType
from src.models.inference_guardrails import (
    FIGHTING_ENERGY_TYPE,
    InferenceGuardrails,
    MIST_ENERGY_CARD_ID,
    POWERFUL_HAND_ATTACK_ID,
    ROCK_FIGHTING_ENERGY_CARD_ID,
    SampledSearchGuardrails,
    TEAM_ROCKET_ARTICUNO_CARD_ID,
)
from src.env.env_wrapper import PokemonTCGEnv


def _card(card_id, serial=1):
    return SimpleNamespace(id=card_id, serial=serial)


def _pokemon(serial, *, card_id=999, energy_card_ids=(), hp=100):
    return SimpleNamespace(
        id=card_id,
        serial=serial,
        hp=hp,
        energyCards=[_card(card_id, serial=100 + index) for index, card_id in enumerate(energy_card_ids)],
    )


def _log(**values):
    defaults = {
        "type": None,
        "attackId": None,
        "playerIndex": None,
        "serial": None,
        "head": None,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def _observation(
    *,
    target,
    turn=5,
    actor=0,
    logs=(),
    attack_ids=(POWERFUL_HAND_ATTACK_ID, None),
    min_count=1,
    opponent_bench=(),
):
    players = [
        SimpleNamespace(
            active=[_pokemon(10)], bench=[], deckCount=20, prize=[None] * 4, handCount=5
        ),
        SimpleNamespace(
            active=[target],
            bench=list(opponent_bench),
            deckCount=20,
            prize=[None] * 4,
            handCount=5,
        ),
    ]
    return SimpleNamespace(
        current=SimpleNamespace(
            turn=turn,
            turnActionCount=0,
            yourIndex=actor,
            players=players,
        ),
        select=SimpleNamespace(
            option=[
                Option(
                    type=OptionType.ATTACK if attack_id is not None else OptionType.END,
                    attackId=attack_id,
                )
                for attack_id in attack_ids
            ],
            minCount=min_count,
            maxCount=1,
        ),
        logs=list(logs),
        search_begin_input="decision-state",
    )


def _encoded(mask=(1, 1, 0)):
    return {
        "vector": np.zeros(4, dtype=np.float32),
        "action_mask": np.asarray(mask, dtype=np.int8),
    }


class InferenceGuardrailTests(unittest.TestCase):
    def test_mist_energy_masks_powerful_hand_without_mutating_input(self):
        guardrails = InferenceGuardrails()
        obs = _observation(
            target=_pokemon(20, energy_card_ids=(MIST_ENERGY_CARD_ID,)),
        )
        encoded = _encoded()

        guarded, interventions = guardrails.apply(obs, encoded, perspective=0)

        self.assertEqual([1, 1, 0], encoded["action_mask"].tolist())
        self.assertEqual([0, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual(1, len(interventions))
        self.assertEqual(
            "powerful_hand_blocked_by_mist_energy",
            interventions[0].rule,
        )
        metrics = guardrails.metrics_snapshot()
        self.assertEqual(1.0, metrics["proposals_total"])
        self.assertEqual(1.0, metrics["accepted_total"])
        self.assertEqual(
            1.0,
            metrics["by_rule"]["powerful_hand_blocked_by_mist_energy"],
        )

    def test_shadow_mode_records_proposal_without_changing_mask(self):
        guardrails = InferenceGuardrails(mode="shadow")
        obs = _observation(
            target=_pokemon(20, energy_card_ids=(MIST_ENERGY_CARD_ID,)),
        )
        encoded = _encoded()

        guarded, interventions = guardrails.apply(obs, encoded, perspective=0)

        self.assertIs(guarded, encoded)
        self.assertEqual([], interventions)
        metrics = guardrails.metrics_snapshot()
        self.assertEqual(1.0, metrics["proposals_total"])
        self.assertEqual(0.0, metrics["accepted_total"])
        self.assertEqual(1.0, metrics["shadow_total"])

    def test_mist_energy_does_not_mask_other_attacks(self):
        guardrails = InferenceGuardrails()
        obs = _observation(
            target=_pokemon(20, energy_card_ids=(MIST_ENERGY_CARD_ID,)),
            attack_ids=(999, None),
        )

        guarded, interventions = guardrails.apply(obs, _encoded(), perspective=0)

        self.assertEqual([1, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)

    def test_rock_fighting_energy_masks_powerful_hand_on_fighting_pokemon(self):
        fighting_card_id = 676
        guardrails = InferenceGuardrails(
            card_energy_types={fighting_card_id: FIGHTING_ENERGY_TYPE}
        )
        obs = _observation(
            target=_pokemon(
                20,
                card_id=fighting_card_id,
                energy_card_ids=(ROCK_FIGHTING_ENERGY_CARD_ID,),
            ),
        )

        guarded, interventions = guardrails.apply(obs, _encoded(), perspective=0)

        self.assertEqual([0, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual(1, len(interventions))
        self.assertEqual(
            "powerful_hand_blocked_by_rock_fighting_energy",
            interventions[0].rule,
        )

    def test_rock_fighting_energy_does_not_mask_on_non_fighting_pokemon(self):
        non_fighting_card_id = 741
        guardrails = InferenceGuardrails(
            card_energy_types={non_fighting_card_id: 5}
        )
        obs = _observation(
            target=_pokemon(
                20,
                card_id=non_fighting_card_id,
                energy_card_ids=(ROCK_FIGHTING_ENERGY_CARD_ID,),
            ),
        )

        guarded, interventions = guardrails.apply(obs, _encoded(), perspective=0)

        self.assertEqual([1, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)

    def test_production_option_contract_works_for_player_one_perspective(self):
        guardrails = InferenceGuardrails()
        obs = _observation(target=_pokemon(20), actor=1)
        obs.current.players[0].active = [
            _pokemon(10, energy_card_ids=(MIST_ENERGY_CARD_ID,))
        ]

        guarded, interventions = guardrails.apply(obs, _encoded(), perspective=1)

        self.assertEqual([0, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual(
            "powerful_hand_blocked_by_mist_energy",
            interventions[0].rule,
        )

    def test_repelling_veil_masks_powerful_hand_against_articuno_itself(self):
        guardrails = InferenceGuardrails()
        guardrails.set_basic_team_rocket_card_ids({TEAM_ROCKET_ARTICUNO_CARD_ID})
        obs = _observation(
            target=_pokemon(20, card_id=TEAM_ROCKET_ARTICUNO_CARD_ID),
        )

        guarded, interventions = guardrails.apply(obs, _encoded(), perspective=0)

        self.assertEqual([0, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual(1, len(interventions))
        self.assertEqual(
            "powerful_hand_blocked_by_repelling_veil",
            interventions[0].rule,
        )

    def test_repelling_veil_masks_basic_team_rocket_target_from_bench(self):
        basic_team_rocket_card_id = 999
        guardrails = InferenceGuardrails(
            basic_team_rocket_card_ids={basic_team_rocket_card_id}
        )
        obs = _observation(
            target=_pokemon(20, card_id=basic_team_rocket_card_id),
            opponent_bench=(
                _pokemon(21, card_id=TEAM_ROCKET_ARTICUNO_CARD_ID),
            ),
        )

        guarded, interventions = guardrails.apply(obs, _encoded(), perspective=0)

        self.assertEqual([0, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual(
            "powerful_hand_blocked_by_repelling_veil",
            interventions[0].rule,
        )

    def test_repelling_veil_does_not_mask_non_basic_team_rocket_target(self):
        guardrails = InferenceGuardrails(basic_team_rocket_card_ids=set())
        obs = _observation(
            target=_pokemon(20, card_id=999),
            opponent_bench=(
                _pokemon(21, card_id=TEAM_ROCKET_ARTICUNO_CARD_ID),
            ),
        )

        guarded, interventions = guardrails.apply(obs, _encoded(), perspective=0)

        self.assertEqual([1, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)

    def test_synthetic_lethal_hint_does_not_hard_mask(self):
        guardrails = InferenceGuardrails()
        obs = _observation(target=_pokemon(20))
        obs.select.option = [
            SimpleNamespace(attackId=101, isLethal=False),
            SimpleNamespace(attackId=102, isLethal=True),
        ]
        encoded = {"action_mask": np.array([1, 1], dtype=np.int8)}

        guarded, interventions = guardrails.apply(obs, encoded, perspective=0)

        self.assertEqual([1, 1], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)

    def test_synthetic_recoil_hint_does_not_hard_mask(self):
        guardrails = InferenceGuardrails()
        obs = _observation(target=_pokemon(20))
        obs.current.players[0].active[0].hp = 30
        obs.select.option = [
            SimpleNamespace(attackId=101, recoilHp=50),
            SimpleNamespace(attackId=102, recoilHp=0),
        ]
        encoded = {"action_mask": np.array([1, 1], dtype=np.int8)}

        guarded, interventions = guardrails.apply(obs, encoded, perspective=0)

        self.assertEqual([1, 1], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)

    def test_synthetic_search_hint_does_not_hard_mask(self):
        guardrails = InferenceGuardrails()
        obs = _observation(target=_pokemon(20))
        obs.current.players[0].deckCount = 0
        obs.select.option = [
            SimpleNamespace(attackId=None, isSearch=True),
            SimpleNamespace(attackId=102, isSearch=False),
        ]
        encoded = {"action_mask": np.array([1, 1], dtype=np.int8)}

        guarded, interventions = guardrails.apply(obs, encoded, perspective=0)

        self.assertEqual([1, 1], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)

    def test_synthetic_wasted_attach_hint_does_not_hard_mask(self):
        guardrails = InferenceGuardrails()
        obs = _observation(target=_pokemon(20))
        obs.select.option = [
            SimpleNamespace(attackId=None, isWastedAttach=True),
            SimpleNamespace(attackId=None, isWastedAttach=False),
        ]
        encoded = {"action_mask": np.array([1, 1], dtype=np.int8)}

        guarded, interventions = guardrails.apply(obs, encoded, perspective=0)

        self.assertEqual([1, 1], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)

    def test_synthetic_sequence_hint_does_not_hard_mask(self):
        guardrails = InferenceGuardrails()
        obs = _observation(target=_pokemon(20))
        obs.select.option = [
            SimpleNamespace(attackId=None, isDiscardDraw=True),
            SimpleNamespace(attackId=None, isPlayableItem=True),
        ]
        encoded = {"action_mask": np.array([1, 1], dtype=np.int8)}

        guarded, interventions = guardrails.apply(obs, encoded, perspective=0)

        self.assertEqual([1, 1], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)

    def test_synthetic_bench_priority_hint_does_not_hard_mask(self):
        guardrails = InferenceGuardrails()
        obs = _observation(target=_pokemon(20))
        obs.current.players[0].bench = [_pokemon(30), _pokemon(31)]
        obs.current.players[0].maxBench = 3
        obs.select.option = [
            SimpleNamespace(attackId=None, isBench=True, isLowPriorityBench=True),
            SimpleNamespace(attackId=None, isBench=False),
        ]
        encoded = {"action_mask": np.array([1, 1], dtype=np.int8)}

        guarded, interventions = guardrails.apply(obs, encoded, perspective=0)

        self.assertEqual([1, 1], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)

    def test_synthetic_recycle_hint_does_not_hard_mask(self):
        guardrails = InferenceGuardrails()
        obs = _observation(target=_pokemon(20))
        obs.current.players[0].discardCount = 0
        obs.current.players[0].discard = []
        obs.select.option = [
            SimpleNamespace(attackId=None, isRecycleItem=True),
            SimpleNamespace(attackId=102, isRecycleItem=False),
        ]
        encoded = {"action_mask": np.array([1, 1], dtype=np.int8)}

        guarded, interventions = guardrails.apply(obs, encoded, perspective=0)

        self.assertEqual([1, 1], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)

    def test_synthetic_draw_hint_does_not_hard_mask(self):
        guardrails = InferenceGuardrails()
        obs = _observation(target=_pokemon(20))
        obs.current.players[0].deckCount = 3
        obs.select.option = [
            SimpleNamespace(attackId=None, isHeavyDraw=True, cardsToDraw=5, isLethal=False),
            SimpleNamespace(attackId=102, isHeavyDraw=False, cardsToDraw=0, isLethal=False),
        ]
        encoded = {"action_mask": np.array([1, 1], dtype=np.int8)}

        guarded, interventions = guardrails.apply(obs, encoded, perspective=0)

        self.assertEqual([1, 1], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)

    def test_synthetic_win_condition_hint_does_not_hard_mask(self):
        guardrails = InferenceGuardrails()
        obs = _observation(target=_pokemon(20))
        obs.select.option = [
            SimpleNamespace(attackId=None, isSoleStage2Discard=True),
            SimpleNamespace(attackId=None, isSoleStage2Discard=False),
        ]
        encoded = {"action_mask": np.array([1, 1], dtype=np.int8)}

        guarded, interventions = guardrails.apply(obs, encoded, perspective=0)

        self.assertEqual([1, 1], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)

    def test_synthetic_energy_priority_hint_does_not_hard_mask(self):
        guardrails = InferenceGuardrails()
        obs = _observation(target=_pokemon(20))
        obs.select.option = [
            SimpleNamespace(attackId=None, isBenchEnergyAttach=True),
            SimpleNamespace(attackId=None, isActiveLethalAttach=True),
        ]
        encoded = {"action_mask": np.array([1, 1], dtype=np.int8)}

        guarded, interventions = guardrails.apply(obs, encoded, perspective=0)

        self.assertEqual([1, 1], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)


class _FixedRng:

    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


class SampledSearchGuardrailTests(unittest.TestCase):
    @staticmethod
    def _search_child(target_serial=20, target_hp=100, hp_change=0):
        target = _pokemon(target_serial, hp=target_hp)
        current = SimpleNamespace(
            players=[
                SimpleNamespace(active=[_pokemon(10)], bench=[]),
                SimpleNamespace(active=[target], bench=[]),
            ]
        )
        return SimpleNamespace(
            observation=SimpleNamespace(
                current=current,
                logs=[SimpleNamespace(type=16, serial=target_serial, value=hp_change)],
            )
        )

    def test_samples_risky_state_and_masks_search_confirmed_no_effect(self):
        guardrails = SampledSearchGuardrails(sample_rate=0.075)
        obs = _observation(target=_pokemon(20, hp=100))
        root = SimpleNamespace(searchId=7)
        child = self._search_child()

        with (
            patch("src.cg.api.search_begin", return_value=root) as search_begin,
            patch("src.cg.api.search_step", return_value=child) as search_step,
            patch("src.cg.api.search_end") as search_end,
        ):
            guarded, interventions = guardrails.apply(
                obs,
                _encoded(),
                perspective=0,
                rng=_FixedRng(0.01),
            )

        self.assertEqual([0, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual("powerful_hand_zero_effect_search", interventions[0].rule)
        self.assertEqual(1, guardrails.risky_state_count)
        self.assertEqual(1, guardrails.sampled_state_count)
        self.assertEqual(1, guardrails.search_begin_count)
        self.assertEqual(1, guardrails.search_step_count)
        self.assertEqual(1, guardrails.intervention_count)
        search_begin.assert_called_once()
        search_step.assert_called_once_with(7, [0])
        search_end.assert_called_once()

    def test_decision_cache_prevents_resampling_and_research(self):
        guardrails = SampledSearchGuardrails(sample_rate=0.075)
        obs = _observation(target=_pokemon(20, hp=100))
        root = SimpleNamespace(searchId=7)

        with (
            patch("src.cg.api.search_begin", return_value=root) as search_begin,
            patch("src.cg.api.search_step", return_value=self._search_child()),
            patch("src.cg.api.search_end"),
        ):
            first, _ = guardrails.apply(
                obs, _encoded(), perspective=0, rng=_FixedRng(0.01)
            )
            second, _ = guardrails.apply(
                obs, _encoded(), perspective=0, rng=_FixedRng(0.99)
            )

        self.assertEqual([0, 1, 0], first["action_mask"].tolist())
        self.assertEqual([0, 1, 0], second["action_mask"].tolist())
        self.assertEqual(1, guardrails.risky_state_count)
        self.assertEqual(1, guardrails.sampled_state_count)
        search_begin.assert_called_once()

    def test_unsampled_risky_state_has_no_search_overhead(self):
        guardrails = SampledSearchGuardrails(sample_rate=0.075)
        obs = _observation(target=_pokemon(20, hp=100))

        with patch("src.cg.api.search_begin") as search_begin:
            guarded, interventions = guardrails.apply(
                obs,
                _encoded(),
                perspective=0,
                rng=_FixedRng(0.50),
            )

        self.assertEqual([1, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)
        self.assertEqual(1, guardrails.risky_state_count)
        self.assertEqual(0, guardrails.sampled_state_count)
        search_begin.assert_not_called()

    def test_search_failure_fails_open(self):
        guardrails = SampledSearchGuardrails(sample_rate=0.075)
        obs = _observation(target=_pokemon(20, hp=100))

        with patch("src.cg.api.search_begin", side_effect=RuntimeError("search failed")):
            guarded, interventions = guardrails.apply(
                obs,
                _encoded(),
                perspective=0,
                rng=_FixedRng(0.01),
            )

        self.assertEqual([1, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)
        self.assertEqual(1, guardrails.failure_count)
        self.assertIn("search failed", guardrails.last_error)

    def test_episode_reset_preserves_lifetime_search_metrics(self):
        guardrails = SampledSearchGuardrails(sample_rate=0.075)
        obs = _observation(target=_pokemon(20, hp=100))

        with patch("src.cg.api.search_begin", side_effect=RuntimeError("search failed")):
            guardrails.apply(
                obs,
                _encoded(),
                perspective=0,
                rng=_FixedRng(0.01),
            )
        guardrails.reset()

        metrics = guardrails.metrics_snapshot()
        self.assertEqual(1.0, metrics["decisions_total"])
        self.assertEqual(1.0, metrics["search_failures_total"])


class InferenceGuardrailStateTests(unittest.TestCase):
    def test_splashing_dodge_heads_persists_for_the_protected_turn(self):
        guardrails = InferenceGuardrails()
        target = _pokemon(20)
        splash_logs = [
            _log(type=15, attackId=1266, playerIndex=1, serial=20),
            _log(type=16, playerIndex=0, serial=10),
            _log(type=22, playerIndex=1, head=True),
        ]
        obs = _observation(target=target, turn=5, actor=0, logs=splash_logs)

        guarded, interventions = guardrails.apply(obs, _encoded(), perspective=0)

        self.assertEqual([0, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual(
            "powerful_hand_blocked_by_splashing_dodge",
            interventions[0].rule,
        )

        later_same_turn = _observation(target=target, turn=5, actor=0, logs=[])
        guarded, interventions = guardrails.apply(
            later_same_turn,
            _encoded(),
            perspective=0,
        )
        self.assertEqual([0, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual(1, len(interventions))

    def test_splashing_dodge_tails_does_not_mask(self):
        guardrails = InferenceGuardrails()
        logs = [
            _log(type=15, attackId=244, playerIndex=1, serial=20),
            _log(type=22, playerIndex=1, head=False),
        ]
        obs = _observation(target=_pokemon(20), turn=5, actor=0, logs=logs)

        guarded, interventions = guardrails.apply(obs, _encoded(), perspective=0)

        self.assertEqual([1, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)

    def test_splashing_dodge_expires_after_the_protected_turn(self):
        guardrails = InferenceGuardrails()
        target = _pokemon(20)
        logs = [
            _log(type=15, attackId=1266, playerIndex=1, serial=20),
            _log(type=22, playerIndex=1, head=True),
        ]
        guardrails.apply(
            _observation(target=target, turn=5, actor=0, logs=logs),
            _encoded(),
            perspective=0,
        )

        guarded, interventions = guardrails.apply(
            _observation(target=target, turn=7, actor=0, logs=[]),
            _encoded(),
            perspective=0,
        )

        self.assertEqual([1, 1, 0], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)

    def test_guardrail_fails_open_if_it_would_break_required_selection(self):
        guardrails = InferenceGuardrails()
        obs = _observation(
            target=_pokemon(20, energy_card_ids=(MIST_ENERGY_CARD_ID,)),
            attack_ids=(POWERFUL_HAND_ATTACK_ID,),
        )
        encoded = _encoded(mask=(1, 0, 0))

        guarded, interventions = guardrails.apply(obs, encoded, perspective=0)

        self.assertIs(guarded, encoded)
        self.assertEqual([1, 0, 0], guarded["action_mask"].tolist())
        self.assertEqual([], interventions)
        metrics = guardrails.metrics_snapshot()
        self.assertEqual(1.0, metrics["proposals_total"])
        self.assertEqual(1.0, metrics["rolled_back_total"])
        self.assertEqual(0.0, metrics["accepted_total"])


class PokemonTCGEnvGuardrailIntegrationTests(unittest.TestCase):
    def test_env_guardrails_integration_and_metrics(self):
        deck = np.loadtxt(
            "decks/deck_bank/bank_38.csv", delimiter=",", dtype=np.int64
        ).reshape(-1).tolist()
        env = PokemonTCGEnv(deck, deck, inference_guardrails=True)
        self.assertIsNotNone(env.inference_guardrail)
        self.assertTrue(env.inference_guardrail._card_energy_types)
        self.assertTrue(env.inference_guardrail._basic_team_rocket_card_ids)
        env.reset()
        with patch.object(
            env.inference_guardrail,
            "apply",
            wraps=env.inference_guardrail.apply,
        ) as apply_guardrails:
            env._get_obs_cpp(perspective=env.learner_perspective)
        apply_guardrails.assert_called_once()
        opponent_observation = {"action_mask": np.ones(2, dtype=np.int8)}
        with patch.object(env.inference_guardrail, "apply") as opponent_apply:
            returned = env._apply_guardrails_to_observation(
                SimpleNamespace(),
                opponent_observation,
                perspective=1 - env.learner_perspective,
                pending_selection=(),
            )
        self.assertIs(returned, opponent_observation)
        opponent_apply.assert_not_called()
        metrics = env.get_guardrail_metrics()
        self.assertIn("total_interventions", metrics)
        self.assertEqual(metrics["total_interventions"], 0.0)
        self.assertIn("known_rules", metrics)
        self.assertIn("powerful_hand_blocked_by_mist_energy", metrics["known_rules"])
        env.close()


if __name__ == "__main__":
    unittest.main()
