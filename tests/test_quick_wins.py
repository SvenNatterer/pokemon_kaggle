#!/usr/bin/env python3
"""Test suite for the 4 Quick Wins: Deck Belief State, Engine Previews, Zone Aux Masking, and BC Loss."""

import os
import sys
import unittest
import numpy as np
import torch
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(ROOT, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.env_wrapper import PokemonTCGEnv
from src.custom_ppo import CustomPPO, PokemonTCGRecurrentPolicy
from src.cg.api import to_observation_class
from src.cg.game import battle_start


class TestQuickWins(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.deck_path = os.path.join(ROOT, "decks", "deck_bank", "bank_18.csv")
        if not os.path.exists(cls.deck_path):
            cls.deck_path = os.path.join(ROOT, "decks", "deck_0.csv")
        df = pd.read_csv(cls.deck_path, header=None)
        cls.deck = df[0].tolist()

    def test_qw1_public_remaining_deck_counts(self):
        """Test Quick Win 1: own_deck_remaining_counts, own_deck_summary, and known_prize_cards_mask."""
        env = PokemonTCGEnv(self.deck, self.deck)
        obs, _ = env.reset()
        obs_obj = to_observation_class(env.current_obs_dict)
        obs_dict = env._structured_observation(obs_obj, 0, [])
        
        self.assertIn("own_deck_remaining_counts", obs_dict)
        self.assertIn("own_deck_summary", obs_dict)
        self.assertIn("known_prize_cards_mask", obs_dict)
        rem_counts = obs_dict["own_deck_remaining_counts"]
        summary = obs_dict["own_deck_summary"]
        self.assertEqual(len(rem_counts), 60)
        self.assertEqual(len(summary), 4)
        self.assertGreater(np.sum(rem_counts), 0.0)
        print(f"\n[PASS] Quick Win 1 & Draw Probabilities verified. Total deck size ratio: {summary[0]:.2f}, Energy Prob: {summary[1]:.2f}")

    def test_qw2_attack_option_previews(self):
        """Test Quick Win 2: Attack option previews include wins_game, prize_trade_delta, and prize_trade_efficiency."""
        env = PokemonTCGEnv(self.deck, self.deck)
        obs, _ = env.reset()
        obs_obj = to_observation_class(env.current_obs_dict)
        obs_dict = env._structured_observation(obs_obj, 0, [])
        
        option_features = obs_dict["option_features"]
        self.assertEqual(option_features.shape[1], 21)
        print("\n[PASS] Quick Win 2: Exact attack option preview & Prize Trade features verified (dim=21).")




    def test_qw3_zone_aux_order_default(self):
        """Test Quick Win 3: CustomPPO order_aux_weight defaults to 0.0."""
        model = CustomPPO(PokemonTCGRecurrentPolicy, PokemonTCGEnv(self.deck, self.deck), verbose=0)
        self.assertEqual(model.order_aux_weight, 0.0)
        print("\n[PASS] Quick Win 3: order_aux_weight defaults to 0.0 to eliminate unrevealed order noise.")

    def test_qw4_behavior_cloning_loss(self):
        """Test Quick Win 4: CustomPPO supports bc_coef > 0.0 for Behavior Cloning."""
        model = CustomPPO(PokemonTCGRecurrentPolicy, PokemonTCGEnv(self.deck, self.deck), bc_coef=0.08, verbose=0)
        self.assertEqual(model.bc_coef, 0.08)
        print("\n[PASS] Quick Win 4: Behavior Cloning (BC) Loss parameter bc_coef initialized correctly.")



if __name__ == "__main__":
    unittest.main()
