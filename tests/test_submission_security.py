import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import importlib.util
import tempfile
import time
import tracemalloc
import unittest

from src.cg.api import to_observation_class
from src.cg.game import battle_start, battle_select

# Alias src.cg in sys.modules so main.py uses the exact same C-extension singleton
if "src.cg" in sys.modules:
    sys.modules["cg"] = sys.modules["src.cg"]
    sys.modules["cg.api"] = sys.modules["src.cg.api"]
    sys.modules["cg.game"] = sys.modules["src.cg.game"]

from src.league.tournament import evaluate_vs_opponent
from src.league.model_paths import resolve_deck_model_path
import pandas as pd





class TestSubmissionSecurity(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.submission_main_path = os.path.join(ROOT, "submission", "main.py")
        cls.baseline_model_path = os.path.join(
            ROOT, "models", "ppo_v6_mewtwo_quickwins_devpool.zip"
        )
        cls.deck_path = os.path.join(ROOT, "decks", "deck_bank", "bank_18.csv")
        if not os.path.exists(cls.deck_path):
            cls.deck_path = os.path.join(ROOT, "decks", "deck_0.csv")
        
        df = pd.read_csv(cls.deck_path, header=None)
        cls.deck = df[0].tolist()

    def _load_submission_module(self):
        """Helper to load submission main module freshly."""
        spec = importlib.util.spec_from_file_location("sub_main", self.submission_main_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _create_sample_obs_dict(self):
        obs_dict, _ = battle_start(self.deck, self.deck)
        return obs_dict

    def test_01_deck_selection_reset(self):
        """Verify deck selection resets match time and lstm state."""
        mod = self._load_submission_module()
        mod.match_accumulated_time = 123.45
        mod.fallback_mode = True
        
        deck = mod.agent({"select": None})
        self.assertIsInstance(deck, list)
        self.assertEqual(len(deck), 60)
        self.assertEqual(mod.match_accumulated_time, 0.0)
        self.assertFalse(mod.fallback_mode)
        self.assertEqual(mod.step_counter, 0)
        print("\n[PASS] Deck selection reset test")

    def test_02_time_tracking_accumulation(self):
        """Verify match_accumulated_time increases on decision steps."""
        mod = self._load_submission_module()
        mod.agent({"select": None})  # Reset
        
        obs_dict = self._create_sample_obs_dict()
        
        mod.agent(obs_dict)
        self.assertGreater(mod.match_accumulated_time, 0.0)
        self.assertEqual(mod.step_counter, 1)
        print(f"\n[PASS] Time tracking test: accumulated {mod.match_accumulated_time:.6f}s after 1 step")

    def test_03_time_limit_fallback_trigger(self):
        """Verify fallback triggers when match_accumulated_time > MAX_MATCH_TIME_SECONDS."""
        mod = self._load_submission_module()
        mod.agent({"select": None})
        mod.match_accumulated_time = 545.0  # Force > 540s cutoff
        
        obs_dict = self._create_sample_obs_dict()
        
        res = mod.agent(obs_dict)
        self.assertTrue(mod.fallback_mode)
        self.assertIsInstance(res, list)
        self.assertGreaterEqual(len(res), 1)
        print(f"\n[PASS] Time limit fallback triggered cleanly. Selection: {res}")

    def test_04_low_overage_time_fallback_trigger(self):
        """Verify fallback triggers when remainingOverageTime < 30.0s."""
        mod = self._load_submission_module()
        mod.agent({"select": None})
        
        obs_dict = self._create_sample_obs_dict()
        obs_dict["remainingOverageTime"] = 15.0
        
        res = mod.agent(obs_dict)
        self.assertTrue(mod.fallback_mode)
        self.assertIsInstance(res, list)
        print(f"\n[PASS] Low remainingOverageTime fallback triggered cleanly. Selection: {res}")

    def test_05_prediction_exception_fallback(self):
        """Verify unhandled error during model prediction triggers fast fallback without crashing."""
        mod = self._load_submission_module()
        mod.agent({"select": None})
        
        # Inject mock model that raises an Exception on predict
        class BrokenModel:
            def predict(self, *args, **kwargs):
                raise RuntimeError("Simulated PyTorch CUDA out of memory error!")
                
        mod.model = BrokenModel()
        
        obs_dict = self._create_sample_obs_dict()
        
        res = mod.agent(obs_dict)
        self.assertTrue(mod.fallback_mode)
        self.assertIsInstance(res, list)
        print(f"\n[PASS] Exception handling fallback test passed. Returned valid response: {res}")


    def test_06_self_play_matches(self):
        """Execute self-play matches (PPO Agent vs PPO Agent copy) to verify stability."""
        print(f"\n=== Executing PPO vs PPO Self-Play Test ({self.baseline_model_path}) ===")
        self.assertTrue(os.path.exists(self.baseline_model_path), f"Baseline model not found: {self.baseline_model_path}")
        self.assertTrue(os.path.exists(self.deck_path), f"Deck not found: {self.deck_path}")

        num_games = 5
        tracemalloc.start()
        start_t = time.monotonic()
        
        result, details = evaluate_vs_opponent(
            self.baseline_model_path,
            self.deck_path,
            self.baseline_model_path,
            self.deck_path,
            num_games,
            return_details=True,
        )
        
        elapsed = time.monotonic() - start_t
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        wins, losses, draws, pw1, dw1, bw1, pw2, dw2, bw2 = result
        crashes = int(details.get("crashes", 0)) if isinstance(details, dict) else 0
        health = details.get("health", {}) if isinstance(details, dict) else {}
        
        print(f"Self-play completed in {elapsed:.2f}s ({elapsed / num_games:.2f}s per game)")
        print(f"Result: Wins={wins}, Losses={losses}, Draws={draws}, Crashes={crashes}")
        print(f"Peak RAM during self-play: {peak_mem / (1024 * 1024):.2f} MB")
        
        self.assertEqual(crashes, 0, "Self-play games encountered crashes!")
        self.assertEqual(wins + losses + draws, num_games, "Not all self-play games completed!")
        self.assertLess(elapsed / num_games, 120.0, "Game duration far too slow for 600s limit!")
        print("[PASS] Self-play match test passed with 0 crashes and 0 timeouts!")

if __name__ == "__main__":
    unittest.main()
