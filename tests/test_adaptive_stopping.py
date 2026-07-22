import math
import unittest

from src.env.adaptive_stopping import LowKLAndEntropyStagnationCallback


class FakeLogger:
    def __init__(self):
        self.name_to_value = {}

    def record(self, key, value, *args, **kwargs):
        self.name_to_value[key] = value


class FakeModel:
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        self._n_updates = 20
        self.logger = FakeLogger()


class AdaptiveStoppingTests(unittest.TestCase):
    def make_callback(self, **overrides):
        options = {
            "kl_threshold": 0.01,
            "entropy_trend_threshold": 0.002,
            "min_steps": 100,
            "patience": 3,
        }
        options.update(overrides)
        model = FakeModel()
        callback = LowKLAndEntropyStagnationCallback(**options)
        callback.init_callback(model)
        return model, callback

    def update(self, model, callback, *, steps, kl, entropy):
        model.num_timesteps += steps
        model._n_updates += 1
        model.logger.name_to_value["train/approx_kl"] = kl
        model.logger.name_to_value["train/entropy_loss"] = entropy
        return callback.on_step()

    def test_stops_only_after_minimum_budget_and_full_patience(self):
        model, callback = self.make_callback()
        self.assertTrue(self.update(model, callback, steps=50, kl=0.001, entropy=-1.000))
        self.assertEqual(0, callback.patience_counter)
        self.assertTrue(self.update(model, callback, steps=50, kl=0.001, entropy=-1.001))
        self.assertEqual(1, callback.patience_counter)
        self.assertTrue(self.update(model, callback, steps=20, kl=0.001, entropy=-1.0015))
        self.assertEqual(2, callback.patience_counter)
        self.assertFalse(self.update(model, callback, steps=20, kl=0.001, entropy=-1.0017))
        self.assertTrue(callback.triggered)
        self.assertEqual("low_kl_and_entropy_loss_stagnation", callback.stop_reason)
        self.assertEqual(140, callback.stop_run_steps)

    def test_low_kl_alone_does_not_stop_when_entropy_keeps_moving(self):
        model, callback = self.make_callback(min_steps=0)
        for entropy in (-1.0, -1.01, -1.02, -1.03, -1.04):
            self.assertTrue(self.update(model, callback, steps=20, kl=0.001, entropy=entropy))
        self.assertEqual(3, callback.patience_counter)
        self.assertAlmostEqual(-0.01, callback.last_entropy_trend)
        self.assertFalse(callback.triggered)

    def test_noisy_entropy_can_stop_when_rolling_trend_is_flat(self):
        model, callback = self.make_callback(min_steps=0, patience=5)
        for entropy in (-1.03, -0.97, -1.00, -0.97):
            self.assertTrue(
                self.update(model, callback, steps=20, kl=0.001, entropy=entropy)
            )
        self.assertFalse(
            self.update(model, callback, steps=20, kl=0.001, entropy=-1.03)
        )
        self.assertAlmostEqual(0.0, callback.last_entropy_trend)

    def test_high_kl_resets_patience(self):
        model, callback = self.make_callback(min_steps=0)
        self.update(model, callback, steps=20, kl=0.001, entropy=-1.0)
        self.update(model, callback, steps=20, kl=0.001, entropy=-1.001)
        self.assertEqual(2, callback.patience_counter)
        self.update(model, callback, steps=20, kl=0.02, entropy=-1.0015)
        self.assertEqual(0, callback.patience_counter)

    def test_each_ppo_update_is_counted_once(self):
        model, callback = self.make_callback(min_steps=0)
        self.update(model, callback, steps=20, kl=0.001, entropy=-1.0)
        self.update(model, callback, steps=20, kl=0.001, entropy=-1.001)
        self.assertEqual(2, callback.patience_counter)
        for _ in range(20):
            model.num_timesteps += 8
            self.assertTrue(callback.on_step())
        self.assertEqual(2, callback.patience_counter)

    def test_missing_or_non_finite_metric_cannot_stop(self):
        for bad_value in (None, math.nan, math.inf):
            model, callback = self.make_callback(min_steps=0, patience=2)
            self.update(model, callback, steps=20, kl=0.001, entropy=-1.0)
            self.assertTrue(
                self.update(model, callback, steps=20, kl=bad_value, entropy=-1.0)
            )
            self.assertEqual(0, callback.patience_counter)
            self.assertFalse(callback.triggered)

    def test_patience_cannot_be_one_quiet_update(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            LowKLAndEntropyStagnationCallback(
                kl_threshold=0.01,
                entropy_trend_threshold=0.002,
                min_steps=0,
                patience=1,
            )


if __name__ == "__main__":
    unittest.main()
