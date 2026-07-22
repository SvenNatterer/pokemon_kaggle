from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RemovedArchitectureTests(unittest.TestCase):
    def test_train_has_no_periodic_checkpoint_interface(self):
        source = (ROOT / "src" / "training" / "train.py").read_text(encoding="utf-8")
        for forbidden in ("--checkpoint-interval", "--keep-checkpoints", "RotatingCheckpointCallback", "latest_checkpoint"):
            self.assertNotIn(forbidden, source)

    def test_training_shell_scripts_do_not_use_checkpoint_flags(self):
        for path in (ROOT / "scripts").glob("*.sh"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("--checkpoint-interval", source, path.as_posix())
            self.assertNotIn("--keep-checkpoints", source, path.as_posix())

    def test_arena_worker_does_not_exist(self):
        self.assertFalse((ROOT / "src" / "arena").exists())


if __name__ == "__main__":
    unittest.main()
