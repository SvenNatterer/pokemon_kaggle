import json
from pathlib import Path
import tempfile
import unittest

from src.training.train import (
    DEFAULT_QUEUE_PATH,
    add_to_queue,
    build_arg_parser,
    parse_args_with_config,
    parse_queue_item,
    pop_next_queue_item,
)


class TrainingQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.queue_path = str(Path(self.temp_dir.name) / "test_queue.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_and_pop_queue_items(self):
        add_to_queue(self.queue_path, "configs/experiments/exp_001_lookahead_distill.yaml")
        add_to_queue(self.queue_path, '{"deck": "decks/deck_18.csv", "timesteps": 5000}')

        item1 = pop_next_queue_item(self.queue_path)
        self.assertEqual(item1, "configs/experiments/exp_001_lookahead_distill.yaml")

        item2 = pop_next_queue_item(self.queue_path)
        self.assertEqual(item2, {"deck": "decks/deck_18.csv", "timesteps": 5000})

        item3 = pop_next_queue_item(self.queue_path)
        self.assertIsNone(item3)

    def test_parse_queue_item_dict(self):
        parser = build_arg_parser()
        item = {"deck": "decks/deck_18.csv", "timesteps": 250000, "lr": 3e-4}
        args = parse_queue_item(item, parser)

        self.assertEqual(args.deck, "decks/deck_18.csv")
        self.assertEqual(args.timesteps, 250000)
        self.assertAlmostEqual(args.lr, 3e-4)

    def test_parse_queue_item_yaml_config(self):
        parser = build_arg_parser()
        yaml_path = "configs/train_config.yaml"
        args = parse_queue_item(yaml_path, parser)

        self.assertAlmostEqual(args.lr, 0.0001)
        self.assertEqual(args.n_steps, 512)
        self.assertEqual(args.batch_size, 1024)

    def test_cli_queue_flags(self):
        parser = build_arg_parser()
        args1 = parser.parse_args(["--queue"])
        self.assertEqual(args1.queue, DEFAULT_QUEUE_PATH)

        args2 = parser.parse_args(["--cue", "custom_queue.json"])
        self.assertEqual(args2.queue, "custom_queue.json")

        args3 = parser.parse_args(["--add-to-cue", "exp_002.yaml"])
        self.assertEqual(args3.add_to_queue, "exp_002.yaml")


if __name__ == "__main__":
    unittest.main()
