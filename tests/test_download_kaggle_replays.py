import tempfile
import json
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from scripts.download_kaggle_replays import (
    discover_submission_ids,
    run,
    terminal_episodes,
)


def episode(episode_id, state="COMPLETED", submission_id=10, agent_index=0):
    return SimpleNamespace(
        id=episode_id,
        state=state,
        end_time=datetime(2026, 7, episode_id, tzinfo=timezone.utc),
        agents=[SimpleNamespace(submission_id=submission_id, index=agent_index)],
    )


class FakeApi:
    def __init__(self):
        self.replays = []
        self.logs = []

    def competition_submissions(self, competition, page_number, page_size):
        del competition, page_size
        return {
            1: [
                SimpleNamespace(ref=10, description="First bot", date=None, status="complete", file_name="a.tar.gz"),
                SimpleNamespace(ref=20, description="Second bot", date=None, status="complete", file_name="b.tar.gz"),
            ],
            2: [],
        }[page_number]

    def competition_list_episodes(self, submission_id):
        return [episode(2, submission_id=submission_id), episode(1, submission_id=submission_id)]

    def competition_episode_replay(self, episode_id, path, quiet):
        del quiet
        self.replays.append((episode_id, path))
        Path(path, f"episode-{episode_id}-replay.json").write_text("{}")

    def competition_episode_agent_logs(self, episode_id, agent_index, path, quiet):
        del quiet
        self.logs.append((episode_id, agent_index, path))
        Path(path, f"episode-{episode_id}-agent-{agent_index}-logs.json").write_text("{}")


class DownloadKaggleReplaysTests(unittest.TestCase):
    def test_discovers_unique_submission_ids(self):
        self.assertEqual(discover_submission_ids(FakeApi(), "competition", 100), [10, 20])

    def test_keeps_terminal_episodes_newest_first(self):
        values = [episode(1), episode(3, "CREATED"), episode(2, "ERRORED")]
        self.assertEqual([value.id for value in terminal_episodes(values, 1)], [2])

    def test_metadata_only_writes_descriptions_without_downloading_replays(self):
        api = FakeApi()
        with tempfile.TemporaryDirectory() as temp_dir:
            args = Namespace(
                competition="competition",
                submission_id=[],
                output=Path(temp_dir),
                limit=None,
                include_logs=False,
                force=False,
                dry_run=False,
                metadata_only=True,
                page_size=100,
            )
            stats = run(args, api)
            metadata = json.loads(Path(temp_dir, "submissions.json").read_text())

        self.assertEqual((stats.downloaded, stats.skipped, stats.failed), (0, 0, 0))
        self.assertEqual(api.replays, [])
        self.assertEqual(metadata["submissions"]["20"]["description"], "Second bot")

    def test_downloads_replay_and_own_log_and_skips_existing_files(self):
        api = FakeApi()
        with tempfile.TemporaryDirectory() as temp_dir:
            args = Namespace(
                competition="competition",
                submission_id=[10],
                output=Path(temp_dir),
                limit=None,
                include_logs=True,
                force=False,
                dry_run=False,
                metadata_only=False,
                page_size=100,
            )
            first = run(args, api)
            second = run(args, api)
            metadata = json.loads(Path(temp_dir, "submissions.json").read_text())

        self.assertEqual((first.downloaded, first.skipped, first.failed), (4, 0, 0))
        self.assertEqual((second.downloaded, second.skipped, second.failed), (0, 4, 0))
        self.assertEqual(len(api.replays), 2)
        self.assertEqual(len(api.logs), 2)
        self.assertEqual(metadata["submissions"]["10"]["description"], "First bot")


if __name__ == "__main__":
    unittest.main()
