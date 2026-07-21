from pathlib import Path
from unittest.mock import patch

from tools.migrations.repair_compact_factory_scrape_bug import (
    REPAIR_TARGET_IDS,
    target_assignments,
)
from scripts.run_compact_opponent_factory import current_selection


def test_repair_targets_match_the_pre_fix_affected_targets():
    assert REPAIR_TARGET_IDS == (
        "bank_33",
        "bank_49",
        "bank_61",
        "bank_25",
        "bank_36",
        "bank_84",
        "bank_2",
        "bank_99",
    )


def test_target_assignments_preserve_round_robin_base_order():
    config = {
        "targets": [
            {"deck_id": "bank_33"},
            {"deck_id": "bank_38"},
            {"deck_id": "bank_49"},
            {"deck_id": "bank_61"},
        ]
    }
    bases = [{"id": "compact_c"}, {"id": "compact_a"}, {"id": "compact_b"}]

    assignments = target_assignments(config, bases)

    assert [(target["deck_id"], base["id"]) for target, base in assignments] == [
        ("bank_33", "compact_c"),
        ("bank_38", "compact_a"),
        ("bank_49", "compact_b"),
        ("bank_61", "compact_c"),
    ]


def test_selection_without_input_hashes_is_stale(tmp_path: Path):
    selection = tmp_path / "selection.json"
    selection.write_text(
        '{"selected_base_ids":["compact_c","compact_a","compact_b"],'
        '"base_fingerprints":{"same":true}}',
        encoding="utf-8",
    )

    with (
        patch(
            "scripts.run_compact_opponent_factory.BASE_SELECTION",
            selection,
        ),
        patch(
            "scripts.run_compact_opponent_factory.model_fingerprints",
            return_value={"same": True},
        ),
        patch(
            "scripts.run_compact_opponent_factory.selection_input_fingerprints",
            return_value={"deck.csv": {"sha256": "new"}},
        ),
    ):
        assert current_selection({}) is None
