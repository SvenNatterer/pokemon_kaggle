from scripts.freeze_compact_opponent_factory import (
    build_manifests,
    selected_bases_from_summary,
    target_assignments,
)


def test_target_assignments_preserve_the_validated_base_order():
    selected = selected_bases_from_summary(
        {"selected_base_ids": ["compact_c", "compact_a", "compact_b"]}
    )
    config = {
        "targets": [
            {"deck_id": "bank_33", "split": "training"},
            {"deck_id": "bank_38", "split": "training"},
            {"deck_id": "bank_49", "split": "training"},
            {"deck_id": "bank_61", "split": "training"},
        ]
    }

    assignments = target_assignments(config, selected)

    assert [(target["deck_id"], base["id"]) for target, base in assignments] == [
        ("bank_33", "compact_c"),
        ("bank_38", "compact_a"),
        ("bank_49", "compact_b"),
        ("bank_61", "compact_c"),
    ]


def test_manifests_keep_training_validation_and_holdout_disjoint():
    def entry(deck_id):
        return {
            "label": f"bot_{deck_id}",
            "deck_id": deck_id,
            "deck_path": f"decks/deck_bank/{deck_id}.csv",
            "model_path": f"models/{deck_id}.zip",
        }

    entries = {
        "training": [entry("bank_33"), entry("bank_38")],
        "validation": [entry("bank_11"), entry("bank_25")],
        "holdout": [entry("bank_99"), entry("bank_3")],
    }

    manifests = build_manifests(entries)

    training = manifests["training_pool_v6.json"]
    validation = manifests["validation_opponents_v6.json"]["opponents"]
    holdout = manifests["final_holdout_opponents_v6.json"]["opponents"]
    assert {row["deck"] for row in training}.isdisjoint(
        {row["deck_path"] for row in validation + holdout}
    )
    assert {row["deck_id"] for row in validation}.isdisjoint(
        {row["deck_id"] for row in holdout}
    )
    assert all(row["weight"] == 1.0 for row in training)
