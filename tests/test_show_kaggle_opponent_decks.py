from scripts import show_kaggle_opponent_decks as MODULE


def test_reconstruct_cards_deduplicates_serials_and_nested_cards():
    steps = [
        [
            {
                "visualize": [
                    {
                        "current": {
                            "players": [
                                {},
                                {"deck": [{"id": 40, "serial": 65, "playerIndex": 1}]},
                            ]
                        }
                    }
                ],
                "observation": {
                    "current": {
                        "players": [
                            {},
                            {
                                "hand": [{"id": 10, "serial": 61, "playerIndex": 1}],
                                "active": [
                                    {
                                        "id": 20,
                                        "serial": 62,
                                        "playerIndex": 1,
                                        "energyCards": [
                                            {"id": 3, "serial": 63, "playerIndex": 1}
                                        ],
                                    }
                                ],
                            },
                        ]
                    }
                }
            },
            {
                "observation": {
                    "current": {
                        "players": [
                            {},
                            {"hand": [{"id": 10, "serial": 61, "playerIndex": 1}]},
                        ]
                    },
                    "logs": [
                        {"cardId": 30, "serial": 64, "playerIndex": 1},
                        {"cardId": 99, "serial": 1, "playerIndex": 0},
                    ],
                }
            },
        ]
    ]

    assert MODULE.reconstruct_cards(steps, 1) == {10: 1, 20: 1, 3: 1, 30: 1, 40: 1}


def test_read_replay_selects_the_non_owner_as_opponent(tmp_path):
    replay = tmp_path / "episode-123-replay.json"
    replay.write_text(
        """{
          "info": {"EpisodeId": 123, "TeamNames": ["Opponent", "Sven Natterer"]},
          "rewards": [1, -1],
          "steps": [[
            {"observation": {"current": {"players": [
              {"hand": [{"id": 42, "serial": 1, "playerIndex": 0}]}, {}
            ]}}},
            {"observation": null}
          ]]
        }""",
        encoding="utf-8",
    )

    deck = MODULE.read_replay(replay)

    assert deck is not None
    assert deck.opponent_name == "Opponent"
    assert deck.opponent_index == 0
    assert deck.cards == {42: 1}
    assert deck.result == "win"


def test_card_kind_recognizes_all_three_groups():
    assert MODULE.card_kind("Basic Pokémon") == "Pokémon"
    assert MODULE.card_kind("Pokémon Tool") == "Trainer"
    assert MODULE.card_kind("Special Energy") == "Energy"
    assert MODULE.card_kind("Item") == "Trainer"


def test_infers_archetypes_and_writes_dependency_free_svg(tmp_path):
    card_data = {
        1: MODULE.CardInfo(1, "Riolu", "MEG", "76", "Pokémon", "Basic Pokémon", 80),
        2: MODULE.CardInfo(2, "Mega Lucario ex", "MEG", "77", "Pokémon", "Stage 1 Pokémon", 340),
        3: MODULE.CardInfo(3, "Ultra Ball", "SVI", "196", "Trainer", "Item", 0),
    }
    assert MODULE.infer_archetype({1: 4, 2: 2, 3: 4}, card_data) == "Mega Lucario ex"

    output = tmp_path / "decks.svg"
    MODULE.write_pie_chart(
        output,
        {"Mega Lucario ex": 4, "Dragapult ex": 3, "Other deck": 3},
        "12345",
    )
    svg = output.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "Mega Lucario ex" in svg
    assert "40.0%" in svg
