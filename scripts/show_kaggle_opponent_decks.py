#!/usr/bin/env python3
"""Plot opponent deck archetypes from downloaded Kaggle replays.

Kaggle's privileged visualization state usually contains both complete decks.
When it is absent, visible card instances are deduplicated by replay serial and
unrevealed cards remain explicitly unknown instead of being guessed.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_ROOT = ROOT / "replays" / "kaggle"
DEFAULT_CARD_DATA = ROOT / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv"
DEFAULT_REPORT_DIR = ROOT / "reports" / "deck_analysis"
DECK_SIZE = 60


@dataclass(frozen=True)
class CardInfo:
    card_id: int
    name: str
    expansion: str
    collection_number: str
    kind: str
    stage: str
    hp: int


@dataclass(frozen=True)
class ReplayDeck:
    episode_id: str
    opponent_name: str
    opponent_index: int
    replay_path: Path
    cards: Counter[int]
    result: str

    @property
    def observed_count(self) -> int:
        return sum(self.cards.values())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a pie chart showing the opponent deck archetype distribution "
            "for one downloaded Kaggle bot/submission."
        )
    )
    parser.add_argument(
        "submission_id",
        nargs="?",
        help="Kaggle submission ID (directory below replays/kaggle)",
    )
    parser.add_argument(
        "--opponent",
        help="Only show opponents whose name contains this text (case-insensitive)",
    )
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=DEFAULT_REPLAY_ROOT,
        help=f"Replay root (default: {DEFAULT_REPLAY_ROOT})",
    )
    parser.add_argument(
        "--card-data",
        type=Path,
        default=DEFAULT_CARD_DATA,
        help=f"EN_Card_Data.csv path (default: {DEFAULT_CARD_DATA})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of creating a pie chart",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print the individual reconstructed deck lists instead of a pie chart",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "SVG output path "
            "(default: reports/deck_analysis/opponent-decks-SUBMISSION_ID.svg)"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=9,
        help="Maximum number of pie slices including Other (default: 9)",
    )
    return parser.parse_args(argv)


def load_card_data(path: Path) -> dict[int, CardInfo]:
    cards: dict[int, CardInfo] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                card_id = int(row["Card ID"])
            except (KeyError, TypeError, ValueError):
                continue
            stage = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
            try:
                hp = int(row.get("HP", ""))
            except (TypeError, ValueError):
                hp = 0
            cards[card_id] = CardInfo(
                card_id=card_id,
                name=row.get("Card Name", "") or f"Card {card_id}",
                expansion=row.get("Expansion", ""),
                collection_number=row.get("Collection No.", ""),
                kind=card_kind(stage),
                stage=stage,
                hp=hp,
            )
    return cards


def card_kind(stage: str) -> str:
    lowered = stage.casefold()
    if "energy" in lowered:
        return "Energy"
    if lowered.endswith("pokémon") or lowered.endswith("pokemon"):
        return "Pokémon"
    return "Trainer"


def iter_card_instances(value: Any, player_index: int) -> Iterable[tuple[int, int]]:
    """Yield ``(serial, card_id)`` pairs recursively from replay structures."""
    if isinstance(value, dict):
        if {"id", "serial", "playerIndex"}.issubset(value):
            try:
                if int(value["playerIndex"]) == player_index:
                    yield int(value["serial"]), int(value["id"])
            except (TypeError, ValueError):
                pass

        # Logs encode cards as cardId/serial rather than id/serial.
        if {"cardId", "serial", "playerIndex"}.issubset(value):
            try:
                if int(value["playerIndex"]) == player_index:
                    yield int(value["serial"]), int(value["cardId"])
            except (TypeError, ValueError):
                pass

        for child in value.values():
            yield from iter_card_instances(child, player_index)
    elif isinstance(value, list):
        for child in value:
            yield from iter_card_instances(child, player_index)


def reconstruct_cards(steps: Any, player_index: int) -> Counter[int]:
    """Return observed card counts for one player, deduplicated by serial."""
    by_serial: dict[int, int] = {}
    if not isinstance(steps, list):
        return Counter()

    for step in steps:
        if not isinstance(step, list):
            continue
        for agent_state in step:
            if not isinstance(agent_state, dict):
                continue
            # The first Kaggle step can contain a privileged ``visualize`` state
            # with both complete decks. Later steps still provide public/private
            # observations, so scan the complete state and deduplicate by serial.
            for serial, card_id in iter_card_instances(agent_state, player_index):
                previous = by_serial.setdefault(serial, card_id)
                if previous != card_id:
                    # A replay serial identifies one physical card. Keep the first
                    # valid mapping if malformed input later reuses the serial.
                    continue
    return Counter(by_serial.values())


def replay_result(data: dict[str, Any], opponent_index: int) -> str:
    rewards = data.get("rewards")
    if not isinstance(rewards, list) or len(rewards) <= opponent_index:
        return "unknown"
    try:
        opponent_reward = float(rewards[opponent_index])
        other_rewards = [float(value) for i, value in enumerate(rewards) if i != opponent_index]
    except (TypeError, ValueError):
        return "unknown"
    if not other_rewards or opponent_reward > max(other_rewards):
        return "win"
    if opponent_reward == max(other_rewards):
        return "draw"
    return "loss"


def read_replay(path: Path, own_name: str | None = None) -> ReplayDeck | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return None

    info = data.get("info", {})
    names = info.get("TeamNames") or [agent.get("Name", "Unknown") for agent in info.get("Agents", [])]
    if not isinstance(names, list) or len(names) != 2:
        print(f"warning: no two player names in {path}", file=sys.stderr)
        return None

    own_index = identify_own_index(names, own_name)
    opponent_index = 1 - own_index
    return ReplayDeck(
        episode_id=str(info.get("EpisodeId") or path.stem.removeprefix("episode-").removesuffix("-replay")),
        opponent_name=str(names[opponent_index]),
        opponent_index=opponent_index,
        replay_path=path,
        cards=reconstruct_cards(data.get("steps"), opponent_index),
        result=replay_result(data, opponent_index),
    )


def identify_own_index(names: list[Any], own_name: str | None) -> int:
    if own_name:
        matches = [i for i, name in enumerate(names) if str(name) == own_name]
        if len(matches) == 1:
            return matches[0]

    # Downloaded replay directories belong to this project's Kaggle account.
    # This also covers self-play, where either seat is equivalent as an opponent.
    matches = [i for i, name in enumerate(names) if str(name) == "Sven Natterer"]
    return matches[0] if matches else 0


def load_submission_name(replay_root: Path, submission_id: str) -> str | None:
    metadata_path = replay_root / "submissions.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        entry = metadata.get("submissions", {}).get(str(submission_id), {})
        owner_name = entry.get("owner_name")
        return str(owner_name) if owner_name else None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def discover_replays(replay_root: Path, submission_id: str) -> list[Path]:
    directory = replay_root / str(submission_id)
    return sorted(directory.glob("*-replay.json")) if directory.is_dir() else []


def list_submissions(replay_root: Path) -> int:
    descriptions: dict[str, str] = {}
    try:
        metadata = json.loads((replay_root / "submissions.json").read_text(encoding="utf-8"))
        descriptions = {
            str(key): str(value.get("description", ""))
            for key, value in metadata.get("submissions", {}).items()
        }
    except (OSError, json.JSONDecodeError, AttributeError):
        pass

    rows = []
    if replay_root.is_dir():
        for directory in replay_root.iterdir():
            if directory.is_dir() and directory.name.isdigit():
                rows.append((directory.name, len(list(directory.glob("*-replay.json"))), descriptions.get(directory.name, "")))
    if not rows:
        print(f"No downloaded submissions found below {replay_root}", file=sys.stderr)
        return 1

    print("Available Kaggle submissions:\n")
    print(f"{'Submission':<12} {'Replays':>7}  Description")
    print(f"{'-' * 12} {'-' * 7}  {'-' * 40}")
    for submission_id, count, description in sorted(rows, key=lambda row: int(row[0])):
        print(f"{submission_id:<12} {count:>7}  {description}")
    print("\nUsage: python scripts/show_kaggle_opponent_decks.py SUBMISSION_ID")
    print("The default output is an SVG pie chart in the current directory.")
    return 0


def card_sort_key(item: tuple[int, int], card_data: dict[int, CardInfo]) -> tuple[int, str, int]:
    card_id, _count = item
    card = card_data.get(card_id)
    kind_order = {"Pokémon": 0, "Trainer": 1, "Energy": 2}
    return (kind_order.get(card.kind if card else "", 3), (card.name if card else "").casefold(), card_id)


def pokemon_strength(card: CardInfo) -> int:
    stage = card.stage.casefold()
    if stage.startswith("stage 2"):
        stage_bonus = 200
    elif stage.startswith("stage 1"):
        stage_bonus = 100
    else:
        stage_bonus = 0
    return stage_bonus + card.hp


def infer_archetype(cards: Counter[int], card_data: dict[int, CardInfo]) -> str:
    """Name a deck after its strongest observed evolution/attacker."""
    candidates = [
        (card, count)
        for card_id, count in cards.items()
        if (card := card_data.get(card_id)) is not None and card.kind == "Pokémon"
    ]
    if not candidates:
        return "Unknown"
    card, _count = max(
        candidates,
        key=lambda item: (
            pokemon_strength(item[0]),
            item[1],
            " ex" in item[0].name.casefold(),
            item[0].name.casefold(),
        ),
    )
    return card.name


def archetype_distribution(
    decks: list[ReplayDeck], card_data: dict[int, CardInfo]
) -> Counter[str]:
    return Counter(infer_archetype(deck.cards, card_data) for deck in decks)


def collapse_distribution(distribution: Counter[str], top: int) -> list[tuple[str, int]]:
    rows = sorted(distribution.items(), key=lambda item: (-item[1], item[0].casefold()))
    if len(rows) <= top:
        return rows
    kept = rows[: max(1, top - 1)]
    return kept + [("Other", sum(count for _name, count in rows[len(kept) :]))]


def write_pie_chart(
    path: Path,
    distribution: Counter[str],
    submission_id: str,
    top: int = 9,
) -> None:
    if top < 2:
        raise ValueError("--top must be at least 2")
    rows = collapse_distribution(distribution, top)
    total = sum(distribution.values())
    if total <= 0:
        raise ValueError("cannot plot an empty distribution")

    width, height = 1000, 650
    cx, cy, radius = 300, 340, 220
    colors = [
        "#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed",
        "#0891b2", "#db2777", "#65a30d", "#64748b",
    ]
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">Opponent deck archetypes for Kaggle submission {html.escape(submission_id)}</title>',
        f'<desc id="desc">Pie chart of {total} matches grouped by inferred opponent deck archetype.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033}'
        '.title{font-size:26px;font-weight:600}.sub{font-size:15px;fill:#64748b}'
        '.legend{font-size:16px}.value{font-weight:600}</style>',
        f'<text x="40" y="48" class="title">Opponent deck archetypes · {html.escape(submission_id)}</text>',
        f'<text x="40" y="76" class="sub">{total} Kaggle matches · strongest Pokémon per reconstructed deck</text>',
    ]

    start = -math.pi / 2
    for index, (name, count) in enumerate(rows):
        fraction = count / total
        end = start + 2 * math.pi * fraction
        color = colors[index % len(colors)]
        if fraction >= 0.999999:
            pieces.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{color}"/>')
        else:
            x1, y1 = cx + radius * math.cos(start), cy + radius * math.sin(start)
            x2, y2 = cx + radius * math.cos(end), cy + radius * math.sin(end)
            large_arc = 1 if fraction > 0.5 else 0
            pieces.append(
                f'<path d="M {cx} {cy} L {x1:.3f} {y1:.3f} '
                f'A {radius} {radius} 0 {large_arc} 1 {x2:.3f} {y2:.3f} Z" '
                f'fill="{color}" stroke="#ffffff" stroke-width="3"/>'
            )
        start = end

        legend_y = 145 + index * 52
        percentage = 100 * count / total
        pieces.extend(
            [
                f'<rect x="590" y="{legend_y - 15}" width="18" height="18" rx="4" fill="{color}"/>',
                f'<text x="622" y="{legend_y}" class="legend">{html.escape(name)}</text>',
                f'<text x="940" y="{legend_y}" text-anchor="end" class="legend value">'
                f'{percentage:.1f}% <tspan fill="#64748b">({count})</tspan></text>',
            ]
        )

    pieces.extend(
        [
            f'<circle cx="{cx}" cy="{cy}" r="92" fill="#ffffff"/>',
            f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" class="title">{total}</text>',
            f'<text x="{cx}" y="{cy + 25}" text-anchor="middle" class="sub">matches</text>',
            '</svg>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def print_distribution(distribution: Counter[str]) -> None:
    total = sum(distribution.values())
    for name, count in sorted(distribution.items(), key=lambda item: (-item[1], item[0].casefold())):
        print(f"{name:<32} {count:>3}  {100 * count / total:>5.1f}%")


def print_text(decks: list[ReplayDeck], card_data: dict[int, CardInfo], submission_id: str) -> None:
    print(f"Opponent decks for Kaggle submission {submission_id} ({len(decks)} matches)\n")
    for deck in decks:
        unknown = max(0, DECK_SIZE - deck.observed_count)
        print(f"=== {deck.opponent_name} | episode {deck.episode_id} | opponent {deck.result} ===")
        print(f"Observed: {deck.observed_count}/{DECK_SIZE} cards | Unknown: {unknown}")
        print(f"{'Qty':>3}  {'Type':<8} {'Card':<34} {'Set':<8} {'ID':>5}")
        print(f"{'---':>3}  {'-' * 8} {'-' * 34} {'-' * 8} {'-' * 5}")
        for card_id, count in sorted(deck.cards.items(), key=lambda item: card_sort_key(item, card_data)):
            card = card_data.get(card_id)
            name = card.name if card else f"Unknown card {card_id}"
            kind = card.kind if card else "Unknown"
            card_set = f"{card.expansion} {card.collection_number}".strip() if card else ""
            print(f"{count:>3}  {kind:<8} {name[:34]:<34} {card_set[:8]:<8} {card_id:>5}")
        if unknown:
            print(f"{unknown:>3}  {'Unknown':<8} {'Not revealed in this replay':<34}")
        print()


def json_payload(decks: list[ReplayDeck], card_data: dict[int, CardInfo], submission_id: str) -> dict[str, Any]:
    matches = []
    for deck in decks:
        cards = []
        for card_id, count in sorted(deck.cards.items(), key=lambda item: card_sort_key(item, card_data)):
            card = card_data.get(card_id)
            cards.append(
                {
                    "count": count,
                    "card_id": card_id,
                    "name": card.name if card else None,
                    "type": card.kind if card else None,
                    "expansion": card.expansion if card else None,
                    "collection_number": card.collection_number if card else None,
                }
            )
        matches.append(
            {
                "episode_id": deck.episode_id,
                "opponent": deck.opponent_name,
                "archetype": infer_archetype(deck.cards, card_data),
                "opponent_result": deck.result,
                "observed_cards": deck.observed_count,
                "unknown_cards": max(0, DECK_SIZE - deck.observed_count),
                "cards": cards,
                "replay": str(deck.replay_path),
            }
        )
    distribution = archetype_distribution(decks, card_data)
    return {
        "submission_id": submission_id,
        "archetype_distribution": dict(distribution.most_common()),
        "matches": matches,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    replay_root = args.replay_root.expanduser().resolve()
    if not args.submission_id:
        return list_submissions(replay_root)

    replay_paths = discover_replays(replay_root, args.submission_id)
    if not replay_paths:
        print(
            f"No replay files found for submission {args.submission_id} below {replay_root}",
            file=sys.stderr,
        )
        return 1

    try:
        card_data = load_card_data(args.card_data.expanduser().resolve())
    except OSError as exc:
        print(f"Cannot read card data: {exc}", file=sys.stderr)
        return 1

    own_name = load_submission_name(replay_root, args.submission_id)
    decks = [deck for path in replay_paths if (deck := read_replay(path, own_name)) is not None]
    if args.opponent:
        query = args.opponent.casefold()
        decks = [deck for deck in decks if query in deck.opponent_name.casefold()]
    if not decks:
        suffix = f" matching {args.opponent!r}" if args.opponent else ""
        print(f"No opponent replays{suffix} found.", file=sys.stderr)
        return 1

    decks.sort(key=lambda deck: (deck.opponent_name.casefold(), deck.episode_id))
    if args.json:
        print(json.dumps(json_payload(decks, card_data, str(args.submission_id)), ensure_ascii=False, indent=2))
    elif args.details:
        print_text(decks, card_data, str(args.submission_id))
    else:
        distribution = archetype_distribution(decks, card_data)
        output = (
            args.output
            or DEFAULT_REPORT_DIR / f"opponent-decks-{args.submission_id}.svg"
        ).expanduser().resolve()
        try:
            write_pie_chart(output, distribution, str(args.submission_id), args.top)
        except (OSError, ValueError) as exc:
            print(f"Cannot create pie chart: {exc}", file=sys.stderr)
            return 1
        print_distribution(distribution)
        print(f"\nPie chart: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
