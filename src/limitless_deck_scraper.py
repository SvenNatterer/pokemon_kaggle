from __future__ import annotations

import csv
import html
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CARD_DATA_PATH = ROOT / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv"

DECK_CARD_PATTERN = re.compile(
    r'<div class="decklist-card"(?P<attrs>[^>]*)>'
    r'.*?<span class="card-count">(?P<count>\d+)</span>'
    r'\s*<span class="card-name">(?P<name>[^<]+)</span>',
    re.DOTALL,
)
ATTRIBUTE_PATTERN = re.compile(r'data-([a-z-]+)="([^"]*)"')


@dataclass(frozen=True)
class LimitlessCardEntry:
    count: int
    name: str
    set_code: str | None
    collection_number: str | None
    basic_energy_id: int | None


def normalize_name(name: str) -> str:
    normalized = (
        html.unescape(name)
        .replace("’", "'")
        .replace("\xa0", " ")
        .strip()
        .casefold()
    )
    aliases = {
        "telepathic psychic energy": "telepath psychic energy",
    }
    return aliases.get(normalized, normalized)


def normalize_collection_number(number: str | int | None) -> str:
    if number is None:
        return ""
    normalized = str(number).strip().upper()
    match = re.fullmatch(r"0*(\d+)([A-Z]*)", normalized)
    if match:
        return f"{int(match.group(1))}{match.group(2)}"
    return normalized


def parse_limitless_deck_html(page_html: str) -> list[LimitlessCardEntry]:
    entries: list[LimitlessCardEntry] = []
    for match in DECK_CARD_PATTERN.finditer(page_html):
        attributes = dict(ATTRIBUTE_PATTERN.findall(match.group("attrs")))
        basic_energy = attributes.get("basic-energy")
        entries.append(
            LimitlessCardEntry(
                count=int(match.group("count")),
                name=html.unescape(match.group("name")).strip(),
                set_code=(attributes.get("set") or "").strip().upper() or None,
                collection_number=attributes.get("number"),
                basic_energy_id=int(basic_energy) if basic_energy else None,
            )
        )
    return entries


class CardPrintResolver:
    def __init__(
        self,
        cards: Iterable[object],
        card_data_path: Path = DEFAULT_CARD_DATA_PATH,
    ) -> None:
        self.cards_by_id = {int(card.cardId): card for card in cards}
        self.ids_by_name: dict[str, set[int]] = defaultdict(set)
        for card_id, card in self.cards_by_id.items():
            self.ids_by_name[normalize_name(card.name)].add(card_id)

        self.ids_by_print: dict[tuple[str, str], set[int]] = defaultdict(set)
        with Path(card_data_path).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                card_id = int(row["Card ID"])
                print_key = (
                    row["Expansion"].strip().upper(),
                    normalize_collection_number(row["Collection No."]),
                )
                self.ids_by_print[print_key].add(card_id)

    def resolve(self, entry: LimitlessCardEntry) -> int:
        if entry.basic_energy_id is not None:
            card = self.cards_by_id.get(entry.basic_energy_id)
            if card is None:
                raise ValueError(
                    f"{entry.name}: unknown Basic Energy ID {entry.basic_energy_id}"
                )
            return entry.basic_energy_id

        if entry.set_code and entry.collection_number:
            print_key = (
                entry.set_code.upper(),
                normalize_collection_number(entry.collection_number),
            )
            print_ids = self.ids_by_print.get(print_key, set())
            if len(print_ids) == 1:
                card_id = next(iter(print_ids))
                card_name = normalize_name(self.cards_by_id[card_id].name)
                requested_name = normalize_name(entry.name)
                if card_name == requested_name:
                    return card_id
            if len(print_ids) > 1:
                matching_name_ids = {
                    card_id
                    for card_id in print_ids
                    if normalize_name(self.cards_by_id[card_id].name)
                    == normalize_name(entry.name)
                }
                if len(matching_name_ids) == 1:
                    return next(iter(matching_name_ids))

        name_ids = self.ids_by_name.get(normalize_name(entry.name), set())
        if len(name_ids) == 1:
            return next(iter(name_ids))
        if not name_ids:
            raise ValueError(f"{entry.name}: card is missing from the engine database")
        raise ValueError(
            f"{entry.name}: ambiguous engine IDs {sorted(name_ids)}; "
            "Limitless set and collection number did not resolve a print"
        )


def resolve_deck_entries(
    entries: Iterable[LimitlessCardEntry],
    resolver: CardPrintResolver,
) -> list[int]:
    deck: list[int] = []
    for entry in entries:
        card_id = resolver.resolve(entry)
        deck.extend([card_id] * entry.count)
    return deck
