import unittest

from src.cg.api import all_card_data
from src.data.limitless_deck_scraper import (
    CardPrintResolver,
    LimitlessCardEntry,
    parse_limitless_deck_html,
    resolve_deck_entries,
)


class LimitlessDeckScraperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolver = CardPrintResolver(all_card_data())

    def test_parses_print_metadata_and_basic_energy(self):
        page = """
        <div class="decklist-card" data-set="SSP" data-number="87" data-lang="en">
          <span class="card-count">1</span>
          <span class="card-name">Dedenne</span>
        </div>
        <div class="decklist-card" data-set="MEE" data-number="5"
             data-lang="en" data-basic-energy="5">
          <span class="card-count">2</span>
          <span class="card-name">Psychic Energy</span>
        </div>
        """
        entries = parse_limitless_deck_html(page)

        self.assertEqual(("SSP", "87", None), (
            entries[0].set_code,
            entries[0].collection_number,
            entries[0].basic_energy_id,
        ))
        self.assertEqual(5, entries[1].basic_energy_id)
        self.assertEqual([222, 5, 5], resolve_deck_entries(entries, self.resolver))

    def test_exact_print_resolves_duplicate_pokemon_names(self):
        expected = {
            ("Dedenne", "SSP", "087"): 222,
            ("Genesect", "SFA", "40"): 142,
            ("Shaymin", "DRI", "010"): 343,
            ("Dedenne", "POR", "29"): 1038,
            ("Genesect", "PFL", "8"): 785,
        }
        for (name, set_code, number), card_id in expected.items():
            with self.subTest(name=name, set_code=set_code, number=number):
                entry = LimitlessCardEntry(1, name, set_code, number, None)
                self.assertEqual(card_id, self.resolver.resolve(entry))

    def test_ambiguous_name_without_known_print_is_rejected(self):
        entry = LimitlessCardEntry(1, "Dedenne", None, None, None)

        with self.assertRaisesRegex(ValueError, "ambiguous engine IDs"):
            self.resolver.resolve(entry)

    def test_unique_name_can_fall_back_when_print_is_a_reprint(self):
        entry = LimitlessCardEntry(
            1,
            "Buddy-Buddy Poffin",
            "UNKNOWN",
            "999",
            None,
        )

        self.assertEqual(1086, self.resolver.resolve(entry))

    def test_unique_name_can_fall_back_when_local_print_number_differs(self):
        entry = LimitlessCardEntry(
            1,
            "Telepathic Psychic Energy",
            "POR",
            "88",
            None,
        )

        self.assertEqual(19, self.resolver.resolve(entry))


if __name__ == "__main__":
    unittest.main()
