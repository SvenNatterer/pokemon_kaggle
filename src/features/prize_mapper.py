import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Sequence


class PrizeMapper:
    """
    Deduces the probability and count of cards trapped in the player's 6 Prize Cards.
    
    Formula:
        Prizes = Starting Deck - (Hand + Active + Bench + Discard + Known Deck)
    
    When a deck search occurs (e.g., Nest Ball, Ultra Ball), the remaining deck contents
    become 100% known, making the Prize Card distribution completely deterministic.
    """

    def __init__(self, starting_deck: Sequence[str]):
        self.starting_deck = list(starting_deck)
        self.total_cards = len(self.starting_deck)
        self.initial_deck_counts = Counter(self.starting_deck)
        self.deck_thinned_exact = False
        self.exact_prize_counts: Optional[Counter] = None

    def reset(self, starting_deck: Optional[Sequence[str]] = None):
        if starting_deck is not None:
            self.starting_deck = list(starting_deck)
            self.total_cards = len(self.starting_deck)
            self.initial_deck_counts = Counter(self.starting_deck)
        self.deck_thinned_exact = False
        self.exact_prize_counts = None

    def update(
        self,
        hand: Sequence[str],
        active: Sequence[str],
        bench: Sequence[str],
        discard: Sequence[str],
        known_deck: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Updates card location tracking and returns a normalized 60-element prize probability vector.
        """
        visible_counts = Counter(list(hand) + list(active) + list(bench) + list(discard))
        
        if known_deck is not None:
            visible_counts.update(known_deck)
            self.deck_thinned_exact = True

        remaining_unaccounted = Counter()
        for card, initial_count in self.initial_deck_counts.items():
            seen = visible_counts.get(card, 0)
            unseen = max(0, initial_count - seen)
            if unseen > 0:
                remaining_unaccounted[card] = unseen

        # Vectorize based on 60 starting deck slots
        vector = np.zeros(60, dtype=np.float32)
        total_unseen = sum(remaining_unaccounted.values())
        
        if total_unseen > 0:
            for idx, card in enumerate(self.starting_deck):
                if card in remaining_unaccounted:
                    # Probability of card slot being in Prize Cards
                    # Ratio of remaining unseen instances over total unaccounted
                    unseen_count = remaining_unaccounted[card]
                    vector[idx] = min(1.0, float(unseen_count) / max(1.0, float(total_unseen)) * 6.0)

        return vector

    def get_summary(self) -> Dict[str, float]:
        """Returns dict mapping card names to estimated prize probability."""
        if not self.starting_deck:
            return {}
        vec = self.update([], [], [], [])
        return {card: float(vec[idx]) for idx, card in enumerate(self.starting_deck)}
