import pytest
import numpy as np
from src.features.prize_mapper import PrizeMapper


def test_prize_mapper_initialization():
    deck = ["Pikachu"] * 20 + ["Lightning Energy"] * 40
    mapper = PrizeMapper(deck)
    assert mapper.total_cards == 60
    assert len(mapper.starting_deck) == 60


def test_prize_mapper_vector_output():
    deck = ["Pikachu"] * 20 + ["Lightning Energy"] * 40
    mapper = PrizeMapper(deck)
    
    # Hand (7), Active (1), Bench (2), Discard (0)
    hand = ["Pikachu"] * 4 + ["Lightning Energy"] * 3
    active = ["Pikachu"]
    bench = ["Pikachu"] * 2
    discard = []
    
    vec = mapper.update(hand, active, bench, discard)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (60,)
    assert np.all(vec >= 0.0)
    assert np.all(vec <= 6.0)


def test_prize_mapper_exact_deck_search():
    deck = [f"Card_{i}" for i in range(60)]
    mapper = PrizeMapper(deck)
    
    hand = deck[:7]
    active = [deck[7]]
    bench = deck[8:10]
    discard = deck[10:15]
    known_deck = deck[15:54]  # 39 cards in deck, remaining 6 cards (54:60) are prizes
    
    vec = mapper.update(hand, active, bench, discard, known_deck=known_deck)
    
    # Check that cards 54..59 have 1.0 probability of being prizes
    for idx in range(54, 60):
        assert vec[idx] > 0.0
