import pytest
import numpy as np
from src.models.deck_archetypes import ArchetypePredictor, detect_deck_archetypes, TR_MEWTWO_EX_CARD_ID, TR_SPIDOPS_CARD_ID


def test_archetype_predictor_initialization():
    predictor = ArchetypePredictor()
    assert len(predictor.DEFAULT_ARCHETYPES) == 8


def test_archetype_predictor_prediction():
    predictor = ArchetypePredictor()
    predictor.register_archetype("starmie_ex", ["starmie_ex", "water_energy", "misty"])
    revealed = ["starmie_ex", "water_energy", "misty"]
    probs = predictor.predict(revealed)
    
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (8,)
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[0] > probs[1]


def test_detect_deck_archetypes():
    # Empty deck
    assert detect_deck_archetypes([]) == set()
    
    # Deck with only Mewtwo ex
    assert detect_deck_archetypes([TR_MEWTWO_EX_CARD_ID, 1, 2]) == set()
    
    # Deck with both Mewtwo ex AND Spidops
    archetypes = detect_deck_archetypes([TR_MEWTWO_EX_CARD_ID, TR_SPIDOPS_CARD_ID, 1, 2])
    assert "TR38_MEWTWO" in archetypes
