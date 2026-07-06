#!/bin/bash
source venv/bin/activate

# Der Champion (z.B. Dragapult Dusknoir, Deck 2)
CHAMPION_DECK="decks/deck_2.csv"
CHAMPION_MODEL="models/ppo_deck_2" # train_vs.py erwartet den Pfad ohne .zip für das lernende Modell

# Die Gegner aus dem aktuellen Kader
declare -a opponent_decks=("decks/deck_3.csv" "decks/deck_4.csv" "decks/deck_5.csv" "decks/deck_6.csv" "decks/deck_7.csv")
declare -a opponent_models=("models/ppo_deck_3.zip" "models/ppo_deck_4.zip" "models/ppo_deck_5.zip" "models/ppo_deck_6.zip" "models/ppo_deck_7.zip")

# Wie viele Trainingsschritte der Champion gegen jeden Gegner absolvieren soll
TIMESTEPS=15000

for i in "${!opponent_decks[@]}"; do
    opp_deck="${opponent_decks[$i]}"
    opp_model="${opponent_models[$i]}"
    
    # Überspringe falls ein Gegnerdeck nicht existiert (z.B. durch Auto-Tourney eliminiert)
    if [ ! -f "$opp_deck" ] || [ ! -f "$opp_model" ]; then
        continue
    fi
    
    echo "=========================================================="
    echo "🥋 Training Champion ($CHAMPION_DECK) vs Opponent ($opp_deck)"
    echo "=========================================================="
    
    python src/train_vs.py \
        --learning-deck "$CHAMPION_DECK" \
        --learning-model "$CHAMPION_MODEL" \
        --opponent-deck "$opp_deck" \
        --opponent-model "$opp_model" \
        --timesteps "$TIMESTEPS"
done

echo "=========================================================="
echo "🏆 Champion Training abgeschlossen! Das Modell wurde iterativ verbessert."
echo "=========================================================="
