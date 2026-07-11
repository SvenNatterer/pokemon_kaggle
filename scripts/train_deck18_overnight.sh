#!/usr/bin/env bash
# train_deck18_overnight.sh – Overnight-Training für deck_bank_18
# ================================================================
# Strategie: Das Modell stagnierte, weil Rewards und Perspektive
# kaputt waren. Dieses Skript nutzt die frisch gefixte Pipeline:
#   ✓ Korrekte Win/Loss-Erkennung (auch Deck-Out)
#   ✓ Perspektiv-Rotation (50/50 Player 0/1)
#   ✓ Skalierte Auxiliary Rewards (10x kleiner)
#
# Phase 1: "Aufwach-Phase" – Höhere LR, um aus dem lokalen Minimum
#           zu kommen. Gegen Random-Gegner mit Perspektiv-Rotation.
# Phase 2: "Stärkung" – Gegen diverse Modell-Gegner.
# Phase 3: "Härtung" – Gegen die stärksten verfügbaren Gegner.
# Phase 4: "Politur" – Niedrige LR, Self-Play zum Feinschliff.
#
# Geschätzte Laufzeit: ~6-7 Stunden
# ================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# ─── Konfiguration ──────────────────────────────────────────────
DECK="decks/deck_bank/bank_18.csv"
# Wir kopieren das aktuelle Modell als Ausgangspunkt und trainieren
# in eine NEUE Datei, damit das alte erhalten bleibt.
SOURCE_MODEL="models/backup/ppo_v4_deck_bank_18_robust.zip"
MODEL="models/ppo_v4_deck_bank_18_v2.zip"

NUM_ENVS=8
N_STEPS=2048
BATCH_SIZE=1024
N_EPOCHS=4
BELIEF_DIM=64
AUX_COEF=0.10
EVAL_GAMES="${EVAL_GAMES:-100}"

# Gegner-Modelle (nicht im Holdout!)
OPP_BANK47="models/ppo_v4_deck_bank_47.zip"
OPP_BANK47_CK1="models/backup/ppo_v4_deck_bank_47_checkpoint_1.zip"
OPP_BANK19="models/backup/ppo_v4_deck_bank_19.zip"
OPP_BANK37="models/backup/ppo_v4_deck_bank_37_opp_stage2.zip"
OPP_BANK79="models/backup/ppo_v4_deck_bank_79.zip"
OPP_BANK100="models/backup/ppo_v4_deck_bank_100.zip"

# ─── Vorbereitung ───────────────────────────────────────────────
if [ ! -f "$MODEL" ]; then
  echo "Kopiere Ausgangsmodell: $SOURCE_MODEL → $MODEL"
  cp "$SOURCE_MODEL" "$MODEL"
fi

echo "============================================================"
echo " Deck 18 Overnight-Training (v2, mit Fixes)"
echo " Start: $(date)"
echo "============================================================"
echo " Modell:      ${MODEL}"
echo " Quelle:      ${SOURCE_MODEL}"
echo " Perspektive: Rotation (50/50 P0/P1)"
echo " Rewards:     Gefixt (sparse outcomes + skalierte aux)"
echo "============================================================"
echo ""

run_stage() {
  local stage="$1" label="$2" steps="$3" opp_deck="$4"
  local opp_model="$5" lr="$6" ent="$7" clip="$8" kl="$9" sparse="${10}"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Stage ${stage}: ${label}  [${steps} steps]  ($(date +%H:%M))"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  export WANDB_NAME="deck18_v2_stage${stage}_${label}"
  export WANDB_RUN_GROUP="deck18_v2_overnight"

  local cmd=(
    python src/train.py
    --deck "$DECK" --model-name "$MODEL"
    --opp-deck "$opp_deck"
    --timesteps "$steps"
    --num-envs "$NUM_ENVS" --n-steps "$N_STEPS"
    --batch-size "$BATCH_SIZE" --n-epochs "$N_EPOCHS"
    --checkpoint-interval 300000 --keep-checkpoints 2
    --lr "$lr" --ent-coef "$ent"
    --clip-range "$clip" --target-kl "$kl"
    --belief-actor --belief-dim "$BELIEF_DIM"
    --aux-coef "$AUX_COEF"
    --rotate-perspective
  )
  [ -n "$opp_model" ] && cmd+=(--opp-model "$opp_model")
  [ "$sparse" = "1" ] && cmd+=(--sparse-rewards)

  "${cmd[@]}"
}

run_eval() {
  local stage="$1"
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  local out="logs/eval_deck18_v2_stage${stage}_${ts}.json"
  mkdir -p logs
  echo ""
  echo "▶ Holdout-Evaluierung nach Stage ${stage} → ${out}"
  python3 scripts/evaluate_overnight_references.py \
    --reference models/holdout/ppo_v4_deck_4.zip \
    --reference models/holdout/ppo_v4_deck_7.zip \
    --reference models/holdout/ppo_v4_deck_12.zip \
    --reference models/holdout/ppo_v4_deck_bank_1.zip \
    --reference models/holdout/ppo_v4_deck_bank_16.zip \
    --reference models/holdout/ppo_v4_deck_bank_20.zip \
    --reference models/holdout/ppo_v4_deck_bank_30.zip \
    --candidate "$MODEL" --games "$EVAL_GAMES" --results-file "$out"
}

# ================================================================
# PHASE 1: AUFWACH-PHASE (aus dem lokalen Minimum rauskommen)
# ================================================================
# Höhere LR + Entropy gegen Random-Gegner mit Perspektiv-Rotation.
# Das Model muss zunächst die neuen, korrekten Reward-Signale lernen.

# Stage 1: Random-Gegner, hohe LR → "Aufwecken" (1M steps, ~33min)
run_stage 1 "random_warmup" 1000000 \
  "$DECK" "" \
  "5e-5" "0.015" "0.20" "0.05" "0"

run_eval 1

# ================================================================
# PHASE 2: STÄRKUNG (diverse Gegner)
# ================================================================
# Mittlere LR, diverse Gegner aus verschiedenen Spielstilen.

# Stage 2: Abra (schwächer) – aggressiver Gegner (800k, ~27min)
run_stage 2 "vs_abra_warm" 800000 \
  "decks/deck_bank/bank_47.csv" "$OPP_BANK47_CK1" \
  "3e-5" "0.010" "0.15" "0.04" "0"

# Stage 3: N's Zorua – Kontroll-Stil (800k, ~27min)
run_stage 3 "vs_zorua" 800000 \
  "decks/deck_bank/bank_19.csv" "$OPP_BANK19" \
  "3e-5" "0.008" "0.15" "0.04" "0"

# Stage 4: Murkrow – Disruption (800k, ~27min)
run_stage 4 "vs_murkrow" 800000 \
  "decks/deck_bank/bank_37.csv" "$OPP_BANK37" \
  "3e-5" "0.008" "0.15" "0.04" "0"

# Stage 5: Slowpoke – defensiv (600k, ~20min)
run_stage 5 "vs_slowpoke" 600000 \
  "decks/deck_bank/bank_79.csv" "$OPP_BANK79" \
  "2e-5" "0.006" "0.12" "0.035" "0"

run_eval 5

# ================================================================
# PHASE 3: HÄRTUNG (stärkste Gegner, sparse rewards)
# ================================================================
# Niedrigere LR, sparse rewards → reine Win/Loss-Optimierung.

# Stage 6: Abra (stark) – sparse (1.5M, ~50min)
run_stage 6 "vs_abra_hard" 1500000 \
  "decks/deck_bank/bank_47.csv" "$OPP_BANK47" \
  "2e-5" "0.004" "0.12" "0.03" "1"

# Stage 7: Nochmal Random → nicht verlernen (500k, ~17min)
run_stage 7 "vs_random_robust" 500000 \
  "$DECK" "" \
  "2e-5" "0.005" "0.12" "0.03" "0"

run_eval 7

# ================================================================
# PHASE 4: POLITUR (Self-Play, niedrige LR)
# ================================================================

# Stage 8: Self-Play – gegen sich selbst (1.5M, ~50min)
run_stage 8 "selfplay" 1500000 \
  "$DECK" "$MODEL" \
  "1e-5" "0.003" "0.10" "0.025" "0"

# Stage 9: Finaler Abra-Test (800k, ~27min)
run_stage 9 "vs_abra_final" 800000 \
  "decks/deck_bank/bank_47.csv" "$OPP_BANK47" \
  "1e-5" "0.003" "0.10" "0.025" "1"

# Stage 10: Random zum Schluss (300k, ~10min)
run_stage 10 "random_final" 300000 \
  "$DECK" "" \
  "1e-5" "0.003" "0.10" "0.025" "0"

run_eval 10

# ================================================================
# FINALE EVALUIERUNG
# ================================================================
echo ""
echo "============================================================"
echo " Training abgeschlossen! $(date)"
echo " Logs: logs/eval_deck18_v2_stage*.json"
echo "============================================================"
echo ""
echo "Gesamtlaufzeit der Stages:"
echo "  1: 1.0M  (random warmup)         ~33min"
echo "  2: 0.8M  (abra warm)             ~27min"
echo "  3: 0.8M  (zorua)                  ~27min"
echo "  4: 0.8M  (murkrow)               ~27min"
echo "  5: 0.6M  (slowpoke)              ~20min"
echo "  6: 1.5M  (abra hard, sparse)     ~50min"
echo "  7: 0.5M  (random robust)         ~17min"
echo "  8: 1.5M  (selfplay)              ~50min"
echo "  9: 0.8M  (abra final, sparse)    ~27min"
echo " 10: 0.3M  (random final)          ~10min"
echo " ────────────────────────────────────"
echo " Gesamt: 8.6M steps               ~288min (~4h 48min)"
echo " + 4x Eval à ~15min               ~60min"
echo " = Gesamt:                          ~5h 48min"
