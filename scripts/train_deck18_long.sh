#!/usr/bin/env bash
# train_deck18_long.sh – 4-5 Stunden Training für deck_bank_18
# Ziel: Schwäche gegen bank_47 beheben + diverse Spielkonzepte lernen
# Holdout-safe: deck_4, deck_7, deck_12, bank_1, bank_16, bank_20, bank_30 werden NICHT genutzt
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# ─── Konfiguration ───────────────────────────────────────────────────────────
DECK="decks/deck_bank/bank_18.csv"
MODEL="${MODEL:-models/ppo_v4_deck_bank_18_robust.zip}"
NUM_ENVS=8
N_STEPS=2048
BATCH_SIZE=1024
N_EPOCHS=4
BELIEF_DIM=64
AUX_COEF=0.10
EVAL_GAMES="${EVAL_GAMES:-100}"

# Sichere Gegner (alle NICHT im Holdout)
OPP_BANK47="models/ppo_v4_deck_bank_47.zip"
OPP_BANK47_CK1="models/backup/ppo_v4_deck_bank_47_checkpoint_1.zip"
OPP_BANK19="models/backup/ppo_v4_deck_bank_19.zip"
OPP_BANK37="models/backup/ppo_v4_deck_bank_37_opp_stage2.zip"
OPP_BANK79="models/backup/ppo_v4_deck_bank_79.zip"

# Schritte pro Stage → Ziel: ~4.5h Gesamtzeit
# Mit 8 envs / ~500fps: 1M Steps ≈ 33 min
STEPS_ANTI_47_WARM=800000     # Stage 1: ~27 min – Abra warm (checkpoint_1)
STEPS_ANTI_47_MAIN=1500000    # Stage 2: ~50 min – Abra stark (bestes Modell)
STEPS_DIVERSE_ZORUA=600000    # Stage 3: ~20 min – N's Zorua  (Kontroll)
STEPS_DIVERSE_MURKROW=600000  # Stage 4: ~20 min – Murkrow    (Disruption)
STEPS_DIVERSE_SLOWPOKE=600000 # Stage 5: ~20 min – Slowpoke   (defensiv)
STEPS_ANTI_47_FINAL=1500000   # Stage 6: ~50 min – Abra nochmal (Festigung)
STEPS_HEURISTIC=500000        # Stage 7: ~17 min – Heuristik  (Grundlagen)
# Gesamt: ~7.1M Steps ≈ 4h 24min

echo "============================================================"
echo " Deck 18 Langzeit-Training (~264 Minuten erwartet)"
echo "============================================================"
echo " Modell:  ${MODEL}"
echo " Stages:  7 (Anti-Abra x3, Diverse x3, Heuristik x1)"
echo "============================================================"
echo ""

run_stage() {
  local stage="$1" label="$2" steps="$3" opp_deck="$4"
  local opp_model="$5" lr="$6" ent="$7" clip="$8" kl="$9" sparse="${10}"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Stage ${stage}: ${label}  [${steps} steps]"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  export WANDB_NAME="deck18_long_stage${stage}_${label}"
  export WANDB_RUN_GROUP="deck18_long_run"

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
  )
  [ -n "$opp_model" ] && cmd+=(--opp-model "$opp_model")
  [ "$sparse" = "1" ] && cmd+=(--sparse-rewards)

  "${cmd[@]}"
}

run_eval() {
  local stage="$1"
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  local out="logs/eval_deck18_long_stage${stage}_${ts}.json"
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

# Stage 1: Abra Aufwärm (schwächerer Checkpoint – sanfter Einstieg)
run_stage 1 "vs_abra_warmup" "$STEPS_ANTI_47_WARM" \
  "decks/deck_bank/bank_47.csv" "$OPP_BANK47_CK1" \
  "3e-5" "0.008" "0.15" "0.035" "0"

# Stage 2: Abra Main (stärkstes Modell – direkt gegen die Schwäche)
run_stage 2 "vs_abra_main" "$STEPS_ANTI_47_MAIN" \
  "decks/deck_bank/bank_47.csv" "$OPP_BANK47" \
  "2e-5" "0.004" "0.12" "0.030" "1"

run_eval 2

# Stage 3: N's Zorua (Kontroll-Stil)
run_stage 3 "vs_zorua_control" "$STEPS_DIVERSE_ZORUA" \
  "decks/deck_bank/bank_19.csv" "$OPP_BANK19" \
  "2e-5" "0.005" "0.12" "0.030" "0"

# Stage 4: Team Rocket's Murkrow (Disruption)
run_stage 4 "vs_murkrow_disruption" "$STEPS_DIVERSE_MURKROW" \
  "decks/deck_bank/bank_37.csv" "$OPP_BANK37" \
  "2e-5" "0.005" "0.12" "0.030" "0"

# Stage 5: Slowpoke (defensiver Stil)
run_stage 5 "vs_slowpoke_defensive" "$STEPS_DIVERSE_SLOWPOKE" \
  "decks/deck_bank/bank_79.csv" "$OPP_BANK79" \
  "2e-5" "0.005" "0.12" "0.030" "0"

run_eval 5

# Stage 6: Abra nochmal (Festigung – nach diversen Erfahrungen)
run_stage 6 "vs_abra_reinforce" "$STEPS_ANTI_47_FINAL" \
  "decks/deck_bank/bank_47.csv" "$OPP_BANK47" \
  "1e-5" "0.003" "0.10" "0.025" "1"

# Stage 7: Stochastischer Gegner (Robustheit gegen ungewöhnliche Züge)
run_stage 7 "vs_random_final" "$STEPS_HEURISTIC" \
  "$DECK" "" \
  "1e-5" "0.003" "0.10" "0.025" "0"

run_eval 7

echo ""
echo "============================================================"
echo " Training abgeschlossen!"
echo " Logs: logs/eval_deck18_long_stage*.json"
echo "============================================================"
