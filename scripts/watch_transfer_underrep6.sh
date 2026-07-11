#!/usr/bin/env bash
set -euo pipefail

cd /Users/svennatterer/Documents/GitHub/Kaggle/pokemon_kaggle

source venv/bin/activate || true
export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

INTERVAL_SECONDS="${WATCH_INTERVAL_SECONDS:-900}"
TRAIN_LOG="logs/train_transfer_underrep6_20260709_184020.log"
WATCHDOG_LOG="logs/train_transfer_underrep6_watchdog.log"
SOURCE_MODEL="models/ppo_v4_base_brain_endless.zip"
TARGET_DECKS=(
  "decks/deck_bank/bank_79.csv"
  "decks/deck_bank/bank_47.csv"
  "decks/deck_bank/bank_37.csv"
  "decks/deck_bank/bank_19.csv"
  "decks/deck_bank/bank_100.csv"
  "decks/deck_bank/bank_18.csv"
)

TRANSFER_PATTERN="bash scripts/train_transfer.sh ${SOURCE_MODEL} ${TARGET_DECKS[*]}"
TRAIN_PATTERN="src/train.py --deck decks/deck_bank/bank_(79|47|37|19|100|18).csv"
COMPLETION_MARKER="Successfully completed two-stage training for decks/deck_bank/bank_18.csv!"

log_status() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$WATCHDOG_LOG"
}

is_training_running() {
  pgrep -f "$TRANSFER_PATTERN" >/dev/null || pgrep -f "$TRAIN_PATTERN" >/dev/null
}

is_training_complete() {
  [ -f "$TRAIN_LOG" ] && grep -qF "$COMPLETION_MARKER" "$TRAIN_LOG"
}

restart_training() {
  log_status "training not running; restarting transfer job"
  nohup bash scripts/train_transfer.sh "$SOURCE_MODEL" "${TARGET_DECKS[@]}" >> "$TRAIN_LOG" 2>&1 &
  log_status "restart pid=$!"
}

log_status "watchdog started; interval=${INTERVAL_SECONDS}s"

while true; do
  if is_training_complete; then
    log_status "training complete; watchdog exiting"
    exit 0
  fi

  if is_training_running; then
    log_status "training running"
  else
    restart_training
  fi

  sleep "$INTERVAL_SECONDS"
done
