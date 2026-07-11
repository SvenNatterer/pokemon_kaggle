# Pokémon TCG Arena

The project has one queue-free arena, one holdout evaluator, and one training
entry point. Historical models are participants; their filenames do not imply
that the arena may create or rotate new intermediate saves.

## Start

Create/activate the existing environment first (`source venv/bin/activate`).

```bash
# Backend and dashboard together (recommended)
PYTHONPATH=src venv/bin/python src/server.py
# Open http://127.0.0.1:8050/dashboard/dashboard.html

# Optional static dashboard on port 8080 (API remains on 8050)
python3 dashboard/serve_dashboard.py

# Arena without the dashboard API (state must be set through the API normally)
PYTHONPATH=src venv/bin/python -m src.arena_worker

# Holdout evaluation
PYTHONPATH=src venv/bin/python scripts/evaluate_submission.py \
  --candidate models/ppo_v5_deck_bank_18.zip --games 30

# New training run (fails if the target already exists)
PYTHONPATH=src venv/bin/python src/train.py \
  --deck decks/deck_bank/bank_18.csv \
  --model-name models/my_final_model.zip \
  --opp-deck decks/deck_bank/bank_47.csv \
  --opp-model models/ppo_v4_deck_bank_47.zip \
  --timesteps 500000

# Explicitly continue exactly that final target model
PYTHONPATH=src venv/bin/python src/train.py \
  --deck decks/deck_bank/bank_18.csv \
  --model-name models/my_final_model.zip \
  --continue-existing --timesteps 500000
```

`scripts/train_v5.sh` is the small preset wrapper. The holdout-safe staged
workflow is `scripts/train_v5_curriculum_bank18.sh --dry-run` and then the same
command without `--dry-run`.

## Arena architecture

- `src/arena_core.py`: participants, atomic JSON, Wilson score, ranking, and matchmaking.
- `src/arena_control.py`: start/pause/stop/reset and PID-aware process control.
- `src/arena_worker.py`: the one file-locked worker; it executes matches directly.
- `src/arena_match.py`: match subprocess, Elo update, journal record, and bot cooldown.
- `src/server.py`: thin HTTP API and static files.
- `scripts/evaluate_submission.py`: the single frozen-holdout evaluator.
- `src/train.py`: the single training implementation.

Runtime data lives in ignored `arena_data/`. Writes use a flushed sibling temp
file followed by `os.replace`. There is no persistent or in-memory match queue.
The worker chooses the next pair only after the preceding match is persisted.
A PID file plus an advisory file lock prevents duplicate workers.

## Participants and failures

`decks/arena_agents.json` contains participants that cannot be inferred safely,
including the normal rule-based bot and the foundation model. PPO models are
also discovered in `models/`, `models/backup/`, and the historical curriculum
snapshot directory. `models/holdout/` is deliberately excluded.

ZIP integrity and deck resolution are reported in the dashboard. A failed match
does not stop the arena: both involved bots enter a five-minute cooldown and are
then eligible for another attempt. The failure remains in the match journal.

## Matchmaking

The first bot is sampled from the least-played group (within five games of the
minimum). The second is chosen by a small score combining close Elo, low match
count, and rare prior pairing; 15% of selections are random exploration.
Self-matches and unloadable/disabled bots are excluded. The extra player-0 game
in odd-sized batches alternates for each pair.

## Ranking

Draws count as half a success in both win rate and Wilson lower bound:

```text
effective_wins = wins + 0.5 * draws
n = wins + losses + draws
```

The single ranking formula in `rank_participants` is:

```text
35% arena Wilson lower bound
25% min-max normalized Elo
15% arena effective win rate
25% holdout Wilson lower bound
```

All components are in `[0, 1]`. Identical Elo values normalize to `0.5`.
Missing holdout data is visible and receives the conservative value `0.35`.

## Persistence and reset

`arena_data/matches.json` is the authoritative match journal; leaderboard data
is reproducible from it. A record contains IDs, timestamp, both bots/decks/types,
aggregate W/L/D, reason, total turns, perspective/start player, Elo before/after,
error, replay reference, and schema version. Evaluation progress/history is
separate and never enters arena matches or training data.

Factory Reset requires the exact phrase `RESET ARENA`. It resets arena matches,
ranking, health cooldowns, and state. Models, decks, training data, and evaluation
results are preserved. Replay deletion is a separate explicit request flag.

The previous arena JSON files are not deleted or rewritten. At cleanup time no
complete old match journal existed, so fabricating a lossy migration from Elo and
pairwise aggregates would not be reliable; those legacy artifacts remain in place.
