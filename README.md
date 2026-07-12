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

## Reliable training and evaluation workflow

The project deliberately separates **arena**, **validation**, and **final
holdout**. They answer different questions and must not be mixed.

| Layer | Purpose | May influence training/model choice? |
| --- | --- | --- |
| Arena | Continuous regression detection, Elo and replay inspection | Yes |
| Validation league | Choose a checkpoint or training variant repeatedly | Yes |
| Final holdout | One final, independent performance statement | No |

### 1. Create a validation league

The validation league is a frozen list of opponents that must never occur in
training. Build it from locally available models, excluding the final holdout
and every deck used by the current curriculum:

```bash
PYTHONPATH=. venv/bin/python scripts/build_validation_manifest.py \
  --exclude-deck bank_18 --exclude-deck bank_47 --exclude-deck bank_19 \
  --exclude-deck bank_37 --exclude-deck bank_79 --count 8
```

This writes `decks/validation_opponents.json` and copies its frozen PPO assets
to `models/validation/`. Both `src/train.py` and the V5
curriculum reject an overlap with this file or `decks/holdout_opponents.json`.
Keep these manifests under version control; changing them changes the meaning
of every comparison.

The model folders have distinct responsibilities:

- `models/validation/`: frozen opponents for repeatable model selection.
- `models/holdout/`: final evaluation opponents; never use during selection or training.
- `models/stage_snapshots/`: candidate checkpoints produced by training.
- `models/archive-*/`: historical storage only; manifests must not point here.

### 2. Train with provenance and immutable stage snapshots

Every training run now writes `models/experiments/<model>.json`. It records the
model path, command-line parameters, Git revision, reserved manifests and final
step count. The V5 curriculum additionally copies the model after each completed
stage to `models/stage_snapshots/`. Existing snapshot names are never replaced.

The final model is still the only mutable target. Snapshots are candidates for
evaluation, not automatic promotions.

### 3. Evaluate candidates on validation

Evaluate competing snapshots with equal numbers of player-0 and player-1 games.
Use at least 100 games per opponent for a decision and 200 for close results:

```bash
PYTHONPATH=. venv/bin/python scripts/evaluate_submission.py \
  --holdout-file decks/validation_opponents.json \
  --candidate models/stage_snapshots/ppo_v5b_deck_bank_18_stage7_mixed_league.zip \
  --candidate models/stage_snapshots/ppo_v5b_deck_bank_18_stage9_sparse_league_final.zip \
  --games 100 \
  --results-file logs/v5_validation.json \
  --best-candidate-file logs/v5_selection.json
```

The result contains a Wilson lower confidence bound, worst matchup, per-opponent
scores, score difference between player perspectives, mean turns, and win/loss
reasons. Candidate selection orders by Wilson lower bound, worst matchup and
overall score. A large perspective gap (normally above 10 percentage points) is
a regression signal, not evidence of a stronger model.

The underlying native game engine does not currently expose deterministic RNG
seeding. Player perspectives are balanced, but exact paired game seeds cannot
yet be guaranteed; do not describe repeated runs as bit-for-bit reproducible.

### 4. Promote only through gates

The current champion is a small pointer file, not a copy of a model. Promotion
is rejected if the candidate has excessive player-perspective bias or does not
beat the existing champion's Wilson lower bound by the requested margin:

```bash
PYTHONPATH=. venv/bin/python scripts/promote_champion.py \
  --selection logs/v5_selection.json \
  --champion-file models/champion.json \
  --min-wilson-improvement 0.01 \
  --max-perspective-gap 0.10
```

Run it first with `--dry-run` when introducing the process. The champion is
never overwritten by training itself.

### 5. Use the arena for diagnosis, not final selection

Keep the champion, its predecessor, historical strong snapshots and diverse
rule/PPO bots in the arena. After 20--50 arena batches, inspect ELO/Wilson,
the most frequent loss reason and replays for the worst matchup. Use the result
to formulate one fine-tuning hypothesis at a time. Validate that hypothesis in
the validation league before promotion.

Only after selecting a champion through validation should it play a new,
previously unseen final holdout. Once final-holdout results influence a training
or model-selection decision, freeze a new final holdout for the next report.

### Dashboard workflow

The dashboard now automates the normal evaluation flow. Open it through the
server, select one or more PPO models in **Evaluation & Champion**, leave the
mode on **Validation**, set `100` games, then click **Evaluate**. On completion,
the dashboard stores the detailed result and its selected winner. **Promote
champion** applies the Wilson-improvement and perspective-bias gates. A crowned
row in the arena leaderboard is the current champion.

An arena cooldown does not block validation or final-holdout evaluation. A PPO
model in cooldown remains selectable because the cooldown only prevents the
arena worker from scheduling it temporarily. Disabled participants, missing or
invalid model archives, and other load failures remain unavailable for
evaluation.

Use **Final holdout** only once a validation winner has been promoted. Arena
matches continue independently throughout: they are for regression alerts and
replays, not automatic champion selection.

## Controlled fine-tuning

Do not continue training the last model by default. The stored V5 results show
that `checkpoint_1` outperformed the final model, so fine-tuning starts from a
frozen selected parent and compares independent variants. The helper below runs
three variants against the same mixed league:

| Variant | Single change from parent |
| --- | --- |
| `epochs2` | Two PPO epochs instead of one |
| `aux0` | Auxiliary belief loss disabled |
| `sparse` | Terminal-only rewards |

After all three arms finish, it automatically evaluates them against the
validation manifest and stores the selection for the dashboard. Set the parent
explicitly if the default V5 checkpoint is not compatible with the currently
installed Observation/Policy architecture.

```bash
BASE_MODEL=models/ppo_v5_deck_bank_18_checkpoint_1.zip \
STEPS=500000 EVAL_GAMES=100 \
bash scripts/run_validation_finetune.sh
```

Run `bash scripts/run_validation_finetune.sh --dry-run` first. The dashboard
will then show all arms as normal PPO candidates; use **Evaluation & Champion**
to inspect the recorded validation result and promote only its winner. The
original parent model is never overwritten.

## Roadmap / To-dos

This is the implementation order for making model improvements credible. Do not
start a larger training run merely because it is convenient; complete the
corresponding measurement step first.

### P0 — establish trustworthy comparison sets

- [ ] **Create a validation league with 6--10 opponents.** It is used repeatedly
  for checkpoint choice, curriculum weighting and hyperparameter experiments.
- [ ] **Create a separate final holdout with 6--10 different opponents.** It is
  used only after a validation winner has been selected. If its result changes a
  model-choice decision, retire it and create a fresh final holdout.
- [ ] **Add enough eligible frozen PPO bots.** The current local model inventory
  contains too few non-training/non-holdout opponents for a statistically useful
  validation league. Train or retain diverse historical bots before creating the
  manifest.
- [ ] **Version the manifests.** Commit `decks/validation_opponents.json` and
  `decks/final_holdout_opponents.json` and never silently edit them during an
  experiment series.
- [ ] **Evaluate 100 games/opponent for normal selection, 200 for close calls.**
  Always balance player 0 and player 1, compare Wilson lower bound first, then
  worst matchup and perspective gap.

#### How to construct good validation and holdout sets

Choose opponents by *strategic diversity*, not simply by the currently highest
Elo. Both sets should contain a balanced mix of:

| Opponent type | Why it belongs in the set |
| --- | --- |
| Fast aggressive decks | Tests setup consistency, tempo and prize racing. |
| Slow/tanky decks | Tests resource planning and endgame conversion. |
| Control/disruption decks | Tests recovery after hand, energy or board disruption. |
| Different energy/type requirements | Tests whether the policy generalizes beyond one deck's sequencing. |
| Historical PPO checkpoints | Detects forgetting and exploits that only work against the newest league. |
| Rule-based bot | A stable floor and a diagnostic baseline, not the only serious opponent. |

Rules:

1. No exact deck/model pair may appear in training and validation.
2. No validation opponent may appear in the final holdout.
3. Do not select all opponents from the same deck family or all from one PPO
   generation.
4. Freeze the exact ZIP files used by a manifest; replacing a model at the same
   path changes the benchmark.
5. Keep a small written deck/archetype inventory next to the manifest so that
   future selections do not accidentally duplicate the same matchup.

The validation league can include known historical checkpoints because it is for
model selection. The final holdout should include frozen opponents that were not
inspected during training decisions. A rule-based opponent is useful in both,
but should make up no more than one or two opponents in each set.

### P0 — evaluate and improve the rule-based bot

- [ ] **Create a benchmark matrix for the rule bot.** Run it against every PPO
  candidate and across at least 5 distinct decks, with 100 games per pair and
  balanced perspectives.
- [ ] **Record decision categories, not just W/L.** At minimum: opening setup,
  active energy attachment, bench energy attachment, evolution, attack choice,
  retreat/switch, search target and end-turn decision.
- [ ] **Add deterministic scenario tests.** Construct small game states where a
  clearly legal best action is known; tests should assert the rule bot's chosen
  action rather than relying on aggregate win rate.
- [ ] **Use replay review for losses.** Classify a loss as setup, resource,
  target-selection, tempo, deckout or rules-coverage failure before adding a
  heuristic.
- [ ] **Apply it to other decks through card metadata, not deck IDs.** Rules
  should query attack costs, damage, retreat cost, evolution relation, board
  state and legal options. Deck-specific overrides are acceptable only as named,
  tested exceptions.
- [ ] **Keep the rule bot deterministic.** If two legal actions tie, use a
  documented stable tie-breaker. Random fallback makes it a noisy benchmark.

The rule bot is appropriate as a baseline, regression test and weak curriculum
anchor. It should not dominate PPO training: a policy that beats one heuristic
can still fail against a diverse PPO league.

### P1 — controlled PPO and reward optimization

- [ ] Run the three existing fine-tuning arms from the best frozen parent:
  `epochs2`, `aux0`, and `sparse`.
- [ ] Promote a winner only through the validation gate. Keep the parent and all
  arms in the arena until the regression picture is clear.
- [ ] If no arm wins convincingly, test one additional factor at a time:
  `gamma` (`0.995` vs `0.999`), entropy coefficient, then clip range.
- [ ] Add training-health review for KL, clip fraction, entropy, explained
  variance, value loss, auxiliary loss and invalid-action fallback count. A run
  with a collapsed entropy or unstable value estimate is not a good candidate
  even if one small evaluation happens to look good.
- [ ] Use failure-weighted opponent sampling only after validation identifies a
  repeatable weak matchup. Cap the weight of one matchup so that it does not
  cause catastrophic forgetting elsewhere.
- [ ] Do reward ablations, not intuition-only reward edits. Compare terminal
  reward, current weak shaping, and one potential-based alternative from the
  same parent with the same evaluation league.

### P1 — transfer learning strategy

Use PPO-to-PPO transfer, not rule-based weights, as the default. A rule-based
bot has no neural representation to transfer; it supplies demonstrations,
tests and an opponent policy instead.

Recommended sequence:

1. Train a **base PPO** on a broad mixed league and several representative
   decks, using the same Observation/Action version as downstream models.
2. Freeze and archive this base model with its experiment metadata.
3. Copy it for a target deck and fine-tune with a lower learning rate, mixed
   historical opponents and perspective rotation.
4. Compare transfer against a fresh model with the same target-deck budget.
   Transfer is useful only if it improves validation sample efficiency without
   reducing final target-deck strength.
5. Maintain a small reservoir of base and historical target snapshots in the
   opponent league to prevent specialized fine-tunes from forgetting basic play.

Rule-based knowledge can still help transfer indirectly:

- train against it during early mechanics learning;
- use its deterministic scenarios as regression tests;
- optionally collect state/action demonstrations for a separate behaviour-
  cloning warm start experiment.

Do not mix behaviour cloning, PPO, reward changes and architecture changes in
one experiment. A behaviour-cloning warm start should be its own ablation with
the same downstream PPO budget as the no-cloning control.

### P1 — Observation and Action V6

- [ ] **Replace the 1,000-way action space for new models.** The current
  `STOP_ACTION=999` preserves legacy checkpoint compatibility but leaves a very
  large masked action head while only up to 65 options are structurally encoded.
  V6 should use roughly `MAX_ENCODED_OPTIONS + 1` actions, with STOP immediately
  after the encoded option range.
- [ ] **Make V6 explicitly incompatible.** Give it a new observation/policy
  version; retain V5 models as frozen opponents and benchmarks. Do not silently
  load a V5 actor into a V6 action space.
- [ ] **Measure option-count coverage first.** Log the maximum and percentile
  count of engine options. If real games regularly exceed 65 choices, increase
  the encoded range before shrinking the action space.
- [ ] **Ablate the legacy scalar vector.** The structured card/entity/option
  encoder is the preferred path. Measure whether the 1,500-value legacy vector
  still adds validation value or only increases model capacity and noise.
- [ ] **Keep hidden information honest.** Own deck composition and publicly
  revealed/discarded cards are valid inputs; opponent hand and unrevealed prize
  identities must never leak into observations.

### P0/P1 — missing test coverage

- [ ] **Run the functional observation tests under pytest in CI.** The current
  environment has no pytest package in the project venv, so function-style tests
  are not included in the `unittest` suite by default.
- [ ] **Environment invariants:** run long random/legal rollouts and assert
  eventual termination, legal action masks, no duplicate selected option,
  non-negative zone sizes, 60-card conservation and correct winner perspective.
- [ ] **Perspective symmetry:** mirror a fixed game state and verify that
  observations/rewards map consistently between player 0 and player 1.
- [ ] **Selection edge cases:** test min/max counts, STOP legality, repeated
  selections, zero-option states and more-than-65-option behavior explicitly.
- [ ] **Evaluation tests:** verify multi-candidate progress accounting,
  validation/final manifest selection, promotion rejection on perspective gap,
  and champion marking in the API/dashboard payload.
- [ ] **Rule bot scenario tests:** one deterministic test for each decision
  category listed above, plus a regression replay corpus for previously fixed
  mistakes.
- [ ] **Model compatibility tests:** V5 checkpoint loads only into V5; V6 must
  fail clearly when given a V5 checkpoint rather than padding/truncating data.
- [ ] **Native engine RNG:** expose a seed API in the CG wrapper if possible.
  Until then, report evaluations as repeated stochastic estimates, not exact
  paired-seed experiments.

### Definition of done for the next model generation

- A versioned validation league and a separate final holdout exist.
- The rule bot has a scenario suite and benchmark matrix.
- The three fine-tune arms have completed with 100-game/opponent validation.
- One champion has passed the promotion gate and remains stable in arena matches.
- A final-holdout report is produced once, after all selection decisions.
- V6 action/observation migration has an explicit compatibility test and an
  ablation result before replacing V5 as the default architecture.
