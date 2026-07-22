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
  --candidate models/ppo_v5_deck_bank_18.zip

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

## Kaggle submission

Build the compressed submission archive locally, then upload that archive
directly through the competition API. This is the only supported submission
workflow; do not create a Kaggle Dataset, Kernel, or Notebook wrapper.

The outer archive must remain `.tar.gz`, which is the format already accepted
by the competition. The PPO model inside it remains a `.zip` file.

```bash
# Build an explicitly named submission archive.
scripts/build_submission.sh \
  bank_38 \
  models/ppo_v6_deck_bank_38.zip \
  artifacts/submissions/submission_v6_bank38_YYYYMMDD.tar.gz

# Upload the newest built archive directly to pokemon-tcg-ai-battle.
scripts/submit_latest_submission.sh

# Or select an archive and description explicitly.
scripts/submit_latest_submission.sh \
  artifacts/submissions/submission_v6_bank38_YYYYMMDD.tar.gz \
  "V6 bank38 YYYY-MM-DD"
```

## Arena architecture

- `src/arena_core.py`: participants, atomic JSON, Wilson score, ranking, and matchmaking.
- `src/arena_control.py`: start/pause/stop/reset and PID-aware process control.
- `src/arena_worker.py`: the one file-locked worker; it executes matches directly.
- `src/arena_match.py`: match subprocess, Elo update, journal record, and bot cooldown.
- `src/server.py`: thin HTTP API and static files.
- `scripts/evaluate_submission.py`: the single frozen-holdout evaluator.
- `src/train.py`: the single training implementation.

Runtime data lives in ignored `evaluation_results/`. Writes use a flushed sibling temp
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
50% arena Wilson lower bound
35% Elo expected score versus the fixed 1200 baseline
15% arena effective win rate
```

All components are in `[0, 1]`. Elo 1200 maps to `0.5`; unlike min-max
normalization, adding or removing an unrelated bot does not change this
component. Validation and final-holdout results are displayed separately and do
not affect the arena ranking.

## Persistence and reset

`evaluation_results/matches.json` is the authoritative match journal; leaderboard data
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
  --candidate models/stage_snapshots/ppo_v5b_deck_bank_18_stage9_final_league.zip \
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

### 4. Use the arena for diagnosis, not final selection

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
two variants against the same mixed league:

| Variant | Single change from parent |
| --- | --- |
| `epochs2` | Two PPO epochs instead of one |
| `aux0` | Auxiliary belief loss disabled |

After both arms finish, it automatically evaluates them against the
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

## Current RL architecture and cost/benefit status

This section records the repository audit from 2026-07-15. It describes the
current V6 working state, including changes that are not yet committed or
portable to every supported platform. A component listed as present is not
automatically production-ready; the P0 correctness and portability gates below
still take precedence over additional training or architecture work.

The implemented policy stack is already close to the intended target:

```text
structured card/entity/option observation
                 -> shared card and option encoders
                 -> 256-unit actor and critic LSTMs
                 -> state-conditioned scoring of legal engine options
                 -> hard action mask and autoregressive STOP selection
```

| Component | Current status | Remaining gap |
| --- | --- | --- |
| Recurrent PPO | Present | Separate one-layer, 256-unit actor and critic LSTMs are already used. |
| Structured observation | Present | Twelve field entities, attachments, hand, discard, known/revealed cards, logs, own deck and legal options are encoded. |
| Relational representation | Partial | Static attack, skill, evolution and attachment relations exist; learned entity-to-entity attention/message passing does not. |
| Dynamic action representation | Present | V6 scores each legal engine option with shared weights and uses 65 option slots plus STOP. |
| Action masking | Present | Pending selections are resolved by actor ownership in both player perspectives; regression coverage checks the mask and STOP invariant. |
| Hidden-card belief auxiliary task | Present | It is optional during fine-tuning and must remain isolated from actor-visible simulator truth. |
| Reward design | Potential-based | Training pays terminal rewards plus the discounted change in board potential. |
| Historical opponent pool | Present | Static weighted PPO and rule-based opponents are sampled per episode. |
| Self-play | Partial | Frozen self-play and PFSP-lite sampling exist; controlled static-vs-PFSP validation, iterative league insertion and exploiters remain. |
| Validation and holdout evaluation | Present | Balanced perspectives, Wilson lower bounds, worst matchup, promotion gates and separate manifests exist. |
| Behaviour cloning / DAgger | Missing | Replays are not yet a complete `(actor observation, legal mask, expert action)` dataset. |
| Entity/temporal Transformer | Missing by design | A small entity-attention ablation is later work; a temporal Transformer/GTrXL is not a current priority. |
| Population-based training | Missing by design | PBT remains last because it multiplies compute before the cheaper opponent-sampling and correctness work is exhausted. |

### Measured V6 architecture trade-off

The equal-budget validation ablation in
`logs/v6_architecture_ablation/results_100_games.json` compared the same deck,
seed, league and PPO settings over 800 games per candidate:

| Variant | Policy parameters | Validation score | Decision |
| --- | ---: | ---: | --- |
| Full | 13.30 M | 88.75% | Best current strength; keep for champion candidates. |
| Compact | 4.82 M | 85.88% | Best efficiency candidate when many cheaper opponent policies are needed. |
| Balanced | 6.42 M | 85.75% | No measured advantage over Compact in this run. |
| Compact without legacy vector | 4.01 M | 64.63% | Rejected; the structured path does not yet replace all global information. |

Full achieved an 86.37% Wilson lower bound, a 77% worst-matchup score and only
a three-point player-perspective gap. Compact removes about 64% of the
parameters for a 2.88-point aggregate score reduction. Do not remove the legacy
global vector until the missing information has been identified and represented
structurally.

### Audit snapshot of the V6 opponent factory

The completed Compact/Potential V6 factory has four configured foundations and
16 target decks split into four training, six validation and six holdout
targets. All 16 targets have completion markers and were evaluated for 240
games each without crashes. The freeze step produced disjoint manifests and
frozen model copies with SHA-256 hashes under
`decks/generated/opponent_factory_v6_compact_potential/`. Validation is active;
the final holdout remains intentionally inactive until model selection ends.

### Priority order by expected cost/benefit

| Rank | Work item | Expected benefit | Remaining effort |
| ---: | --- | --- | --- |
| 1 | Fix rotated-perspective pending-selection masks and add the invariant test. **Completed.** | Very high: removes known invalid training samples. | Done |
| 2 | Add native-symbol detection, Python fallback and reproducible dependencies; verify Linux. **Completed.** | Very high: preserves the measured speed-up without breaking Kaggle. | Done |
| 3 | Finish, smoke-test and freeze the existing V6 opponent factory outputs. **Completed for Compact/Potential V6.** | Very high: converts spent compute into trustworthy leagues. | Done |
| 4 | Add health counters and adaptive run-level stopping. | High: catches corruption and avoids training after learning stalls. | Low--medium |
| 5 | Replace static opponent weights with a capped PFSP-lite historical league. **Implemented; controlled ablation is pending.** | High: spends samples on informative weaknesses without forgetting old strategies. | Low, mostly evaluation |
| 6 | Complete rule-bot scenarios and its cross-play matrix. | High: improves diagnosis and supplies a stable curriculum/baseline. | Medium |
| 7 | Test one small entity-attention arm after the P0 gates. | Potentially high: addresses the largest architecture gap. | Medium--high plus retraining |
| 8 | Test behaviour-cloning warm start from replay/expert data. | Medium--high after a legal observation/action dataset exists. | Medium--high |
| 9 | Temporal Transformer/GTrXL and population-based training. | Uncertain at the current maturity level. | Very high; defer |

This order reuses the factory, evaluation and recurrent-policy work already in
the repository. A new temporal architecture is not justified until the known
correctness issue and the cheaper training-loop improvements are resolved.

## Roadmap / To-dos

This is the implementation order for making model improvements credible. Do not
start a larger training run merely because it is convenient; complete the
corresponding measurement step first.

### P0 — fix V6 correctness and Kaggle portability

These items have the highest priority. They protect every later reward,
opponent-pool and architecture comparison from invalid samples or a local-only
implementation.

- [x] **Fix pending-selection ownership under perspective rotation.** Native and
  Python observation paths now select pending state by actor ownership instead
  of assuming that the learner is player 0. This applies to future runs; models
  and samples produced before the fix are unchanged.
- [x] **Add the regression invariant for both perspectives.**
  `tests/test_action_space_v6.py` verifies for learner player 0 and player 1 that
  an already selected option stays masked, the other actor's pending selection
  is not mixed in, and STOP agrees with the active autoregressive selection.
- [x] **Make native observation encoding an optional optimization.** The loader
  detects a ctypes-compatible `GetV6Observation` export and otherwise uses the
  tested Python encoder. The x86-64 Linux library is reproducibly cross-compiled
  with the native symbol; submissions intentionally encode Kaggle's external
  observation dictionary in Python because it has no local engine pointer.
- [x] **Make the JSON fast path reproducible.** Add `orjson` to the installation
  requirements or retain a tested standard-library `json` fallback.
- [x] **Promote the parity scripts into automated tests.** Cover both player
  perspectives, autoregressive pending selections, hidden-information rules and
  option overflow. Hundreds of local Python/C++ states matched exactly, which is
  encouraging but not yet sufficient automated coverage.
- [x] **Expose training-health counters and gates.** Training now logs invalid
  learner actions, selected opponent labels, option-count percentiles/overflows
  and native engine errors to W&B and the experiment record. The default health
  gate stops a corrupted run without saving its target model; evaluation,
  candidate selection and champion promotion reject any crash, native engine
  error, invalid learner action or option overflow. `--no-health-gate` is only
  for diagnostics: its saved model remains ineligible for promotion.

Do not silently patch a running target. Apply the correction at a model boundary,
then run a controlled pre-fix/post-fix comparison and decide from validation
whether affected foundations or targets need retraining.

### P0 — establish trustworthy comparison sets

- [x] **Create the historical V1 validation league with 6--10 opponents.** The
  committed manifest currently contains eight frozen PPO/rule-based opponents.
- [x] **Create a separate historical V1 final holdout.** The committed
  `decks/holdout_opponents.json` contains nine opponents and remains separate
  from validation and training.
- [x] **Retain enough eligible frozen V1 PPO/rule-based bots.** The current
  historical manifests are large enough for the existing benchmark protocol.
- [x] **Version the current manifests.** Both `decks/validation_opponents.json`
  and `decks/holdout_opponents.json` are tracked; never replace a frozen model at
  the same path during an experiment series.
- [x] **Finish and freeze the new V6 comparison sets.** The Compact/Potential
  factory completed all 16 targets, evaluated every target for 240 games with
  zero crashes, and wrote disjoint training/validation/holdout manifests with
  exact model hashes. The V6 final holdout remains inactive until selection is
  complete; the old V1 holdout remains historical evidence.
- [x] **Evaluate 100 games/opponent for normal selection, 200 for close calls.**
  The recorded V6 architecture comparison ran 100 games against each of eight
  validation opponents, balanced 50/50 between player 0 and player 1, for 800
  games per candidate without crashes. Close future promotion decisions still
  require 200 games per opponent.

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

### P1 — evaluate and improve the rule-based bot

- [ ] **Create a benchmark matrix for the rule bot.** Run it against every PPO
  candidate and across at least 5 distinct decks, with 100 games per pair and
  balanced perspectives.
- [x] **Record decision categories, not just W/L.** At minimum: opening setup,
  active energy attachment, bench energy attachment, evolution, attack choice,
  retreat/switch, search target and end-turn decision.
- [x] **Add deterministic scenario tests.** Construct small game states where a
  clearly legal best action is known; tests should assert the rule bot's chosen
  action rather than relying on aggregate win rate.
- [ ] **Use replay review for losses.** Classify a loss as setup, resource,
  target-selection, tempo, deckout or rules-coverage failure before adding a
  heuristic.
- [x] **Apply it to other decks through card metadata, not deck IDs.** Rules
  should query attack costs, damage, retreat cost, evolution relation, board
  state and legal options. Deck-specific overrides are acceptable only as named,
  tested exceptions.
- [x] **Keep the rule bot deterministic.** If two legal actions tie, use a
  documented stable tie-breaker. Random fallback makes it a noisy benchmark.

The rule bot is appropriate as a baseline, regression test and weak curriculum
anchor. It should not dominate PPO training: a policy that beats one heuristic
can still fail against a diverse PPO league.

#### Rule-v4 meta league and coefficient tuning

`decks/rule_bot_meta_pool_v1.json` defines 20 deterministic Rule-v4 opponents:
two variants for each of the eight leading archetypes in the local Kaggle
`>450` report and four rotating tail opponents. Its meta weights reproduce the
265-match archetype distribution; training probabilities mix 60% meta
frequency, 20% uniform coverage and a 20% PFSP allocation. Exact validation and
holdout deck lists are rejected by tests. Full reconstructed Kaggle lists supply
the missing Grimmsnarl, Archaludon and independent Starmie decks.
`decks/rule_bot_training_pool_v1.json` is the directly loadable training form
with those derived initial probabilities; use it with `--opp-pool` and optional
`--pfsp-lite`.

Rule-v4 model specs keep legacy aliases compatible and add explicit archetype,
variant and optional bounded coefficient overrides, for example
`rule_based:v4:dragapult:tempo?attack_knockout=42`. Tactical wins and knockouts
are hard priorities, so a generic setup preference can no longer suppress a
winning attack. Decisions report auditable categories for setup, attachment,
evolution, attacks, retreat, target selection and end-turn behavior.

Validate the matrix without playing games, run a small integration smoke test,
then launch the full balanced-perspective benchmark:

```bash
venv/bin/python scripts/benchmark_rule_bots.py --dry-run
venv/bin/python scripts/benchmark_rule_bots.py --smoke \
  --output reports/rule_bot_benchmark_v1_smoke.json
venv/bin/python scripts/benchmark_rule_bots.py --games 100 \
  --output reports/rule_bot_benchmark_v1.json
```

`decks/rule_bot_generalization_v1.json` repeats the same eight core archetypes
on deck lists excluded from both the tuning pool and the PPO reserved sets:

```bash
venv/bin/python scripts/benchmark_rule_bots.py \
  --pool decks/rule_bot_generalization_v1.json --games 100 \
  --output reports/rule_bot_generalization_v1.json
```

The coefficient tuner uses a seeded cross-entropy population, a rule/PPO
development league, worst-matchup and safety penalties, and a hall of fame. It
does not read the validation or holdout manifests:

```bash
venv/bin/python scripts/tune_rule_bots.py \
  --archetype dragapult --variant tempo \
  --deck decks/deck_bank/bank_10.csv --dry-run
venv/bin/python scripts/tune_rule_bots.py \
  --archetype dragapult --variant tempo \
  --deck decks/deck_bank/bank_10.csv
```

### P1 — controlled PPO and reward optimization

- [ ] **Repeat the two historical fine-tuning arms as a clean V6 ablation:**
  `epochs2` and `aux0`. Historical V5/V5b artifacts already exist;
  do not treat them as a controlled V6 result or spend compute repeating them
  before the P0 fixes and factory freeze.
- [ ] **Promote a V6 winner only through the validation gate.** Keep the parent
  and all arms in the arena until the regression picture is clear; never use the
  final holdout to choose an arm.
- [ ] If no arm wins convincingly, test one additional factor at a time:
  `gamma` (`0.995` vs `0.999`), entropy coefficient, then clip range.
- [x] **Enforce the P0 training-health gate for every arm.** Review KL, clip
  fraction, entropy, explained variance, value loss, auxiliary loss, invalid
  learner actions, option overflow and native engine errors. A corrupted or
  collapsed run is not a candidate even if one small evaluation looks good.
- [ ] Do reward ablations, not intuition-only reward edits. Compare terminal
  reward, current weak shaping, and one potential-based alternative from the
  same parent with the same evaluation league.

#### PFSP-lite historical league

PFSP-lite is the default for training pools and updates weights only from
completed training games after the P0 masks and opponent labels are trustworthy.
Use `--no-pfsp-lite` only for controlled static-baseline comparisons:

- [x] Aggregate games, wins, losses and uncertainty per training opponent over
  each completed segment; do not use validation or holdout games to set weights.
- [x] Prefer under-sampled opponents and matchups with neither near-certain wins
  nor near-certain losses, while retaining a random-sampling floor.
- [x] Cap the probability of any one opponent and retain frozen historical and
  rule-based opponents so that one weakness cannot cause catastrophic
  forgetting.
- [ ] Insert a new policy snapshot only after it passes a validation gate or
  adds a materially different cross-play vector; keep roughly 5--10 useful
  snapshots instead of every checkpoint.
- [ ] Compare static sampling and PFSP-lite from the same parent, seed, step
  budget and training manifest before adopting it.

The prepared Deck 38 ablation uses the frozen Deck 38 parent, a 1,000,000-step
budget, the same seed, and the versioned six-opponent pool in
`experiments/2026-07/deck38_static_pfsp_pool_20260718.json`. It adds the
measured V6 Alakazam snapshot (bank 54) and the available historical V5b Mega
Lucario snapshot (bank 18) to the active V6 pool, without touching the frozen
factory configuration. The runner creates byte-identical copies of the parent,
then changes only PFSP-lite sampling in the second arm:

```bash
venv/bin/python scripts/run_training_pool_ablation.py --dry-run
venv/bin/python scripts/run_training_pool_ablation.py --wandb-mode online
```

It checks both reserved leagues before it starts, applies the training-health
gate to both arms, evaluates only against the validation league, and writes a
comparison report before creating either `.complete` marker.

#### Next training iteration: adaptive stopping

The next training improvement should be a run-level controller instead of
another unconditional increase in the step budget. The existing `target_kl`
protects an individual PPO update from an excessively large policy change; it
does not stop a whole run when KL remains very small.

- [x] Add a low-KL early-stopping callback with a minimum training budget and a
  patience window. Stop only when rolling `train/approx_kl` remains below a
  configured threshold for several consecutive PPO updates; never stop on one
  quiet rollout.
- [x] Combine low KL with entropy-loss stagnation. The patience counter advances
  only while KL is below its threshold; once the window is full, a linear trend
  over the rolling `train/entropy_loss` values must also be nearly flat.
- [x] Log the thresholds, minimum steps, patience counter, actual final step and
  stop reason to W&B and `models/experiments/`. An external interrupt, worker
  crash or engine error must remain failed/incomplete and must never create a
  `.complete` marker.
- [x] Save the model normally when the configured early-stop condition is met,
  then pass it through the same validation gate as a fixed-budget model.
- [ ] Run the first comparison as a controlled ablation: identical parent,
  seed, opponent pool and maximum budget, with only adaptive stopping changed.
  Adopt it only if it saves compute without reducing validation strength or
  increasing perspective bias.

The controller remains opt-in via `--adaptive-stop`; defaults are a 250,000-step
minimum, eight qualifying PPO updates, KL below `0.001`, and an absolute
entropy-loss trend of at most `0.002` per update across the rolling window. The
legacy CLI spelling
`--adaptive-entropy-delta` remains an alias for `--adaptive-entropy-trend`.
`scripts/run_adaptive_stopping_ablation.py` creates both arms
from byte-identical copies of one parent, runs the shared validation evaluator,
and writes `adoption_decision.json`. It leaves both outputs incomplete if
training is interrupted or validation crashes, and recommends adoption only
when compute is saved without a validation-strength or perspective-gap
regression.

The first experiment is prepared around the current 2M-step PFSP Alakazam
parent, its frozen development pool and seed `20260721`. Review its exact
commands without spending compute, then start it when ready:

```bash
venv/bin/python scripts/run_adaptive_stopping_ablation.py --dry-run
venv/bin/python scripts/run_adaptive_stopping_ablation.py --wandb-mode online
```

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

Current implementation status:

- [x] The opponent factory can copy compatible PPO foundations and fine-tune
  separate target models without overwriting the parent.
- [ ] Compare V6 transfer against scratch with equal target-deck budget and the
  same frozen evaluation league.
- [ ] Build a replay/demonstration dataset containing the actor-visible
  observation, legal mask, selected legal option, perspective and outcome.
- [ ] Only then run behaviour cloning as a warm-start ablation; keep DAgger and
  human labelling out of the first experiment.

#### V6 opponent factory

The opponent factory builds four independent compact-action V6 foundations,
evaluates them, selects the best three that pass strength and perspective gates,
and only then fine-tunes frozen opponents for three disjoint purposes. If fewer
than three pass the first evaluation, it trains Bases E and F once and repeats
the evaluation across all six candidates. V6 has
exactly 66 actions: option indices `0..64` and STOP at `65`. V5/V5b checkpoints
remain valid opponents but are intentionally incompatible as V6 training
parents.

Current Compact/Potential factory status (2026-07-17):

- [x] Four primary V6 bases were trained and evaluated. All four passed the
  configured gates; Bases A, C and B were selected. Fallback Bases E and F were
  not needed.
- [x] Complete all 16 target fine-tunes: 4/4 training, 6/6 validation and 6/6
  holdout targets have `.complete` markers.
- [x] Evaluate every target, freeze the models and generate/version the three
  V6 manifests with exact model hashes. The recorded target evaluation contains
  240 games per target and zero crashes.

```bash
# Validate every source, split and command without starting training.
venv/bin/python scripts/run_opponent_factory.py --dry-run --wandb-mode offline

# The full run uses online W&B by default. Log in before starting it.
venv/bin/wandb login
venv/bin/python scripts/run_opponent_factory.py --wandb-mode online
```

Individual phases can be resumed safely:

```bash
venv/bin/python scripts/run_opponent_factory.py --stage bases
venv/bin/python scripts/run_opponent_factory.py --stage evaluate-bases
venv/bin/python scripts/run_opponent_factory.py --stage targets --split training
venv/bin/python scripts/run_opponent_factory.py --stage targets --split validation
venv/bin/python scripts/run_opponent_factory.py --stage targets --split holdout
venv/bin/python scripts/run_opponent_factory.py --stage freeze
```

Completed outputs have a sibling `.complete` marker. An existing incomplete
output causes the factory to stop instead of silently continuing it; inspect it
and use `--force` only when deliberately restarting that arm. Target training
is blocked until at least three bases pass the configured evaluation gates.

After freezing, inspect the three files under
`decks/generated/opponent_factory_v6/`. The final holdout models live below
`models/holdout/v6/`, so the arena cannot schedule them. Activate the staged
validation and final holdout only after all target runs and independent smoke
evaluations have passed. The old manifests remain historical V1 benchmarks.

The completed architecture ablation compared Full Base A with Compact,
Balanced and Compact-without-legacy variants under an equal budget and over 800
validation games per candidate. Full won; Compact remains the compute-efficient
opponent option, and removing the legacy 1,500-value vector is rejected for now.

```bash
venv/bin/python scripts/run_v6_architecture_ablation.py \
  --wait-for-base-a \
  --stop-factory-screen pokemon_opponent_factory_v6_bases \
  --steps 1000000 --games 30 --wandb-mode online
```

The command is retained for reproducibility. Its watcher stops queued Full
bases after Base A, trains the configured alternatives with the same deck,
seed, league and PPO settings, writes parameter profiles, and evaluates them on
the frozen validation manifest. The final holdout is not touched during this
architecture decision.

### P1 — Observation and Action V6

- [x] **Implement the compact V6 action space.** Current V6 models and the
  opponent factory use option indices `0..64` plus STOP at `65`; the 1,000-action
  path remains for V5 checkpoint compatibility.
- [x] **Make V5 and V6 explicitly incompatible for continuation training.** A
  V5 actor is not silently padded or truncated into a V6 action space; V5 models
  remain valid frozen opponents and benchmarks.
- [x] **Make V6 the training default.** The general training CLI and its default
  profile regression test now select V6 with 66 actions and the Compact feature
  variant. V5 remains available explicitly for compatible historical models.
- [x] **Measure option-count coverage.** Every training and evaluation records
  the maximum plus p50/p90/p95/p99 learner option counts, overflow count and a
  zero-overflow acceptance gate. If real games exceed 65 choices, increase the
  encoded range before treating the model as a candidate.
- [x] **Ablate the legacy scalar vector.** Removing it reduced validation score
  from 88.75% to 64.63%; retain it until the missing global information is
  represented structurally.
- [ ] **Keep hidden information honest.** Own deck composition and publicly
  revealed/discarded cards are valid inputs; opponent hand and unrevealed prize
  identities must never leak into observations. Add mirrored-state tests that
  change simulator-private truth while actor-visible inputs remain unchanged.
- [ ] **Run one small relational ablation after P0.** Add one or two lightweight
  self-attention/message-passing blocks over the twelve field entities while
  keeping PPO, LSTM, reward, deck and opponent pool fixed.
- [ ] **Defer temporal Transformer/GTrXL, asymmetric critic and PBT.** Revisit
  them only if the smaller entity-attention arm wins across seeds and the
  training/evaluation pipeline is stable.

#### Encoder FFN hyperparameter experiments

Run these only after the P0 health counters are available. Both experiments
must start from the same data split, deck, opponent pool, seed set, PPO/LSTM
settings and step budget. They require fresh model paths because changing the
encoder FFN changes checkpoint tensor shapes.

- [ ] **Experiment 1 — encoder FFN width.** Keep the current Compact encoder as
  the control (`combined -> 512 -> 256`) and compare hidden widths 256 and 768.
  Change no other architecture or PPO hyperparameter. Record parameter count,
  training FPS, peak memory, validation Wilson lower bound, worst matchup and
  perspective gap; keep a variant only if its strength/compute trade-off is
  better across at least three seeds.
- [ ] **Experiment 2 — encoder FFN depth.** Use the winning width from Experiment
  1, keep ReLU and the 256-dimensional encoder output fixed, and compare the
  one-hidden-layer control with a two-hidden-layer FFN. Match all training and
  evaluation settings and reject the deeper arm if it only adds parameters or
  latency without a repeatable validation gain.

### P0/P1 — missing test coverage

- [ ] **Run the functional observation tests under pytest in CI.** Local audit
  runs passed 35 targeted tests plus 29 generated subtests, but the repository
  still needs a reproducible CI command and declared test dependencies.
- [ ] **Environment invariants:** run long random/legal rollouts and assert
  eventual termination, legal action masks, no duplicate selected option,
  non-negative zone sizes, 60-card conservation and correct winner perspective.
- [ ] **Perspective symmetry:** mirror a fixed game state and verify that
  observations/rewards map consistently between player 0 and player 1. This is
  a P0 gate because the audit reproduced invalid masked actions only for the
  rotated learner perspective.
- [ ] **Selection edge cases:** test min/max counts, STOP legality, repeated
  selections, zero-option states and more-than-65-option behavior explicitly.
- [ ] **Evaluation tests:** verify multi-candidate progress accounting,
  validation/final manifest selection, promotion rejection on perspective gap,
  and champion marking in the API/dashboard payload.
- [ ] **Rule bot scenario tests:** one deterministic test for each decision
  category listed above, plus a regression replay corpus for previously fixed
  mistakes.
- [ ] **Model compatibility tests:** A synthetic 1,000-action rejection test
  exists. Extend it to load one real frozen V5 and one real V6 archive end to
  end, verifying both valid same-version loads and clear cross-version failures.
- [ ] **Native engine RNG:** expose a seed API in the CG wrapper if possible.
  Until then, report evaluations as repeated stochastic estimates, not exact
  paired-seed experiments.

### Definition of done for the next model generation

- No sampled masked action is rejected in long rollouts from either learner
  perspective, including pending multi-selection states.
- Every candidate has a passing persisted training/evaluation health record:
  zero invalid learner actions, option overflows, native engine errors and
  evaluation crashes.
- The native encoder has automated parity tests plus a safe Python fallback, and
  a clean Kaggle/Linux environment imports and runs it from declared
  dependencies.
- Fresh V6 training, validation and final-holdout manifests are disjoint,
  versioned and reference frozen model hashes.
- All 16 factory targets are complete and independently smoke-tested.
- The rule bot has a scenario suite and benchmark matrix.
- The three V6 fine-tune arms have completed with 100-game/opponent validation.
- One champion has passed the promotion gate and remains stable in arena matches.
- A final-holdout report is produced once, after all selection decisions.
- V6 has real archived-checkpoint compatibility, option-overflow and hidden-
  information tests before replacing V5 as the default architecture.
