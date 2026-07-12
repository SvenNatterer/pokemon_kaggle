# V5b night training notes

The new effect/card relation encoder is incompatible with the existing
`models/ppo_v5_deck_bank_18.zip`. The curriculum therefore defaults to the new
target `models/ppo_v5b_deck_bank_18.zip`. Do not point it back at the old V5
checkpoint: loading that checkpoint fails because the encoder dimensions changed.

## Before the full run

Validate all files and commands:

```bash
bash scripts/train_v5_curriculum_bank18.sh --dry-run
```

Run a one-rollout smoke test with a separate disposable model name:

```bash
WANDB_MODE=disabled MODEL=models/ppo_v5b_smoke_bank_18.zip \
STAGE1_STEPS=16384 START_STAGE=1 END_STAGE=1 \
bash scripts/train_v5_curriculum_bank18.sh
```

The smoke test must finish, save, and reload successfully before starting the
full curriculum. The full run is:

```bash
bash scripts/train_v5_curriculum_bank18.sh
```

The league consists mainly of frozen PPO opponents. The generic `heuristic`
agent is included only as a low-weight stability anchor because there is no
separate suite of rule-based league bots.

Default total budget is 4.1M environment steps. Expected wall time is uncertain
because the encoder became much larger. Measure FPS in the smoke run; approximate
hours are `4,100,000 / effective_FPS / 3600`. Rollout collection in
the first smoke test was about 309 FPS, but four PPO epochs made the update the
dominant cost. The overnight curriculum therefore defaults to one PPO epoch and
spends the saved compute on fresher environment experience. Override `N_EPOCHS=2`
for a longer follow-up run if evaluation shows underfitting.

If the full run cannot fit overnight, use `END_STAGE=7`; stages 8-9 can be resumed
later with `START_STAGE=8`. Never start at stage 8 unless stages 1-7 completed and
the target model exists.
