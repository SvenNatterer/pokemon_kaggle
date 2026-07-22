# Experiment Log

This document tracks all hyperparameter, architectural, and reward experiments. Every experiment should be recorded here with Git commit references and rollback steps.

---

## Experiment Template

Copy this block when starting a new experiment:

```markdown
### [EXP-XXX] <Title / Short Description>
- **Date**: YYYY-MM-DD
- **Git Branch / Commit**: `exp/xxx` / `<commit-hash>`
- **Config File**: `configs/experiments/<config_name>.yaml`
- **Hypothesis**: <What change is being tested and expected outcome>
- **Changes Made**:
  - `src/...`: <Brief explanation>
- **Results**:
  - Win Rate vs Dev Pool: XX%
  - Average Reward: XX
- **Status**: `[ ] Adopted` | `[ ] Reverted` | `[ ] In Progress`
- **Rollback Instructions**:
  ```bash
  git checkout main
  # or: git revert <commit-hash>
  ```
```

---

## Experiment Log History

### [EXP-000] Baseline V6 Compact Setup
- **Date**: 2026-07-22
- **Git Branch / Commit**: `main` / `HEAD`
- **Config File**: `configs/train_compact.yaml`
- **Hypothesis**: Baseline environment state space (Compact V6) and default reward configuration.
- **Results**:
  - Baseline model trained and validated against dev pool.
- **Status**: Adopted (Baseline)
- **Rollback Instructions**: N/A (Baseline state)

### [EXP-001] On-The-Fly Lookahead Tree Search Policy Distillation
- **Date**: 2026-07-22
- **Git Branch / Commit**: `exp/001-lookahead-distill` / `HEAD`
- **Config File**: `configs/experiments/exp_001_lookahead_distill.yaml`
- **Hypothesis**: Distilling bounded minimax tree search decisions into the PPO policy at 50% sampling rate on complex decisions improves tactical sequencing and Win Rate vs Dev Pool.
- **Changes Made**:
  - `configs/experiments/exp_001_lookahead_distill.yaml`: Config file with lookahead distillation hyperparameters.
  - `src/training/custom_ppo.py`: Added distillation loss head and rollout teacher sampling.
  - `src/env/env_wrapper.py`: Added teacher target tracking in environment observations.
- **Results**:
  - Training in progress / pending validation.
- **Status**: `[ ] In Progress`
- **Rollback Instructions**:
  ```bash
  git checkout main
  rm configs/experiments/exp_001_lookahead_distill.yaml
  ```

