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
