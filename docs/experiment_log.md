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
- **Git Branch / Commit**: `exp/001-lookahead-distill` / `937242f13`
- **Config File**: `configs/experiments/exp_001_lookahead_distill.yaml`
- **Hypothesis**: Distill bounded depth-3 minimax tree search decisions into PPO policy at 1% sample rate for high training FPS and tactical sequencing balance.
- **Changes Made**:
  - `configs/experiments/exp_001_lookahead_distill.yaml`: Set `teacher_sample_rate: 0.01` (1%), `max_depth: 3`, `beam_width: 2`, `node_budget: 16`.
  - `src/env/env_wrapper.py`: Fixed `teacher_sample_rate` gating so branching heuristic does not bypass sample rate.
  - `src/training/custom_ppo.py`: Added distillation loss head and rollout teacher sampling.
- **Results**:
  - Win Rate vs Dev Pool: 50%
  - **700-Game Validation Pool Win Rate**: **66.7%** (467W-233L)
  - Average Rollout Throughput: ~293 FPS
  - Saved model: `models/ppo_v6_exp001_lookahead_distill.zip`
- **Status**: `[x] Adopted`
- **Rollback Instructions**:
  ```bash
  git checkout main
  rm configs/experiments/exp_001_lookahead_distill.yaml
  ```

### [EXP-002] Strong Lookahead Tree Search Policy Distillation
- **Date**: 2026-07-22
- **Git Branch / Commit**: `exp/002-lookahead-distill-strong` / `HEAD`
- **Config File**: `configs/experiments/exp_002_lookahead_distill_strong.yaml`
- **Hypothesis**: Increasing distillation loss coefficient to `0.3` (3x stronger teacher weighting) accelerates tactical policy imitation without sacrificing RL exploration.
- **Changes Made**:
  - `configs/experiments/exp_002_lookahead_distill_strong.yaml`: Created config with `distill_coef: 0.3`, `teacher_sample_rate: 0.01`.
  - `scratch/queue_experiments.sh`: Queued for execution following EXP-001 completion.
- **Results**:
  - Win Rate vs Dev Pool: 55%
  - **700-Game Validation Pool Win Rate**: **68.4%** (479W-220L-1D)
  - Average FPS: ~280 FPS
  - Model saved to `models/ppo_v6_exp002_lookahead_distill_strong.zip`
- **Status**: `[x] Adopted`
- **Rollback Instructions**:
  ```bash
  git checkout main
  rm configs/experiments/exp_002_lookahead_distill_strong.yaml
  ```

### [EXP-003] 1 Million Step Baseline Training Run (No Teacher)
- **Date**: 2026-07-22
- **Git Branch / Commit**: `main` / `HEAD`
- **Config File**: N/A (Standard PPO baseline CLI parameters)
- **Hypothesis**: Benchmark standard PPO reinforcement learning over 1,000,000 steps without lookahead teacher distillation as a clean long-term comparison baseline.
- **Changes Made**:
  - `scratch/queue_experiments.sh`: Executed 1,000,000 timesteps baseline training.
- **Results**:
  - Win Rate vs Dev Pool: 74%
  - **700-Game Validation Pool Win Rate**: **34.6%** (242W-458L)
  - Average FPS: ~326 FPS
  - Model saved to `models/ppo_v6_1m_baseline.zip`
- **Status**: `[x] Adopted`
- **Rollback Instructions**: N/A (Standard baseline model output `models/ppo_v6_1m_baseline.zip`)

### [EXP-004] Lookahead Teacher Policy Distillation (3% Teacher Sample Rate, aux_coef=0.1)
- **Date**: 2026-07-22
- **Git Branch / Commit**: `main` / `HEAD`
- **Config File**: `configs/experiments/exp_004_lookahead_distill_3pct.yaml`
- **Hypothesis**: Increasing teacher sampling rate to 3% while keeping aux_coef at 0.1 provides higher quality teacher guidance without degrading PPO policy convergence over 500,000 steps.
- **Changes Made**:
  - Created `configs/experiments/exp_004_lookahead_distill_3pct.yaml` with `teacher_sample_rate: 0.03`, `aux_coef: 0.1`, `distill_coef: 0.1`.
- **Results**:
  - Win Rate vs Dev Pool: **65.14%** (228W-122L)
  - **700-Game Validation Pool Win Rate**: **65.71%** (460W-240L)
  - Model saved to `models/ppo_v6_exp004_lookahead_distill_3pct.zip`
- **Status**: `[x] Adopted`
- **Rollback Instructions**:
  ```bash
  git checkout main
  rm configs/experiments/exp_004_lookahead_distill_3pct.yaml
  ```
### [EXP-005] Shallow High-Frequency 1-Step Lookahead Policy Distillation (5% Sample Rate)
- **Date**: 2026-07-22
- **Git Branch / Commit**: `main` / `HEAD`
- **Config File**: `configs/experiments/exp_005_shallow_lookahead_5pct.yaml`
- **Hypothesis**: Shallow 1-step lookahead (`max_depth: 1`) is computationally lightweight, permitting a 5x higher teacher sampling rate (5% vs 1%) for frequent 1-ply tactical distillation over 500,000 steps without dropping training FPS.
- **Changes Made**:
  - Created `configs/experiments/exp_005_shallow_lookahead_5pct.yaml` with `teacher_sample_rate: 0.05`, `max_depth: 1`, `node_budget: 8`.
  - Queued job using `--add-to-queue`.
- **Results**:
  - **500k-Step Dev Validation Win Rate**: 70.29% (492W-208L-0D)
  - **1,000,000-Step Dev Validation Win Rate**: **79.14%** (554W-145L-1D) — **NEW DOMINANT ALL-TIME CHAMPION**
  - **Unseen Holdout Pool Win Rate**: **63.00%** (378W-220L-2D) (vs 32.33% Baseline)
  - Average FPS: ~173 FPS
  - Saved model: `models/ppo_v6_exp005_1m.zip` / `models/ppo_v6_exp005_shallow_lookahead_5pct.zip`
- **Status**: `[x] Adopted` (New Champion - 79.14% Dev / 63.00% Holdout)
- **Rollback Instructions**:
  ```bash
  git checkout main
  rm configs/experiments/exp_005_shallow_lookahead_5pct.yaml
  ```

### [EXP-006] Overfitting & Generalization Detection Concept
- **Date**: 2026-07-22
- **Git Branch / Commit**: `main` / `HEAD`
- **Config File**: `configs/experiments/exp_006_overfit_detection.yaml` (Planned)
- **Hypothesis**: Tracking win-rate gap between self-play training and held-out validation pool (along with normalized action entropy $H(\pi)$) enables automated early stopping before policy memorization and validation win-rate collapse occur.
- **Planned Concept / Architecture**:
  - **Holdout Validation**: Periodic evaluation against held-out bot pool every 50k steps.
  - **Overfit Metric**: $\Delta_{\text{overfit}} = \text{WinRate}_{\text{train\_pool}} - \text{WinRate}_{\text{holdout\_pool}}$. Early stop if gap $> 25\%$.
  - **Action Entropy Monitoring**: Detect rapid entropy drops ($H(\pi) < 0.15$) signaling deterministic action loops.
  - **WandB / 1DB Notification**: Trigger an explicit WandB alert/notification (`overfit/alert_triggered = 1.0`, run tag `overfit_detected`) whenever overfitting condition is met.
  - **Standalone Callback**: Isolated in `src/training/overfit_detector.py` to persist across git rollbacks.
- **Status**: `[ ] Planned` (Documented concept, code implementation pending approval)
- **Rollback Instructions**: N/A (Documentation phase)





