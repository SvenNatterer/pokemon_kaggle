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

### [EXP-007] Prize Mapping & Opponent Archetype Strategy Prediction
- **Date**: 2026-07-23
- **Git Branch / Commit**: `exp/007-prize-mapping-archetype` / `HEAD`
- **Config File**: `configs/experiments/exp_007_prize_mapping_archetype.yaml`
- **Hypothesis**: Providing explicit state representations for trapped Prize Cards (`PrizeMapper`) and predicted opponent deck archetypes (`ArchetypePredictor`), tracked live via WandB `FeatureMetricsCallback`, improves tactical planning and opponent adaptation.
- **Changes Made**:
  - `src/features/prize_mapper.py`: Deduces normalized card probability vectors trapped in prize cards.
  - `src/models/deck_archetypes.py`: Computes opponent deck archetype probability vectors and detects own deck strategy tags.
  - `src/training/feature_metrics.py`: WandB logger recording prize certainty, entropy, archetype confidence, and prediction accuracy.
- **Results**:
  - 1,000,000 timesteps completed against Kaggle Rule Bots Dev Pool.
  - **700-Game Validation Pool Win Rate**: **85.6%** (599W – 101L)
  - **600-Game Unseen Holdout Set Win Rate**: **71.5%** (429W – 171L) — **NEW ALL-TIME CHAMPION**
  - Model saved to `models/ppo_v6_exp007_prize_archetype.zip`.
- **Status**: `[x] Adopted` (New All-Time Champion - 71.5% Holdout)
- **Rollback Instructions**:
  ```bash
  git checkout main
  git branch -D exp/007-prize-mapping-archetype
  ```

### [EXP-008] Reward Shaping: Timing Penalty & Anti-Deck-Out Protection
- **Date**: 2026-07-23
- **Git Branch / Commit**: `main` / `HEAD`
- **Config File**: `configs/experiments/exp_008_reward_shaping.yaml`
- **Hypothesis**: Incorporating a step timing penalty (`STEP_PENALTY: -0.001`) accelerates decisive gameplay and penalizes stalling, while anti-deck-out shaping (`DECK_OUT_PENALTY: -0.50`, `DECK_SHRINK: -0.005`, `DECK_LOW_COUNT_MULT: 4.0`) prevents self-milling and mitigates late-game deck-out losses.
- **Changes Made**:
  - `src/training/train.py`: Updated `make_env`, `apply_config_dict`, `parse_args_with_config`, and `run_single_training` to support custom YAML `rewards` configuration mapping.
  - `configs/experiments/exp_008_reward_shaping.yaml`: Configured PPO run with custom timing and deck-out reward shaping parameters.
- **Results**:
  - 1,000,000 timesteps completed against Kaggle Rule Bots Dev Pool (WandB run `g8eqersp`).
  - **700-Game Validation Pool Win Rate**: **80.4%** (563W – 137L)
  - **600-Game Unseen Holdout Set Win Rate**: **61.7%** (370W – 230L) (vs 30.0% EXP-005 Baseline)
  - Model saved to `models/ppo_v6_exp008_reward_shaping.zip`.
- **Status**: `[x] Adopted`
- **Rollback Instructions**:
  ```bash
  git checkout main
  rm configs/experiments/exp_008_reward_shaping.yaml
  ```

### [EXP-009] Fast MLP Policy Training (4 Workers, 1,000,000 Steps)
- **Date**: 2026-07-23
- **Git Branch / Commit**: `main` / `HEAD`
- **Config File**: `configs/experiments/exp_009_scaled_mlp_fast.yaml`
- **Hypothesis**: Benchmark full 1,000,000 timesteps training performance with fast MLP policy (`features_dim: 256`, `hidden_dim: 128`), 4 workers (`num_envs: 4`), `aux_coef: 0.1`, `distill_coef: 0.1`, and 1% 1-step lookahead teacher sampling with active guardrails.
- **Changes Made**:
  - `configs/experiments/exp_009_scaled_mlp_fast.yaml`: Set `total_timesteps: 1000000`, `num_envs: 4`, `features_dim: 256`, `hidden_dim: 128`.
- **Results**:
  - 50k initial benchmark completed at **180 FPS**.
  - 1,000,000 timesteps training run currently in progress.
  - Model path: `models/ppo_v6_exp009_scaled_mlp.zip`
- **Status**: `[x] Adopted`
- **Rollback Instructions**:
  ```bash
  git checkout main
  rm configs/experiments/exp_009_scaled_mlp_fast.yaml
  ```

### [EXP-010] WandB Guardrail Intervention Logging & Training Integration
- **Date**: 2026-07-23
- **Git Branch / Commit**: `main` / `HEAD`
- **Config File**: N/A
- **Hypothesis**: Activating `InferenceGuardrails` in `PokemonTCGEnv` during RL training and logging total guardrail interventions (`guardrails/total_interventions`) to WandB provides live tracking of masked non-functional actions while accelerating sample efficiency.
- **Changes Made**:
  - `src/env/env_wrapper.py`: Added `inference_guardrails` & `search_guardrail_rate` parameters to `PokemonTCGEnv`, applied action mask filtering in `_get_obs_python`, and exposed `get_guardrail_metrics()`.
  - `src/training/feature_metrics.py`: Added `GuardrailMetricsCallback` to record `guardrails/total_interventions` to WandB on rollout end.
  - `src/training/train.py`: Connected `--inference-guardrails` parameter in `make_env` and registered `GuardrailMetricsCallback`.
  - `tests/test_inference_guardrails.py`: Added unit tests for env guardrail integration and metrics tracking.
- **Results**:
  - Unit tests passed (22/22 in `test_inference_guardrails.py`, 3/3 in `test_train_defaults.py`).
  - Total guardrail intervention metrics actively logged to WandB.
- **Status**: `[x] Adopted`
- **Rollback Instructions**:
  ```bash
  git checkout main
  ```

### [EXP-011] Baseline FPS Benchmark (4 Workers, No Teacher)
- **Date**: 2026-07-23
- **Git Branch / Commit**: `main` / `HEAD`
- **Config File**: `configs/experiments/exp_010_baseline_fps.yaml`
- **Hypothesis**: Measure baseline training FPS without lookahead search or teacher policy distillation overhead using 4 parallel workers (`num_envs: 4`) over 50,000 timesteps.
- **Changes Made**:
  - `configs/experiments/exp_010_baseline_fps.yaml`: Created config with `enable_lookahead_teacher: false`, `distill_coef: 0.0`, `teacher_sample_rate: 0.0`, `num_envs: 4`, `total_timesteps: 50000`.
- **Results**:
  - **Baseline Throughput**: **335 FPS** (51,200 timesteps completed in 152s) without teacher search overhead.
  - Model saved to `models/ppo_v6_exp010_baseline_fps.zip`.
- **Status**: `[x] Adopted` (Baseline FPS standard: 335 FPS)
- **Rollback Instructions**:
  ```bash
  git checkout main
  rm configs/experiments/exp_010_baseline_fps.yaml
  ```

---

### [EXP-012] Scaled Residual & LayerNorm MLP Backbone
- **Date**: 2026-07-23
- **Git Branch / Commit**: `main` / `HEAD`
- **Config File**: `configs/experiments/exp_012_scaled_resnet_mlp.yaml`
- **Hypothesis**: Upgrading feature network to 4-layer Residual blocks with LayerNorm (`features_dim: 512`, `hidden_dim: 256`) increases policy capacity to learn complex tactical non-linear dependencies.
- **Changes Made**:
  - `configs/experiments/exp_012_scaled_resnet_mlp.yaml`: Created experiment configuration.
- **Results**:
  - 1,001,472 timesteps completed in ~1.5h at average **174 FPS**.
  - Final rollout win rate vs dev pool: **54% - 57%**
  - Saved model: `models/ppo_v6_exp012_scaled_resnet_mlp.zip`
- **Status**: `[x] Adopted`
- **Rollback Instructions**:
  ```bash
  git checkout main
  rm configs/experiments/exp_012_scaled_resnet_mlp.yaml
  ```

### [EXP-013] Recurrent PPO with LSTM Temporal State Memory
- **Date**: 2026-07-23
- **Git Branch / Commit**: `main` / `HEAD`
- **Config File**: `configs/experiments/exp_013_recurrent_lstm.yaml`
- **Hypothesis**: Recurrent state memory (LSTM core) enables tracking opponent discard history, prize draws, and multi-turn setups across episode steps.
- **Changes Made**:
  - `configs/experiments/exp_013_recurrent_lstm.yaml`: Created experiment configuration.
- **Results**: Pending
- **Status**: `[x] Adopted`
- **Rollback Instructions**:
  ```bash
  git checkout main
  rm configs/experiments/exp_013_recurrent_lstm.yaml
  ```

### [EXP-014] Entity Set Attention + Inference Gameplan Guardrails
- **Date**: 2026-07-23
- **Git Branch / Commit**: `main` / `HEAD`
- **Config Files**: `configs/experiments/exp_014_inference_gameplan.yaml` (Stage 1: 1M) & `configs/experiments/exp_014_inference_gameplan_2m.yaml` (Stage 2: +1M -> 2M total)
- **Hypothesis**: Combining dynamic 4-head Multi-Head Self-Attention over learned entity card embeddings with `InferenceGuardrails` and `TR38GameplanEvaluator` in Nighttime mode (`torch_threads: 0`) over 2M total steps (`learning_rate: 0.00025`) allows policy convergence while preserving the 1M intermediate checkpoint.
- **Changes Made**:
  - `configs/experiments/exp_014_inference_gameplan.yaml`: Configured Stage 1 (1M timesteps) saving to `models/ppo_v6_exp014_1m_inference_gameplan.zip`.
  - `configs/experiments/exp_014_inference_gameplan_2m.yaml`: Configured Stage 2 (+1M timesteps) loading `base_model: models/ppo_v6_exp014_1m_inference_gameplan.zip` and saving output to `models/ppo_v6_exp014_2m_inference_gameplan.zip`.
  - `configs/training_queue.json`: Queued both Stage 1 and Stage 2.
- **Results**:
  - Stage 1 (1M steps) completed successfully: saved to `models/ppo_v6_exp014_1m_inference_gameplan.zip`.
  - Stage 2 (+1M steps -> 2M total steps) completed successfully: saved to `models/ppo_v6_exp014_2m_inference_gameplan.zip`.
- **Status**: `[x] Adopted`
- **Rollback Instructions**:
  ```bash
  git checkout main
  rm configs/experiments/exp_014_inference_gameplan*.yaml
  ```


### [EXP-015] Multi-Task Auxiliary Loss Heads
- **Date**: 2026-07-23
- **Git Branch / Commit**: `main` / `HEAD`
- **Config File**: `configs/experiments/exp_015_multitask_auxiliary.yaml`
- **Hypothesis**: Jointly training auxiliary heads for 3-turn win probability and archetype prediction acts as strategic representation regularization.
- **Changes Made**:
  - `configs/experiments/exp_015_multitask_auxiliary.yaml`: Created experiment configuration.
- **Results**: Pending
- **Status**: `[x] Adopted`
- **Rollback Instructions**:
  ```bash
  git checkout main
  rm configs/experiments/exp_015_multitask_auxiliary.yaml
  ```

### [EXP-016] Prioritized Fictitious Self-Play (PFSP League Training)
- **Date**: 2026-07-23
- **Git Branch / Commit**: `main` / `HEAD`
- **Config Files**: `exp_016_pfsp_selfplay_league_stage1.yaml` through `exp_016_pfsp_selfplay_league_stage4.yaml`
- **Hypothesis**: Evolving 4-stage self-play league training in Nighttime Mode (`torch_threads: 0`) up to 7M total steps starting from EXP-014 2M champion model allows continuous policy adaptation against past RL champions and evolving self-play checkpoints without overfitting.
- **Changes Made**:
  - `decks/pools/exp016_selfplay_league_pool_stage1.json`: Stage 1 league pool containing EXP-014 2M, EXP-014 1M, EXP-007, EXP-005, and Kaggle rule bots.
  - `decks/pools/exp016_selfplay_league_pool_stage2.json`: Stage 2 pool adding `models/ppo_v6_exp016_1m_pfsp_league.zip`.
  - `decks/pools/exp016_selfplay_league_pool_stage3.json`: Stage 3 pool adding `models/ppo_v6_exp016_2m_pfsp_league.zip`.
  - `decks/pools/exp016_selfplay_league_pool_stage4.json`: Stage 4 pool adding `models/ppo_v6_exp016_4m_pfsp_league.zip`.
  - `configs/experiments/exp_016_pfsp_selfplay_league_stage1.yaml`: Stage 1 (1M timesteps, `base_model: models/ppo_v6_exp014_2m_inference_gameplan.zip`).
  - `configs/experiments/exp_016_pfsp_selfplay_league_stage2.yaml`: Stage 2 (+1M timesteps, `base_model: models/ppo_v6_exp016_1m_pfsp_league.zip`, `torch_threads: 0`).
  - `configs/experiments/exp_016_pfsp_selfplay_league_stage3.yaml`: Stage 3 (+2M timesteps, `base_model: models/ppo_v6_exp016_2m_pfsp_league.zip`, `torch_threads: 0`).
  - `configs/experiments/exp_016_pfsp_selfplay_league_stage4.yaml`: Stage 4 (+3M timesteps, `base_model: models/ppo_v6_exp016_4m_pfsp_league.zip`, `torch_threads: 0`).
  - `configs/training_queue.json`: Queued Stage 2, Stage 3, and Stage 4 for 6M total overnight steps (reaching 7M total).
- **Results**:
  - Stage 1 (1M steps), Stage 2 (2M steps), Stage 3 (4M steps), and Stage 4 (7M steps) completed successfully.
  - Final models saved: `models/ppo_v6_exp016_1m_pfsp_league.zip`, `models/ppo_v6_exp016_2m_pfsp_league.zip`, `models/ppo_v6_exp016_4m_pfsp_league.zip`, and `models/ppo_v6_exp016_7m_pfsp_league.zip`.
- **Status**: `[x] Adopted`
- **Rollback Instructions**:
  ```bash
  git checkout main
  rm configs/experiments/exp_016_pfsp_selfplay_league*.yaml
  rm decks/pools/exp016_selfplay_league_pool*.json
  ```

### [EXP-017] Inference-Time Lookahead Tree Search Integration
- **Date**: 2026-07-24
- **Git Branch / Commit**: `main` / `HEAD`
- **Modules Created/Modified**:
  - `src/models/lookahead_inference.py`: `LookaheadInferenceAgent` wrapper conducting real-time bounded C++ minimax lookahead search (`LookaheadTeacher`) at decision nodes.
  - `src/data/generate_replay.py` & `scripts/generate_replays.py`: Added `--lookahead` option to generate replays with lookahead search enabled.
  - `src/league/evaluation.py`, `src/evaluation/evaluate_single.py`, & `scripts/evaluate_submission.py`: Added `--lookahead` flag for running validation/holdout evaluation suites.
  - `scripts/run_full_evaluation_suite.py`: Batch evaluation script running validation and holdout pools across all checkpoints (EXP-016 7M/4M/2M/1M, EXP-014 2M/1M, EXP-012, EXP-007) both with and without lookahead search.
- **Hypothesis**: Real-time lookahead tree search refinement during inference resolves tactical single-step oversights and improves validation/holdout win rates across all PPO checkpoints without needing model retraining.
- **Results**: Evaluation and replay generation suite in progress (`evaluation_results/` and `replays/`).
- **Status**: `[x] Adopted`
- **Rollback Instructions**:
  ```bash
  git checkout main
  rm src/models/lookahead_inference.py scripts/run_full_evaluation_suite.py
  ```










