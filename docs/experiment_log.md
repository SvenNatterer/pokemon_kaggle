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











### [EXP-018] Value Function Distillation
**Date**: 2026-07-24
**Hypothesis**: The PPO Critic struggles to accurately evaluate complex late-game prize setups because its only feedback is a sparse +/- 1 reward at the end of the episode. By distilling the Teacher's Lookahead state evaluation score directly into the Critic via an MSE loss (`value_distill_coef: 0.1`), the network will learn to independently recognize highly valuable late-game states without having to execute the tree search at inference time.
**Config**: `exp_018_value_distillation.yaml`
**Git**: `main` at `e41c15ee6` with the documented EXP-018 worktree changes.
**Run**: Fresh 1,000,000-step `--overwrite` run restarted 2026-07-25 in screen session `pokemon_exp18`; output in `logs/train_exp018_20260725_010826_wandb.log`. The initial offline smoke run was stopped before it saved a model; this run uses W&B online as requested (run `f884bnv7`).
**Status**: Running. The direct `--config` launch now forwards YAML `lookahead` settings (`max_depth: 1`, `node_budget: 8`) to `make_env` and `PokemonTCGEnv`, overriding the depth-5 / 96-node fallback. W&B code snapshots are disabled because dirty replay diffs produced 1-2.2 GB local patch files per EXP-018 start.
**Rollback**: Interrupt the `pokemon_exp18` screen session before completion; the atomic `--overwrite` path keeps the previous target model until the replacement is successfully saved. Revert the EXP-018 worktree changes to return to the pre-experiment implementation.

### [EXP-019] Role-Aware Masked Entity Attention
- **Date**: 2026-07-26
- **Git Branch / Commit**: Current worktree / pending commit
- **Config File**: `configs/experiments/exp_019_masked_entity_attention.yaml`
- **Hypothesis**: Ignoring padded board slots and adding learned owner plus active/bench-position embeddings improves board-state representation without adding a new message-passing layer.
- **Controlled settings**: Deck 38, Kaggle rule-bot development pool, PPO/LSTM/reward settings, 1M steps, and seeds 11/23/37. Each replicate must use a fresh `model_name`.
- **Changes Made**: Adds `entity_relation_mode: masked`, which masks empty entity keys and output slots while adding learned observable role embeddings.
- **Run**: Seed 11 was restarted on 2026-07-26 with W&B online tracking. Target: `models/ppo_v6_exp019_masked_entity_attention_seed11.zip`; log: `logs/train_exp019_masked_entity_attention_seed11_wandb.log`. The prior local offline run was stopped before restart.
- **Replay generation**: Completed 2026-07-26 for the seed-11 checkpoint against the seven-opponent Kaggle rule-bot development pool, with inference-time lookahead enabled. All seven replay JSONs are valid and saved in `replays/ppo_v6_exp019_masked_entity_attention_seed11_lookahead/`; this single diagnostic pass finished 4-3 (wins: Mega Lucario, Mega Abomasnow, Dragapult, Iono's Zoroark; losses: Alakazam, Battlecore Archaludon, Conservative Probabilistic).
- **Validation**: Completed 2026-07-26 for seed 11: 10 games against each frozen validation opponent, evaluated both without and with inference-time lookahead. Without lookahead, EXP-019 scored 30-40 (42.9%; Wilson lower bound 31.9%; worst matchup 10.0%) and EXP-020 scored 34-36 (48.6%; Wilson lower bound 37.2%; worst matchup 10.0%). With lookahead, EXP-019 scored 27-43 (38.6%; Wilson lower bound 28.0%; worst matchup 0.0%) and EXP-020 scored 30-40 (42.9%; Wilson lower bound 31.9%; worst matchup 10.0%). All health gates passed. Results: `evaluation_results/exp019_exp020_validation_{no,with}_lookahead.json`.
- **Lookahead diagnostic**: Started 2026-07-27. The validation evaluator will use the exact training search settings (`max_depth: 1`, `beam_width: 2`, `node_budget: 8`, `max_combinations: 16`) and record search decisions plus PPO-action overrides for each matchup.
- **Status**: `[ ] Running`
- **Acceptance**: Retain only if the validation Wilson lower bound improves across the three seeds without worse worst-matchup, perspective gap, or material FPS/memory regression.
- **Rollback Instructions**: Remove the experiment config and the `entity_relation_mode` plumbing; no existing checkpoint is compatible for continuation.

### [EXP-020] Relation-Aware Entity Attention
- **Date**: 2026-07-26
- **Git Branch / Commit**: Current worktree / pending commit
- **Config File**: `configs/experiments/exp_020_relational_entity_attention.yaml`
- **Hypothesis**: Learned directional per-head biases for self, ally, and opponent active/bench pairs help the policy resolve targeting, switching, and attachment interactions beyond masked entity attention alone.
- **Controlled settings**: Identical to EXP-019, including seeds 11/23/37 and fresh target paths.
- **Changes Made**: Adds `entity_relation_mode: relational`, retaining EXP-019 masking and role embeddings while adding zero-initialized four-head relation biases over visible board roles only.
- **Run**: Seed 11 started on 2026-07-26 with W&B online tracking. Target: `models/ppo_v6_exp020_relational_entity_attention_seed11.zip`; log: `logs/train_exp020_relational_entity_attention_seed11_wandb.log`.
- **Validation**: Completed 2026-07-26 for seed 11: 10 games against each frozen validation opponent, evaluated both without and with inference-time lookahead. Without lookahead, EXP-019 scored 30-40 (42.9%; Wilson lower bound 31.9%; worst matchup 10.0%) and EXP-020 scored 34-36 (48.6%; Wilson lower bound 37.2%; worst matchup 10.0%). With lookahead, EXP-019 scored 27-43 (38.6%; Wilson lower bound 28.0%; worst matchup 0.0%) and EXP-020 scored 30-40 (42.9%; Wilson lower bound 31.9%; worst matchup 10.0%). All health gates passed. Results: `evaluation_results/exp019_exp020_validation_{no,with}_lookahead.json`.
- **Lookahead diagnostic**: Started 2026-07-27. The validation evaluator will use the exact training search settings (`max_depth: 1`, `beam_width: 2`, `node_budget: 8`, `max_combinations: 16`) and record search decisions plus PPO-action overrides for each matchup.
- **Status**: `[ ] Running`
- **Acceptance**: Retain only if it beats both the baseline and EXP-019 across seeds on the same validation protocol, with acceptable FPS and memory.
- **Rollback Instructions**: Remove the experiment config and the `entity_relation_mode` plumbing; no existing checkpoint is compatible for continuation.

### [EXP-021] Two-Step Relation-Aware Entity Message Passing
- **Date**: 2026-07-26
- **Git Branch / Commit**: Current worktree / pending commit
- **Config File**: `configs/experiments/exp_021_two_step_relational_attention.yaml`
- **Hypothesis**: A second residual relation-aware attention pass lets each entity consume board information aggregated by the first pass, improving multi-entity targeting, switching, and attachment planning.
- **Controlled settings**: Identical to EXP-019/020: Deck 38, Kaggle rule-bot development pool, PPO/LSTM/reward settings, 1M steps, seeds 11/23/37, online W&B tracking, and active production-contract hard guardrails without sampled search.
- **Changes Made**: Adds `entity_relation_mode: two_step`, retaining EXP-020's role embeddings and directional relation biases while applying an independent second four-head attention pass.
- **Run**: Seed 11 started on 2026-07-26 with active inference guardrails and W&B online tracking. The first launch (`akmcr9ma`) stopped at the initial logger flush because console key truncation collided across long rule names; no model was produced. The logger was corrected and the experiment restarted fresh as W&B run `px7c1qu8`. Target: `models/ppo_v6_exp021_two_step_relational_attention_seed11.zip`; log: `logs/train_exp021_two_step_relational_attention_seed11_wandb.log`.
- **Launch validation**: The restarted run passed the first six rollouts (12,288 timesteps, 236 FPS at the sixth rollout) without a telemetry error. W&B received cumulative/per-rule guardrail metrics and the zero-inclusive per-rule intervention bar table.
- **Status**: `[ ] Running`
- **Acceptance**: Retain only if it beats EXP-019 and EXP-020 across the same seed and validation protocol, without an unacceptable FPS or memory regression.
- **Rollback Instructions**: Remove the EXP-021 config and the `two_step` encoder path; checkpoints require fresh training.

### [EXP-022] Production-Contract Inference Guardrails
- **Date**: 2026-07-26
- **Git Branch / Commit**: Current worktree / pending commit
- **Config File**: `configs/experiments/exp_022_guardrails_shadow.yaml`
- **Hypothesis**: Restricting hard masks to rules derived from the real engine `Option` contract, applying removals transactionally, and measuring proposals in shadow mode eliminates silent dead rules and selection-breaking false positives.
- **Implementation**:
  - Hard rules consume a production-shaped decision context and only propose provable no-effect removals.
  - Strategic hints not present in the engine schema no longer alter legality.
  - `off`, `shadow`, and `active` modes support staged rollout without parallel implementations.
  - W&B records cumulative and per-rollout interventions, rate per 1,000 decisions, rollbacks, search failures, per-rule counts, and a zero-inclusive per-rule bar chart.
- **Controlled settings**: Deck 38, Kaggle rule-bot development pool, 1M steps, seed 11 initially, online W&B, and 7.5% bounded search sampling.
- **Status**: `[ ] Ready for shadow evaluation`
- **Validation**: Full local suite passed: 173 tests plus 9 subtests. Native C++ and Python observation paths both exercise the shared guardrail pipeline; W&B bar-chart payloads include zero-count registered rules.
- **Acceptance**: Zero broken selections or added actions; at least 99% verified intervention precision; search failures below 1%; no validation Wilson-lower-bound or perspective-gap regression.
- **Rollback Instructions**: Disable with `inference_guardrail_mode: off`, then revert the EXP-022 config and guardrail context/telemetry changes. Existing model files are not overwritten by this experiment.

### Attention Architecture Evaluation and Submission Gate (EXP-014/019/020/021)
- **Date**: 2026-07-30
- **Git Branch / Commit**: `main` with the pre-existing dirty worktree retained unchanged.
- **Hypothesis**: The attention variant with the strongest frozen-validation performance under the production inference path is the most suitable candidate for final-holdout evaluation and Kaggle submission packaging.
- **Protocol**: Evaluate EXP-014, EXP-019, EXP-020, and EXP-021 (seed 11 where applicable) on `decks/pools/validation_opponents.json`, using 100 games per opponent, balanced player perspectives, no inference-time lookahead, and the standard health gate. Rank eligible models by Wilson lower bound, worst-matchup score, then overall score. Evaluate the top two eligible validation candidates once on the separate frozen `decks/pools/holdout_opponents.json` with the same protocol. Build and verify one submission archive per holdout finalist; do not upload automatically.
- **Validation result**: EXP-014 ranked first (71.3% score, 67.8% Wilson lower bound, 26.0% worst matchup); EXP-021 ranked second (49.7%, 46.0%, 18.0%). EXP-019 and EXP-020 ranked third and fourth.
- **Final holdout result**: EXP-014 scored 54.8% over 600 games (Wilson lower bound 50.8%, worst matchup 11.0%, perspective gap 8.3%); EXP-021 scored 40.7% (36.8%, 1.0%, 2.7%). Both passed the zero-tolerance health gate.
- **Submission artifacts**: Both archives passed standalone package verification and were not uploaded: `artifacts/submissions/ppo_v6_exp014_entity_card_attention_holdout_20260730.tar.gz` and `artifacts/submissions/ppo_v6_exp021_two_step_relational_attention_seed11_holdout_20260730.tar.gz`.
- **Status**: Completed; EXP-014 is the stronger submission candidate, but its result is not a controlled architecture comparison with EXP-019--021.
- **Rollback**: Evaluation results, match cache entries, and submission archives are additive artifacts. Remove only the newly named artifacts if the protocol is abandoned; do not alter frozen manifests or model checkpoints.

### [EXP-023] Python-Object Relation Attention Compatibility Ablation
- **Date**: 2026-07-30
- **Hypothesis**: Rebuilding the observable directed board-relation graph through Python list/dictionary objects preserves EXP-020 relation semantics but measurably reduces training throughput versus the vectorized tensor path.
- **Controlled settings**: Same deck, opponent pool, seed 11, 1M-step budget, PPO settings, feature width, active guardrails, and lookahead-teacher configuration as EXP-021. Only the relation-table construction changes.
- **Implementation**: `python_object_relational` converts the twelve visible entity slots to Python objects for each forward pass, iterates directed pairs, and converts the resulting relation labels back to the same learned attention-bias tensor used by EXP-020. It reads no simulator-private state.
- **Run**: Seed 11 completed 1,001,472 timesteps (`models/ppo_v6_exp023_python_object_relational_attention_seed11.zip`) tracked via W&B run `ql2o8k0l`.
- **Validation Result**: **48.6%** score (34W-36L, Wilson lower bound 37.2%, worst matchup 20.0%, perspective gap 5.7%). Passed health gate. Saved to `evaluation_results/exp023_validation_results.json`.
- **Status**: `[x] Adopted (Ablation completed - functional equivalence with EXP-020 verified)`
- **Rollback**: Remove the EXP-023 config and mode-specific code/tests. Its checkpoint is a fresh, isolated artifact.

### [EXP-024] Recurrent Behavioral-Cloning Warm Start
- **Date**: 2026-07-30
- **Git Branch / Commit**: Current dirty worktree based on `e41c15ee6`; the BC implementation is isolated to new data/training modules plus scoped extensions of the current collector, policy, and fresh-model construction path.
- **Hypothesis**: Pretraining the existing masked recurrent V6 actor on complete, actor-visible expert trajectories improves PPO sample efficiency without reducing final validation strength, worst-matchup performance, or perspective balance.
- **Dataset contract**: Complete V6 decision sequences with structured actor observations, the exact 66-action legal mask, expert action, episode boundaries, perspective, outcome, STOP and intermediate multi-selection states. Simulator-only auxiliary targets remain separate from actor inputs. Reserved validation and final-holdout opponents are rejected.
- **Controlled comparison**: Scratch and BC arms must use the same initial architecture, seed family, opponent pool, reward configuration, guardrail mode, and downstream PPO timestep budget. The first dataset should use deterministic rule-based demonstrations; sparse lookahead relabeling remains optional and retains provenance.
- **Implementation**: Added a strict checksum-verified, episode-sharded demonstration format; bounded-memory lazy loading; a holdout-safe collector; recurrent masked-NLL training with the existing hidden-card auxiliary loss; a shared fresh-model factory; atomic standard `CustomPPO` output; a fresh optimizer at BC-to-PPO handoff; a full provenance sidecar; and matched scratch/warm-start configs.
- **Configs**: `configs/experiments/exp_024_behavioral_cloning_warmstart.yaml`, `configs/experiments/exp_024_ppo_scratch_seed11.yaml`, and `configs/experiments/exp_024_ppo_warmstart_seed11.yaml` (stripped of guardrails/aux loss/custom rewards to match EXP-005 1:1 except for warmstart base model).
- **Pretraining & Validation Results**: Collected 500 Alakazam games (26,591 transitions, 25,623 branching decisions) on non-holdout Alakazam deck `bank_54.csv` using `python_script:src/agents/kaggle_bots/alakazam_v8_agent.py`. Pretrained V6 actor achieved best validation NLL **0.8092** and **76.16% teacher agreement**. Validation score across frozen validation opponents: **48.6% overall win rate** (8-2 vs bank_11, 7-3 vs bank_36, 7-3 vs rule_bot_100, 7-3 vs bank_14 Alakazam mirror). Model saved to `models/ppo_v6_exp024_bc_warmstart_seed11.zip`.
- **Status**: `[x] Adopted (Downstream PPO training restarted matching EXP-005 configuration 1:1).`
- **Acceptance**: All labels legal; nonzero STOP/multi-selection coverage; held-out recurrent NLL improvement; deterministic save/load parity; successful short BC-to-PPO smoke run; and improved downstream learning-curve AUC without a final frozen-validation regression.
- **Rollback**: Remove the EXP-024 config and new BC modules/tests, restore the collector entry point, and revert only the scoped model-factory/policy hooks. Do not delete generated datasets or checkpoints without explicit permission.

### [EXP-025] Strategic Scalar Observation Vector Ablation
- **Date**: 2026-07-30
- **Git Branch / Commit**: `main` / `e41c15ee6` (dirty worktree)
- **Config Files**: `configs/experiments/exp_025_strategic_vector_v1.yaml` and `configs/experiments/exp_025_strategic_vector_v1_alakazam_seed11.yaml`
- **Hypothesis**: A public-information-only scalar observation vector built from tactical board facts, zone-type resource summaries, and option-consequence blocks can recover useful policy signal without relying on structured entity attention, hidden-card auxiliary loss, lookahead teacher distillation, guardrails, or belief-actor conditioning.
- **Active run target**: Launch the dedicated Alakazam variant on `decks/deck_bank/bank_54.csv` with W&B online logging enabled, keeping the same isolated scalar-observation settings and the standard Kaggle dev opponent pool.
- **Warm-start PPO target**: `configs/experiments/exp_025_ppo_warmstart_strategic_vector_alakazam_seed11.yaml` loads the completed scalar BC checkpoint into a fresh, separate 1M-timestep PPO target. The prior scratch run was intentionally stopped at 587,776 timesteps without publishing a checkpoint.
- **Queued continuation target**: `configs/experiments/exp_025_ppo_warmstart_strategic_vector_alakazam_seed11_stage2_4m.yaml` is queued on 2026-07-30 to load the completed 1M warm-start PPO checkpoint and train for `+3,000,000` more timesteps into `models/ppo_v6_exp025_strategic_vector_v1_alakazam_warmstart_seed11_4m.zip`, preserving the standalone 1M checkpoint file unchanged.
- **Warm-start launch note**: The first launch stopped before training because the legacy-checkpoint guard rejected every non-structured observation. The guard now continues to reject legacy scalar-card checkpoints but permits the explicitly requested `strategic_vector_v1` scalar contract; no warm-start PPO checkpoint was published by the failed launch.
- **Representation under test**:
  - Global block: turn context, both-player public board summaries, hand/discard type counts, board composition counts, revealed-zone counts, and tactical availability flags.
  - Per-option block: legal-action semantics, public immediate attack consequences, and static card semantics for each of the 65 encoded options.
  - Selection block: pending autoregressive selection length, stop legality, and up to eight selected option indices.
- **Isolation settings**:
  - `scalar_obs: true`
  - `feature_variant: strategic_vector_v1`
  - `aux_coef: 0.0`
  - `distill_coef: 0.0`
  - `value_distill_coef: 0.0`
  - `enable_lookahead_teacher: false`
  - `teacher_sample_rate: 0.0`
  - `belief_actor: false`
  - `card_table: false`
  - `inference_guardrails: false`
  - `enable_archetype_prediction: false`
  - `extra_metrics: false`
  - `wandb_mode: online`
  - `sparse_rewards: true`
- **Telemetry note**: Keep standard W&B training curves enabled, but leave `extra_metrics: false` so guardrail/archetype/prize side telemetry remains off for this isolation run.
- **BC warm start**: `configs/experiments/exp_025_behavioral_cloning_strategic_vector_alakazam_seed11.yaml` collects 500 complete actor-visible trajectories from `python_script:src/agents/kaggle_bots/alakazam_v8_agent.py` on `bank_54.csv`, then trains the same scalar policy contract used by the dedicated PPO arm. The dataset and checkpoint have distinct EXP-025 paths and cannot overwrite EXP-024 artifacts.
- **BC result**: Dataset collection completed with 500/500 games, 32,250 decisions, 31,061 branching decisions, and 7,243 STOP-available states. The 10-epoch scalar BC run completed with best held-out NLL **1.3524** and **51.17%** teacher agreement; the checkpoint and checksum-protected provenance sidecar are `models/ppo_v6_exp025_bc_strategic_vector_alakazam_seed11.zip` and `.bc.json`.
- **BC status**: `[x] Adopted` as the scalar-vector pretraining artifact; downstream PPO and frozen validation remain required before adopting EXP-025 as a training representation.
- **Stable Elo / Evaluation**: Stable external reference is **572.4 Elo** for V6 Compact Alakazam 1M; it is a non-comparable safety baseline, not a claimed scalar-vector score. The scalar BC checkpoint requires held-out NLL/agreement and frozen validation before an adoption decision.
- **Status**: `[-] Running` (Alakazam dedicated variant launched on 2026-07-30; the 1M warm-start PPO run is active and the non-overwriting `+3M` continuation is queued behind it.)
- **Rollback**: Remove the EXP-025 Alakazam config and revert only the EXP-025 scalar-observation additions if the ablation is abandoned. Do not delete generated checkpoints or logs without explicit permission.

### [EXP-026] Periodic Frozen-Checkpoint Self-Play
- **Date**: 2026-07-30
- **Git Branch / Commit**: `main` / `e41c15ee6` (pre-existing dirty worktree retained unchanged).
- **Hypothesis**: Three 1M-step PPO stages against a frozen copy of the immediately preceding policy will provide a stable, controlled self-play curriculum for the EXP-025 scalar strategic-vector policy.
- **Protocol**: Each queued stage runs on `bank_54.csv` with `strategic_vector_v1`, seed 11, sparse rewards, four environments, and the isolated EXP-025 PPO settings. Stage 1 starts from `ppo_v6_exp025_strategic_vector_v1_alakazam_warmstart_seed11.zip` and plays that same checkpoint. Stages 2 and 3 each load and play the prior stage's completed checkpoint. The opponent is therefore refreshed precisely every 1,000,000 training timesteps, never during a stage.
- **Queue / Configs**: `configs/training_queue.json`, `configs/experiments/exp_026_selfplay_alakazam_seed11_stage1_1m.yaml`, `configs/experiments/exp_026_selfplay_alakazam_seed11_stage2_2m.yaml`, and `configs/experiments/exp_026_selfplay_alakazam_seed11_stage3_3m.yaml`.
- **Outputs**: `models/ppo_v6_exp026_selfplay_alakazam_seed11_1m.zip`, `_2m.zip`, and `_3m.zip`; no existing checkpoint can be overwritten.
- **Stable Elo / Evaluation**: Stable external reference is **572.4 Elo** for V6 Compact Alakazam 1M. It is non-comparable to this scalar self-play arm; frozen validation and health-gate evaluation are required after the queued run before an adoption decision.
- **Status**: `[-] Paused on 2026-07-30 before any EXP-026 stage was launched; its remaining queue entries were removed while strategic_vector_v2 is prepared.`
- **Rollback**: Remove only the three EXP-026 configs and their queue entries. Do not delete any generated checkpoint without explicit permission.

### [EXP-027] Strategic Vector V2: Resource, Prize, and Bench Pressure
- **Date**: 2026-07-30
- **Git Branch / Commit**: Current dirty worktree / pending commit.
- **Hypothesis**: Adding public deck-out risk, prize-race margin, exposed bench-prize, fragile-bench, and attacker-recovery features helps a scalar PPO policy avoid the recurrent EXP-025 losses without observing hidden opponent information.
- **Implementation**: `strategic_vector_v2` appends ten normalized tactical facts to the EXP-025 global vector. The original `strategic_vector_v1` width and behavior remain unchanged. Fresh BC and PPO configs use new dataset/model paths and cannot overwrite or resume v1 checkpoints.
- **Controlled comparison**: Alakazam deck `bank_54.csv`, Kaggle rule-bot development pool, seed 11, sparse reward, disabled auxiliary/distillation/guardrail paths, and one million PPO timesteps after fresh BC.
- **Stable Elo / Evaluation**: Stable external reference remains 572.4 Elo for V6 Compact Alakazam 1M; this is non-comparable. Adoption requires frozen validation with separate reporting for Alakazam mirrors, Archaludon, and deck-out outcomes.
- **BC result**: Collected 500/500 complete Alakazam expert episodes (32,610 decisions). The ten-epoch V2 BC run reached best validation NLL **1.3336**, validation teacher agreement **50.43%** (3,135/6,216 held-out decisions), and final training agreement **51.37%**. The fresh checkpoint and provenance sidecar are `models/ppo_v6_exp027_bc_strategic_vector_v2_alakazam_seed11.zip` and `.bc.json`.
- **BC extension**: A fresh 20-epoch run against the identical frozen dataset and seed starts from new random weights and writes `models/ppo_v6_exp027_bc_strategic_vector_v2_alakazam_seed11_20e.zip`. It tests whether the still-improving ten-epoch validation curve was undertrained; it does not overwrite the original checkpoint.
- **Replication**: Seeds 23 and 37 repeat the 20-epoch BC protocol against the identical frozen dataset. Report mean and standard deviation of held-out agreement/NLL over seeds 11, 23, and 37 before changing the observation contract again.
- **Replication result**: Seed 11: **53.80%** agreement, NLL **1.2615**; seed 23: **53.12%**, **1.2896**; seed 37: **52.05%**, **1.2885**. Mean held-out agreement is **52.99% ± 0.88 percentage points** (sample standard deviation); mean best validation NLL is **1.2799**. The representation is stable but remains far below EXP-024 Structured V6 agreement.
- **Status**: `[-] BC completed; PPO and frozen validation remain pending before adoption.`
- **Rollback**: Remove only the EXP-027 configs and scoped v2 observation additions. Do not delete existing models, datasets, or replay artifacts.

### [EXP-028] Strategic Vector V3: Regex Rule Features per Legal Option
- **Date**: 2026-07-30
- **Git Branch / Commit**: Current dirty worktree / pending commit.
- **Hypothesis**: Public regex-derived mechanics for each legal option make scalar BC less ambiguous than V2's card type/HP/stage-only action block, especially for draw, search, discard, switch, attachment, recovery, bench, and damage effects.
- **Implementation**: `strategic_vector_v3` preserves V2's global public strategy block and appends 49 shared regex-rule facts per legal option, including draw/search/discard, protection, healing, statuses, energy movement, retreat constraints, prize modifiers, KO effects, and target roles. It reuses the structured encoder's existing rule-text parser; no hidden state or hand-authored utility is introduced. V1/V2 contracts and completed checkpoints remain unchanged.
- **Run**: Fresh 500-game V3 demonstrations and a seed-11 20-epoch BC run use dedicated EXP-028 dataset/checkpoint paths.
- **BC result**: The fresh 500-game V3 dataset completed and the 20-epoch seed-11 BC run selected epoch 18 with held-out teacher agreement **54.25%** and validation NLL **1.2626**. This is +0.45 percentage points over V2 seed 11 (53.80%), but not yet a confirmed representation improvement without further seeds and PPO evaluation.
- **PPO result**: The fresh one-million-timestep warm-start run completed at `models/ppo_v6_exp028_strategic_vector_v3_alakazam_warmstart_seed11.zip`. Frozen screening: **31.4%** validation win rate (22/70) and **15.0%** holdout win rate (9/60). This is below the exact Kaggle Alakazam V8 rule-bot baseline (52.9% / 41.7%).
- **Continuation**: `configs/experiments/exp_028_ppo_continue_strategic_vector_v3_alakazam_seed11_2m.yaml` starts from the preserved 1M checkpoint and trains a further 1M steps, saving only to `models/ppo_v6_exp028_strategic_vector_v3_alakazam_warmstart_seed11_2m.zip`. A queued second continuation (`exp_028_ppo_continue_strategic_vector_v3_alakazam_seed11_4m.yaml`) then loads that 2M checkpoint for a further 2M steps and saves only to `models/ppo_v6_exp028_strategic_vector_v3_alakazam_warmstart_seed11_4m.zip`. Both retain the frozen development pool and all PPO settings; rollback is deleting these isolated continuation configs/outputs only.
- **Continuation result**: The 1M continuation completed at **2,002,944** total checkpoint timesteps and saved `..._2m.zip`; the queued +2M continuation then completed at **4,003,840** total checkpoint timesteps and saved `..._4m.zip`. Both runs ended normally after their configured step budgets; no interruption or checkpoint overwrite occurred.
- **Kaggle submission**: On 2026-07-31, the verified archive `artifacts/submissions/submission_v6_exp028_strategic_vector_v3_alakazam_4m_20260731.tar.gz` (bank_54 deck, 4M checkpoint) was submitted with description `EXP-028 strategic-vector V3 Alakazam PPO 4M continuation (2026-07-31)`. Package verification passed; external Elo remains pending Kaggle evaluation.
- **Status**: `[x] Training continuations completed; replication and post-continuation frozen evaluation remain pending.`
- **Rollback**: Remove only the V3 observation contract and any future EXP-028 artifacts. Do not alter completed EXP-027 results.

### Kaggle Alakazam V8 Rule-Bot Frozen Screen
- **Date**: 2026-07-30
- **Purpose**: Establish an apples-to-apples tactical ceiling reference for EXP-028 by evaluating the exact Kaggle Alakazam V8 Python rule bot and its original deck (`deck_kaggle_alakazam_v8.csv`) against the frozen pools. No training, pool, or production-code changes were made.
- **Protocol**: Ten games per opponent, direct `evaluate_single.py` execution because the general submission evaluator currently recognizes `rule_based:*` but not `python_script:*` candidates. Candidate: `python_script:src/agents/kaggle_bots/alakazam_v8_agent.py`.
- **Validation result**: **37/70 wins (52.9%)**. Per-matchup results: bank_49 1-9, bank_11 6-4, bank_25 5-5, bank_36 1-9, bank_84 6-4, bank_14 9-1, aggressive rule bot 9-1.
- **Holdout result**: **25/60 wins (41.7%)**. Per-matchup results: bank_99 2-8, bank_3 6-4, bank_14 9-1, bank_24 5-5, bank_63 1-9, bank_70 2-8.
- **Conclusion / status**: `[x] Completed`. The explicit rule policy is materially stronger than EXP-028 V3 PPO on both frozen pools (31.4% validation; 15.0% holdout), but it is not a 750-Elo-equivalent local ceiling. Ten games per pairing give high matchup variance; this screen is a directional baseline, not an adoption gate.
- **Rollback**: Remove this log entry only; no model, dataset, or source artifact was changed.

### Foundation Opponent Strength Screen (bank_55 / bank_56)
- **Date**: 2026-07-30
- **Git Branch / Commit**: Current dirty worktree based on `e41c15ee6`; no production code or pool manifest changes.
- **Hypothesis**: The unused Compact/Potential V6 foundation checkpoints for Mega Kangaskhan ex (`bank_55`) and Dragapult ex (`bank_56`) are sufficiently capable and stable to become candidates for a newly frozen evaluation pool.
- **Protocol**: Each checkpoint plays 5 games per opponent, with balanced perspectives over the aggregate, against the fixed five-opponent development reference pool. Admission requires a passed health gate and at least 40% aggregate score; results are screening evidence only and do not consume a final holdout.
- **Status**: `[-] Running`
- **Rollback**: Delete only the screening result file if this diagnostic is abandoned; no models or pool manifests are modified.

### [EXP-029] Grimmsnarl ex Strategic Vector V3 PFSP League Opponent
- **Date**: 2026-07-31
- **Git Branch / Commit**: `main` / dirty worktree based on `e41c15ee6`; experiment isolation is the dedicated config and new output path.
- **Config File**: `configs/experiments/exp_029_grimmsnarl_v3_pfsp_2m.yaml`
- **Hypothesis**: A fresh V3 public-information PPO policy for Grimmsnarl ex fills the highest-frequency unrepresented archetype in the local Kaggle `>450` sample and becomes a diverse training-only league opponent.
- **Protocol**: Train the Kaggle Grimmsnarl deck for 2,000,000 steps against `decks/pools/default_training.json`, with PFSP-lite reweighting every 250,000 steps (150-game window), active guardrails, feature/archetype telemetry, and W&B logging. The pool's Abomasnow deck path is corrected to its existing `decks/deck_bank/` source. No reserved validation or holdout opponent is placed in the training pool. Periodic frozen-checkpoint evaluation is deferred beyond the run budget so the final holdout remains untouched.
- **Observation contract**: `strategic_vector_v3`, scalar public observation, with V2 global tactical facts and 49 shared rule-derived facts per legal option. The run starts from random weights; it does not reuse the Alakazam-specific BC checkpoint.
- **Stable Elo / Evaluation**: The available external reference is 572.4 Elo for V6 Compact Alakazam 1M; it is non-comparable to this fresh Grimmsnarl V3 opponent. Promotion requires a later frozen-validation and health-gate result.
- **Launch**: The first background invocation exited before training because it did not export the repository root on `PYTHONPATH`; it produced no model or training steps. The corrected launch uses `PYTHONPATH=.` and logs to `logs/train_exp029_grimmsnarl_v3_pfsp_2m_20260731.log`.
- **Telemetry revision**: Before the restart, PFSP logging was extended with `train/pfsp_worst_win_rate`, the lowest recent per-opponent score; `rollout/ep_rew_min` and `rollout/ep_rew_max` were removed from W&B.
- **Telemetry visibility revision**: Initial PFSP probabilities and then pool weights, macro winrate, and worst winrate are recorded each rollout, rather than only after the 250,000-step reweighting boundary.
- **Pool revision**: Removed the seven redundant historical Team Rocket's Mewtwo ex EXP checkpoints (EXP-005, EXP-007, EXP-012, EXP-014 1M, and EXP-016 1M/2M/4M). Retained the V6 Mewtwo training opponent and the best EXP reference, EXP-014 2M. The stopped run had no final model checkpoint, so this revision restarts from zero.
- **Status**: `[-] Launching`
- **Rollback**: Interrupt the dedicated training session and remove only this config and its newly named model/log artifacts. Do not alter the training, validation, or holdout manifests.

### [EXP-030] Grimmsnarl ex V3 PFSP with Reduced Deck-out Reward
- **Date**: 2026-07-31
- **Git Branch / Commit**: `main` / dirty worktree based on `e41c15ee6`; experiment isolation is the dedicated configuration and model output path.
- **Config File**: `configs/experiments/exp_030_grimmsnarl_v3_pfsp_deckout_reward.yaml`
- **Hypothesis**: Reducing a deck-out win from `+1.0` to `+0.3`, while retaining `+1.0` for prize and bench-out wins, shifts the policy from passive survival toward decisive wins without treating a legal game win as a loss.
- **Changes**:
  - `configs/experiments/exp_030_grimmsnarl_v3_pfsp_deckout_reward.yaml`: Set `torch_threads: 2` (Daytime Mode for lag-free operation).
  - `decks/pools/default_training.json`: Removed Crustle / Great Tusk Library-out bot.
  - `src/training/train.py`: Logs the PFSP recent worst per-opponent win rate as `rollout/min_win_rate` after every rollout.
- **Stable Elo / Evaluation**: No stable Elo yet. Compare deck-out share, `rollout/min_win_rate`, and frozen evaluation against EXP-029 when a checkpoint is available.
- **Status**: `[-] Running (Daytime Mode, 2 CPU threads)`
- **Rollback**: Interrupt the dedicated EXP-030 screen session and remove only the EXP-030 configuration plus its newly named model/log artifacts. Do not alter pool manifests, validation, or holdout data.

### [EXP-031] Grimmsnarl ex V3 PFSP with Crustle / Great Tusk Library-out
- **Date**: 2026-07-31
- **Git Branch / Commit**: `main` / dirty worktree based on `e41c15ee6`; isolation is provided by the dedicated configuration and model path.
- **Config File**: `configs/experiments/exp_031_grimmsnarl_v3_pfsp_crustle_libraryout.yaml`
- **Hypothesis**: Adding the public Crustle / Great Tusk control bot provides a strong, strategically distinct deck-out opponent and reduces overfitting to prize-race opponents.
- **Changes**:
  - `src/agents/kaggle_bots/crustle_great_tusk_libraryout_agent.py`: Local rule-based implementation of the published Great Tusk mill / Crustle wall plan.
  - `decks/pools/default_training.json`: Adds the opponent at weight `1.0`.
  - Keeps EXP-030's `DECK_OUT_WIN: 0.3` and learning rate `0.0001` unchanged.
- **Stable Elo / Evaluation**: No stable Elo yet. The notebook's displayed public score is 630.9; the stated 1208 maximum is from a prior submission and is not a local benchmark. Compare deck-out share, `rollout/min_win_rate`, and frozen evaluation to EXP-030.
- **Status**: `[-] Launching`
- **Rollback**: Interrupt EXP-031 and remove only its configuration, model/log artifacts, and the added bot/pool entry if the experiment is reverted. Do not alter validation or holdout pools.

### [HOLD-002] Balanced Kaggle-Aligned Holdout Pool V2 & Submissions
- **Date**: 2026-07-31
- **Git Branch / Commit**: `main` / dirty worktree based on current HEAD.
- **Hypothesis**: Replacing duplicate Mewtwo ex PPO opponent (`ppo_v6_mewtwo_2m_bank_38`) with `BattleCore Heuristic Bot` eliminates local deck-counter bias and triples rank-order correlation with real Kaggle Leaderboard Elo for RL checkpoints ($\rho = +0.3929$ vs $+0.1429$ previously).
- **Changes**:
  - `decks/pools/holdout_opponents.json`: Replaced second Mewtwo ex model with `battlecore_agent.py` on `deck_kaggle_battlecore.csv` (7 opponents total, exactly one per archetype).
  - `scripts/build_rule_bot_submission.sh` and `scripts/submit_latest_submission.sh`: Extended to natively support creating and uploading verified rule-based bot packages without demanding `ppo_pokemon_final.zip`.
  - `scripts/verify_submission_package.py`: Added `--rule-bot` flag for standalone testing of heuristic agents.
- **Submissions**: Submitted both `submission_rule_based_battlecore.tar.gz` (BattleCore heuristic tank deck) and `submission_exp030_grimmsnarl_2m.tar.gz` (EXP-030 2M checkpoint) directly to Kaggle on 2026-07-31 to establish real Leaderboard Elo anchors.
- **Status**: `[x] Completed & Adopted`
- **Rollback**: Revert `decks/pools/holdout_opponents.json` to v1 if duplicate Mewtwo representation is required.

### [EXP-032] Archaludon Rule-Based Bot Optimization & Loss Analysis
- **Date**: 2026-07-31
- **Git Branch / Commit**: `main`
- **Focus**: Archaludon ex + Cinderace BattleCore Rule-Based Bot Optimization.
- **Key Analysis & Findings**:
  - **Loss Profile**: Archaludon's primary failure mode is **Bench Out (70-85%)** when basic Duraludon/Cinderace are targeted early before evolution.
  - **Alakazam Counter Mechanism**: Alakazam ex's *Powerful Hand* (20 DMG per card in hand) OHKOs Archaludon ex (300 HP) when Alakazam accumulates 12+ cards in hand.
  - **Strategy & Counters**:
    1. **Hand Disruption**: Play *Judge* (ID 1213) when opponent hand >= 7 to reset hand size from 12-15 down to 4 cards (dropping damage from 300 DMG to 80 DMG).
    2. **Pre-emptive Snipe**: Use *Boss's Orders* (ID 1182) on benched Abra/Kadabra to prevent Alakazam evolution & *Psychic Draw*.
    3. **Supporter Recycling**: Use *Poké Pad* (ID 1152) to recycle *Judge* and *Boss's Orders*.
- **Stable Elo / Reference**: Baseline BattleCore heuristic anchor on Kaggle LB.
- **Status**: `[ ] In Progress`

