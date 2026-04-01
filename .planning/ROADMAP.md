# Roadmap: AlphaZero vs PPO on Othello

## Overview

This project delivers a rigorous comparison of search-based (AlphaZero) vs search-free (PPO) self-play RL on 8x8 Othello, culminating in an AAAI-format paper. The roadmap follows the natural dependency chain: game environment and infrastructure first, then baselines and evaluation scaffolding, then the shared CNN, then each RL algorithm independently, then full evaluation, then paper writing. The hard deadline is April 22, 2026 (22 days from project start).

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Game Environment + Infrastructure** - Working Othello game, agent interface, GPU environment, reproducibility setup
- [x] **Phase 2: Baselines + Evaluation Framework** - Random/minimax agents, tournament system, Elo ratings, training curve logging
- [x] **Phase 3: Shared CNN** - Dual-head CNN architecture validated and locked down for both algorithms
- [ ] **Phase 4: AlphaZero Agent + Training** - MCTS with PUCT, self-play training loop, trained agent beating baselines
- [ ] **Phase 5: PPO Agent + Training** - PPO from scratch with opponent pool self-play, trained agent beating baselines
- [ ] **Phase 6: Full Evaluation + Experiments** - Complete tournament, all tables/figures, ablation studies
- [ ] **Phase 7: Paper Writing** - AAAI-format paper with all sections, submitted with reproducible code

## Phase Details

### Phase 1: Game Environment + Infrastructure
**Goal**: A working Othello game environment with canonical board representation, illegal move masking, and a shared agent interface that all future agents will implement -- plus GPU environment and reproducibility infrastructure validated
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03
**Success Criteria** (what must be TRUE):
  1. An Othello game can be played to completion programmatically (moves, legal action queries, terminal detection, winner determination all work correctly)
  2. Board states are correctly encoded as (3, 8, 8) tensors in canonical form (current player's perspective) and round-trip back to board state without information loss
  3. Illegal move masking sets invalid action logits to -inf and the resulting softmax produces a valid probability distribution over only legal moves
  4. A common agent interface exists (get_action(board) -> action) that random, minimax, and RL agents will all implement
  5. GPU environment (Google Cloud T4) is configured, PyTorch sees CUDA, and seeding infrastructure produces deterministic results
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md -- Game core: alpha-zero-general game files, OthelloEnv wrapper, tensor encoding
- [x] 01-02-PLAN.md -- Action masking, agent interface, seeding, config, and comprehensive test suite

### Phase 2: Baselines + Evaluation Framework
**Goal**: Fixed baseline agents and a complete evaluation pipeline exist so that RL training progress can be measured against known opponents from day one
**Depends on**: Phase 1
**Requirements**: BASE-01, BASE-02, BASE-03, EVAL-01, EVAL-02, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):
  1. Random agent plays legal moves uniformly and completes games without errors
  2. Minimax depth-4 agent consistently beats random agent (>85% win rate), and minimax depth-6 is not weaker than depth-4 under the shared heuristic baseline
  3. Round-robin tournament system runs N games per matchup (both colors), producing win/draw/loss counts
  4. Win rate tables display 95% confidence intervals and Elo ratings rank all agents on a single scale
  5. Training curve logging infrastructure records win rate vs iteration against fixed baselines and produces plots
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md -- Random/minimax baseline agents, arena, tournament engine, and evaluation defaults
- [x] 02-02-PLAN.md -- Wilson CIs, Elo ratings, training-curve logging/plotting, and Phase 2 validation suite

### Phase 3: Shared CNN
**Goal**: A single dual-head CNN architecture (policy + value) is validated and locked down as the controlled variable for both AlphaZero and PPO
**Depends on**: Phase 1
**Requirements**: NET-01, NET-02
**Success Criteria** (what must be TRUE):
  1. SharedCNN accepts (batch, 3, 8, 8) tensor input and outputs policy logits (batch, 65) and value (batch, 1) in the correct ranges (logits are real-valued, value in [-1, +1])
  2. The same SharedCNN class is instantiated by both AlphaZero and PPO code paths with identical architecture (same layer count, channel count, residual blocks)
  3. Forward pass completes in <10ms on T4 GPU for a single board state (fast enough for MCTS rollouts)
  4. Gradients flow correctly through both heads -- a toy training step reduces loss on synthetic data
**Plans**: 1 plan

Plans:
- [x] 03-01-PLAN.md -- SharedCNN module, future AlphaZero/PPO network builders, and validation/benchmark checks

### Phase 4: AlphaZero Agent + Training
**Goal**: A trained AlphaZero agent that uses MCTS with the shared CNN and demonstrably beats random and minimax baselines through self-play training
**Depends on**: Phase 2, Phase 3
**Requirements**: AZ-01, AZ-02, AZ-03, AZ-04
**Success Criteria** (what must be TRUE):
  1. MCTS with PUCT selection uses CNN policy prior and value estimates to select moves, and increasing simulation count produces stronger play (more sims = higher win rate vs random)
  2. Self-play training loop generates (state, pi_mcts, z) tuples, stores them in a replay buffer, and trains the CNN from replay buffer samples
  3. Training curves show monotonic improvement in win rate vs random and minimax-depth-4 baselines over training iterations
  4. Final trained AlphaZero agent achieves >90% win rate vs random and >60% win rate vs minimax-depth-4
  5. Compute cost per training iteration is profiled and total training fits within GPU budget (~$150, half of $300)
**Plans**: 2 plans

Plans:
- [x] 04-01-PLAN.md — MCTS with PUCT selection, replay buffer, and core component tests
- [x] 04-02-PLAN.md — Self-play episode generation, trainer, AlphaZeroAgent, and training pipeline

### Phase 5: PPO Agent + Training
**Goal**: A trained PPO agent (implemented from scratch) with opponent pool self-play that demonstrably beats random and minimax baselines
**Depends on**: Phase 2, Phase 3
**Requirements**: PPO-01, PPO-02, PPO-03, PPO-04
**Success Criteria** (what must be TRUE):
  1. PPO implementation uses clipped surrogate objective and GAE advantage estimation, with importance sampling ratio verified to be 1.0 on the first update step (correctness check)
  2. Opponent pool self-play checkpoints the policy every K steps, samples opponents uniformly from the pool, and the pool grows over training
  3. Actor-critic uses the same SharedCNN architecture as AlphaZero (verified identical layer structure)
  4. Training curves show improvement in win rate vs random and minimax-depth-4 baselines over training iterations (no strategy collapse)
  5. Final trained PPO agent achieves >90% win rate vs random and >50% win rate vs minimax-depth-4
**Plans**: 2 plans

Plans:
- [x] 05-01-PLAN.md — Core PPO algorithm: RolloutBuffer, collect_episode, compute_gae, ppo_loss, IS ratio test
- [x] 05-02-PLAN.md — PPOTrainer with opponent pool, PPOAgent (BaseAgent), package exports, training loop

### Phase 6: Full Evaluation + Experiments
**Goal**: All experimental results for the paper are generated -- head-to-head comparison, round-robin tournament, ablation studies, and all figures/tables
**Depends on**: Phase 4, Phase 5
**Requirements**: EVAL-05, EVAL-06, EVAL-07, EVAL-08, ABL-01, ABL-02
**Success Criteria** (what must be TRUE):
  1. Head-to-head AlphaZero vs PPO results exist with 200+ games and statistical significance reported (p-value or CI)
  2. Compute efficiency analysis compares performance vs wall-clock GPU-hours for both algorithms
  3. Color asymmetry analysis shows win rates disaggregated by playing color for all matchups
  4. Policy entropy and loss curves over training are plotted for both algorithms
  5. Ablation results exist: AlphaZero at 50/200/800 MCTS sims, PPO with opponent pool size 1/5/20
**Plans**: 3 plans

Plans:
- [x] 06-01-PLAN.md — Infrastructure gaps: PPO save/load, deterministic mode, scipy, significance testing, color analysis, loss logger
- [x] 06-02-PLAN.md — Experiment scripts: head-to-head, MCTS ablation, PPO pool ablation, ablation/color plotting
- [x] 06-03-PLAN.md — Compute efficiency timing, loss/entropy curve logging, multi-panel loss/entropy plots

### Phase 7: Paper Writing
**Goal**: A complete AAAI-format research paper with all sections, figures, tables, and reproducibility details, ready for submission with code
**Depends on**: Phase 6
**Requirements**: PAP-01, PAP-02, PAP-03
**Success Criteria** (what must be TRUE):
  1. Paper has all required sections (abstract, introduction, background, related work, methods, experiments, conclusion) in AAAI 2-column format
  2. All experimental results from Phase 6 are presented as properly formatted tables and figures in the paper
  3. Reproducibility section includes hyperparameter tables, training duration, hardware specs, and random seeds
  4. Code is organized, documented with a README, and ready for submission alongside the paper
**Plans**: TBD

Plans:
- [ ] 07-01: TBD
- [ ] 07-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
Note: Phases 2 and 3 both depend only on Phase 1 and could overlap. Phases 4 and 5 are independent but sequential (solo execution).

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Game Environment + Infrastructure | 2/2 | Complete | 2026-03-31 |
| 2. Baselines + Evaluation Framework | 2/2 | Complete | 2026-03-31 |
| 3. Shared CNN | 1/1 | Complete | 2026-03-31 |
| 4. AlphaZero Agent + Training | 2/2 | Complete | 2026-04-01 |
| 5. PPO Agent + Training | 2/2 | Complete | 2026-04-01 |
| 6. Full Evaluation + Experiments | 3/3 | Complete | 2026-04-01 |
| 7. Paper Writing | 0/2 | Not started | - |
