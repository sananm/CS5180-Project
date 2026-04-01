# Requirements: AlphaZero vs PPO on Othello

**Defined:** 2026-03-31
**Core Value:** A rigorous, statistically grounded comparison of search-based vs search-free self-play RL on Othello

## v1 Requirements

### Game Environment

- [x] **ENV-01**: Othello game with board logic, legal-action queries, and terminal detection (from alpha-zero-general)
- [x] **ENV-02**: Canonical board representation as (3, 8, 8) tensor — current player discs, opponent discs, turn indicator
- [x] **ENV-03**: Illegal move masking — set invalid action logits to -inf before softmax

### Neural Network

- [x] **NET-01**: Shared CNN architecture with convolutional backbone, policy head (65 action logits + softmax-compatible output), and value head (scalar in [-1, +1])
- [x] **NET-02**: Same network architecture used by both AlphaZero and PPO for fair comparison

### AlphaZero

- [x] **AZ-01**: MCTS with PUCT selection rule using CNN policy prior and value estimates
- [x] **AZ-02**: Self-play training loop generating (state, pi_mcts, z) tuples stored in replay buffer
- [ ] **AZ-03**: Network training minimizing combined policy + value loss from replay buffer samples
- [ ] **AZ-04**: Trained AlphaZero agent that beats random and minimax baselines

### PPO

- [x] **PPO-01**: PPO with clipped surrogate objective and GAE advantage estimation, implemented from scratch
- [ ] **PPO-02**: Opponent pool self-play — checkpoint every K steps, sample opponents uniformly from pool
- [x] **PPO-03**: Actor-critic using shared CNN architecture (same backbone as AlphaZero)
- [ ] **PPO-04**: Trained PPO agent that beats random and minimax baselines

### Baselines

- [x] **BASE-01**: Random agent selecting uniformly among legal moves
- [x] **BASE-02**: Minimax agent with alpha-beta pruning at depth 4, positional heuristic (corners/edges weighted)
- [x] **BASE-03**: Minimax agent with alpha-beta pruning at depth 6, same heuristic

### Evaluation

- [x] **EVAL-01**: Round-robin tournament system running 200+ games per matchup (both colors)
- [x] **EVAL-02**: Win rate tables with 95% confidence intervals
- [x] **EVAL-03**: Elo rating system ranking all agents on a single scale
- [x] **EVAL-04**: Training curves — win rate vs training iterations against fixed baselines
- [x] **EVAL-05**: Head-to-head AlphaZero vs PPO with statistical significance
- [ ] **EVAL-06**: Compute efficiency analysis — performance vs wall-clock GPU-hours
- [x] **EVAL-07**: Color asymmetry analysis — win rates disaggregated by color
- [x] **EVAL-08**: Policy entropy and loss curve plots over training

### Ablations

- [ ] **ABL-01**: AlphaZero MCTS simulation count ablation (50/200/800 sims at evaluation time)
- [ ] **ABL-02**: PPO opponent pool size ablation (1/5/20) — requires separate training runs

### Paper

- [ ] **PAP-01**: AAAI-format LaTeX paper with all sections (abstract, intro, background, related work, methods, experiments, conclusion)
- [ ] **PAP-02**: Reproducibility details — hyperparameter tables, training duration, hardware specs, random seeds
- [ ] **PAP-03**: Submitted code alongside paper

## v2 Requirements

### Enhanced Evaluation

- **EVAL-09**: Action probability heatmap visualizations for select board positions
- **EVAL-10**: Value function visualization over game trajectories
- **EVAL-11**: Game replay annotation for qualitative analysis

### Statistical Robustness

- **STAT-01**: Multiple random seeds (3+) per algorithm for variance reporting
- **STAT-02**: External engine comparison (Edax)

## Out of Scope

| Feature | Reason |
|---------|--------|
| MuZero / Gumbel variants | MiniZero already did this comparison; project value is AlphaZero vs PPO |
| Hyperparameter search / AutoML | No time; use literature-based defaults |
| Deep networks (20+ ResBlocks) | Overkill for 8x8 Othello; wastes compute |
| Distributed / multi-GPU training | Solo course project scope |
| Web interface or interactive demo | Not graded; wastes time |
| Multiple board games | Project is Othello-specific depth over breadth |
| Complex opponent sampling (PSRO) | Over-engineering; uniform pool is sufficient |
| OpenSpiel integration | Using alpha-zero-general per approved proposal |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Complete (01-01) |
| ENV-02 | Phase 1 | Complete (01-01) |
| ENV-03 | Phase 1 | Complete (01-02) |
| NET-01 | Phase 3 | Complete (03-01) |
| NET-02 | Phase 3 | Complete (03-01) |
| AZ-01 | Phase 4 | Complete (04-01) |
| AZ-02 | Phase 4 | Complete (04-01) |
| AZ-03 | Phase 4 | Pending |
| AZ-04 | Phase 4 | Pending |
| PPO-01 | Phase 5 | Complete (05-01) |
| PPO-02 | Phase 5 | Pending |
| PPO-03 | Phase 5 | Complete (05-01) |
| PPO-04 | Phase 5 | Pending |
| BASE-01 | Phase 2 | Complete (02-01) |
| BASE-02 | Phase 2 | Complete (02-01) |
| BASE-03 | Phase 2 | Complete (02-01) |
| EVAL-01 | Phase 2 | Complete (02-01) |
| EVAL-02 | Phase 2 | Complete (02-02) |
| EVAL-03 | Phase 2 | Complete (02-02) |
| EVAL-04 | Phase 2 | Complete (02-02) |
| EVAL-05 | Phase 6 | Complete |
| EVAL-06 | Phase 6 | Pending |
| EVAL-07 | Phase 6 | Complete |
| EVAL-08 | Phase 6 | Complete |
| ABL-01 | Phase 6 | Pending |
| ABL-02 | Phase 6 | Pending |
| PAP-01 | Phase 7 | Pending |
| PAP-02 | Phase 7 | Pending |
| PAP-03 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 29 total
- Mapped to phases: 29
- Unmapped: 0

---
*Requirements defined: 2026-03-31*
*Last updated: 2026-03-31 after Plan 05-01 execution (PPO-01 and PPO-03 completed; rollout, GAE, loss, IS ratio test implemented)*
