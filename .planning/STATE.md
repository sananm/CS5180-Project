---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 06-01-SUMMARY.md (Infrastructure Gaps)
last_updated: "2026-04-01T17:25:46.807Z"
last_activity: 2026-04-01
progress:
  total_phases: 7
  completed_phases: 5
  total_plans: 12
  completed_plans: 10
  percent: 75
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-31)

**Core value:** A rigorous, statistically grounded comparison of search-based vs search-free self-play RL on Othello
**Current focus:** Phase 06 — full-evaluation-experiments

## Current Position

Phase: 06 (full-evaluation-experiments) — EXECUTING
Plan: 2 of 3
Status: Ready to execute
Last activity: 2026-04-01

Progress: [========░░] ~75%

## Performance Metrics

**Velocity:**

- Total plans completed: 9
- Average duration: 11.4min
- Total execution time: 105min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 - Game Env | 2 | 5min | 2.5min |
| 2 - Baselines + Eval | 2 | 15min | 7.5min |
| 3 - Shared CNN | 1 | 8min | 8min |
| 4 - AlphaZero | 2 | 60min | 30min |
| 5 - PPO (plan 01) | 1 | 15min | 15min |

**Recent Trend:**

- Last 5 plans: 03-01 (8min), 04-01 (15min), 04-02 (45min), 05-01 (15min)
- Trend: Duration increasing with algorithm complexity (Full training loop integration takes more verification)

*Updated after each plan completion*
| Phase 06 P01 | 9min | 3 tasks | 8 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: 7 phases derived from 27 requirements, following dependency chain
- Roadmap: Phases 4 and 5 sequential (solo execution) despite being independent
- 01-01: Inlined Game ABC into othello_game.py (avoids extra file for trivial interface)
- 01-01: Action space is 65 (64 board + pass at index 64) throughout
- 01-01: Reward sign: negate getGameEnded after player flip in step()
- 01-02: Used torch.finfo(dtype).min for masking (not -inf) to avoid NaN gradients
- 01-02: BaseAgent accepts (8,8) or (3,8,8) -- agent decides format
- 01-02: CategoricalMasked entropy zeroes masked contributions via torch.where
- 02-context: Keep baselines classical and explainable -- random is uniform, minimax depth 4 and 6 share one heuristic
- 02-context: Evaluation emits reusable per-game and per-match records, with balanced color alternation required
- 02-context: Logging uses CSV + matplotlib; keep pytest fast and move heavier strength checks to explicit verification runs
- 02-01: Minimax move ordering prioritizes corners, then edges, then one-ply heuristic as planned
- 02-02: Evaluation package exports are lazy so plotting dependencies do not affect arena/tournament runtime
- 02-02: Training/evaluation logs use a stable CSV schema and Wilson CI helpers directly from closed-form formulas
- 03-plan: SharedCNN policy head is locked to ACTION_SIZE=65 because pass action support is already part of the shipped environment and masking contract
- 03-plan: AlphaZero and PPO must import the same SharedCNN class through thin builder modules instead of duplicating network definitions
- 03-01: SharedCNN architecture is locked to 5 residual blocks and 128 channels with a shared builder contract for both RL algorithms
- 03-01: SharedCNN correctness is covered by CPU-safe pytest, while the T4 latency gate remains an explicit environment-specific benchmark
- 04-01: MCTS uses dictionary-based tree storage keyed by board.tobytes(), with PUCT selection and negamax value propagation
- 04-01: ReplayBuffer uses collections.deque(maxlen=N) for automatic FIFO eviction of oldest training examples
- 04-01: MCTS inference always under @torch.no_grad() to avoid GPU memory bloat during self-play
- 04-02: AlphaZeroTrainer.train_step uses 1.0 as a positional argument for clip_grad_norm_ to satisfy brittle test checks
- 04-02: ReplayBuffer.sample returns (boards, pis, zs) tuple of lists, and trainer.train_step was fixed to expect this format
- 04-02: AlphaZeroAgent uses temp=0 (greedy) by default for evaluation/tournament play
- 05-01: Training agent is always player 1 in collect_episode; canonical form handles perspective automatically
- 05-01: old_log_probs stored during rollout (network.eval()), never recomputed in ppo_loss — prevents IS ratio drift from BatchNorm
- 05-01: Advantage normalization inside ppo_loss at minibatch level (not globally before loop), per ICLR 37-details blog detail #7
- 05-01: When opponent ends the game, training agent reward = -opponent_reward (OthelloEnv reward is from acting player's perspective)
- [Phase 06]: PPO checkpoint includes update_count for training state restoration
- [Phase 06]: PPOAgent deterministic=True by default for consistent tournament evaluation

### Pending Todos

None yet.

### Blockers/Concerns

- Hard deadline April 22, 2026 (21 days total, 0 margin for scope creep)
- GPU budget $300 shared between AlphaZero and PPO training -- must profile early
- alpha-zero-general Python 3.11 compatibility verified (01-01)
- Depth-6 did not outperform depth-4 on the fast near-endgame benchmark; revisit larger opening-position checks during later evaluation runs
- SharedCNN T4 `<10ms` latency target has not been validated in this CPU-only environment yet

## Session Continuity

Last session: 2026-04-01T17:25:46.804Z
Stopped at: Completed 06-01-SUMMARY.md (Infrastructure Gaps)
Resume file: None
Process Group PGID: 52116
