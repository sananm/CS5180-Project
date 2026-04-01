# RL Project Plan: Self-Play on Othello (AlphaZero vs PPO)

---

## Overview

**Course**: CS 5180 — Reinforcement Learning (Spring 2026)
**Student**: Mohammed Sanan Moinuddin (solo project)
**Proposal due**: February 23, 2026 ✓ (submitted)
**Final paper due**: April 22, 2026 (no extensions allowed)

**Big picture**: Implement two fundamentally different self-play RL algorithms — AlphaZero (search-based) and PPO (search-free) — on 8×8 Othello, then rigorously compare their performance. The deliverable is an AAAI-format research paper with reproducible code.

---

## 1. Why Othello (Reversi)?

Othello is a 2-player board game played on an 8×8 grid. Players take turns placing black or white discs. When you place a disc, all of the opponent's discs between your new disc and another of your discs (in any direction) get flipped to your color. The player with the most discs at the end wins.

**Why it's a great RL domain:**
- It is a **zero-sum game** — one player's gain is exactly the other's loss. This is a clean, well-defined objective for RL.
- It is **not a solved game** — unlike Connect4 (where perfect play is known), there's no known perfect Othello player. So there's no artificial ceiling on what our agents can achieve.
- It has **no randomness** — the outcome is entirely determined by the players' decisions, not luck. This makes it easier to evaluate whether an agent genuinely learned good strategy.
- It requires **long-horizon planning** — a move that looks good now might be bad 10 moves later. This makes it a serious test for RL.
- It has a **natural self-play setup** — you don't need any external data or human demonstrations. The agent just plays against itself.
- It is **well-supported** — the alpha-zero-general framework includes a complete Othello implementation with board logic, legal-action queries, and terminal detection.

---

## 2. Formal Problem Definition (MDP)

Reinforcement learning is built on the framework of a **Markov Decision Process (MDP)**. Here's how Othello maps onto it:

| MDP Component | Othello Definition |
|---|---|
| **State (s)** | The current board configuration — which cells are black, white, or empty — plus whose turn it is |
| **Action (a)** | The cell index where the current player places their disc (only legal moves are valid) |
| **Reward (r)** | +1 if you win, −1 if you lose, 0 if draw. No intermediate rewards — only given at the end of the game |
| **Transition** | Deterministic — placing a disc has a fixed, rule-based outcome (flip opponent discs) |
| **Horizon** | Max 60 moves (the board has 64 squares, 4 are pre-filled at the start) |

**State representation for the neural network:**
We represent the board as a tensor of shape **(3, 8, 8)**:
- Channel 0: 1 where current player has a disc, 0 elsewhere
- Channel 1: 1 where opponent has a disc, 0 elsewhere
- Channel 2: all 1s if it is the current player's turn, all 0s otherwise (turn indicator)

This is the standard representation used in AlphaZero-style systems. Using the "current player's perspective" (rather than fixed black/white) lets the same neural network play as either color.

**Action space:**
Integer in {0, ..., 63} for disc placement. Invalid moves are masked (set to −∞ before softmax) so the agent can never play an illegal move.

---

## 3. Algorithms

### AlphaZero (Search-Based)

#### What is AlphaZero?
AlphaZero (Silver et al., 2017) is the algorithm DeepMind used to master Chess, Shogi, and Go — starting from zero human knowledge and learning purely by playing against itself. It combines two ideas:
1. **A neural network** that predicts (a) which moves are likely good, and (b) who is likely to win from the current position.
2. **Monte Carlo Tree Search (MCTS)** that uses the neural network's predictions to intelligently search ahead and find the best move.

#### The Neural Network (Shared CNN)
A **Convolutional Neural Network (CNN)** takes the board state as input and outputs two things:
- **Policy head**: A probability distribution over all possible actions — `P(s, a)` — representing which moves look promising.
- **Value head**: A single scalar in [−1, +1] — `V(s)` — representing the estimated probability of winning from this state.

Architecture:
```
Input (3×8×8 board)
    ↓
Convolutional backbone (residual blocks)
    ↓
   / \
Policy head    Value head
(softmax over  (tanh scalar)
 64 actions)
```

The same CNN architecture is used by both AlphaZero and PPO for a fair comparison.

#### Monte Carlo Tree Search (MCTS)
Instead of just playing the move the network thinks is best, AlphaZero uses MCTS to look several moves ahead. The process for each move:

1. **Run N simulations** (e.g., 25-50 during training, 200-800 during evaluation) from the current state, building a search tree.
2. In each simulation, use the **PUCT formula** to select which branch to explore:

   ```
   PUCT(s, a) = Q(s,a) + c_puct × P(s,a) × sqrt(N(s)) / (1 + N(s,a))
   ```

   - `Q(s,a)` = average value of past visits to action a from state s (exploitation)
   - `P(s,a)` = prior probability from the neural network (guidance)
   - `N(s)` = total visits to state s
   - `N(s,a)` = visits to action a from state s
   - `c_puct` = exploration constant (hyperparameter, typically ~1.0)

3. At the end of the simulation tree, use the neural network's value head to estimate who wins.
4. Backpropagate the result up the tree to update Q values.
5. After all N simulations, select the action with the most visits as the actual move.

#### Self-Play Training Loop
```
Repeat:
  1. Play a full game using MCTS (both sides)
     → Collect (state, π_mcts, z) for every move:
        - state: the board at that point
        - π_mcts: the normalized visit counts from MCTS (better than raw network policy)
        - z: the final game outcome (+1 or -1) from this player's perspective
  2. Add all (state, π_mcts, z) tuples to a replay buffer
  3. Sample a batch from the replay buffer
  4. Train the network to minimize:
     Loss = (V(s) - z)² - π_mcts · log P(s)
            [value loss]   [policy loss]
  5. Periodically evaluate the new network vs the old one
     → Keep the new network if it wins > 55% of games
```

#### Implementation
Adapted from the **alpha-zero-general** framework (github.com/suragnair/alpha-zero-general), which provides Coach, MCTS, Arena, and OthelloGame. The CNN and hyperparameters are tuned for 8×8 Othello.

---

### PPO with Self-Play (Search-Free)

#### What is PPO?
**Proximal Policy Optimization** (Schulman et al., 2017) is one of the most popular and reliable deep RL algorithms. It is a **policy gradient** method — it directly optimizes the policy (the function mapping states to actions) using gradient ascent on expected reward.

The key innovation of PPO over older policy gradient methods is a **clipping mechanism** that prevents the policy from changing too drastically in a single update. This makes training much more stable.

#### PPO Objective
```
r(θ) = π_new(a|s) / π_old(a|s)   (probability ratio between new and old policy)

L_CLIP = E[ min(r(θ) × A, clip(r(θ), 1-ε, 1+ε) × A) ]
```

The clip ensures `r(θ)` stays within `[1-ε, 1+ε]` (typically ε=0.2), so the policy never changes too fast in a single update step.

#### The Neural Network (same architecture as AlphaZero for fair comparison)
```
Input (3×8×8 board)
    ↓
Convolutional backbone (same residual blocks as AlphaZero)
    ↓
   / \
Actor head     Critic head
(policy π)     (value V(s))
```
- **Actor**: outputs a probability distribution over moves (the policy)
- **Critic**: outputs a scalar estimating the expected return from this state (used to compute advantage A via GAE)

#### Self-Play with Opponent Pool
PPO alone doesn't know what opponent to train against. We use an **opponent pool**:

1. Start: agent plays against a random opponent.
2. Every K training steps, save a checkpoint of the current policy to the pool.
3. When generating training data, randomly sample an opponent from the pool (not always the latest — this prevents **forgetting** how to beat weaker strategies).
4. The agent gets +1 reward for winning, −1 for losing, 0 for draw.
5. Run PPO updates on the collected game data.

This is similar to how OpenAI trained Dota 2 bots (OpenAI Five, 2019).

#### Implementation
PPO is **implemented from scratch** (not using Stable-Baselines3 or RLlib) to demonstrate understanding and because those libraries' single-agent design fights the self-play opponent pool requirement. CleanRL and the "37 Implementation Details of PPO" paper serve as implementation references.

---

### How AlphaZero and PPO Differ

| | AlphaZero | PPO |
|---|---|---|
| **Type** | Planning + learning | Pure learning (no lookahead) |
| **Uses search?** | Yes — MCTS looks many moves ahead | No — picks move directly from policy |
| **Sample efficiency** | Low (MCTS is expensive per move) | Higher (no MCTS overhead) |
| **Move quality** | Higher (search compensates for imperfect network) | Depends entirely on network quality |
| **Compute per move** | High | Low |
| **Training stability** | Moderate | High (PPO is very stable) |
| **Key strength** | Strong lookahead in complex positions | Fast training, simpler implementation |

This difference makes the comparison scientifically interesting: does the planning ability of MCTS outweigh its computational cost vs. PPO's pure learned intuition?

---

## 4. Baselines

### Baseline 1: Random Agent
Picks uniformly at random from all legal moves. This is the **lower bound** — any reasonable RL agent should beat this easily. If it doesn't, something is wrong.

### Baseline 2: Minimax with Alpha-Beta Pruning
A classical game tree search algorithm that:
- Searches the game tree to a fixed depth (**depth 4** and **depth 6**)
- Uses an **Othello positional heuristic** to evaluate non-terminal positions: corners are worth a lot, edges are worth some
- Alpha-beta pruning prunes branches that can't possibly be better than what's already found

This is a **strong classical baseline** — it has domain knowledge built in (via the heuristic) but no learning.

### Baseline 3: Head-to-Head
AlphaZero vs PPO directly. After both are trained, run 200+ games between them and report the win rate with confidence intervals. This is the core comparison.

---

## 5. What Results We Expect to Show

### Core Metrics
- **Win rate tables** (with 95% CIs) from round-robin tournaments of 200+ games per matchup
- **Elo ratings** ranking all agents on a single scale
- **Training curves** (win rate vs training iterations against fixed baselines)
- **Compute efficiency** (performance vs wall-clock GPU-hours)
- **Color asymmetry** (win rates disaggregated by playing color)
- **Policy entropy & loss curves** over training

### Ablation Studies
- AlphaZero: effect of number of MCTS simulations (50 vs 200 vs 800) at evaluation time
- PPO: effect of opponent pool size (1 vs 5 vs 20) — requires separate training runs

### Hypothesis
```
AlphaZero > PPO ≫ Minimax (depth 6) > Minimax (depth 4) > Random
```
MCTS lookahead should provide an edge over a pure policy network. However, the magnitude of this gap — and whether PPO's faster training compensates at lower compute budgets — is the central open question.

### Risks and Mitigations
| Risk | Mitigation |
|---|---|
| MCTS compute cost blows GPU budget | Profile one full iteration on day 1; use 15-25 sims during training, increase for evaluation |
| PPO self-play collapse (strategy cycling) | Opponent pool from day 1; monitor vs fixed baselines every eval cycle |
| PPO implementation bugs (37+ silent details) | Validate ratio=1.0 on first step; test on CartPole before Othello; use CleanRL as cross-reference |
| Unfair AlphaZero vs PPO comparison | Report multiple MCTS budgets; include raw-network (0 sims) comparison; report wall-clock per move |
| Slow convergence | Early sanity-check runs; reduced training if needed |
| Inconclusive results | 200+ games per matchup for statistical significance |

---

## 6. Tech Stack

**Framework**: [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) — provides Othello game logic, MCTS, Coach, Arena
**Language**: Python 3.12
**ML Framework**: PyTorch 2.6 (pin NumPy <2.0 for alpha-zero-general compatibility)
**PPO**: Implemented from scratch (CleanRL as reference, not dependency)
**Compute**: Google Cloud T4 GPU ($300 credits, ~850 GPU-hours)
**Paper**: LaTeX (AAAI 2-column format)

**Key version constraints:**
- NumPy <2.0 (alpha-zero-general compatibility)
- PyTorch 2.6.x (stable, battle-tested)
- Do NOT use PyTorch 2.11 (too new) or Stable-Baselines3 (wrong tool for self-play)

---

## 7. Implementation Roadmap

| Phase | Goal | Requirements | Target |
|-------|------|--------------|--------|
| **1. Game Environment + Infrastructure** | Working Othello game, agent interface, GPU setup, reproducibility | ENV-01..03 | Apr 1-3 |
| **2. Baselines + Evaluation Framework** | Random/minimax agents, tournament system, Elo, training curve logging | BASE-01..03, EVAL-01..04 | Apr 3-6 |
| **3. Shared CNN** | Dual-head CNN validated for both algorithms | NET-01..02 | Apr 6-7 |
| **4. AlphaZero Agent + Training** | MCTS + CNN self-play, trained agent beats baselines | AZ-01..04 | Apr 7-12 |
| **5. PPO Agent + Training** | PPO from scratch + opponent pool, trained agent beats baselines | PPO-01..04 | Apr 12-16 |
| **6. Full Evaluation + Experiments** | Head-to-head, ablations, all figures/tables | EVAL-05..08, ABL-01..02 | Apr 16-18 |
| **7. Paper Writing** | AAAI paper with all sections, code submission | PAP-01..03 | Apr 18-22 |

**Phase dependencies:**
- Phases 1 → 2 → 3 are sequential (each builds on the prior)
- Phases 2 and 3 both depend only on Phase 1 (could overlap)
- Phases 4 and 5 are independent (both depend on 2 + 3) but sequential for solo execution
- Phase 6 depends on both 4 and 5
- Phase 7 depends on 6 (partial writing can start earlier)

---

## 8. Final Paper (Due April 22) — Structure

Written in **AAAI format** (two-column).

1. **Abstract** (~150 words): What problem, what methods, what results
2. **Introduction**: Why Othello? Why self-play? Why compare AlphaZero vs PPO?
3. **Background**: MDP basics, MCTS algorithm, policy gradient overview
4. **Related Work**: AlphaGo → AlphaZero → MuZero lineage; PPO self-play (OpenAI Five); prior Othello RL papers (van der Ree & Wiering, Liskowski et al., MiniZero)
5. **Methods**: Full algorithm details with equations (PUCT formula, PPO clip objective, network architecture, training loops)
6. **Experiments**:
   - Setup (hardware, hyperparameters, number of training steps)
   - Win rate tables vs all baselines (with 95% CIs)
   - Training curves
   - Elo ratings
   - Compute & sample efficiency
   - Color asymmetry analysis
   - Ablation studies (MCTS sims, opponent pool size)
   - Analysis: where does each algorithm succeed or fail?
7. **Conclusion**: What we learned, which algorithm worked better and why, future directions

Also submit all code as a zip or GitHub link.

---

## 9. References

- Silver, D. et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." *arXiv:1712.01815* — **the AlphaZero paper**
- Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347* — **the PPO paper**
- Berner, C. et al. (2019). "Dota 2 with Large Scale Deep Reinforcement Learning." *arXiv:1912.06680* — PPO + self-play at scale (OpenAI Five)
- Wu, T. et al. (2023). "MiniZero: Comparative Analysis of AlphaZero and MuZero on Go, Othello, and Atari Games." *arXiv:2310.11305* — closest comparable work
- van der Ree, M. & Wiering, M. (2013). "Reinforcement Learning in the Game of Othello." *IEEE ADPRL* — foundational Othello RL reference
- Liskowski, P. et al. (2018). "Learning to Play Othello with Deep Neural Networks." *IEEE Trans. Games* — CNN Othello reference
- Huang, S. et al. (2022). "The 37 Implementation Details of Proximal Policy Optimization." *ICLR Blog Track* — PPO implementation reference
- Lanctot, M. et al. (2019). "OpenSpiel: A Framework for Reinforcement Learning in Games." *arXiv:1908.09453*
