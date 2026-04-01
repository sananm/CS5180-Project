# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Always update CLAUDE.md with the current status of the tasks.

## Git Commit Policy

**IMPORTANT**: Never auto-generate git commit messages. Always ask the user for the commit message. Never include "Co-Authored-By: Claude" or any AI attribution in commits, code, or comments. The user writes all commit messages.

## Project Overview

CS 5180 Reinforcement Learning course project (Northeastern, Spring 2026). Solo project comparing AlphaZero vs PPO on Othello/Reversi (8×8) via self-play. Final deliverable is an AAAI-format paper + reproducible code.

**Plan**: Sonnet-plan.md
**Deadlines**: Proposal Feb 23, 2026 | Final April 22, 2026

## Current Status

- Proposal: complete (proposal.tex / proposal.pdf)
- Phase 1 COMPLETE: Game environment, tensor encoding, action masking, agent interface, seeding, config, 28-test suite
- Phase 2 COMPLETE: random/minimax baselines, balanced arena/tournament engine, Wilson CIs, Elo, CSV training-curve logging, matplotlib plotting
- Phase 2 validation COMPLETE: 47-test suite passes and explicit benchmark checks ran against random/minimax4/minimax6
- Benchmark note: on the fast fixed near-endgame benchmark, minimax4 and minimax6 both went 10-2 vs random and split 6-6 head-to-head
- Current gap: depth-6 is not yet demonstrated stronger than depth-4 from the opening position; revisit with larger evaluation runs later
- Phase 3 COMPLETE: SharedCNN, residual blocks, config constants, AlphaZero/PPO builders, and an 8-test validation suite
- Phase 3 verification COMPLETE on CPU: SharedCNN tests pass, full repo test suite is 55/55 passing, CPU forward smoke benchmark is ~2.154 ms
- Remaining hardware gate: verify the `<10ms on T4 GPU` latency target in the actual cloud environment before making final performance claims
- Phase 4 COMPLETE: MCTS with PUCT, replay buffer, AlphaZero trainer, AlphaZeroAgent, self-play pipeline, 87-test suite
- Phase 5 COMPLETE: PPO core (plan 01) + PPOTrainer with opponent pool, PPOAgent, package exports (plan 02) — IS ratio=1.0 confirmed, smoke test passes, 106-test suite
- Phase 6 COMPLETE: All evaluation infrastructure + experiment scripts — significance testing, color analysis, loss logging, compute timing, ablation scripts, plotting (141-test suite)
- Next step: Phase 7 — Paper Writing (/gsd:plan-phase 7)

## Tech Stack

- **Language**: Python
- **ML Framework**: PyTorch
- **Game/AlphaZero**: alpha-zero-general framework (includes Othello game logic + MCTS with PUCT)
- **PPO**: implemented from scratch
- **Compute**: Google Cloud (GPU)
- **Paper**: LaTeX (AAAI 2-column format)

## Architecture (Planned)

- **Shared**: Othello game (from alpha-zero-general), random agent, minimax (alpha-beta, depth 4/6), Elo rating, evaluation scripts
- **AlphaZero**: adapt alpha-zero-general's MCTS engine + CNN, tune self-play training loop
- **PPO**: full implementation from scratch with opponent pool self-play

## Key Algorithms & References

- **AlphaZero**: MCTS with PUCT selection + CNN policy/value heads (Silver et al., 2017)
- **PPO**: clipped surrogate objective + opponent pool self-play (Schulman et al., 2017)

## Evaluation

- Win rates (95% CIs) from 200+ games per matchup
- Elo ratings across all agents
- Training curves vs fixed baselines
- Ablation studies (stretch goal): MCTS sim count, opponent pool size
