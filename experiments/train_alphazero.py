#!/usr/bin/env python3
"""
AlphaZero training script for GCP T4 GPU.

Trains AlphaZero via self-play and evaluates periodically against Random and
Minimax-4 baselines. Saves checkpoints and training-curve CSVs.

Target thresholds (Phase 4 success criteria):
  - Win rate vs Random    >= 90%
  - Win rate vs Minimax-4 >= 60%

Recommended GCP run:
    python3 experiments/train_alphazero.py \
        --device cuda \
        --num-iterations 200 \
        --games-per-iter 100 \
        --num-sims 100 \
        --eval-every 10 \
        --eval-games 50 \
        --checkpoint-dir checkpoints \
        --output-dir results/alphazero_training

Resume from checkpoint:
    python3 experiments/train_alphazero.py ... --resume checkpoints/az_iter050.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from othello_rl.game.othello_game import OthelloGame
from othello_rl.models.shared_cnn import SharedCNN
from othello_rl.alphazero.trainer import AlphaZeroTrainer
from othello_rl.alphazero.agent import AlphaZeroAgent
from othello_rl.agents.random_agent import RandomAgent
from othello_rl.agents.minimax_agent import MinimaxAgent
from othello_rl.evaluation.arena import Arena
from othello_rl.evaluation.logging import TrainingCurveLogger
from othello_rl.evaluation.loss_logger import LossLogger, LossRow
from othello_rl.utils.seed import set_seed
from othello_rl.config.default import (
    AZ_NUM_SIMS, AZ_CPUCT, AZ_LEARNING_RATE, AZ_BATCH_SIZE,
    AZ_REPLAY_BUFFER_SIZE, AZ_GAMES_PER_ITER, AZ_EPOCHS_PER_ITER,
    AZ_TEMP_THRESHOLD, DEFAULT_SEED,
)


def evaluate(trainer: AlphaZeroTrainer, game: OthelloGame, num_sims: int,
             device: str, eval_games: int) -> dict:
    """Evaluate current network vs Random and Minimax-4. Returns win rate dict."""
    network = trainer.network
    az_agent = AlphaZeroAgent(game, network, num_sims=num_sims, device=device)
    random_agent = RandomAgent()
    minimax4_agent = MinimaxAgent(depth=4)
    arena = Arena()

    m_random = arena.play_match(az_agent, random_agent, eval_games,
                                agent_a_label='AlphaZero', agent_b_label='random')
    score_r = m_random.wins_a + 0.5 * m_random.draws
    wr_random = score_r / m_random.games_played

    m_mm4 = arena.play_match(az_agent, minimax4_agent, eval_games,
                             agent_a_label='AlphaZero', agent_b_label='minimax4')
    score_m = m_mm4.wins_a + 0.5 * m_mm4.draws
    wr_mm4 = score_m / m_mm4.games_played

    return {
        'vs_random': (m_random.wins_a, m_random.draws, m_random.wins_b, wr_random),
        'vs_minimax4': (m_mm4.wins_a, m_mm4.draws, m_mm4.wins_b, wr_mm4),
    }


def main():
    parser = argparse.ArgumentParser(description='Train AlphaZero on Othello')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--num-iterations', type=int, default=200,
                        help='Total training iterations')
    parser.add_argument('--num-sims', type=int, default=AZ_NUM_SIMS,
                        help='MCTS simulations per move during self-play')
    parser.add_argument('--games-per-iter', type=int, default=AZ_GAMES_PER_ITER,
                        help='Self-play games per iteration')
    parser.add_argument('--epochs-per-iter', type=int, default=AZ_EPOCHS_PER_ITER,
                        help='Training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=AZ_BATCH_SIZE)
    parser.add_argument('--max-buffer-size', type=int, default=AZ_REPLAY_BUFFER_SIZE,
                        help='Replay buffer capacity (smaller = faster, stabilizes sooner)')
    parser.add_argument('--lr', type=float, default=AZ_LEARNING_RATE)
    parser.add_argument('--eval-every', type=int, default=10,
                        help='Evaluate every N iterations')
    parser.add_argument('--eval-games', type=int, default=50,
                        help='Games per eval matchup (must be even)')
    parser.add_argument('--eval-sims', type=int, default=200,
                        help='MCTS sims used during evaluation (higher = stronger eval)')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--output-dir', default='results/alphazero_training',
                        help='Directory for CSV logs')
    parser.add_argument('--resume', default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.eval_games % 2 != 0:
        print('ERROR: --eval-games must be even', file=sys.stderr)
        sys.exit(1)

    set_seed(args.seed)
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('WARNING: CUDA not available, falling back to CPU')
        device = 'cpu'

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    game = OthelloGame(8)
    network = SharedCNN().to(device)

    trainer = AlphaZeroTrainer(
        game=game,
        network=network,
        device=device,
        num_sims=args.num_sims,
        cpuct=AZ_CPUCT,
        lr=args.lr,
        batch_size=args.batch_size,
        max_buffer_size=args.max_buffer_size,
        games_per_iter=args.games_per_iter,
        epochs_per_iter=args.epochs_per_iter,
        temp_threshold=AZ_TEMP_THRESHOLD,
        augment_symmetries=True,
    )

    start_iter = 0
    if args.resume:
        print(f'Resuming from {args.resume}')
        trainer.load_checkpoint(args.resume)
        # Infer start iteration from filename if possible (e.g. az_iter050.pt -> 50)
        stem = Path(args.resume).stem
        if 'iter' in stem:
            try:
                start_iter = int(stem.split('iter')[1])
            except ValueError:
                pass
        print(f'  Continuing from iteration {start_iter}')

    win_logger = TrainingCurveLogger(out_dir / 'training_curves.csv')
    loss_logger = LossLogger(out_dir / 'loss_curves.csv')

    print(f'\nAlphaZero Training')
    print(f'  Device:          {device}')
    print(f'  Iterations:      {args.num_iterations}')
    print(f'  Games/iter:      {args.games_per_iter}')
    print(f'  MCTS sims:       {args.num_sims} (train) / {args.eval_sims} (eval)')
    print(f'  Eval every:      {args.eval_every} iters ({args.eval_games} games each)\n')

    best_wr_random = 0.0
    best_wr_mm4 = 0.0

    for iteration in tqdm(range(start_iter + 1, start_iter + args.num_iterations + 1),
                          desc='AlphaZero', initial=start_iter,
                          total=start_iter + args.num_iterations):
        # Self-play
        num_examples = trainer.run_self_play()

        # Training
        avg_loss = trainer.run_training()

        # Compute rough policy entropy from last batch for logging
        # (use placeholder; full entropy tracking requires modifying train_step)
        loss_logger.log(LossRow(
            iteration=iteration,
            algorithm='AlphaZero',
            policy_loss=0.0,   # run_training returns avg combined loss only
            value_loss=0.0,    # breakdown not exposed by run_training
            entropy=0.0,       # not computed by AlphaZeroTrainer.run_training
            total_loss=avg_loss,
            wall_clock_seconds=0.0,  # timing handled by run_compute_efficiency.py
        ))

        # Periodic evaluation
        if iteration % args.eval_every == 0:
            print(f'\n[Iter {iteration}] Evaluating ({args.eval_games} games each)...')
            results = evaluate(trainer, game, args.eval_sims, device, args.eval_games)

            ckpt_path = ckpt_dir / f'az_iter{iteration:04d}.pt'
            trainer.save_checkpoint(str(ckpt_path))

            w_r, d_r, l_r, wr_random = results['vs_random']
            w_m, d_m, l_m, wr_mm4 = results['vs_minimax4']

            win_logger.log_from_counts(
                iteration=iteration, agent_name='AlphaZero', opponent='random',
                wins=w_r, draws=d_r, losses=l_r, checkpoint_path=str(ckpt_path)
            )
            win_logger.log_from_counts(
                iteration=iteration, agent_name='AlphaZero', opponent='minimax4',
                wins=w_m, draws=d_m, losses=l_m, checkpoint_path=str(ckpt_path)
            )

            if wr_random > best_wr_random:
                best_wr_random = wr_random
                trainer.save_checkpoint(str(ckpt_dir / 'az_best_vs_random.pt'))

            if wr_mm4 > best_wr_mm4:
                best_wr_mm4 = wr_mm4
                trainer.save_checkpoint(str(ckpt_dir / 'az_best_vs_minimax4.pt'))

            print(f'  vs Random:    {wr_random:.1%} ({w_r}W/{d_r}D/{l_r}L)'
                  f'  [best: {best_wr_random:.1%}]')
            print(f'  vs Minimax-4: {wr_mm4:.1%} ({w_m}W/{d_m}D/{l_m}L)'
                  f'  [best: {best_wr_mm4:.1%}]')

            # Stop early if targets met
            if wr_random >= 0.90 and wr_mm4 >= 0.60:
                print(f'\nTarget thresholds reached at iteration {iteration}! Stopping.')
                break

    # Final checkpoint
    final_path = ckpt_dir / 'az_final.pt'
    trainer.save_checkpoint(str(final_path))
    print(f'\nTraining complete.')
    print(f'  Final checkpoint: {final_path}')
    print(f'  Best vs Random:    {best_wr_random:.1%}')
    print(f'  Best vs Minimax-4: {best_wr_mm4:.1%}')
    print(f'  Training curves:   {out_dir}/training_curves.csv')

    if best_wr_random < 0.90:
        print('\nWARNING: Target win rate vs Random (90%) not reached.')
    if best_wr_mm4 < 0.60:
        print('WARNING: Target win rate vs Minimax-4 (60%) not reached.')


if __name__ == '__main__':
    main()
