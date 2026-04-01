#!/usr/bin/env python3
"""
PPO training script for GCP T4 GPU.

Trains PPO via opponent-pool self-play and evaluates periodically against
Random and Minimax-4 baselines. Saves checkpoints and training-curve CSVs.

Target thresholds (Phase 5 success criteria):
  - Win rate vs Random    >= 90%
  - Win rate vs Minimax-4 >= 50%

The --pool-size argument controls ABL-02 opponent pool size:
  - pool_size=1  : no checkpointing (always plays self)
  - pool_size=5  : checkpoint every n_updates//4 updates
  - pool_size=20 : checkpoint every n_updates//19 updates  (main run)

Recommended GCP runs:
    # Main run (pool_size=20, also serves as ABL-02 pool=20)
    python3 experiments/train_ppo.py \
        --device cuda \
        --n-updates 500 \
        --pool-size 20 \
        --eval-every 25 \
        --eval-games 50 \
        --checkpoint-dir checkpoints \
        --output-dir results/ppo_training

    # ABL-02: pool_size=1
    python3 experiments/train_ppo.py --pool-size 1 --run-name ppo_pool1 ...

    # ABL-02: pool_size=5
    python3 experiments/train_ppo.py --pool-size 5 --run-name ppo_pool5 ...
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from othello_rl.game.othello_env import OthelloEnv
from othello_rl.game.othello_game import OthelloGame
from othello_rl.models.shared_cnn import SharedCNN
from othello_rl.ppo.trainer import PPOTrainer
from othello_rl.ppo.agent import PPOAgent
from othello_rl.agents.random_agent import RandomAgent
from othello_rl.agents.minimax_agent import MinimaxAgent
from othello_rl.evaluation.arena import Arena
from othello_rl.evaluation.logging import TrainingCurveLogger
from othello_rl.evaluation.loss_logger import LossLogger, LossRow
from othello_rl.utils.seed import set_seed
from othello_rl.config.default import (
    PPO_CLIP_EPS, PPO_GAMMA, PPO_LAM, PPO_EPOCHS, PPO_MINIBATCH_SIZE,
    PPO_LEARNING_RATE, PPO_VALUE_COEF, PPO_ENTROPY_COEF, PPO_MAX_GRAD_NORM,
    PPO_EPISODES_PER_UPDATE, DEFAULT_SEED,
)


def compute_checkpoint_every(n_updates: int, pool_size: int) -> int:
    """Compute checkpoint_every so the pool reaches exactly pool_size entries.

    Pool starts with 1 entry (initial weights). Each checkpoint adds 1 more.
    So we need (pool_size - 1) checkpoints over n_updates updates.

    pool_size=1 : never checkpoint -> set checkpoint_every > n_updates
    pool_size=k : checkpoint every n_updates // (k - 1) updates
    """
    if pool_size <= 1:
        return n_updates + 1  # never triggers
    return max(1, n_updates // (pool_size - 1))


def evaluate(network, game: OthelloGame, device: str, eval_games: int) -> dict:
    """Evaluate current PPO network (deterministic) vs Random and Minimax-4."""
    ppo_agent = PPOAgent(network, device=device, deterministic=True)
    random_agent = RandomAgent()
    minimax4_agent = MinimaxAgent(depth=4)
    arena = Arena()

    m_random = arena.play_match(ppo_agent, random_agent, eval_games,
                                agent_a_label='PPO', agent_b_label='random')
    score_r = m_random.wins_a + 0.5 * m_random.draws
    wr_random = score_r / m_random.games_played

    m_mm4 = arena.play_match(ppo_agent, minimax4_agent, eval_games,
                             agent_a_label='PPO', agent_b_label='minimax4')
    score_m = m_mm4.wins_a + 0.5 * m_mm4.draws
    wr_mm4 = score_m / m_mm4.games_played

    return {
        'vs_random': (m_random.wins_a, m_random.draws, m_random.wins_b, wr_random),
        'vs_minimax4': (m_mm4.wins_a, m_mm4.draws, m_mm4.wins_b, wr_mm4),
    }


def main():
    parser = argparse.ArgumentParser(description='Train PPO on Othello')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--n-updates', type=int, default=500,
                        help='Total PPO update steps')
    parser.add_argument('--pool-size', type=int, default=20,
                        help='Target opponent pool size (controls checkpoint_every)')
    parser.add_argument('--episodes-per-update', type=int, default=PPO_EPISODES_PER_UPDATE,
                        help='Episodes to collect before each update')
    parser.add_argument('--ppo-epochs', type=int, default=PPO_EPOCHS,
                        help='PPO gradient epochs per update')
    parser.add_argument('--minibatch-size', type=int, default=PPO_MINIBATCH_SIZE)
    parser.add_argument('--lr', type=float, default=PPO_LEARNING_RATE)
    parser.add_argument('--eval-every', type=int, default=25,
                        help='Evaluate every N updates')
    parser.add_argument('--eval-games', type=int, default=50,
                        help='Games per eval matchup (must be even)')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--output-dir', default='results/ppo_training',
                        help='Directory for CSV logs')
    parser.add_argument('--run-name', default=None,
                        help='Name prefix for output files (default: ppo_pool{pool_size})')
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

    run_name = args.run_name or f'ppo_pool{args.pool_size}'
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_every = compute_checkpoint_every(args.n_updates, args.pool_size)

    env = OthelloEnv()
    game = OthelloGame(8)
    network = SharedCNN().to(device)

    trainer = PPOTrainer(
        env=env,
        network=network,
        device=device,
        gamma=PPO_GAMMA,
        lam=PPO_LAM,
        clip_eps=PPO_CLIP_EPS,
        value_coef=PPO_VALUE_COEF,
        entropy_coef=PPO_ENTROPY_COEF,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        episodes_per_update=args.episodes_per_update,
        checkpoint_every=checkpoint_every,
    )

    start_update = 0
    if args.resume:
        print(f'Resuming from {args.resume}')
        trainer.load_checkpoint(args.resume)
        stem = Path(args.resume).stem
        if 'upd' in stem:
            try:
                start_update = int(stem.split('upd')[1])
            except ValueError:
                pass
        print(f'  Continuing from update {start_update}')

    win_logger = TrainingCurveLogger(out_dir / 'training_curves.csv')
    loss_logger = LossLogger(out_dir / 'loss_curves.csv')

    print(f'\nPPO Training ({run_name})')
    print(f'  Device:             {device}')
    print(f'  Updates:            {args.n_updates}')
    print(f'  Episodes/update:    {args.episodes_per_update}')
    print(f'  Pool size target:   {args.pool_size}')
    print(f'  Checkpoint every:   {checkpoint_every} updates')
    print(f'  Eval every:         {args.eval_every} updates ({args.eval_games} games each)\n')

    best_wr_random = 0.0
    best_wr_mm4 = 0.0

    for update in tqdm(range(start_update + 1, start_update + args.n_updates + 1),
                       desc=f'PPO ({run_name})', initial=start_update,
                       total=start_update + args.n_updates):
        # Collect rollout and update
        buffer = trainer.collect_rollout()
        info = trainer.update(buffer)

        # Log losses every update
        loss_logger.log(LossRow(
            iteration=update,
            algorithm=f'PPO_pool{args.pool_size}',
            policy_loss=info['policy_loss'],
            value_loss=info['value_loss'],
            entropy=abs(info['entropy_loss']) / PPO_ENTROPY_COEF,  # recover raw entropy
            total_loss=info['policy_loss'] + PPO_VALUE_COEF * info['value_loss'] + info['entropy_loss'],
            wall_clock_seconds=0.0,  # timing via run_compute_efficiency.py
        ))

        # Periodic evaluation
        if update % args.eval_every == 0:
            print(f'\n[Update {update}] Pool size: {len(trainer.pool)} | '
                  f'Evaluating ({args.eval_games} games each)...')
            results = evaluate(trainer.network, game, device, args.eval_games)

            ckpt_path = ckpt_dir / f'{run_name}_upd{update:05d}.pt'
            trainer.save_checkpoint(str(ckpt_path))

            w_r, d_r, l_r, wr_random = results['vs_random']
            w_m, d_m, l_m, wr_mm4 = results['vs_minimax4']

            win_logger.log_from_counts(
                iteration=update, agent_name=f'PPO_pool{args.pool_size}',
                opponent='random', wins=w_r, draws=d_r, losses=l_r,
                checkpoint_path=str(ckpt_path)
            )
            win_logger.log_from_counts(
                iteration=update, agent_name=f'PPO_pool{args.pool_size}',
                opponent='minimax4', wins=w_m, draws=d_m, losses=l_m,
                checkpoint_path=str(ckpt_path)
            )

            if wr_random > best_wr_random:
                best_wr_random = wr_random
                trainer.save_checkpoint(str(ckpt_dir / f'{run_name}_best_vs_random.pt'))

            if wr_mm4 > best_wr_mm4:
                best_wr_mm4 = wr_mm4
                trainer.save_checkpoint(str(ckpt_dir / f'{run_name}_best_vs_minimax4.pt'))

            print(f'  vs Random:    {wr_random:.1%} ({w_r}W/{d_r}D/{l_r}L)'
                  f'  [best: {best_wr_random:.1%}]')
            print(f'  vs Minimax-4: {wr_mm4:.1%} ({w_m}W/{d_m}D/{l_m}L)'
                  f'  [best: {best_wr_mm4:.1%}]')

            # Stop early if targets met
            if wr_random >= 0.90 and wr_mm4 >= 0.50:
                print(f'\nTarget thresholds reached at update {update}! Stopping.')
                break

    # Final checkpoint
    final_path = ckpt_dir / f'{run_name}_final.pt'
    trainer.save_checkpoint(str(final_path))
    print(f'\nTraining complete ({run_name}).')
    print(f'  Final checkpoint: {final_path}')
    print(f'  Final pool size:  {len(trainer.pool)}')
    print(f'  Best vs Random:    {best_wr_random:.1%}')
    print(f'  Best vs Minimax-4: {best_wr_mm4:.1%}')
    print(f'  Training curves:   {out_dir}/training_curves.csv')
    print(f'  Loss curves:       {out_dir}/loss_curves.csv')

    if best_wr_random < 0.90:
        print('\nWARNING: Target win rate vs Random (90%) not reached.')
    if best_wr_mm4 < 0.50:
        print('WARNING: Target win rate vs Minimax-4 (50%) not reached.')


if __name__ == '__main__':
    main()
