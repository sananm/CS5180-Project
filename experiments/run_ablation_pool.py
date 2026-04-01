#!/usr/bin/env python3
"""
ABL-02: PPO opponent pool size ablation.

IMPORTANT: Unlike ABL-01, this ablation requires SEPARATE TRAINING RUNS.
This script documents the protocol and provides a training helper.

Pool Size Calculation:
    pool_size = 1 + floor(n_updates / checkpoint_every)
    
    For n_updates=100:
    - Pool size 1:  checkpoint_every > 100 (never checkpoint)
    - Pool size 5:  checkpoint_every = 25  (100 / (5-1) = 25)
    - Pool size 20: checkpoint_every = 5   (100 / (20-1) ≈ 5)

Usage:
    # Run training for each pool size configuration:
    python experiments/run_ablation_pool.py train --pool-size 1 --n-updates 100 --output checkpoints/ppo_pool1.pt
    python experiments/run_ablation_pool.py train --pool-size 5 --n-updates 100 --output checkpoints/ppo_pool5.pt
    python experiments/run_ablation_pool.py train --pool-size 20 --n-updates 100 --output checkpoints/ppo_pool20.pt

    # Evaluate all trained checkpoints:
    python experiments/run_ablation_pool.py eval \
        --checkpoints checkpoints/ppo_pool1.pt checkpoints/ppo_pool5.pt checkpoints/ppo_pool20.pt \
        --pool-sizes 1 5 20 \
        --output-dir results/ablation_pool
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

from othello_rl.models.shared_cnn import SharedCNN
from othello_rl.ppo.agent import PPOAgent
from othello_rl.ppo.trainer import PPOTrainer
from othello_rl.ppo.network import build_ppo_network
from othello_rl.game.othello_env import OthelloEnv
from othello_rl.agents.random_agent import RandomAgent
from othello_rl.agents.minimax_agent import MinimaxAgent
from othello_rl.evaluation.arena import Arena
from othello_rl.evaluation.stats import wilson_interval


def compute_checkpoint_every(pool_size: int, n_updates: int) -> int:
    """Compute checkpoint_every to achieve target pool size.
    
    pool_size = 1 + floor(n_updates / checkpoint_every)
    => checkpoint_every = n_updates / (pool_size - 1)  for pool_size > 1
    => checkpoint_every > n_updates                    for pool_size = 1
    """
    if pool_size <= 1:
        return n_updates + 1  # Never checkpoint
    return max(1, n_updates // (pool_size - 1))


def train_with_pool_size(pool_size: int, n_updates: int, output_path: str, device: str):
    """Train PPO with specific pool size configuration."""
    checkpoint_every = compute_checkpoint_every(pool_size, n_updates)
    
    print(f"Training PPO with target pool_size={pool_size}")
    print(f"  n_updates={n_updates}, checkpoint_every={checkpoint_every}")
    
    env = OthelloEnv()
    network = build_ppo_network()
    trainer = PPOTrainer(
        env=env,
        network=network,
        device=device,
        checkpoint_every=checkpoint_every,
    )
    
    trainer.train(n_updates=n_updates, progress=True)
    
    actual_pool_size = len(trainer.pool)
    print(f"  Actual pool size: {actual_pool_size}")
    
    trainer.save_checkpoint(output_path)
    print(f"  Saved to: {output_path}")
    
    return actual_pool_size


def evaluate_checkpoints(checkpoint_paths: list[str], pool_sizes: list[int], 
                         num_games: int, output_dir: Path, device: str):
    """Evaluate multiple PPO checkpoints against baselines."""
    random_agent = RandomAgent()
    minimax4_agent = MinimaxAgent(depth=4)
    arena = Arena()
    results = []

    for ckpt_path, pool_size in zip(checkpoint_paths, pool_sizes):
        print(f"\n=== PPO pool_size={pool_size} ({ckpt_path}) ===")
        
        network = SharedCNN()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        network.load_state_dict(ckpt['network_state_dict'])
        network.to(device)
        network.eval()
        
        ppo_agent = PPOAgent(network, device=device, deterministic=True)
        ppo_label = f'PPO_pool{pool_size}'

        # vs Random
        print(f"  vs Random ({num_games} games)...")
        match_random = arena.play_match(ppo_agent, random_agent, num_games,
                                        agent_a_label=ppo_label, agent_b_label='Random')
        score_random = match_random.wins_a + 0.5 * match_random.draws
        wr_random = score_random / match_random.games_played
        ci_random = wilson_interval(score_random, match_random.games_played)

        # vs Minimax-4
        print(f"  vs Minimax-4 ({num_games} games)...")
        match_mm4 = arena.play_match(ppo_agent, minimax4_agent, num_games,
                                     agent_a_label=ppo_label, agent_b_label='Minimax4')
        score_mm4 = match_mm4.wins_a + 0.5 * match_mm4.draws
        wr_mm4 = score_mm4 / match_mm4.games_played
        ci_mm4 = wilson_interval(score_mm4, match_mm4.games_played)

        results.append({
            'pool_size': pool_size,
            'vs_random_wr': wr_random,
            'vs_random_ci_low': ci_random[0],
            'vs_random_ci_high': ci_random[1],
            'vs_minimax4_wr': wr_mm4,
            'vs_minimax4_ci_low': ci_mm4[0],
            'vs_minimax4_ci_high': ci_mm4[1],
        })

    # Save CSV
    csv_path = output_dir / 'ablation_results.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Save summary
    summary_path = output_dir / 'summary.txt'
    with summary_path.open('w') as f:
        f.write("=" * 70 + "\n")
        f.write("ABL-02: PPO Opponent Pool Size Ablation\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Pool Size':<12} {'vs Random':<25} {'vs Minimax-4':<25}\n")
        f.write("-" * 62 + "\n")
        for r in results:
            rand_str = f"{r['vs_random_wr']:.1%} [{r['vs_random_ci_low']:.2f}-{r['vs_random_ci_high']:.2f}]"
            mm4_str = f"{r['vs_minimax4_wr']:.1%} [{r['vs_minimax4_ci_low']:.2f}-{r['vs_minimax4_ci_high']:.2f}]"
            f.write(f"{r['pool_size']:<12} {rand_str:<25} {mm4_str:<25}\n")

    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='PPO opponent pool size ablation')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train PPO with specific pool size')
    train_parser.add_argument('--pool-size', type=int, required=True, help='Target pool size (1/5/20)')
    train_parser.add_argument('--n-updates', type=int, default=100, help='Number of training updates')
    train_parser.add_argument('--output', required=True, help='Output checkpoint path')
    train_parser.add_argument('--device', default='cpu', help='Device for training')

    # Eval subcommand
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained checkpoints')
    eval_parser.add_argument('--checkpoints', nargs='+', required=True, help='Checkpoint paths')
    eval_parser.add_argument('--pool-sizes', nargs='+', type=int, required=True, help='Pool sizes for each checkpoint')
    eval_parser.add_argument('--num-games', type=int, default=100, help='Games per matchup')
    eval_parser.add_argument('--output-dir', default='results/ablation_pool', help='Output directory')
    eval_parser.add_argument('--device', default='cpu', help='Device for inference')

    args = parser.parse_args()

    if args.command == 'train':
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        train_with_pool_size(args.pool_size, args.n_updates, args.output, args.device)
    elif args.command == 'eval':
        if len(args.checkpoints) != len(args.pool_sizes):
            print("ERROR: Number of checkpoints must match number of pool sizes", file=sys.stderr)
            sys.exit(1)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluate_checkpoints(args.checkpoints, args.pool_sizes, args.num_games, output_dir, args.device)


if __name__ == '__main__':
    main()
