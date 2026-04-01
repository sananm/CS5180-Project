#!/usr/bin/env python3
"""
EVAL-06: Compute efficiency analysis - performance vs wall-clock GPU-hours.

Measures wall-clock time per training iteration for both algorithms
and logs cumulative training cost.

Usage:
    python experiments/run_compute_efficiency.py \
        --algorithm az \
        --n-iterations 100 \
        --output-dir results/compute_efficiency \
        --device cuda

    python experiments/run_compute_efficiency.py \
        --algorithm ppo \
        --n-iterations 100 \
        --output-dir results/compute_efficiency \
        --device cuda

Outputs:
    - results/compute_efficiency/{algorithm}_timing.csv: Per-iteration timing
    - results/compute_efficiency/summary.txt: Summary statistics
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from othello_rl.game.othello_game import OthelloGame
from othello_rl.game.othello_env import OthelloEnv
from othello_rl.models.shared_cnn import SharedCNN
from othello_rl.alphazero.trainer import AlphaZeroTrainer
from othello_rl.ppo.trainer import PPOTrainer
from othello_rl.evaluation.compute_timing import timed_az_iteration, timed_ppo_iteration


def run_az_timing(n_iterations: int, device: str, output_dir: Path):
    """Measure AlphaZero training iteration times."""
    game = OthelloGame(8)
    network = SharedCNN()
    trainer = AlphaZeroTrainer(
        game, network,
        device=device,
        num_sims=100,  # Standard evaluation sim count
        games_per_iter=10,
        epochs_per_iter=5,
    )

    csv_path = output_dir / 'az_timing.csv'
    total_seconds = 0.0
    
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'iteration', 'wall_clock_seconds', 'cumulative_seconds',
            'num_examples', 'avg_loss'
        ])
        writer.writeheader()
        
        for i in tqdm(range(n_iterations), desc="AlphaZero timing"):
            metrics, elapsed = timed_az_iteration(trainer, device=device)
            total_seconds += elapsed
            writer.writerow({
                'iteration': i + 1,
                'wall_clock_seconds': f'{elapsed:.3f}',
                'cumulative_seconds': f'{total_seconds:.3f}',
                'num_examples': metrics['num_examples'],
                'avg_loss': f"{metrics['avg_loss']:.6f}",
            })
            f.flush()
    
    return total_seconds, n_iterations


def run_ppo_timing(n_iterations: int, device: str, output_dir: Path):
    """Measure PPO training iteration times."""
    env = OthelloEnv()
    network = SharedCNN()
    trainer = PPOTrainer(env, network, device=device)

    csv_path = output_dir / 'ppo_timing.csv'
    total_seconds = 0.0
    
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'iteration', 'wall_clock_seconds', 'cumulative_seconds',
            'policy_loss', 'value_loss', 'entropy_loss'
        ])
        writer.writeheader()
        
        for i in tqdm(range(n_iterations), desc="PPO timing"):
            metrics, elapsed = timed_ppo_iteration(trainer, device=device)
            total_seconds += elapsed
            writer.writerow({
                'iteration': i + 1,
                'wall_clock_seconds': f'{elapsed:.3f}',
                'cumulative_seconds': f'{total_seconds:.3f}',
                'policy_loss': f"{metrics['policy_loss']:.6f}",
                'value_loss': f"{metrics['value_loss']:.6f}",
                'entropy_loss': f"{metrics['entropy_loss']:.6f}",
            })
            f.flush()
    
    return total_seconds, n_iterations


def main():
    parser = argparse.ArgumentParser(description='Compute efficiency analysis')
    parser.add_argument('--algorithm', choices=['az', 'ppo'], required=True,
                        help='Algorithm to profile (az=AlphaZero, ppo=PPO)')
    parser.add_argument('--n-iterations', type=int, default=100,
                        help='Number of training iterations')
    parser.add_argument('--output-dir', default='results/compute_efficiency',
                        help='Output directory')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Profiling {args.algorithm.upper()} on {args.device} for {args.n_iterations} iterations...")
    
    if args.algorithm == 'az':
        total_seconds, n_iters = run_az_timing(args.n_iterations, args.device, output_dir)
    else:
        total_seconds, n_iters = run_ppo_timing(args.n_iterations, args.device, output_dir)

    avg_per_iter = total_seconds / n_iters
    gpu_hours = total_seconds / 3600

    # Append to summary
    summary_path = output_dir / 'summary.txt'
    with summary_path.open('a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Algorithm: {args.algorithm.upper()}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Iterations: {n_iters}\n")
        f.write(f"Total time: {total_seconds:.2f} seconds ({gpu_hours:.4f} GPU-hours)\n")
        f.write(f"Average per iteration: {avg_per_iter:.3f} seconds\n")

    print(f"\nTotal: {total_seconds:.2f}s ({gpu_hours:.4f} GPU-hours)")
    print(f"Average: {avg_per_iter:.3f}s per iteration")
    print(f"Results saved to {output_dir}/")


if __name__ == '__main__':
    main()
