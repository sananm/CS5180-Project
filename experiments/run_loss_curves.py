#!/usr/bin/env python3
"""
EVAL-08: Training loss and entropy curves.

Runs training for both algorithms while logging per-iteration
policy loss, value loss, and policy entropy to CSV.

Usage:
    # Log metrics during AlphaZero training
    python experiments/run_loss_curves.py \
        --algorithm az \
        --n-iterations 50 \
        --output results/loss_curves/az_loss.csv \
        --device cuda

    # Log metrics during PPO training
    python experiments/run_loss_curves.py \
        --algorithm ppo \
        --n-iterations 50 \
        --output results/loss_curves/ppo_loss.csv \
        --device cuda

    # Generate plots from logged CSV
    python experiments/run_loss_curves.py \
        --plot \
        --input results/loss_curves/combined_loss.csv \
        --output results/figures/loss_curves.png

Outputs:
    - CSV with columns: iteration, algorithm, policy_loss, value_loss, entropy, total_loss, wall_clock_seconds
    - PNG multi-panel figure with loss and entropy curves
"""

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from othello_rl.game.othello_game import OthelloGame
from othello_rl.game.othello_env import OthelloEnv
from othello_rl.models.shared_cnn import SharedCNN
from othello_rl.alphazero.trainer import AlphaZeroTrainer
from othello_rl.ppo.trainer import PPOTrainer
from othello_rl.evaluation.loss_logger import LossLogger, LossRow
from othello_rl.evaluation.compute_timing import timed_az_iteration, timed_ppo_iteration
from othello_rl.evaluation.plotting import plot_loss_and_entropy


def compute_policy_entropy(logits: torch.Tensor) -> float:
    """Compute entropy of policy distribution from logits.
    
    H(p) = -sum(p * log(p)) in nats.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    # Avoid log(0) by using where probs > 0
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean().item()


def run_az_with_logging(n_iterations: int, device: str, output_path: Path):
    """Train AlphaZero while logging loss and entropy."""
    game = OthelloGame(8)
    network = SharedCNN()
    trainer = AlphaZeroTrainer(
        game, network,
        device=device,
        num_sims=50,  # Faster for logging runs
        games_per_iter=5,
        epochs_per_iter=3,
        batch_size=32,
    )
    logger = LossLogger(output_path)

    for i in tqdm(range(n_iterations), desc="AlphaZero training"):
        metrics, elapsed = timed_az_iteration(trainer, device=device)
        
        # Compute policy entropy from a sample batch
        if len(trainer.buffer) >= trainer.batch_size:
            boards, _, _ = trainer.buffer.sample(trainer.batch_size)
            from othello_rl.utils.tensor_utils import board_to_tensor
            board_tensors = torch.stack([board_to_tensor(b) for b in boards]).to(device)
            trainer.network.eval()
            with torch.no_grad():
                logits, _ = trainer.network(board_tensors)
            entropy = compute_policy_entropy(logits)
        else:
            entropy = 0.0
        
        total_loss = metrics['policy_loss'] + metrics['value_loss']
        
        logger.log(LossRow(
            iteration=i + 1,
            algorithm='AlphaZero',
            policy_loss=metrics['policy_loss'],
            value_loss=metrics['value_loss'],
            entropy=entropy,
            total_loss=total_loss,
            wall_clock_seconds=elapsed,
        ))


def run_ppo_with_logging(n_iterations: int, device: str, output_path: Path):
    """Train PPO while logging loss and entropy."""
    env = OthelloEnv()
    network = SharedCNN()
    trainer = PPOTrainer(env, network, device=device, episodes_per_update=3)
    logger = LossLogger(output_path)

    for i in tqdm(range(n_iterations), desc="PPO training"):
        metrics, elapsed = timed_ppo_iteration(trainer, device=device)
        
        # PPO entropy: entropy_loss = -entropy_coef * entropy
        # We want raw entropy, so: entropy = -entropy_loss / entropy_coef
        # Default entropy_coef = 0.01
        raw_entropy = -metrics['entropy_loss'] / trainer.entropy_coef
        
        total_loss = metrics['policy_loss'] + metrics['value_loss'] + metrics['entropy_loss']
        
        logger.log(LossRow(
            iteration=i + 1,
            algorithm='PPO',
            policy_loss=metrics['policy_loss'],
            value_loss=metrics['value_loss'],
            entropy=raw_entropy,
            total_loss=total_loss,
            wall_clock_seconds=elapsed,
        ))


def main():
    parser = argparse.ArgumentParser(description='Loss and entropy curve logging')
    parser.add_argument('--algorithm', choices=['az', 'ppo'],
                        help='Algorithm to train and log')
    parser.add_argument('--n-iterations', type=int, default=50,
                        help='Number of training iterations')
    parser.add_argument('--output', help='Output CSV path (for training) or PNG path (for plotting)')
    parser.add_argument('--device', default='cpu', help='Device for training')
    parser.add_argument('--plot', action='store_true', help='Generate plot from CSV')
    parser.add_argument('--input', help='Input CSV path for plotting')
    args = parser.parse_args()

    if args.plot:
        if not args.input or not args.output:
            parser.error("--plot requires --input (CSV) and --output (PNG)")
        plot_loss_and_entropy(args.input, args.output)
        print(f"Plot saved to {args.output}")
    else:
        if not args.algorithm or not args.output:
            parser.error("Training mode requires --algorithm and --output")
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.algorithm == 'az':
            run_az_with_logging(args.n_iterations, args.device, output_path)
        else:
            run_ppo_with_logging(args.n_iterations, args.device, output_path)
        
        print(f"Loss/entropy log saved to {args.output}")


if __name__ == '__main__':
    main()
