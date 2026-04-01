#!/usr/bin/env python3
"""
EVAL-05, EVAL-07: Head-to-head AlphaZero vs PPO with statistical significance and color analysis.

Usage:
    python experiments/run_head_to_head.py \
        --az-checkpoint checkpoints/az_final.pt \
        --ppo-checkpoint checkpoints/ppo_final.pt \
        --num-games 200 \
        --output-dir results/head_to_head

Outputs:
    - results/head_to_head/results.csv: Win/loss/draw counts and statistics
    - results/head_to_head/color_analysis.csv: Per-color win rates
    - results/head_to_head/summary.txt: Human-readable summary with p-value
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

from othello_rl.game.othello_game import OthelloGame
from othello_rl.models.shared_cnn import SharedCNN
from othello_rl.alphazero.agent import AlphaZeroAgent
from othello_rl.ppo.agent import PPOAgent
from othello_rl.evaluation.arena import Arena
from othello_rl.evaluation.significance import binomial_test
from othello_rl.evaluation.color_analysis import disaggregate_by_color


def load_az_agent(checkpoint_path: str, num_sims: int, device: str) -> AlphaZeroAgent:
    """Load AlphaZero agent from checkpoint."""
    game = OthelloGame(8)
    network = SharedCNN()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    network.load_state_dict(ckpt['network_state_dict'])
    network.to(device)
    network.eval()
    return AlphaZeroAgent(game, network, num_sims=num_sims, device=device)


def load_ppo_agent(checkpoint_path: str, device: str) -> PPOAgent:
    """Load PPO agent from checkpoint (deterministic mode for evaluation)."""
    network = SharedCNN()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    network.load_state_dict(ckpt['network_state_dict'])
    network.to(device)
    network.eval()
    return PPOAgent(network, device=device, deterministic=True)


def main():
    parser = argparse.ArgumentParser(description='Head-to-head AlphaZero vs PPO evaluation')
    parser.add_argument('--az-checkpoint', required=True, help='AlphaZero checkpoint path')
    parser.add_argument('--ppo-checkpoint', required=True, help='PPO checkpoint path')
    parser.add_argument('--num-games', type=int, default=200, help='Number of games (must be even)')
    parser.add_argument('--az-sims', type=int, default=200, help='MCTS simulations for AlphaZero')
    parser.add_argument('--output-dir', default='results/head_to_head', help='Output directory')
    parser.add_argument('--device', default='cpu', help='Device for inference')
    args = parser.parse_args()

    if args.num_games % 2 != 0:
        print("ERROR: --num-games must be even for balanced color assignment", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AlphaZero from {args.az_checkpoint} (sims={args.az_sims})...")
    az_agent = load_az_agent(args.az_checkpoint, args.az_sims, args.device)

    print(f"Loading PPO from {args.ppo_checkpoint} (deterministic=True)...")
    ppo_agent = load_ppo_agent(args.ppo_checkpoint, args.device)

    print(f"Running {args.num_games} games...")
    arena = Arena()
    match = arena.play_match(az_agent, ppo_agent, num_games=args.num_games,
                             agent_a_label='AlphaZero', agent_b_label='PPO')

    # Statistical significance
    sig = binomial_test(match.wins_a, match.wins_b, match.draws)

    # Color analysis
    az_color = disaggregate_by_color(match.records, 'AlphaZero')
    ppo_color = disaggregate_by_color(match.records, 'PPO')

    # Save results CSV
    results_csv = output_dir / 'results.csv'
    with results_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['az_wins', match.wins_a])
        writer.writerow(['ppo_wins', match.wins_b])
        writer.writerow(['draws', match.draws])
        writer.writerow(['total_games', match.games_played])
        writer.writerow(['decisive_games', sig.decisive_games])
        writer.writerow(['p_value', f'{sig.p_value:.6f}'])
        writer.writerow(['ci_low', f'{sig.ci_low:.4f}'])
        writer.writerow(['ci_high', f'{sig.ci_high:.4f}'])
        writer.writerow(['significant_at_05', sig.significant_at_05])

    # Save color analysis CSV
    color_csv = output_dir / 'color_analysis.csv'
    with color_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['agent', 'color', 'games', 'wins', 'draws', 'losses', 'win_rate', 'ci_low', 'ci_high'])
        for analysis in [az_color, ppo_color]:
            for color, stats in [('black', analysis.as_black), ('white', analysis.as_white)]:
                writer.writerow([
                    analysis.agent_name, color, stats.games, stats.wins, stats.draws,
                    stats.losses, f'{stats.win_rate:.4f}', f'{stats.ci_low:.4f}', f'{stats.ci_high:.4f}'
                ])

    # Save summary
    summary_txt = output_dir / 'summary.txt'
    with summary_txt.open('w') as f:
        f.write("=" * 60 + "\n")
        f.write("HEAD-TO-HEAD: AlphaZero vs PPO\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Games played: {match.games_played}\n")
        f.write(f"AlphaZero wins: {match.wins_a}\n")
        f.write(f"PPO wins: {match.wins_b}\n")
        f.write(f"Draws: {match.draws}\n\n")
        f.write(f"Decisive games: {sig.decisive_games}\n")
        f.write(f"P-value: {sig.p_value:.6f}\n")
        f.write(f"95% CI: [{sig.ci_low:.4f}, {sig.ci_high:.4f}]\n")
        f.write(f"Significant at α=0.05: {'YES' if sig.significant_at_05 else 'NO'}\n\n")
        f.write("COLOR ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"AlphaZero as Black: {az_color.as_black.wins}/{az_color.as_black.games} wins ({az_color.as_black.win_rate:.1%})\n")
        f.write(f"AlphaZero as White: {az_color.as_white.wins}/{az_color.as_white.games} wins ({az_color.as_white.win_rate:.1%})\n")
        f.write(f"PPO as Black: {ppo_color.as_black.wins}/{ppo_color.as_black.games} wins ({ppo_color.as_black.win_rate:.1%})\n")
        f.write(f"PPO as White: {ppo_color.as_white.wins}/{ppo_color.as_white.games} wins ({ppo_color.as_white.win_rate:.1%})\n")

    print(f"\nResults saved to {output_dir}/")
    print(f"  - results.csv")
    print(f"  - color_analysis.csv")
    print(f"  - summary.txt")
    print(f"\nAlphaZero: {match.wins_a} wins | PPO: {match.wins_b} wins | Draws: {match.draws}")
    print(f"P-value: {sig.p_value:.6f} ({'significant' if sig.significant_at_05 else 'not significant'} at α=0.05)")


if __name__ == '__main__':
    main()
