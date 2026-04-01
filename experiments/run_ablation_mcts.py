#!/usr/bin/env python3
"""
ABL-01: AlphaZero MCTS simulation count ablation.

Evaluates a SINGLE trained AlphaZero checkpoint at different sim counts (50/200/800).
No retraining needed - only evaluation-time parameter change.

Usage:
    python experiments/run_ablation_mcts.py \
        --az-checkpoint checkpoints/az_final.pt \
        --num-games 100 \
        --output-dir results/ablation_mcts

Outputs:
    - results/ablation_mcts/ablation_results.csv: Win rates at each sim count vs baselines
    - results/ablation_mcts/summary.txt: Human-readable table
"""

import argparse
import csv
from pathlib import Path

import torch

from othello_rl.game.othello_game import OthelloGame
from othello_rl.models.shared_cnn import SharedCNN
from othello_rl.alphazero.agent import AlphaZeroAgent
from othello_rl.agents.random_agent import RandomAgent
from othello_rl.agents.minimax_agent import MinimaxAgent
from othello_rl.evaluation.arena import Arena
from othello_rl.evaluation.stats import wilson_interval


SIM_COUNTS = [50, 200, 800]


def load_network(checkpoint_path: str, device: str):
    """Load trained network from checkpoint."""
    network = SharedCNN()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    network.load_state_dict(ckpt['network_state_dict'])
    network.to(device)
    network.eval()
    return network


def main():
    parser = argparse.ArgumentParser(description='MCTS simulation count ablation')
    parser.add_argument('--az-checkpoint', required=True, help='AlphaZero checkpoint path')
    parser.add_argument('--num-games', type=int, default=100, help='Games per matchup (must be even)')
    parser.add_argument('--output-dir', default='results/ablation_mcts', help='Output directory')
    parser.add_argument('--device', default='cpu', help='Device for inference')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    game = OthelloGame(8)
    network = load_network(args.az_checkpoint, args.device)

    # Baseline opponents
    random_agent = RandomAgent()
    minimax4_agent = MinimaxAgent(depth=4)

    arena = Arena()
    results = []

    for num_sims in SIM_COUNTS:
        print(f"\n=== AlphaZero with {num_sims} simulations ===")
        az_agent = AlphaZeroAgent(game, network, num_sims=num_sims, device=args.device)
        az_label = f'AZ_{num_sims}'

        # vs Random
        print(f"  vs Random ({args.num_games} games)...")
        match_random = arena.play_match(az_agent, random_agent, args.num_games,
                                        agent_a_label=az_label, agent_b_label='Random')
        score_random = match_random.wins_a + 0.5 * match_random.draws
        wr_random = score_random / match_random.games_played
        ci_random = wilson_interval(score_random, match_random.games_played)

        # vs Minimax-4
        print(f"  vs Minimax-4 ({args.num_games} games)...")
        match_mm4 = arena.play_match(az_agent, minimax4_agent, args.num_games,
                                     agent_a_label=az_label, agent_b_label='Minimax4')
        score_mm4 = match_mm4.wins_a + 0.5 * match_mm4.draws
        wr_mm4 = score_mm4 / match_mm4.games_played
        ci_mm4 = wilson_interval(score_mm4, match_mm4.games_played)

        results.append({
            'num_sims': num_sims,
            'vs_random_wr': wr_random,
            'vs_random_ci_low': ci_random[0],
            'vs_random_ci_high': ci_random[1],
            'vs_random_wins': match_random.wins_a,
            'vs_random_losses': match_random.wins_b,
            'vs_random_draws': match_random.draws,
            'vs_minimax4_wr': wr_mm4,
            'vs_minimax4_ci_low': ci_mm4[0],
            'vs_minimax4_ci_high': ci_mm4[1],
            'vs_minimax4_wins': match_mm4.wins_a,
            'vs_minimax4_losses': match_mm4.wins_b,
            'vs_minimax4_draws': match_mm4.draws,
        })

    # Save CSV
    csv_path = output_dir / 'ablation_results.csv'
    with csv_path.open('w', newline='') as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Save summary
    summary_path = output_dir / 'summary.txt'
    with summary_path.open('w') as f:
        f.write("=" * 70 + "\n")
        f.write("ABL-01: AlphaZero MCTS Simulation Count Ablation\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Checkpoint: {args.az_checkpoint}\n")
        f.write(f"Games per matchup: {args.num_games}\n\n")
        f.write(f"{'Sims':<8} {'vs Random':<20} {'vs Minimax-4':<20}\n")
        f.write("-" * 48 + "\n")
        for r in results:
            rand_str = f"{r['vs_random_wr']:.1%} [{r['vs_random_ci_low']:.2f}-{r['vs_random_ci_high']:.2f}]"
            mm4_str = f"{r['vs_minimax4_wr']:.1%} [{r['vs_minimax4_ci_low']:.2f}-{r['vs_minimax4_ci_high']:.2f}]"
            f.write(f"{r['num_sims']:<8} {rand_str:<20} {mm4_str:<20}\n")

    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
