"""Tests for experiment runner scripts and new plotting functions."""
from __future__ import annotations

import csv
import os
import sys
import tempfile

import pytest


class TestExperimentImports:
    """Verify all experiment scripts are importable and expose expected symbols."""

    def test_run_head_to_head_imports(self):
        from experiments.run_head_to_head import load_az_agent, load_ppo_agent, main  # noqa: F401

    def test_run_ablation_mcts_imports(self):
        from experiments.run_ablation_mcts import load_network, main, SIM_COUNTS  # noqa: F401

    def test_run_ablation_pool_imports(self):
        from experiments.run_ablation_pool import (  # noqa: F401
            compute_checkpoint_every,
            train_with_pool_size,
            evaluate_checkpoints,
            main,
        )

    def test_sim_counts_values(self):
        from experiments.run_ablation_mcts import SIM_COUNTS
        assert SIM_COUNTS == [50, 200, 800]


class TestComputeCheckpointEvery:
    """Unit tests for pool-size / checkpoint-every calculation."""

    def test_pool_size_1_never_checkpoints(self):
        from experiments.run_ablation_pool import compute_checkpoint_every
        result = compute_checkpoint_every(1, 100)
        assert result == 101

    def test_pool_size_5_correct(self):
        from experiments.run_ablation_pool import compute_checkpoint_every
        result = compute_checkpoint_every(5, 100)
        assert result == 25

    def test_pool_size_20_correct(self):
        from experiments.run_ablation_pool import compute_checkpoint_every
        result = compute_checkpoint_every(20, 100)
        assert result == 5

    def test_pool_size_0_treated_as_1(self):
        from experiments.run_ablation_pool import compute_checkpoint_every
        # pool_size <= 1 should never checkpoint
        result = compute_checkpoint_every(0, 100)
        assert result == 101


class TestArgparseHelp:
    """Verify argparse --help exits with code 0 for all scripts."""

    def _run_help(self, module_path: str):
        saved_argv = sys.argv[:]
        try:
            sys.argv = [module_path, '--help']
            if module_path == 'experiments/run_head_to_head.py':
                from experiments.run_head_to_head import main
            elif module_path == 'experiments/run_ablation_mcts.py':
                from experiments.run_ablation_mcts import main
            else:
                from experiments.run_ablation_pool import main
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0
        finally:
            sys.argv = saved_argv

    def test_head_to_head_help(self):
        self._run_help('experiments/run_head_to_head.py')

    def test_ablation_mcts_help(self):
        self._run_help('experiments/run_ablation_mcts.py')


class TestPlottingFunctions:
    """Tests for new plotting utility functions."""

    def test_plotting_imports(self):
        from othello_rl.evaluation.plotting import (  # noqa: F401
            plot_training_curves,
            plot_ablation_results,
            plot_color_asymmetry,
        )

    def test_plot_ablation_results_creates_png(self):
        from othello_rl.evaluation.plotting import plot_ablation_results

        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, 'ablation.csv')
            png_path = os.path.join(d, 'ablation.png')

            fieldnames = [
                'num_sims', 'vs_random_wr', 'vs_random_ci_low', 'vs_random_ci_high',
                'vs_minimax4_wr', 'vs_minimax4_ci_low', 'vs_minimax4_ci_high',
            ]
            rows = [
                {'num_sims': 50, 'vs_random_wr': 0.8, 'vs_random_ci_low': 0.7,
                 'vs_random_ci_high': 0.9, 'vs_minimax4_wr': 0.5, 'vs_minimax4_ci_low': 0.4,
                 'vs_minimax4_ci_high': 0.6},
                {'num_sims': 200, 'vs_random_wr': 0.9, 'vs_random_ci_low': 0.85,
                 'vs_random_ci_high': 0.95, 'vs_minimax4_wr': 0.6, 'vs_minimax4_ci_low': 0.5,
                 'vs_minimax4_ci_high': 0.7},
            ]
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            plot_ablation_results(csv_path, png_path)
            assert os.path.exists(png_path), "Ablation PNG was not created"

    def test_plot_ablation_results_custom_x_key(self):
        """plot_ablation_results accepts a custom x_key for pool size ablation CSVs."""
        from othello_rl.evaluation.plotting import plot_ablation_results

        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, 'pool_ablation.csv')
            png_path = os.path.join(d, 'pool_ablation.png')

            fieldnames = [
                'pool_size', 'vs_random_wr', 'vs_random_ci_low', 'vs_random_ci_high',
                'vs_minimax4_wr', 'vs_minimax4_ci_low', 'vs_minimax4_ci_high',
            ]
            rows = [
                {'pool_size': 1, 'vs_random_wr': 0.6, 'vs_random_ci_low': 0.5,
                 'vs_random_ci_high': 0.7, 'vs_minimax4_wr': 0.3, 'vs_minimax4_ci_low': 0.2,
                 'vs_minimax4_ci_high': 0.4},
                {'pool_size': 5, 'vs_random_wr': 0.75, 'vs_random_ci_low': 0.65,
                 'vs_random_ci_high': 0.85, 'vs_minimax4_wr': 0.45, 'vs_minimax4_ci_low': 0.35,
                 'vs_minimax4_ci_high': 0.55},
            ]
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            plot_ablation_results(
                csv_path, png_path,
                title="Pool Size Ablation",
                x_label="Pool Size",
                x_key="pool_size",
            )
            assert os.path.exists(png_path), "Pool ablation PNG was not created"

    def test_plot_ablation_results_empty_csv_raises(self):
        from othello_rl.evaluation.plotting import plot_ablation_results

        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, 'empty.csv')
            png_path = os.path.join(d, 'out.png')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['num_sims', 'vs_random_wr'])  # header only

            with pytest.raises(ValueError):
                plot_ablation_results(csv_path, png_path)

    def test_plot_color_asymmetry_creates_png(self):
        from othello_rl.evaluation.plotting import plot_color_asymmetry

        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, 'color.csv')
            png_path = os.path.join(d, 'color.png')

            fieldnames = ['agent', 'color', 'games', 'wins', 'draws', 'losses',
                          'win_rate', 'ci_low', 'ci_high']
            rows = [
                {'agent': 'AZ', 'color': 'black', 'games': 100, 'wins': 60, 'draws': 5,
                 'losses': 35, 'win_rate': 0.625, 'ci_low': 0.52, 'ci_high': 0.72},
                {'agent': 'AZ', 'color': 'white', 'games': 100, 'wins': 55, 'draws': 10,
                 'losses': 35, 'win_rate': 0.60, 'ci_low': 0.50, 'ci_high': 0.70},
            ]
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            plot_color_asymmetry(csv_path, png_path)
            assert os.path.exists(png_path), "Color asymmetry PNG was not created"

    def test_plot_color_asymmetry_multi_agent(self):
        """Two agents produces two bar groups without error."""
        from othello_rl.evaluation.plotting import plot_color_asymmetry

        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, 'color2.csv')
            png_path = os.path.join(d, 'color2.png')

            fieldnames = ['agent', 'color', 'games', 'wins', 'draws', 'losses',
                          'win_rate', 'ci_low', 'ci_high']
            rows = [
                {'agent': 'AZ', 'color': 'black', 'games': 100, 'wins': 60, 'draws': 5,
                 'losses': 35, 'win_rate': 0.625, 'ci_low': 0.52, 'ci_high': 0.72},
                {'agent': 'AZ', 'color': 'white', 'games': 100, 'wins': 55, 'draws': 10,
                 'losses': 35, 'win_rate': 0.60, 'ci_low': 0.50, 'ci_high': 0.70},
                {'agent': 'PPO', 'color': 'black', 'games': 100, 'wins': 45, 'draws': 5,
                 'losses': 50, 'win_rate': 0.475, 'ci_low': 0.38, 'ci_high': 0.57},
                {'agent': 'PPO', 'color': 'white', 'games': 100, 'wins': 40, 'draws': 10,
                 'losses': 50, 'win_rate': 0.45, 'ci_low': 0.36, 'ci_high': 0.54},
            ]
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            plot_color_asymmetry(csv_path, png_path)
            assert os.path.exists(png_path)
