"""Tests for compute timing utilities."""
import time
import pytest
from othello_rl.evaluation.compute_timing import timed_call, TimedResult


class TestTimedCall:
    """Tests for the timed_call utility."""

    def test_returns_result(self):
        """Function return value is captured in result field."""
        def fn():
            return 42
        result = timed_call(fn, device="cpu")
        assert result.result == 42

    def test_measures_time(self):
        """Elapsed time is approximately correct."""
        def fn():
            time.sleep(0.1)
            return "done"
        result = timed_call(fn, device="cpu")
        assert 0.08 < result.wall_clock_seconds < 0.2  # Allow some variance

    def test_records_device(self):
        """Device is recorded in result."""
        result = timed_call(lambda: None, device="cpu")
        assert result.device == "cpu"

    def test_timed_result_dataclass(self):
        """TimedResult is a proper dataclass."""
        tr = TimedResult(result=123, wall_clock_seconds=1.5, device="cuda")
        assert tr.result == 123
        assert tr.wall_clock_seconds == 1.5
        assert tr.device == "cuda"


class TestTimedIterations:
    """Tests for algorithm-specific timing wrappers."""

    def test_timed_ppo_iteration_returns_dict(self):
        """PPO timing returns metrics dict with expected keys."""
        from othello_rl.evaluation.compute_timing import timed_ppo_iteration
        from othello_rl.ppo.trainer import PPOTrainer
        from othello_rl.models.shared_cnn import SharedCNN
        from othello_rl.game.othello_env import OthelloEnv

        env = OthelloEnv()
        network = SharedCNN()
        trainer = PPOTrainer(env, network, episodes_per_update=1)
        
        metrics, elapsed = timed_ppo_iteration(trainer, device="cpu")
        
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy_loss" in metrics
        assert elapsed > 0

    def test_timed_az_iteration_returns_dict(self):
        """AlphaZero timing returns metrics dict with expected keys."""
        from othello_rl.evaluation.compute_timing import timed_az_iteration
        from othello_rl.alphazero.trainer import AlphaZeroTrainer
        from othello_rl.models.shared_cnn import SharedCNN
        from othello_rl.game.othello_game import OthelloGame

        game = OthelloGame(8)
        network = SharedCNN()
        trainer = AlphaZeroTrainer(
            game, network, 
            num_sims=10,  # Fast for testing
            games_per_iter=1,
            epochs_per_iter=1,
            batch_size=8,
        )
        
        metrics, elapsed = timed_az_iteration(trainer, device="cpu")
        
        assert "num_examples" in metrics
        assert "avg_loss" in metrics
        assert elapsed > 0
