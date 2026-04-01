"""Tests for AlphaZero trainer."""

import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from othello_rl.game.othello_game import OthelloGame
from othello_rl.alphazero.network import build_alphazero_network
from othello_rl.alphazero.trainer import AlphaZeroTrainer


@pytest.fixture
def game():
    return OthelloGame(8)


@pytest.fixture
def network():
    return build_alphazero_network()


@pytest.fixture
def trainer(game, network):
    return AlphaZeroTrainer(
        game=game,
        network=network,
        device="cpu",
        num_sims=25,
        batch_size=16,
        lr=1e-3,
        games_per_iter=2,
        epochs_per_iter=2,
    )


def _make_fake_examples(n=64):
    """Generate synthetic (board, pi, z) examples for testing."""
    examples = []
    for _ in range(n):
        board = np.random.choice([-1, 0, 1], size=(8, 8)).astype(np.float64)
        pi = np.random.dirichlet(np.ones(65))
        z = np.random.choice([-1.0, 1.0])
        examples.append((board, pi, z))
    return examples


class TestTrainStep:
    """Test single training step mechanics."""

    def test_training_reduces_loss(self, trainer):
        """Repeated train_step() calls reduce loss on the same batch."""
        examples = _make_fake_examples(32)
        trainer.buffer.push(examples)
        batch = trainer.buffer.sample(16)

        # Collect losses over 20 steps on the same batch
        losses = [trainer.train_step(batch)["loss"] for _ in range(20)]

        # Average of last 5 steps should be lower than average of first 5 steps
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, (
            f"Loss should decrease: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"
        )

    def test_training_loss_components(self, trainer):
        """train_step returns dict with 'loss', 'policy_loss', 'value_loss' keys."""
        examples = _make_fake_examples(32)
        trainer.buffer.push(examples)
        batch = trainer.buffer.sample(16)

        result = trainer.train_step(batch)

        assert "loss" in result
        assert "policy_loss" in result
        assert "value_loss" in result
        assert isinstance(result["loss"], float)
        assert isinstance(result["policy_loss"], float)
        assert isinstance(result["value_loss"], float)
        # Combined loss should be sum of components
        assert abs(result["loss"] - (result["policy_loss"] + result["value_loss"])) < 1e-5


class TestSelfPlay:
    """Test self-play game generation."""

    def test_self_play_iteration(self, trainer):
        """run_self_play() generates examples and adds them to buffer."""
        assert len(trainer.buffer) == 0
        total_examples = trainer.run_self_play(num_games=1)
        assert total_examples > 0
        assert len(trainer.buffer) > 0


class TestTraining:
    """Test training loop mechanics."""

    def test_train_iteration(self, trainer):
        """run_training() samples from buffer and updates network."""
        examples = _make_fake_examples(64)
        trainer.buffer.push(examples)

        # Store initial weights
        initial_params = [p.clone() for p in trainer.network.parameters()]

        avg_loss = trainer.run_training(num_epochs=1)

        # Verify weights changed
        changed = False
        for p_old, p_new in zip(initial_params, trainer.network.parameters()):
            if not torch.allclose(p_old, p_new.data):
                changed = True
                break
        assert changed, "Network weights should change after training"
        assert isinstance(avg_loss, float)

    def test_training_multiple_batches_per_epoch(self, trainer):
        """run_training() does len(buffer)//batch_size batches per epoch, not just 1."""
        examples = _make_fake_examples(64)
        trainer.buffer.push(examples)

        # With batch_size=16 and 64 examples, should do 64//16 = 4 batches per epoch
        # With epochs_per_iter=1, total batches should be 4
        call_count = [0]
        original_train_step = trainer.train_step

        def counting_train_step(batch):
            call_count[0] += 1
            return original_train_step(batch)

        trainer.train_step = counting_train_step
        trainer.run_training(num_epochs=1)

        expected_batches = len(trainer.buffer) // trainer.batch_size
        assert call_count[0] == expected_batches, (
            f"Expected {expected_batches} batches per epoch, got {call_count[0]}"
        )


class TestGradientClipping:
    """Test that gradient clipping is applied."""

    def test_gradient_clipping(self, trainer):
        """Verify trainer clips gradients at max_norm=1.0."""
        examples = _make_fake_examples(32)
        trainer.buffer.push(examples)
        batch = trainer.buffer.sample(16)

        # Train a step and check gradient norms are bounded
        # We need to check DURING the train step, so we patch clip_grad_norm_
        with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as mock_clip:
            trainer.train_step(batch)
            mock_clip.assert_called_once()
            # Verify max_norm argument is 1.0
            args, kwargs = mock_clip.call_args
            assert args[1] == 1.0 or kwargs.get("max_norm") == 1.0


class TestCheckpointing:
    """Test model save/load."""

    def test_save_load_checkpoint(self, trainer, tmp_path):
        """save_checkpoint() and load_checkpoint() round-trip network weights."""
        # Train for a bit to get non-initial weights
        examples = _make_fake_examples(32)
        trainer.buffer.push(examples)
        batch = trainer.buffer.sample(16)
        trainer.train_step(batch)

        # Save checkpoint
        ckpt_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(str(ckpt_path))
        assert ckpt_path.exists()

        # Store current weights
        saved_params = {k: v.clone() for k, v in trainer.network.state_dict().items()}

        # Modify weights
        with torch.no_grad():
            for p in trainer.network.parameters():
                p.add_(torch.randn_like(p))

        # Verify weights changed
        for k, v in trainer.network.state_dict().items():
            if not torch.allclose(saved_params[k], v):
                break
        else:
            pytest.fail("Weights should have changed after modification")

        # Load checkpoint
        trainer.load_checkpoint(str(ckpt_path))

        # Verify weights restored
        for k, v in trainer.network.state_dict().items():
            assert torch.allclose(saved_params[k], v), (
                f"Parameter {k} not restored correctly"
            )
