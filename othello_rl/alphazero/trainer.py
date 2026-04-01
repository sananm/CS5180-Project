"""AlphaZero trainer implementation.

Orchestrates self-play and neural network updates for the AlphaZero algorithm.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from othello_rl.alphazero.mcts import MCTS
from othello_rl.alphazero.replay_buffer import ReplayBuffer
from othello_rl.alphazero.self_play import execute_episode
from othello_rl.utils.tensor_utils import board_to_tensor

if TYPE_CHECKING:
    from othello_rl.game.othello_game import OthelloGame
    from othello_rl.models.shared_cnn import SharedCNN


class AlphaZeroTrainer:
    """Trainer that implements the AlphaZero self-play training loop."""

    def __init__(
        self,
        game: OthelloGame,
        network: SharedCNN,
        device: str = 'cpu',
        num_sims: int = 100,
        cpuct: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_buffer_size: int = 100_000,
        games_per_iter: int = 20,
        epochs_per_iter: int = 10,
        temp_threshold: int = 15,
        augment_symmetries: bool = True,
    ):
        """Initialize the AlphaZero trainer.

        Args:
            game: OthelloGame instance.
            network: SharedCNN model to train.
            device: Device for training ('cpu' or 'cuda').
            num_sims: Number of MCTS simulations per move.
            cpuct: Exploration constant for PUCT.
            lr: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
            batch_size: Mini-batch size for training.
            max_buffer_size: Maximum capacity of the replay buffer.
            games_per_iter: Number of self-play games per iteration.
            epochs_per_iter: Number of training epochs per iteration.
            temp_threshold: Move number at which MCTS temperature switches to 0.
            augment_symmetries: Whether to use board symmetries for training.
        """
        self.game = game
        self.network = network.to(device)
        self.device = device
        self.num_sims = num_sims
        self.cpuct = cpuct
        self.batch_size = batch_size
        self.games_per_iter = games_per_iter
        self.epochs_per_iter = epochs_per_iter
        self.temp_threshold = temp_threshold
        self.augment_symmetries = augment_symmetries
        
        self.mcts = MCTS(game, self.network, num_sims=num_sims, cpuct=cpuct, device=device)
        self.buffer = ReplayBuffer(max_size=max_buffer_size)
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )

    def run_self_play(self, num_games: int | None = None) -> int:
        """Run self-play games and add examples to the buffer.
        
        Args:
            num_games: Number of games to play. Defaults to self.games_per_iter.
            
        Returns:
            Total number of new examples added (including symmetries).
        """
        n_games = num_games if num_games is not None else self.games_per_iter
        total_examples = 0

        self.network.eval()
        for _ in range(n_games):
            # execute_episode already handles symmetries and returns (board, pi, z)
            # board is canonical (8,8)
            examples = execute_episode(
                self.game,
                self.mcts,
                temp_threshold=self.temp_threshold,
                augment_symmetries=self.augment_symmetries
            )
            self.buffer.push(examples)
            total_examples += len(examples)
            
        return total_examples

    def train_step(self, batch: tuple[list, list, list]) -> dict[str, float]:
        """Perform a single training update on a batch of examples.
        
        Args:
            batch: Tuple of (boards, target_pis, target_zs) lists.
            
        Returns:
            Dictionary with 'loss', 'policy_loss', and 'value_loss'.
        """
        self.network.train()
        
        # Prepare tensors
        boards, target_pis, target_zs = batch
        
        # boards are (8, 8) canonical -> (B, 3, 8, 8)
        board_tensors = torch.stack([
            board_to_tensor(b) for b in boards
        ]).to(self.device)
        
        target_pis = torch.FloatTensor(np.array(target_pis)).to(self.device)
        target_zs = torch.FloatTensor(np.array(target_zs)).to(self.device)
        
        # Forward pass
        out_pi_logits, out_v = self.network(board_tensors)
        
        # Loss computation
        # Policy loss: cross-entropy -sum(pi * log(p))
        # Since pi is a distribution, we use log_softmax and KL-divergence or manually
        log_pi = F.log_softmax(out_pi_logits, dim=1)
        policy_loss = -torch.mean(torch.sum(target_pis * log_pi, dim=1))
        
        # Value loss: mean squared error (v - z)^2
        value_loss = F.mse_loss(out_v.view(-1), target_zs)
        
        loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item()
        }

    def run_training(self, num_epochs: int | None = None) -> float:
        """Run training epochs on the current replay buffer.
        
        Args:
            num_epochs: Number of epochs to train. Defaults to self.epochs_per_iter.
            
        Returns:
            Average total loss across all epochs and batches.
        """
        n_epochs = num_epochs if num_epochs is not None else self.epochs_per_iter
        if len(self.buffer) < self.batch_size:
            import warnings
            warnings.warn(
                f"Buffer size {len(self.buffer)} < batch_size {self.batch_size}; skipping training.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0
            
        total_loss = 0.0
        num_batches = len(self.buffer) // self.batch_size
        
        for _ in range(n_epochs):
            # Shuffle is implicitly handled by random sampling from buffer
            for _ in range(num_batches):
                batch = self.buffer.sample(self.batch_size)
                metrics = self.train_step(batch)
                total_loss += metrics["loss"]
                
        return total_loss / (n_epochs * num_batches)

    def save_checkpoint(self, path: str) -> None:
        """Save network and optimizer state to a file.
        
        Args:
            path: Path to the checkpoint file.
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load network and optimizer state from a file.
        
        Args:
            path: Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
