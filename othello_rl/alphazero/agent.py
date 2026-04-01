"""AlphaZero agent implementation.

Wraps MCTS with the BaseAgent interface for evaluation and tournament play.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from othello_rl.agents.base_agent import BaseAgent
from othello_rl.alphazero.mcts import MCTS

if TYPE_CHECKING:
    from othello_rl.game.othello_game import OthelloGame
    from othello_rl.models.shared_cnn import SharedCNN


class AlphaZeroAgent(BaseAgent):
    """Agent that uses Monte Carlo Tree Search for action selection.
    
    Wraps the MCTS class to provide a standard agent interface.
    """

    def __init__(
        self,
        game: OthelloGame,
        network: SharedCNN,
        num_sims: int = 100,
        cpuct: float = 1.0,
        device: str = 'cpu',
    ):
        """Initialize the AlphaZero agent.

        Args:
            game: OthelloGame instance.
            network: SharedCNN model for policy/value guidance.
            num_sims: Number of MCTS simulations per move.
            cpuct: Exploration constant for PUCT.
            device: Device for neural network inference.
        """
        self.mcts = MCTS(game, network, num_sims=num_sims, cpuct=cpuct, device=device)

    def get_action(self, board: np.ndarray, valid_moves: np.ndarray) -> int:
        """Select an action using MCTS.
        
        Args:
            board: (8, 8) canonical board array.
            valid_moves: (65,) binary array of legal moves.
            
        Returns:
            Selected action index {0..64}.
        """
        # Ensure board is (8, 8) if it came in as (3, 8, 8)
        if board.ndim == 3:
            # First channel of (3, 8, 8) is current player pieces
            # Second channel is opponent pieces
            # We reconstruct (8, 8) with 1, -1, 0
            board = board[0] - board[1]

        # MCTS get_action_prob uses temp=0 for greedy selection (default for evaluation)
        probs = self.mcts.get_action_prob(board, temp=0)
        
        # Select action with highest probability
        action = np.argmax(probs)
        return int(action)

    def reset(self):
        """Clear MCTS tree for a new game."""
        self.mcts.reset()

    @property
    def name(self) -> str:
        return f"AlphaZero(sims={self.mcts.num_sims})"
