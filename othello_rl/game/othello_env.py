import numpy as np
import torch

from othello_rl.game.othello_game import OthelloGame
from othello_rl.utils.tensor_utils import board_to_tensor


class OthelloEnv:
    """Wrapper providing (3,8,8) tensor interface over alpha-zero-general OthelloGame."""

    def __init__(self, n=8):
        self.game = OthelloGame(n)
        self.n = n
        self.action_size = n * n + 1  # 65: 64 board positions + pass

        self.board = None
        self.player = None

    def reset(self):
        """Initialize board and return (obs, valid_actions).

        Returns:
            obs: (3, 8, 8) float32 tensor in canonical form
            valid_actions: (65,) int array, 1 for legal moves, 0 for illegal
        """
        self.board = self.game.getInitBoard()
        self.player = 1
        return self.get_observation(), self.get_valid_actions()

    def get_observation(self) -> torch.Tensor:
        """Convert current board to (3, 8, 8) tensor from current player's perspective."""
        canonical = self.game.getCanonicalForm(self.board, self.player)
        return board_to_tensor(canonical)

    def get_valid_actions(self) -> np.ndarray:
        """Returns binary mask of shape (65,) -- 1 for legal, 0 for illegal."""
        return self.game.getValidMoves(self.board, self.player)

    def step(self, action):
        """Execute action, return (obs, reward, done, info).

        Args:
            action: integer in {0..64}, where 64 = pass

        Returns:
            obs: (3, 8, 8) tensor or None if done
            reward: float -- reward for the player who just acted
            done: bool
            info: dict with 'result' key (raw getGameEnded value from new player's perspective)
        """
        self.board, self.player = self.game.getNextState(self.board, self.player, action)
        done_val = self.game.getGameEnded(self.board, self.player)
        done = done_val != 0

        # CRITICAL: done_val is from the NEW player's perspective (getNextState flipped player).
        # Reward for the acting player = -done_val.
        reward = float(-done_val) if done else 0.0

        obs = self.get_observation() if not done else None
        return obs, reward, done, {"result": done_val}
