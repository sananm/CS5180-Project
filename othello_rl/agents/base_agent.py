from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Common interface for all Othello agents."""

    @abstractmethod
    def get_action(self, board: np.ndarray, valid_moves: np.ndarray) -> int:
        """
        Select an action given the current canonical board state.

        Args:
            board: (8, 8) canonical board array (current player = +1)
                   OR (3, 8, 8) tensor -- agent decides which format it needs
            valid_moves: (65,) binary array, 1 = legal action

        Returns:
            action: integer in {0..64}, where 64 = pass
        """
        pass

    def reset(self):
        """Called at the start of each game. Override if agent has state."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
