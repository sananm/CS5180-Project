import numpy as np

from othello_rl.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Uniformly samples from the legal action mask."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    @property
    def name(self) -> str:
        return "random"

    def get_action(self, board: np.ndarray, valid_moves: np.ndarray) -> int:
        legal = np.flatnonzero(valid_moves)
        if legal.size == 0:
            raise ValueError("RandomAgent received no legal moves")
        return int(self.rng.choice(legal))
