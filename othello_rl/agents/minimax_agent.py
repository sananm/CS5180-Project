from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from othello_rl.agents.base_agent import BaseAgent
from othello_rl.game.othello_game import OthelloGame
from othello_rl.utils.tensor_utils import tensor_to_board


@dataclass(frozen=True)
class _ScoredAction:
    action: int
    score: float


class MinimaxAgent(BaseAgent):
    """Classical alpha-beta minimax over canonical Othello boards."""

    POSITIONAL_WEIGHTS = np.array(
        [
            [120, -20, 20, 5, 5, 20, -20, 120],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [20, -5, 15, 3, 3, 15, -5, 20],
            [5, -5, 3, 3, 3, 3, -5, 5],
            [5, -5, 3, 3, 3, 3, -5, 5],
            [20, -5, 15, 3, 3, 15, -5, 20],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [120, -20, 20, 5, 5, 20, -20, 120],
        ],
        dtype=np.int32,
    )
    _CORNERS = {0, 7, 56, 63}
    _EDGES = {
        *(range(1, 7)),
        *(range(57, 63)),
        *(i * 8 for i in range(1, 7)),
        *(i * 8 + 7 for i in range(1, 7)),
    }

    def __init__(
        self,
        depth: int,
        game: OthelloGame | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.depth = depth
        self.game = game or OthelloGame(8)
        self.rng = rng

    @property
    def name(self) -> str:
        return f"minimax{self.depth}"

    def get_action(self, board: np.ndarray | torch.Tensor, valid_moves: np.ndarray) -> int:
        canonical_board = self._coerce_board(board)
        legal_actions = np.flatnonzero(valid_moves)
        if legal_actions.size == 0:
            raise ValueError("MinimaxAgent received no legal moves")
        if legal_actions.size == 1:
            return int(legal_actions[0])

        best_score = -float("inf")
        best_actions: list[int] = []

        for action in self._ordered_actions(canonical_board, valid_moves):
            next_board, next_player = self.game.getNextState(canonical_board, 1, action)
            next_canonical = self.game.getCanonicalForm(next_board, next_player)
            score = -self._search(next_canonical, self.depth - 1, -float("inf"), float("inf"))
            if score > best_score:
                best_score = score
                best_actions = [int(action)]
            elif score == best_score:
                best_actions.append(int(action))

        if self.rng is not None and len(best_actions) > 1:
            return int(self.rng.choice(best_actions))
        return min(best_actions)

    def _coerce_board(self, board: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(board, torch.Tensor):
            if board.ndim == 3:
                return tensor_to_board(board.detach().cpu())
            return board.detach().cpu().numpy()
        arr = np.asarray(board)
        if arr.ndim == 3:
            return tensor_to_board(torch.from_numpy(arr))
        return arr

    def _search(self, board: np.ndarray, depth: int, alpha: float, beta: float) -> float:
        terminal = self.game.getGameEnded(board, 1)
        if terminal != 0:
            return float(10_000 * terminal)
        if depth == 0:
            return float(self._evaluate(board))

        valid_moves = self.game.getValidMoves(board, 1)
        best = -float("inf")
        for action in self._ordered_actions(board, valid_moves):
            next_board, next_player = self.game.getNextState(board, 1, action)
            next_canonical = self.game.getCanonicalForm(next_board, next_player)
            score = -self._search(next_canonical, depth - 1, -beta, -alpha)
            best = max(best, score)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    def _ordered_actions(self, board: np.ndarray, valid_moves: np.ndarray) -> list[int]:
        legal_actions = [int(a) for a in np.flatnonzero(valid_moves)]
        if legal_actions == [self.game.getActionSize() - 1]:
            return legal_actions

        scored_actions: list[_ScoredAction] = []
        for action in legal_actions:
            category = self._action_category(action)
            next_board, next_player = self.game.getNextState(board, 1, action)
            next_canonical = self.game.getCanonicalForm(next_board, next_player)
            heuristic = -self._evaluate(next_canonical)
            score = -(category * 100_000) - heuristic
            scored_actions.append(_ScoredAction(action=action, score=score))

        scored_actions.sort(key=lambda item: (item.score, item.action))
        return [item.action for item in scored_actions]

    def _action_category(self, action: int) -> int:
        if action in self._CORNERS:
            return 2
        if action in self._EDGES:
            return 1
        return 0

    def _evaluate(self, board: np.ndarray) -> float:
        positional = int(self.POSITIONAL_WEIGHTS[board == 1].sum() - self.POSITIONAL_WEIGHTS[board == -1].sum())
        legal_self = int(np.count_nonzero(self.game.getValidMoves(board, 1)[:-1]))
        opp_board = self.game.getCanonicalForm(board, -1)
        legal_opp = int(np.count_nonzero(self.game.getValidMoves(opp_board, 1)[:-1]))
        mobility = legal_self - legal_opp

        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        corners_self = sum(1 for row, col in corners if board[row, col] == 1)
        corners_opp = sum(1 for row, col in corners if board[row, col] == -1)
        corner_score = corners_self - corners_opp

        return float(positional + 2 * mobility + 25 * corner_score)
