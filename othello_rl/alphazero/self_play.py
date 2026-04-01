"""Self-play episode generation for AlphaZero training.

Plays complete games using MCTS, producing (board, pi, z) training examples
with correct outcome assignment and optional symmetry augmentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from othello_rl.config import AZ_TEMP_THRESHOLD

if TYPE_CHECKING:
    from othello_rl.alphazero.mcts import MCTS
    from othello_rl.game.othello_game import OthelloGame


def execute_episode(
    game: OthelloGame,
    mcts: MCTS,
    temp_threshold: int = AZ_TEMP_THRESHOLD,
    augment_symmetries: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play one complete self-play game, returning training examples.

    Args:
        game: OthelloGame instance for state transitions.
        mcts: MCTS instance (with neural network) for action selection.
        temp_threshold: Move number at which temperature switches from 1 to 0.
        augment_symmetries: Whether to augment examples with board symmetries.

    Returns:
        List of (canonical_board, pi, z) tuples where:
        - canonical_board: (8, 8) numpy array from the current player's view
        - pi: (65,) policy vector from MCTS visit counts
        - z: +1 if the player to move at this position won, -1 if lost
    """
    examples = []  # (canonical_board, player, pi)
    board = game.getInitBoard()
    player = 1
    step = 0

    mcts.reset()

    while True:
        step += 1
        canonical = game.getCanonicalForm(board, player)
        temp = 1 if step < temp_threshold else 0

        pi = mcts.get_action_prob(canonical, temp=temp)

        # Store board, player, and policy for later outcome assignment
        examples.append((canonical.copy(), player, pi))

        action = np.random.choice(len(pi), p=pi)
        board, player = game.getNextState(board, player, action)

        result = game.getGameEnded(board, player)
        if result != 0:
            # Game over. result is from the perspective of `player` (the player
            # whose turn it now is, i.e. the player who CANNOT move).
            # result == 1 means `player` wins (has more pieces)
            # result == -1 means `player` loses (has fewer pieces)
            #
            # For each stored example, z should be:
            #   +1 if that example's player won
            #   -1 if that example's player lost
            #
            # If result > 0, `player` won. If the example's stored player
            # matches `player`, z = +1; otherwise z = -1.
            # If result < 0, `player` lost (-player won). If the example's
            # stored player matches `player`, z = -1; otherwise z = +1.
            final_examples = []
            for brd, pl, policy in examples:
                if pl == player:
                    z = result
                else:
                    z = -result
                final_examples.append((brd, policy, z))

            if augment_symmetries:
                augmented = []
                for brd, policy, z in final_examples:
                    syms = game.getSymmetries(brd, policy)
                    for sym_board, sym_pi in syms:
                        augmented.append((sym_board, np.array(sym_pi), z))
                return augmented

            return final_examples
