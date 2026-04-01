import numpy as np
import torch
import pytest

from othello_rl.game.othello_game import OthelloGame
from othello_rl.utils.tensor_utils import board_to_tensor, tensor_to_board


@pytest.fixture
def game():
    return OthelloGame(8)


def test_tensor_shape(game):
    """board_to_tensor returns (3, 8, 8) float32 tensor."""
    board = game.getInitBoard()
    canonical = game.getCanonicalForm(board, 1)
    tensor = board_to_tensor(canonical)
    assert tensor.shape == (3, 8, 8)


def test_tensor_channels(game):
    """Channel 0 = current player (1s), channel 1 = opponent (1s), channel 2 = all 1s."""
    board = game.getInitBoard()
    canonical = game.getCanonicalForm(board, 1)
    tensor = board_to_tensor(canonical)

    # Channel 0: current player pieces
    assert tensor[0].sum().item() == (canonical == 1).sum()
    # Channel 1: opponent pieces
    assert tensor[1].sum().item() == (canonical == -1).sum()
    # Channel 2: turn indicator, all ones
    assert (tensor[2] == 1.0).all()


def test_tensor_roundtrip_initial(game):
    """Initial board round-trips correctly for both players."""
    board = game.getInitBoard()

    # Player 1
    canonical_p1 = game.getCanonicalForm(board, 1)
    tensor_p1 = board_to_tensor(canonical_p1)
    recovered_p1 = tensor_to_board(tensor_p1)
    assert np.array_equal(canonical_p1, recovered_p1)

    # Player -1
    canonical_pm1 = game.getCanonicalForm(board, -1)
    tensor_pm1 = board_to_tensor(canonical_pm1)
    recovered_pm1 = tensor_to_board(tensor_pm1)
    assert np.array_equal(canonical_pm1, recovered_pm1)


def test_tensor_roundtrip_midgame(game):
    """Board after 10 random moves round-trips correctly."""
    np.random.seed(99)
    board = game.getInitBoard()
    player = 1
    for _ in range(10):
        valid = game.getValidMoves(board, player)
        legal = np.where(valid == 1)[0]
        if len(legal) == 0:
            break
        action = int(np.random.choice(legal))
        board, player = game.getNextState(board, player, action)
        if game.getGameEnded(board, player) != 0:
            break

    canonical = game.getCanonicalForm(board, player)
    tensor = board_to_tensor(canonical)
    recovered = tensor_to_board(tensor)
    assert np.array_equal(canonical, recovered)


def test_tensor_no_overlap(game):
    """Channels 0 and 1 never both have 1 at the same position."""
    board = game.getInitBoard()
    canonical = game.getCanonicalForm(board, 1)
    tensor = board_to_tensor(canonical)

    overlap = (tensor[0] == 1.0) & (tensor[1] == 1.0)
    assert not overlap.any(), "Channels 0 and 1 overlap"


def test_tensor_dtype(game):
    """Output is float32."""
    board = game.getInitBoard()
    canonical = game.getCanonicalForm(board, 1)
    tensor = board_to_tensor(canonical)
    assert tensor.dtype == torch.float32
