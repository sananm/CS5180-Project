import numpy as np
import pytest

from othello_rl.game.othello_game import OthelloGame


@pytest.fixture
def game():
    return OthelloGame(8)


def test_initial_board(game):
    """getInitBoard returns (8,8) array with 4 pieces placed correctly."""
    board = game.getInitBoard()
    assert board.shape == (8, 8)
    # Center 4 squares: (3,3)=-1, (3,4)=1, (4,3)=1, (4,4)=-1
    assert board[3][3] == -1
    assert board[3][4] == 1
    assert board[4][3] == 1
    assert board[4][4] == -1
    # Total pieces = 4
    assert np.count_nonzero(board) == 4


def test_initial_valid_moves(game):
    """Player 1 has exactly 4 legal moves from starting position."""
    board = game.getInitBoard()
    valid = game.getValidMoves(board, 1)
    assert valid.shape == (65,)
    # Standard Othello opening: player 1 (white) has 4 legal moves
    assert valid.sum() == 4
    # Pass should not be valid when there are legal board moves
    assert valid[64] == 0


def test_action_size(game):
    """getActionSize() == 65."""
    assert game.getActionSize() == 65


def test_get_next_state(game):
    """Executing a valid move changes the board and flips player."""
    board = game.getInitBoard()
    valid = game.getValidMoves(board, 1)
    legal_actions = np.where(valid == 1)[0]
    action = int(legal_actions[0])

    new_board, new_player = game.getNextState(board, 1, action)
    assert new_player == -1  # player flipped
    assert not np.array_equal(new_board, board)  # board changed


def test_canonical_form(game):
    """getCanonicalForm(board, 1) == board; getCanonicalForm(board, -1) == -board."""
    board = game.getInitBoard()
    canonical_p1 = game.getCanonicalForm(board, 1)
    assert np.array_equal(canonical_p1, board)

    canonical_pm1 = game.getCanonicalForm(board, -1)
    assert np.array_equal(canonical_pm1, -board)


def test_game_ended_not_terminal(game):
    """getGameEnded returns 0 for non-terminal positions."""
    board = game.getInitBoard()
    assert game.getGameEnded(board, 1) == 0
    assert game.getGameEnded(board, -1) == 0


def test_pass_action(game):
    """When a player has no legal moves, only index 64 (pass) is valid."""
    # We construct a board state where one player must pass.
    # Easiest approach: play random games until we find a pass situation,
    # or construct one manually. Let's construct one.
    # A board where player 1 has no moves but player -1 does:
    # Fill board so player 1 is surrounded with no flipping opportunities.
    # Simpler: just play random games and check if pass ever occurs.
    np.random.seed(123)
    found_pass = False
    for _ in range(200):
        board = game.getInitBoard()
        player = 1
        for _ in range(120):
            valid = game.getValidMoves(board, player)
            if valid[64] == 1 and valid.sum() == 1:
                # This player must pass
                found_pass = True
                break
            legal = np.where(valid == 1)[0]
            action = int(np.random.choice(legal))
            board, player = game.getNextState(board, player, action)
            if game.getGameEnded(board, player) != 0:
                break
        if found_pass:
            break
    assert found_pass, "Could not find a pass situation in 200 random games"


def test_play_full_game(game):
    """Play a complete random game, verify it terminates and returns valid result."""
    np.random.seed(42)
    board = game.getInitBoard()
    player = 1
    moves = 0
    max_moves = 128

    while moves < max_moves:
        result = game.getGameEnded(board, player)
        if result != 0:
            break
        valid = game.getValidMoves(board, player)
        legal = np.where(valid == 1)[0]
        assert len(legal) > 0, "No legal moves but game not ended"
        action = int(np.random.choice(legal))
        board, player = game.getNextState(board, player, action)
        moves += 1

    result = game.getGameEnded(board, player)
    assert result != 0, f"Game did not terminate within {max_moves} moves"
    # Result should be +1, -1, or a small float for draw
    assert result == 1 or result == -1 or (isinstance(result, float) and abs(result) < 1)
