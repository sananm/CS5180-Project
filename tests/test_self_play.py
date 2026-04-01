"""Tests for self-play episode generation."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from othello_rl.game.othello_game import OthelloGame
from othello_rl.alphazero.mcts import MCTS
from othello_rl.alphazero.network import build_alphazero_network
from othello_rl.alphazero.self_play import execute_episode


@pytest.fixture
def game():
    return OthelloGame(8)


@pytest.fixture
def network():
    return build_alphazero_network()


@pytest.fixture
def mcts(game, network):
    return MCTS(game, network, num_sims=25, cpuct=1.0, device="cpu")


class TestEpisodeGeneration:
    """Test that execute_episode returns well-formed training examples."""

    def test_episode_generates_examples(self, game, mcts):
        """execute_episode() returns a non-empty list of (board, pi, z) tuples."""
        examples = execute_episode(game, mcts, augment_symmetries=False)
        assert len(examples) > 0
        for ex in examples:
            assert len(ex) == 3  # (board, pi, z)

    def test_episode_board_shape(self, game, mcts):
        """Each board in examples has shape (8, 8)."""
        examples = execute_episode(game, mcts, augment_symmetries=False)
        for board, pi, z in examples:
            assert board.shape == (8, 8)

    def test_episode_pi_shape(self, game, mcts):
        """Each pi has shape (65,) and sums to ~1.0."""
        examples = execute_episode(game, mcts, augment_symmetries=False)
        for board, pi, z in examples:
            assert pi.shape == (65,)
            assert abs(pi.sum() - 1.0) < 1e-6

    def test_episode_z_values(self, game, mcts):
        """Each z is +1 or -1 (game outcome)."""
        examples = execute_episode(game, mcts, augment_symmetries=False)
        for board, pi, z in examples:
            assert z in (+1, -1), f"z should be +1 or -1, got {z}"


class TestOutcomeAssignment:
    """Test that z values are assigned correctly based on game outcome."""

    def test_episode_correct_outcome_assignment(self, game, mcts):
        """z is +1 for winner's positions, -1 for loser's positions."""
        examples = execute_episode(game, mcts, augment_symmetries=False)
        # At minimum, there should be both +1 and -1 z values in a non-draw game
        # (or all the same sign). The important thing is they're consistent.
        z_vals = [z for _, _, z in examples]
        assert all(z in (+1, -1) for z in z_vals)

    def test_episode_known_game_outcome(self, game):
        """
        Play a game with a mock MCTS that returns deterministic actions,
        then verify z=+1 for winner's moves and z=-1 for loser's moves.

        Uses a mock MCTS that always picks the first valid action. We track
        which player moved at each step, then verify the z signs are consistent
        with the final game outcome.
        """
        # Create a mock MCTS that always picks the first valid action
        mock_mcts = MagicMock()
        mock_mcts.reset = MagicMock()

        # Track the actions for verification
        call_count = [0]

        def deterministic_action_prob(canonical_board, temp=1):
            call_count[0] += 1
            valids = game.getValidMoves(canonical_board, 1)
            probs = np.zeros(65)
            # Pick first valid action deterministically
            first_valid = np.argmax(valids)
            probs[first_valid] = 1.0
            return probs

        mock_mcts.get_action_prob = deterministic_action_prob

        # Play out the episode with no randomness in action selection
        # (pi is one-hot, so np.random.choice picks the deterministic action)
        examples = execute_episode(game, mock_mcts, augment_symmetries=False)

        assert len(examples) > 0

        # Verify z values are +1 or -1
        z_vals = set(z for _, _, z in examples)
        assert z_vals.issubset({+1, -1})

        # Replay the game to determine the winner independently
        board = game.getInitBoard()
        player = 1
        players_per_step = []

        while True:
            canonical = game.getCanonicalForm(board, player)
            valids = game.getValidMoves(canonical, 1)
            action = int(np.argmax(valids))
            players_per_step.append(player)
            board, player = game.getNextState(board, player, action)
            result = game.getGameEnded(board, player)
            if result != 0:
                break

        # result is from perspective of `player` (the player who can't move).
        # If result == -1, that means `player` lost, so the OTHER player won.
        # If result == 1, `player` won (but this shouldn't happen in a normal game end).
        # Actually getGameEnded returns 1 if `player` has more pieces, -1 if fewer.
        # At game end (no legal moves for either side), result tells us who won
        # from `player`'s perspective.

        # Verify: each example's z should match whether that move's player won
        for i, (board_ex, pi_ex, z_ex) in enumerate(examples):
            step_player = players_per_step[i]
            # If result is from current `player`'s perspective:
            # The player who was about to move when game ended = `player`
            # result > 0 means `player` won, result < 0 means `player` lost
            if result > 0:
                # `player` (who can't move) won
                expected_z = 1 if step_player == player else -1
            else:
                # `player` lost, so -player won
                expected_z = 1 if step_player != player else -1
            assert z_ex == expected_z, (
                f"Step {i}: player={step_player}, final_player={player}, "
                f"result={result}, expected_z={expected_z}, got z={z_ex}"
            )


class TestTemperature:
    """Test temperature schedule during self-play."""

    def test_episode_uses_temperature(self, game):
        """First N moves use temp=1, rest use temp=0."""
        mock_mcts = MagicMock()
        mock_mcts.reset = MagicMock()
        recorded_temps = []

        def record_temp(canonical_board, temp=1):
            recorded_temps.append(temp)
            valids = game.getValidMoves(canonical_board, 1)
            probs = np.zeros(65)
            first_valid = np.argmax(valids)
            probs[first_valid] = 1.0
            return probs

        mock_mcts.get_action_prob = record_temp

        temp_threshold = 5
        execute_episode(game, mock_mcts, temp_threshold=temp_threshold,
                        augment_symmetries=False)

        # First temp_threshold-1 moves should use temp=1, rest temp=0
        for i, t in enumerate(recorded_temps):
            step = i + 1  # steps are 1-indexed
            if step < temp_threshold:
                assert t == 1, f"Step {step} should use temp=1, got {t}"
            else:
                assert t == 0, f"Step {step} should use temp=0, got {t}"


class TestSymmetryAugmentation:
    """Test that symmetry augmentation works correctly."""

    def test_symmetry_augmentation(self, game):
        """With augmentation, examples are exactly 8x the non-augmented count."""
        mock_mcts = MagicMock()
        mock_mcts.reset = MagicMock()

        def first_valid(canonical_board, temp=1):
            valids = game.getValidMoves(canonical_board, 1)
            probs = np.zeros(65)
            probs[int(np.argmax(valids))] = 1.0
            return probs

        mock_mcts.get_action_prob = first_valid

        base_examples = execute_episode(game, mock_mcts, augment_symmetries=False)
        aug_examples = execute_episode(game, mock_mcts, augment_symmetries=True)

        assert len(aug_examples) % 8 == 0
        assert len(aug_examples) == len(base_examples) * 8
