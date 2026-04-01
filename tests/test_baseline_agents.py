import numpy as np
import pytest

from othello_rl.agents.minimax_agent import MinimaxAgent
from othello_rl.agents.random_agent import RandomAgent
from othello_rl.evaluation.arena import Arena
from othello_rl.game.othello_env import OthelloEnv
from othello_rl.game.othello_game import OthelloGame


CORNER_BOARD = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, -1, 0, 0, 0, -1, 1, -1],
        [0, 0, -1, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, -1, 0],
        [0, 1, 1, 1, -1, -1, -1, 1],
        [0, 1, 0, -1, -1, -1, -1, 0],
        [0, 1, -1, 1, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, -1],
    ],
    dtype=np.int8,
)

PASS_BOARD = np.array(
    [
        [-1, 1, 1, 1, 1, 1, 1, 0],
        [-1, -1, -1, -1, -1, -1, -1, 1],
        [-1, -1, -1, -1, 1, -1, 1, 1],
        [-1, 1, -1, -1, -1, 1, -1, 1],
        [-1, 1, 1, -1, 1, 1, -1, 1],
        [-1, 1, -1, 1, -1, 1, -1, -1],
        [-1, -1, 1, 1, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ],
    dtype=np.int8,
)

SHORT_MATCH_BOARD = np.array(
    [
        [1, 0, 1, 1, -1, 1, 1, 1],
        [-1, -1, 1, 1, 1, -1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1],
        [0, -1, 1, 1, 1, -1, -1, -1],
        [-1, -1, -1, 1, 1, 1, -1, 1],
        [0, -1, -1, 1, -1, 1, 1, 1],
        [-1, -1, -1, -1, -1, 1, 1, 1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ],
    dtype=np.int8,
)


class FixedStateEnv(OthelloEnv):
    def __init__(self, board: np.ndarray, player: int):
        super().__init__()
        self._fixed_board = np.array(board, copy=True)
        self._fixed_player = int(player)

    def reset(self):
        self.board = np.array(self._fixed_board, copy=True)
        self.player = self._fixed_player
        return self.get_observation(), self.get_valid_actions()


@pytest.fixture
def game():
    return OthelloGame(8)


def test_random_agent_only_returns_legal_actions():
    env = OthelloEnv()
    _, valid_moves = env.reset()
    canonical = env.game.getCanonicalForm(env.board, env.player)

    agent = RandomAgent(np.random.default_rng(7))
    action = agent.get_action(canonical, valid_moves)

    assert valid_moves[action] == 1


def test_random_agent_seed_reproducible():
    env = OthelloEnv()
    _, valid_moves = env.reset()
    canonical = env.game.getCanonicalForm(env.board, env.player)

    agent_a = RandomAgent(np.random.default_rng(11))
    agent_b = RandomAgent(np.random.default_rng(11))

    actions_a = [agent_a.get_action(canonical, valid_moves) for _ in range(12)]
    actions_b = [agent_b.get_action(canonical, valid_moves) for _ in range(12)]

    assert actions_a == actions_b


@pytest.mark.parametrize("depth", [4, 6])
def test_minimax_returns_legal_actions_for_depth_4_and_6(depth):
    env = OthelloEnv()
    _, valid_moves = env.reset()
    canonical = env.game.getCanonicalForm(env.board, env.player)

    action = MinimaxAgent(depth=depth).get_action(canonical, valid_moves)

    assert valid_moves[action] == 1


def test_minimax_prefers_corner_when_available(game):
    valid_moves = game.getValidMoves(CORNER_BOARD, 1)
    assert valid_moves[0] == 1
    assert np.count_nonzero(valid_moves[:-1]) > 1

    action = MinimaxAgent(depth=4).get_action(CORNER_BOARD, valid_moves)

    assert action == 0


def test_minimax_handles_pass_action(game):
    valid_moves = game.getValidMoves(PASS_BOARD, 1)
    assert valid_moves[64] == 1
    assert valid_moves.sum() == 1
    assert game.getGameEnded(PASS_BOARD, 1) == 0

    action = MinimaxAgent(depth=4).get_action(PASS_BOARD, valid_moves)

    assert action == 64


def test_depth4_beats_random_in_short_match():
    arena = Arena(env_factory=lambda: FixedStateEnv(SHORT_MATCH_BOARD, 1))
    result = arena.play_match(
        MinimaxAgent(depth=4),
        RandomAgent(np.random.default_rng(123)),
        num_games=2,
        agent_a_label="minimax4",
        agent_b_label="random",
    )

    assert result.games_played == 2
    assert result.wins_a > result.wins_b
