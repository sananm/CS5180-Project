import numpy as np
import torch
import pytest

from othello_rl.game.othello_env import OthelloEnv
from othello_rl.utils.seed import set_seed


@pytest.fixture
def env():
    return OthelloEnv()


def _play_random_game(env, seed=None):
    """Helper: play a full game with random legal moves. Returns (moves, reward, done)."""
    if seed is not None:
        np.random.seed(seed)
    obs, valid = env.reset()
    moves = 0
    max_moves = 128
    while moves < max_moves:
        legal = np.where(valid == 1)[0]
        action = int(np.random.choice(legal))
        obs, reward, done, info = env.step(action)
        moves += 1
        if done:
            return moves, reward, done
        valid = env.get_valid_actions()
    return moves, 0.0, False


def test_reset_shapes(env):
    """reset() returns obs (3,8,8) and valid (65,)."""
    obs, valid = env.reset()
    assert obs.shape == (3, 8, 8)
    assert obs.dtype == torch.float32
    assert valid.shape == (65,)


def test_step_shapes(env):
    """step() returns correct types and shapes."""
    obs, valid = env.reset()
    legal = np.where(valid == 1)[0]
    action = int(legal[0])
    obs2, reward, done, info = env.step(action)

    if not done:
        assert obs2.shape == (3, 8, 8)
        assert obs2.dtype == torch.float32
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "result" in info


def test_play_complete_game(env):
    """Play a full game with random moves, verify termination and valid result."""
    moves, reward, done = _play_random_game(env, seed=42)
    assert done, f"Game did not terminate within 128 moves"
    assert reward in (1.0, -1.0) or (isinstance(reward, float) and abs(reward) < 1 and abs(reward) > 0), \
        f"Unexpected reward: {reward}"


def test_reward_sign(env):
    """When a game ends, verify reward sign convention."""
    # Play multiple games and check that reward is nonzero at termination
    np.random.seed(77)
    for _ in range(10):
        obs, valid = env.reset()
        done = False
        while not done:
            legal = np.where(valid == 1)[0]
            action = int(np.random.choice(legal))
            obs, reward, done, info = env.step(action)
            if not done:
                valid = env.get_valid_actions()
        # At game end, reward should be nonzero
        assert reward != 0.0, "Reward is 0 at game end (no draws expected in most games)"


def test_multiple_games(env):
    """Play 10 random games, all terminate within 128 moves."""
    for i in range(10):
        moves, reward, done = _play_random_game(env, seed=i * 7)
        assert done, f"Game {i} did not terminate within 128 moves"
        assert moves <= 128


def test_deterministic_with_seed(env):
    """Two games played with same seed produce identical move sequences."""
    def play_and_record(seed):
        set_seed(seed, deterministic=False)
        np.random.seed(seed)
        obs, valid = env.reset()
        actions = []
        done = False
        while not done:
            legal = np.where(valid == 1)[0]
            action = int(np.random.choice(legal))
            actions.append(action)
            obs, reward, done, info = env.step(action)
            if not done:
                valid = env.get_valid_actions()
        return actions, reward

    actions1, reward1 = play_and_record(42)
    actions2, reward2 = play_and_record(42)

    assert actions1 == actions2, "Move sequences differ with same seed"
    assert reward1 == reward2, "Rewards differ with same seed"
