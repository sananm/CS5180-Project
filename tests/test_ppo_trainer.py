"""Tests for PPOTrainer, OpponentPool, and the training smoke test."""

import pytest

from othello_rl.ppo.trainer import OpponentPool, PPOTrainer
from othello_rl.ppo.network import build_ppo_network
from othello_rl.ppo.rollout import collect_episode
from othello_rl.game.othello_env import OthelloEnv
from othello_rl.config.default import (
    PPO_CLIP_EPS,
    PPO_GAMMA,
    PPO_LAM,
    PPO_EPOCHS,
    PPO_MINIBATCH_SIZE,
    PPO_LEARNING_RATE,
    PPO_VALUE_COEF,
    PPO_ENTROPY_COEF,
    PPO_MAX_GRAD_NORM,
    PPO_CHECKPOINT_EVERY,
)


@pytest.fixture
def env():
    return OthelloEnv()


@pytest.fixture
def network():
    return build_ppo_network()


@pytest.fixture
def trainer(env, network):
    return PPOTrainer(
        env=env,
        network=network,
        device="cpu",
        gamma=PPO_GAMMA,
        lam=PPO_LAM,
        clip_eps=PPO_CLIP_EPS,
        value_coef=PPO_VALUE_COEF,
        entropy_coef=PPO_ENTROPY_COEF,
        ppo_epochs=PPO_EPOCHS,
        minibatch_size=PPO_MINIBATCH_SIZE,
        lr=PPO_LEARNING_RATE,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        episodes_per_update=1,
        checkpoint_every=PPO_CHECKPOINT_EVERY,
    )


# ---------------------------------------------------------------------------
# OpponentPool tests
# ---------------------------------------------------------------------------

def test_opponent_pool_starts_with_one(network):
    pool = OpponentPool(network.state_dict())
    assert len(pool) == 1


def test_opponent_pool_grows(network):
    pool = OpponentPool(network.state_dict())
    pool.checkpoint(build_ppo_network().state_dict())
    assert len(pool) == 2


def test_opponent_pool_sample_returns_state_dict(network):
    pool = OpponentPool(network.state_dict())
    sd = pool.sample()
    assert isinstance(sd, dict)
    assert len(sd) > 0


# ---------------------------------------------------------------------------
# PPOTrainer rollout / update tests
# ---------------------------------------------------------------------------

def test_collect_rollout_returns_nonempty_buffer(trainer):
    buffer = trainer.collect_rollout()
    assert len(buffer) > 0


def test_update_returns_loss_dict(trainer):
    buffer = trainer.collect_rollout()
    info = trainer.update(buffer)
    assert "policy_loss" in info
    assert "value_loss" in info
    assert "entropy_loss" in info
    for key in ("policy_loss", "value_loss", "entropy_loss"):
        assert isinstance(info[key], float)
        assert info[key] == info[key]  # not NaN


def test_pool_grows_after_checkpoint_every(env, network):
    trainer = PPOTrainer(
        env=env,
        network=network,
        device="cpu",
        episodes_per_update=1,
        checkpoint_every=1,
    )
    buffer = trainer.collect_rollout()
    trainer.update(buffer)                          # update_count=1 → checkpoint
    trainer.update(trainer.collect_rollout())       # update_count=2 → checkpoint
    assert len(trainer.pool) >= 2


def test_train_returns_log(trainer):
    log = trainer.train(n_updates=2, progress=False)
    assert isinstance(log, list)
    assert len(log) == 2
    assert all("policy_loss" in item for item in log)


# ---------------------------------------------------------------------------
# Smoke test: confirm reward signal flows end-to-end (win_rate >= 30%)
# ---------------------------------------------------------------------------

def test_ppo_training_improves():
    """Train for 10 updates on CPU; win rate vs random-weight opponent >= 30%."""
    env = OthelloEnv()
    network = build_ppo_network()
    trainer = PPOTrainer(
        env=env,
        network=network,
        device="cpu",
        gamma=PPO_GAMMA,
        lam=PPO_LAM,
        clip_eps=PPO_CLIP_EPS,
        value_coef=PPO_VALUE_COEF,
        entropy_coef=PPO_ENTROPY_COEF,
        ppo_epochs=2,
        minibatch_size=PPO_MINIBATCH_SIZE,
        lr=PPO_LEARNING_RATE,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        episodes_per_update=3,
        checkpoint_every=PPO_CHECKPOINT_EVERY,
    )
    trainer.train(n_updates=10, progress=False)

    # Evaluate against a fresh random-weight opponent network
    random_opponent = build_ppo_network()
    n_games = 20
    wins = 0
    for _ in range(n_games):
        transitions = collect_episode(env, trainer.network, random_opponent, "cpu")
        if transitions and transitions[-1]["reward"] > 0:
            wins += 1

    win_rate = wins / n_games
    assert win_rate >= 0.30, (
        f"Expected win_rate >= 0.30, got {win_rate:.2f} ({wins}/{n_games})"
    )
