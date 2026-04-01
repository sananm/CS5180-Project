"""Tests for PPOAgent: BaseAgent interface compliance and action validity."""

import numpy as np
import torch
import pytest

from othello_rl.ppo.agent import PPOAgent
from othello_rl.ppo.network import build_ppo_network
from othello_rl.agents.base_agent import BaseAgent
from othello_rl.game.othello_env import OthelloEnv


@pytest.fixture
def env():
    return OthelloEnv()


@pytest.fixture
def agent():
    return PPOAgent(build_ppo_network(), device="cpu")


def test_ppo_agent_is_base_agent(agent):
    assert isinstance(agent, BaseAgent)


def test_ppo_agent_get_action_returns_valid_int(env, agent):
    obs, valid = env.reset()
    board_np = env.game.getCanonicalForm(env.board, env.player)
    action = agent.get_action(board_np, valid)
    assert isinstance(action, int)
    assert 0 <= action <= 64
    assert valid[action] == 1


def test_ppo_agent_name(agent):
    assert agent.name == "PPO"


def test_ppo_agent_accepts_tensor_obs(agent):
    obs_tensor = torch.zeros(3, 8, 8)
    valid = np.ones(65, dtype=np.int64)
    action = agent.get_action(obs_tensor, valid)
    assert 0 <= action <= 64
