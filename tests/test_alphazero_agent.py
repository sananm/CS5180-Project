"""Tests for AlphaZero agent."""

import numpy as np
import pytest
import torch

from othello_rl.agents.base_agent import BaseAgent
from othello_rl.alphazero.agent import AlphaZeroAgent
from othello_rl.game.othello_game import OthelloGame
from othello_rl.alphazero.network import build_alphazero_network


@pytest.fixture
def game():
    return OthelloGame(8)


@pytest.fixture
def network():
    return build_alphazero_network()


@pytest.fixture
def agent(game, network):
    return AlphaZeroAgent(game, network, num_sims=10)


def test_implements_base_agent(agent):
    """AlphaZeroAgent should implement BaseAgent interface."""
    assert isinstance(agent, BaseAgent)
    assert agent.name.startswith("AlphaZero")


def test_get_action_returns_valid_int(agent, game):
    """get_action should return an integer action in [0, 64]."""
    board = game.getInitBoard()
    valids = game.getValidMoves(board, 1)
    
    action = agent.get_action(board, valids)
    
    assert isinstance(action, int)
    assert 0 <= action <= 64
    assert valids[action] == 1


def test_reset_clears_mcts(agent, game):
    """reset() should clear the underlying MCTS tree."""
    board = game.getInitBoard()
    agent.get_action(board, game.getValidMoves(board, 1))
    
    assert len(agent.mcts.Ns) > 0
    agent.reset()
    assert len(agent.mcts.Ns) == 0


def test_accepts_3channel_board(agent, game):
    """get_action should handle (3, 8, 8) tensor board."""
    # Create (3, 8, 8) board
    board_2d = game.getInitBoard()
    board_3d = np.zeros((3, 8, 8), dtype=np.float64)
    board_3d[0] = (board_2d == 1)
    board_3d[1] = (board_2d == -1)
    board_3d[2] = 1.0  # Player 1 turn
    
    valids = game.getValidMoves(board_2d, 1)
    
    # This should not raise an error
    action = agent.get_action(board_3d, valids)
    assert 0 <= action <= 64
