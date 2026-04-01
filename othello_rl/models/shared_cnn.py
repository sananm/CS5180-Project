from __future__ import annotations

import torch
from torch import nn

from othello_rl.config import (
    ACTION_SIZE,
    BOARD_SIZE,
    CHANNELS,
    SHARED_CNN_CHANNELS,
    SHARED_CNN_POLICY_HEAD_CHANNELS,
    SHARED_CNN_RES_BLOCKS,
    SHARED_CNN_VALUE_HIDDEN_DIM,
)


class ResidualBlock(nn.Module):
    """Standard residual block for the Othello CNN backbone."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        return self.relu(x)


class SharedCNN(nn.Module):
    """Shared residual policy/value network for AlphaZero and PPO."""

    def __init__(
        self,
        in_channels: int = CHANNELS,
        board_size: int = BOARD_SIZE,
        action_size: int = ACTION_SIZE,
        num_channels: int = SHARED_CNN_CHANNELS,
        num_res_blocks: int = SHARED_CNN_RES_BLOCKS,
        policy_head_channels: int = SHARED_CNN_POLICY_HEAD_CHANNELS,
        value_hidden_dim: int = SHARED_CNN_VALUE_HIDDEN_DIM,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.board_size = board_size
        self.action_size = action_size
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.policy_head_channels = policy_head_channels
        self.value_hidden_dim = value_hidden_dim

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList(
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        )

        policy_features = policy_head_channels * board_size * board_size
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, policy_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_head_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(policy_features, action_size),
        )

        value_features = board_size * board_size
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(value_features, value_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_in(x)
        for block in self.res_blocks:
            x = block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
