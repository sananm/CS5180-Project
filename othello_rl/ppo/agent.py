"""PPOAgent: wraps a trained SharedCNN behind the BaseAgent interface."""

from __future__ import annotations

import numpy as np
import torch

from othello_rl.agents.base_agent import BaseAgent
from othello_rl.utils.tensor_utils import board_to_tensor, CategoricalMasked


class PPOAgent(BaseAgent):
    """BaseAgent wrapper for a PPO-trained SharedCNN policy.

    Accepts either a 2-D numpy board (canonical 8×8) or a pre-encoded
    3-D tensor (3,8,8) as the `board` argument to get_action().

    Args:
        network: Trained SharedCNN policy/value network.
        device: Device for inference ('cpu' or 'cuda').
        deterministic: If True (default), use argmax for greedy action selection.
                       If False, sample from the masked policy distribution.
    """

    def __init__(self, network, device: str = "cpu", deterministic: bool = True):
        self.network = network.to(device)
        self.device = device
        self.deterministic = deterministic
        self._name = "PPO"

    def get_action(self, board, valid_moves: np.ndarray) -> int:
        """Select an action using the PPO policy.

        Args:
            board: (8, 8) canonical numpy array  OR  (3, 8, 8) tensor.
            valid_moves: (65,) binary array, 1 = legal action.

        Returns:
            Integer action in {0..64}.
        """
        if isinstance(board, np.ndarray) and board.ndim == 2:
            obs = board_to_tensor(board)
        elif isinstance(board, np.ndarray):
            obs = torch.from_numpy(board).float()
        elif isinstance(board, torch.Tensor):
            obs = board.float()
        else:
            obs = torch.as_tensor(board, dtype=torch.float32)

        self.network.eval()
        with torch.no_grad():
            obs_t = obs.unsqueeze(0).to(self.device)        # (1, 3, 8, 8)
            logits, _ = self.network(obs_t)                  # (1, 65), (1, 1)
            mask = torch.tensor(valid_moves, dtype=torch.bool, device=self.device)
            
            if self.deterministic:
                # Greedy: pick highest probability legal action
                masked_logits = logits.squeeze(0).clone()
                masked_logits[~mask] = float('-inf')
                action = torch.argmax(masked_logits)
            else:
                dist = CategoricalMasked(logits=logits.squeeze(0), mask=mask)
                action = dist.sample()
        return action.item()

    @property
    def name(self) -> str:
        return self._name
