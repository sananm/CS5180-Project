import numpy as np
import torch
from torch.distributions import Categorical


def board_to_tensor(canonical_board: np.ndarray) -> torch.Tensor:
    """
    Convert canonical (8,8) board to (3,8,8) tensor.

    In canonical form (after getCanonicalForm), current player pieces = +1,
    opponent pieces = -1, empty = 0.

    Channel 0: current player discs (1 where canonical == 1)
    Channel 1: opponent discs (1 where canonical == -1)
    Channel 2: turn indicator (all 1s -- always current player's turn in canonical form)
    """
    current = (canonical_board == 1).astype(np.float32)
    opponent = (canonical_board == -1).astype(np.float32)
    turn = np.ones_like(current)  # Always 1 in canonical form
    tensor = np.stack([current, opponent, turn], axis=0)  # (3, 8, 8)
    return torch.from_numpy(tensor)


def tensor_to_board(tensor: torch.Tensor) -> np.ndarray:
    """
    Inverse: (3,8,8) tensor back to canonical (8,8) board.
    Useful for verification / round-trip testing.
    """
    arr = tensor.numpy()
    board = np.zeros((8, 8), dtype=np.float64)
    board[arr[0] == 1] = 1    # current player
    board[arr[1] == 1] = -1   # opponent
    return board


def apply_action_mask(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    Set logits of invalid actions to a very large negative value.

    Args:
        logits: (batch, 65) or (65,) raw network output
        valid_mask: same shape as logits, binary (1=valid, 0=invalid)

    Returns:
        masked_logits: same shape, invalid actions set to torch.finfo min
    """
    mask_value = torch.finfo(logits.dtype).min
    return torch.where(valid_mask.bool(), logits, torch.tensor(mask_value, dtype=logits.dtype))


class CategoricalMasked(Categorical):
    """Categorical distribution with action masking support."""

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor = None):
        self.mask = mask
        if mask is None:
            super().__init__(logits=logits)
        else:
            mask_value = torch.finfo(logits.dtype).min
            masked_logits = torch.where(mask.bool(), logits,
                                        torch.full_like(logits, mask_value))
            super().__init__(logits=masked_logits)

    def entropy(self):
        """Entropy computed only over valid actions."""
        if self.mask is None:
            return super().entropy()
        # Zero out contribution from masked actions
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.mask.bool(), p_log_p, torch.zeros_like(p_log_p))
        return -p_log_p.sum(dim=-1)
