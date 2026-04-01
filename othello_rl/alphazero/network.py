from __future__ import annotations

from othello_rl.models.shared_cnn import SharedCNN


def build_alphazero_network(**overrides) -> SharedCNN:
    """Return the shared project CNN for AlphaZero code paths."""
    return SharedCNN(**overrides)
