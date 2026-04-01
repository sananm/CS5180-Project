from __future__ import annotations

from othello_rl.models.shared_cnn import SharedCNN


def build_ppo_network(**overrides) -> SharedCNN:
    """Return the shared project CNN for PPO code paths."""
    return SharedCNN(**overrides)
