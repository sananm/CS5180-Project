"""FIFO replay buffer for AlphaZero self-play training examples.

Stores (board, pi, z) tuples with automatic FIFO eviction when capacity
is exceeded.  Uses ``collections.deque`` with ``maxlen`` for correct and
efficient eviction.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any


class ReplayBuffer:
    """Fixed-capacity FIFO buffer that stores (board, pi, z) training examples.

    Args:
        max_size: Maximum number of examples to retain.  Oldest examples
            are evicted first when capacity is exceeded.
    """

    def __init__(self, max_size: int = 100_000):
        self.buffer: deque[tuple[Any, Any, float]] = deque(maxlen=max_size)

    def push(self, examples: list[tuple[Any, Any, float]]) -> None:
        """Append a list of (board, pi, z) tuples to the buffer."""
        self.buffer.extend(examples)

    def sample(self, batch_size: int) -> tuple[list, list, list]:
        """Return a random sample of (boards, pis, zs) lists.

        If *batch_size* exceeds the current buffer length, returns
        all available examples (randomly ordered).
        """
        k = min(batch_size, len(self.buffer))
        batch = random.sample(list(self.buffer), k)
        boards, pis, zs = zip(*batch)
        return list(boards), list(pis), list(zs)

    def clear(self) -> None:
        """Remove all examples from the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
