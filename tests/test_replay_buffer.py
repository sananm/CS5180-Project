"""Tests for AlphaZero FIFO replay buffer."""

import numpy as np
import pytest

from othello_rl.alphazero.replay_buffer import ReplayBuffer


def _make_examples(n, board_shape=(8, 8), pi_size=65):
    """Create n synthetic (board, pi, z) tuples."""
    examples = []
    for i in range(n):
        board = np.full(board_shape, float(i), dtype=np.float64)
        pi = np.zeros(pi_size, dtype=np.float64)
        pi[i % pi_size] = 1.0
        z = 1.0 if i % 2 == 0 else -1.0
        examples.append((board, pi, z))
    return examples


class TestReplayBufferBasics:
    """Test basic push, length, and clear operations."""

    def test_buffer_push_and_len(self):
        """buffer.push(examples) increases len(buffer) correctly."""
        buf = ReplayBuffer(max_size=100)
        assert len(buf) == 0

        examples = _make_examples(5)
        buf.push(examples)
        assert len(buf) == 5

        more = _make_examples(3)
        buf.push(more)
        assert len(buf) == 8

    def test_buffer_clear(self):
        """buffer.clear() empties the buffer."""
        buf = ReplayBuffer(max_size=100)
        buf.push(_make_examples(10))
        assert len(buf) == 10

        buf.clear()
        assert len(buf) == 0


class TestReplayBufferSampling:
    """Test random sampling behaviour."""

    def test_buffer_sample(self):
        """buffer.sample(batch_size) returns (boards, pis, zs) lists of correct length."""
        buf = ReplayBuffer(max_size=100)
        buf.push(_make_examples(20))

        boards, pis, zs = buf.sample(5)
        assert len(boards) == 5
        assert len(pis) == 5
        assert len(zs) == 5

        # Each element should have the right type/shape
        assert boards[0].shape == (8, 8)
        assert pis[0].shape == (65,)
        assert isinstance(zs[0], float)

    def test_buffer_sample_smaller_than_size(self):
        """Sampling more than buffer size returns min(batch_size, len(buffer))."""
        buf = ReplayBuffer(max_size=100)
        buf.push(_make_examples(3))

        boards, pis, zs = buf.sample(10)
        assert len(boards) == 3  # clamped to buffer size
        assert len(pis) == 3
        assert len(zs) == 3


class TestReplayBufferFIFO:
    """Test FIFO eviction at capacity."""

    def test_buffer_fifo_eviction(self):
        """When buffer exceeds maxlen, oldest items are evicted first."""
        buf = ReplayBuffer(max_size=5)

        # Push 5 examples with board values 0-4
        first_batch = _make_examples(5)
        buf.push(first_batch)
        assert len(buf) == 5

        # Push 3 more examples with board values 5-7
        second_batch = _make_examples(3)
        # Override board values so they are distinguishable
        second_batch_distinct = []
        for i in range(3):
            board = np.full((8, 8), float(i + 100), dtype=np.float64)
            pi = np.zeros(65, dtype=np.float64)
            z = 1.0
            second_batch_distinct.append((board, pi, z))
        buf.push(second_batch_distinct)

        # Buffer should still be size 5 (maxlen)
        assert len(buf) == 5

        # Oldest 3 items (board values 0, 1, 2) should be evicted
        # Remaining: board values 3, 4, 100, 101, 102
        remaining_board_vals = set()
        # Sample all 5 items
        boards, _, _ = buf.sample(5)
        for b in boards:
            remaining_board_vals.add(float(b[0, 0]))

        assert 0.0 not in remaining_board_vals, "Oldest item (0) should be evicted"
        assert 1.0 not in remaining_board_vals, "Second oldest item (1) should be evicted"
        assert 2.0 not in remaining_board_vals, "Third oldest item (2) should be evicted"
        assert 3.0 in remaining_board_vals, "Item 3 should remain"
        assert 4.0 in remaining_board_vals, "Item 4 should remain"
        assert 100.0 in remaining_board_vals, "New item 100 should be present"
