import torch
import pytest

from othello_rl.utils.tensor_utils import apply_action_mask, CategoricalMasked


def test_mask_shape_preserved():
    """Output shape == input shape."""
    logits = torch.randn(1, 65)
    mask = torch.zeros(1, 65)
    mask[0, [19, 26, 37, 44]] = 1
    masked = apply_action_mask(logits, mask)
    assert masked.shape == logits.shape


def test_illegal_moves_zero_prob():
    """After masking + softmax, illegal moves have probability < 1e-30."""
    logits = torch.randn(1, 65)
    mask = torch.zeros(1, 65)
    mask[0, [19, 26, 37, 44]] = 1
    masked = apply_action_mask(logits, mask)
    probs = torch.softmax(masked, dim=-1)
    assert (probs[mask == 0] < 1e-30).all()


def test_legal_moves_sum_to_one():
    """Masked softmax over legal moves sums to ~1.0."""
    logits = torch.randn(1, 65)
    mask = torch.zeros(1, 65)
    mask[0, [19, 26, 37, 44]] = 1
    masked = apply_action_mask(logits, mask)
    probs = torch.softmax(masked, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-6


def test_single_legal_move():
    """When only 1 move is legal, it gets probability ~1.0."""
    logits = torch.randn(1, 65)
    mask = torch.zeros(1, 65)
    mask[0, 30] = 1
    masked = apply_action_mask(logits, mask)
    probs = torch.softmax(masked, dim=-1)
    assert abs(probs[0, 30].item() - 1.0) < 1e-6


def test_pass_only():
    """When only pass (index 64) is legal, it gets probability ~1.0."""
    logits = torch.randn(1, 65)
    mask = torch.zeros(1, 65)
    mask[0, 64] = 1
    masked = apply_action_mask(logits, mask)
    probs = torch.softmax(masked, dim=-1)
    assert abs(probs[0, 64].item() - 1.0) < 1e-6


def test_categorical_masked_sample():
    """CategoricalMasked.sample() only returns legal actions (100 samples)."""
    logits = torch.randn(65)
    mask = torch.zeros(65)
    mask[[10, 20, 30, 40]] = 1
    dist = CategoricalMasked(logits=logits, mask=mask)

    legal_set = {10, 20, 30, 40}
    for _ in range(100):
        action = dist.sample().item()
        assert action in legal_set, f"Sampled illegal action {action}"


def test_categorical_masked_entropy():
    """Entropy with 2 valid actions < entropy with 4 valid actions."""
    logits = torch.zeros(65)  # uniform logits

    mask_2 = torch.zeros(65)
    mask_2[[10, 20]] = 1
    dist_2 = CategoricalMasked(logits=logits, mask=mask_2)
    entropy_2 = dist_2.entropy().item()

    mask_4 = torch.zeros(65)
    mask_4[[10, 20, 30, 40]] = 1
    dist_4 = CategoricalMasked(logits=logits, mask=mask_4)
    entropy_4 = dist_4.entropy().item()

    assert entropy_2 < entropy_4, f"Entropy(2 valid)={entropy_2} >= Entropy(4 valid)={entropy_4}"


def test_batch_masking():
    """Masking works with batch dimension > 1."""
    logits = torch.randn(4, 65)
    mask = torch.zeros(4, 65)
    mask[0, [10, 20]] = 1
    mask[1, [30, 40]] = 1
    mask[2, [5, 15, 25]] = 1
    mask[3, [64]] = 1  # pass only

    masked = apply_action_mask(logits, mask)
    probs = torch.softmax(masked, dim=-1)

    for i in range(4):
        assert abs(probs[i].sum().item() - 1.0) < 1e-6
        illegal = (mask[i] == 0)
        assert (probs[i][illegal] < 1e-30).all()
