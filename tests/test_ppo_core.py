"""Unit tests for core PPO algorithm components.

Tests cover:
    - RolloutBuffer dataclass (clear, len)
    - compute_gae mathematical correctness
    - IS ratio == 1.0 before any gradient step
    - ppo_loss shape and finiteness
"""

import pytest
import torch

from othello_rl.ppo.gae import compute_gae
from othello_rl.ppo.loss import ppo_loss
from othello_rl.ppo.network import build_ppo_network
from othello_rl.ppo.rollout import RolloutBuffer, collect_episode
from othello_rl.utils.tensor_utils import CategoricalMasked


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------


def test_rollout_buffer_clear():
    """Appending data then calling clear() resets len to 0."""
    buf = RolloutBuffer()
    buf.obs.append(torch.zeros(3, 8, 8))
    buf.actions.append(0)
    buf.log_probs.append(-1.0)
    buf.values.append(0.5)
    buf.rewards.append(0.0)
    buf.dones.append(False)
    buf.masks.append(torch.ones(65, dtype=torch.bool))
    assert len(buf) == 1

    buf.clear()
    assert len(buf) == 0
    assert buf.obs == []
    assert buf.actions == []
    assert buf.log_probs == []
    assert buf.values == []
    assert buf.rewards == []
    assert buf.dones == []
    assert buf.masks == []


# ---------------------------------------------------------------------------
# compute_gae
# ---------------------------------------------------------------------------


def test_gae_single_terminal_step():
    """GAE on a single terminal step.

    delta = 1.0 + 0.99*0 - 0.5 = 0.5  (last_val zeroed by done mask)
    advantage = 0.5
    return = 0.5 + 0.5 = 1.0
    """
    adv, ret = compute_gae([1.0], [0.5], [True], gamma=0.99, lam=0.95)
    assert adv[0] == pytest.approx(0.5, abs=1e-5)
    assert ret[0] == pytest.approx(1.0, abs=1e-5)


def test_gae_multi_step_no_terminal():
    """GAE on a 3-step trajectory with terminal only at the end.

    rewards=[0.0, 0.0, 1.0], values=[0.3, 0.4, 0.5], dones=[False, False, True]
    gamma=0.99, lam=0.95.

    Expected (computed by hand / verified by reference implementation):
        adv[2] = 0.5
        adv[1] = 0.56525
        adv[0] = 0.6276176250000001
        ret[2] = 1.0
        ret[1] = 0.96525
        ret[0] = 0.9276176250000001
    """
    adv, ret = compute_gae(
        [0.0, 0.0, 1.0], [0.3, 0.4, 0.5], [False, False, True], gamma=0.99, lam=0.95
    )
    assert adv[0] == pytest.approx(0.6276176250, abs=1e-4)
    assert adv[1] == pytest.approx(0.56525, abs=1e-4)
    assert adv[2] == pytest.approx(0.5, abs=1e-4)
    assert ret[0] == pytest.approx(0.9276176250, abs=1e-4)
    assert ret[1] == pytest.approx(0.96525, abs=1e-4)
    assert ret[2] == pytest.approx(1.0, abs=1e-4)


def test_gae_masks_episode_boundary():
    """Episode boundaries zero out the advantage bootstrap correctly.

    rewards=[1.0, 0.0, -1.0], values=[0.5, 0.5, 0.5], dones=[True, False, True]
    gamma=0.99, lam=0.95.

    Episode 1 (step 0 is terminal):
        delta = 1.0 + 0 - 0.5 = 0.5  =>  adv[0] = 0.5

    Episode 2 (steps 1-2):
        Step 2 (terminal): delta = -1.0 + 0 - 0.5 = -1.5  =>  adv[2] = -1.5
    """
    adv, ret = compute_gae(
        [1.0, 0.0, -1.0], [0.5, 0.5, 0.5], [True, False, True], gamma=0.99, lam=0.95
    )
    assert adv[0] == pytest.approx(0.5, abs=1e-5), (
        f"Episode 1 should be isolated; expected 0.5 got {adv[0]}"
    )
    assert adv[2] == pytest.approx(-1.5, abs=1e-5), (
        f"Episode 2 terminal; expected -1.5 got {adv[2]}"
    )


# ---------------------------------------------------------------------------
# IS ratio == 1.0 before any gradient step
# ---------------------------------------------------------------------------


def test_is_ratio_first_update():
    """IS ratio must be identically 1.0 before the first gradient step.

    This is the standard correctness check for PPO implementations.  If the
    network is in .train() mode during rollout collection, BatchNorm running
    statistics change between passes, causing ratios != 1.0 even with no
    gradient update.  Collecting rollouts with .eval() prevents this.
    """
    torch.manual_seed(0)
    network = build_ppo_network()
    network.eval()

    B = 4
    batch_obs = torch.zeros(B, 3, 8, 8)
    batch_masks = torch.ones(B, 65, dtype=torch.bool)

    # Collect old log-probs (simulating rollout collection in eval mode)
    with torch.no_grad():
        logits, _ = network(batch_obs)
        dist = CategoricalMasked(logits=logits, mask=batch_masks)
        batch_actions = dist.sample()
        batch_old_log_probs = dist.log_prob(batch_actions)

    # Recompute with the SAME network (no gradient step has occurred)
    with torch.no_grad():
        logits2, _ = network(batch_obs)
        dist2 = CategoricalMasked(logits=logits2, mask=batch_masks)
        new_log_probs = dist2.log_prob(batch_actions)
        ratio = torch.exp(new_log_probs - batch_old_log_probs)

    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5), (
        f"IS ratio must be 1.0 before any update; got mean={ratio.mean():.6f}"
    )


# ---------------------------------------------------------------------------
# ppo_loss shapes and finiteness
# ---------------------------------------------------------------------------


def _make_ppo_batch(B: int = 4):
    """Helper: create a minimal valid ppo_loss input batch."""
    torch.manual_seed(1)
    network = build_ppo_network()
    batch_obs = torch.randn(B, 3, 8, 8)
    batch_actions = torch.zeros(B, dtype=torch.long)
    batch_masks = torch.ones(B, 65, dtype=torch.bool)

    network.eval()
    with torch.no_grad():
        logits, _ = network(batch_obs)
        dist = CategoricalMasked(logits=logits, mask=batch_masks)
        old_lp = dist.log_prob(batch_actions)

    batch_adv = torch.ones(B, dtype=torch.float32)
    batch_ret = torch.ones(B, dtype=torch.float32)
    return network, batch_obs, batch_actions, batch_masks, old_lp, batch_adv, batch_ret


def test_ppo_loss_shapes():
    """ppo_loss returns four scalar tensors."""
    network, obs, actions, masks, old_lp, adv, ret = _make_ppo_batch()
    network.train()
    total, p, v, e = ppo_loss(network, obs, actions, masks, old_lp, adv, ret)

    for name, val in [("total", total), ("policy", p), ("value", v), ("entropy", e)]:
        assert val.shape == torch.Size([]), f"{name} loss should be scalar, got {val.shape}"
        assert torch.isfinite(val), f"{name} loss should be finite, got {val.item()}"


def test_ppo_loss_no_nan():
    """ppo_loss total loss must not be NaN or Inf."""
    network, obs, actions, masks, old_lp, adv, ret = _make_ppo_batch()
    network.train()
    total, _, _, _ = ppo_loss(network, obs, actions, masks, old_lp, adv, ret)
    assert not torch.isnan(total), "total loss is NaN"
    assert not torch.isinf(total), "total loss is Inf"
