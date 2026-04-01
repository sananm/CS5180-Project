from __future__ import annotations

import torch
import torch.nn.functional as F

from othello_rl.utils.tensor_utils import CategoricalMasked


def ppo_loss(
    policy_network,
    batch_obs: torch.Tensor,
    batch_actions: torch.Tensor,
    batch_masks: torch.Tensor,
    batch_old_log_probs: torch.Tensor,
    batch_advantages: torch.Tensor,
    batch_returns: torch.Tensor,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO clipped surrogate loss with value loss and entropy bonus.

    Caller is responsible for setting policy_network.train() before calling.
    old_log_probs MUST be stored from rollout collection -- never recomputed here.
    Advantage normalization happens inside this function at minibatch level.

    Args:
        policy_network:       SharedCNN in .train() mode
        batch_obs:            (B, 3, 8, 8) float tensor
        batch_actions:        (B,) long tensor
        batch_masks:          (B, 65) bool tensor of valid actions
        batch_old_log_probs:  (B,) float tensor stored during rollout
        batch_advantages:     (B,) float tensor (unnormalized GAE advantages)
        batch_returns:        (B,) float tensor (GAE bootstrapped returns)
        clip_eps:             PPO clip epsilon (default 0.2)
        value_coef:           weight for value loss (default 0.5)
        entropy_coef:         weight for entropy bonus (default 0.01)

    Returns:
        (total_loss, policy_loss, value_loss, entropy_loss) -- all scalar tensors
    """
    logits, values = policy_network(batch_obs)    # (B, 65), (B, 1)
    dist = CategoricalMasked(logits=logits, mask=batch_masks)
    new_log_probs = dist.log_prob(batch_actions)   # (B,)
    entropy = dist.entropy()                        # (B,)

    # Normalize advantages at minibatch level
    adv_norm = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

    # Importance-sampling ratio
    ratio = torch.exp(new_log_probs - batch_old_log_probs)  # (B,)

    # Clipped surrogate objective
    surr1 = ratio * adv_norm
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss (MSE against bootstrapped returns)
    value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

    # Entropy loss (negated; added with positive coefficient to encourage exploration)
    entropy_loss = -entropy.mean()

    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    return total_loss, policy_loss, value_loss, entropy_loss
