from __future__ import annotations


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[list[float], list[float]]:
    """Compute Generalized Advantage Estimation (GAE) advantages and returns.

    Backward iteration through a trajectory of length T. Episode boundaries
    are handled by zeroing the bootstrap when dones[t] is True.

    Args:
        rewards: length-T list of per-step rewards
        values:  length-T list of critic estimates V(s_t)
        dones:   length-T list of episode-terminal flags
        gamma:   discount factor (default 0.99)
        lam:     GAE lambda (default 0.95)

    Returns:
        advantages: length-T list of GAE advantage estimates
        returns:    length-T list of bootstrapped return targets (used as value targets)
    """
    T = len(rewards)
    advantages = [0.0] * T
    last_adv = 0.0
    last_val = 0.0  # Bootstrap value beyond the last step is zero

    for t in reversed(range(T)):
        mask = 0.0 if dones[t] else 1.0
        last_val = last_val * mask
        last_adv = last_adv * mask

        delta = rewards[t] + gamma * last_val - values[t]
        last_adv = delta + gamma * lam * last_adv
        advantages[t] = last_adv
        last_val = values[t]

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns
