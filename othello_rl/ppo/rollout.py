from __future__ import annotations

from dataclasses import dataclass, field, fields

import torch

from othello_rl.utils.tensor_utils import CategoricalMasked


@dataclass
class RolloutBuffer:
    """Stores transitions collected during PPO rollout episodes.

    Each list entry corresponds to one training-agent step.
    Fields per step:
        obs      -- (3,8,8) float32 tensor (canonical board)
        actions  -- int action index in {0..64}
        log_probs -- float log-probability of sampled action
        values   -- float critic estimate V(s_t)
        rewards  -- float reward (0 until terminal)
        dones    -- bool True at episode end
        masks    -- (65,) bool tensor of valid actions
    """

    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    masks: list = field(default_factory=list)

    def clear(self):
        """Empty all transition lists."""
        for f in fields(self):
            getattr(self, f.name).clear()

    def __len__(self) -> int:
        return len(self.rewards)


def collect_episode(env, policy_network, opponent_network, device) -> list[dict]:
    """Play one complete Othello game and collect PPO transitions for the training agent.

    Training agent is always player 1 (black). Opponent turns advance the
    environment state without generating training data.

    Args:
        env: OthelloEnv instance (will be reset at start)
        policy_network: SharedCNN for the training agent (set to eval inside)
        opponent_network: SharedCNN for the opponent (set to eval inside)
        device: torch.device or str

    Returns:
        List of transition dicts with keys:
            obs       -- torch.Tensor (3,8,8)
            action    -- int
            log_prob  -- float
            value     -- float
            reward    -- float
            done      -- bool
            mask      -- torch.Tensor (65,) bool
    """
    obs, valid = env.reset()
    transitions: list[dict] = []
    done = False

    while not done:
        if env.player == 1:
            # Training agent's turn
            policy_network.eval()
            with torch.no_grad():
                obs_t = obs.unsqueeze(0).to(device)          # (1, 3, 8, 8)
                logits, value = policy_network(obs_t)         # (1, 65), (1, 1)
                mask_t = torch.tensor(valid, dtype=torch.bool, device=device)  # (65,)
                dist = CategoricalMasked(logits=logits.squeeze(0), mask=mask_t)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            obs_next, reward, done, _ = env.step(action.item())

            transitions.append({
                "obs": obs,
                "action": action.item(),
                "log_prob": log_prob.item(),
                "value": value.item(),
                "reward": float(reward),
                "done": bool(done),
                "mask": mask_t.cpu(),
            })

            if not done:
                obs = obs_next
                valid = env.get_valid_actions()

        else:
            # Opponent's turn: advance state, no data collected
            opponent_network.eval()
            with torch.no_grad():
                obs_t = obs.unsqueeze(0).to(device)
                logits, _ = opponent_network(obs_t)
                mask_t = torch.tensor(valid, dtype=torch.bool, device=device)
                dist = CategoricalMasked(logits=logits.squeeze(0), mask=mask_t)
                action = dist.sample()

            obs_next, reward, done, _ = env.step(action.item())

            if not done:
                obs = obs_next
                valid = env.get_valid_actions()
            else:
                # Opponent ended the game. Assign reward from training agent's
                # perspective to the last recorded training-agent transition.
                # OthelloEnv.step() returns reward = -done_val where done_val
                # is from the NEW player's perspective (already negated).
                # Since the opponent just acted and is player 2 at this point,
                # the returned reward is for the opponent.  Training agent gets
                # the negated value (i.e., -reward from here).
                if transitions:
                    transitions[-1]["reward"] = float(-reward)
                    transitions[-1]["done"] = True

    return transitions
