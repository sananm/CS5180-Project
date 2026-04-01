"""PPO training infrastructure: OpponentPool and PPOTrainer."""

from __future__ import annotations

import copy
import random

import torch
import torch.nn.utils as nn_utils
from tqdm import tqdm

from othello_rl.ppo.rollout import RolloutBuffer, collect_episode
from othello_rl.ppo.gae import compute_gae
from othello_rl.ppo.loss import ppo_loss
from othello_rl.ppo.network import build_ppo_network
from othello_rl.config.default import (
    PPO_CLIP_EPS,
    PPO_GAMMA,
    PPO_LAM,
    PPO_EPOCHS,
    PPO_MINIBATCH_SIZE,
    PPO_LEARNING_RATE,
    PPO_VALUE_COEF,
    PPO_ENTROPY_COEF,
    PPO_MAX_GRAD_NORM,
    PPO_EPISODES_PER_UPDATE,
    PPO_CHECKPOINT_EVERY,
)


class OpponentPool:
    """In-memory pool of opponent policy snapshots for self-play.

    Starts with the initial network weights and grows as training progresses.
    Opponents are sampled uniformly at random to prevent overfitting.
    """

    def __init__(self, initial_state_dict: dict):
        self._pool: list[dict] = [copy.deepcopy(initial_state_dict)]

    def checkpoint(self, state_dict: dict) -> None:
        """Add a new opponent snapshot to the pool."""
        self._pool.append(copy.deepcopy(state_dict))

    def sample(self) -> dict:
        """Return a uniformly random state_dict from the pool."""
        return random.choice(self._pool)

    def __len__(self) -> int:
        return len(self._pool)


class PPOTrainer:
    """Trains a PPO agent via opponent-pool self-play on Othello.

    Connects rollout collection, GAE advantage estimation, and the PPO
    clipped surrogate loss into a full training loop. Checkpoints the
    policy into the opponent pool every `checkpoint_every` updates.

    Args:
        env: OthelloEnv instance used for episode collection.
        network: SharedCNN policy/value network (training agent).
        device: Device string for PyTorch tensors.
        gamma: Discount factor for GAE.
        lam: GAE lambda.
        clip_eps: PPO clip epsilon.
        value_coef: Weight for value loss in combined loss.
        entropy_coef: Weight for entropy bonus.
        ppo_epochs: Number of gradient epochs per update.
        minibatch_size: Minibatch size for each gradient step.
        lr: Adam learning rate.
        max_grad_norm: Maximum gradient norm for clipping.
        episodes_per_update: Episodes to collect before each update.
        checkpoint_every: Opponent pool checkpoint frequency (in updates).
    """

    def __init__(
        self,
        env,
        network,
        device: str = "cpu",
        gamma: float = PPO_GAMMA,
        lam: float = PPO_LAM,
        clip_eps: float = PPO_CLIP_EPS,
        value_coef: float = PPO_VALUE_COEF,
        entropy_coef: float = PPO_ENTROPY_COEF,
        ppo_epochs: int = PPO_EPOCHS,
        minibatch_size: int = PPO_MINIBATCH_SIZE,
        lr: float = PPO_LEARNING_RATE,
        max_grad_norm: float = PPO_MAX_GRAD_NORM,
        episodes_per_update: int = PPO_EPISODES_PER_UPDATE,
        checkpoint_every: int = PPO_CHECKPOINT_EVERY,
    ):
        self.env = env
        self.network = network.to(device)
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.episodes_per_update = episodes_per_update
        self.checkpoint_every = checkpoint_every
        self.update_count = 0

        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.opponent_network = build_ppo_network().to(device)
        self.opponent_network.eval()
        self.pool = OpponentPool(copy.deepcopy(network.state_dict()))

    def collect_rollout(self) -> RolloutBuffer:
        """Collect self.episodes_per_update episodes and return a RolloutBuffer.

        Each episode uses a freshly sampled opponent from the pool.
        """
        buffer = RolloutBuffer()
        for _ in range(self.episodes_per_update):
            self.opponent_network.load_state_dict(self.pool.sample())
            self.opponent_network.eval()
            transitions = collect_episode(
                self.env, self.network, self.opponent_network, self.device
            )
            for t in transitions:
                buffer.obs.append(t["obs"])
                buffer.actions.append(t["action"])
                buffer.log_probs.append(t["log_prob"])
                buffer.values.append(t["value"])
                buffer.rewards.append(t["reward"])
                buffer.dones.append(t["done"])
                buffer.masks.append(t["mask"])
        return buffer

    def update(self, buffer: RolloutBuffer) -> dict:
        """Run PPO update on the collected rollout buffer.

        Computes GAE advantages, then iterates ppo_epochs over shuffled
        minibatches. Checkpoints the opponent pool every checkpoint_every
        updates.

        Returns:
            Dict with keys: policy_loss, value_loss, entropy_loss, update, pool_size.
        """
        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            gamma=self.gamma, lam=self.lam,
        )

        N = len(buffer)
        indices = list(range(N))

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            random.shuffle(indices)
            for start in range(0, N, self.minibatch_size):
                mb_idx = indices[start : start + self.minibatch_size]
                if len(mb_idx) < 2:
                    continue

                mb_obs = torch.stack([buffer.obs[i] for i in mb_idx]).to(self.device)
                mb_actions = torch.tensor(
                    [buffer.actions[i] for i in mb_idx], dtype=torch.long, device=self.device
                )
                mb_masks = torch.stack([buffer.masks[i] for i in mb_idx]).to(self.device)
                mb_old_log_probs = torch.tensor(
                    [buffer.log_probs[i] for i in mb_idx], dtype=torch.float32, device=self.device
                )
                mb_advantages = torch.tensor(
                    [advantages[i] for i in mb_idx], dtype=torch.float32, device=self.device
                )
                mb_returns = torch.tensor(
                    [returns[i] for i in mb_idx], dtype=torch.float32, device=self.device
                )

                self.network.train()
                loss, pol_loss, val_loss, ent_loss = ppo_loss(
                    self.network, mb_obs, mb_actions, mb_masks,
                    mb_old_log_probs, mb_advantages, mb_returns,
                    clip_eps=self.clip_eps,
                    value_coef=self.value_coef,
                    entropy_coef=self.entropy_coef,
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn_utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += pol_loss.item()
                total_value_loss += val_loss.item()
                total_entropy_loss += ent_loss.item()
                num_updates += 1

        self.update_count += 1
        if self.update_count % self.checkpoint_every == 0:
            self.pool.checkpoint(self.network.state_dict())

        denom = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy_loss": total_entropy_loss / denom,
            "update": self.update_count,
            "pool_size": len(self.pool),
        }

    def train(self, n_updates: int, progress: bool = True) -> list[dict]:
        """Run the full PPO training loop for n_updates iterations.

        Args:
            n_updates: Number of collect-rollout → update cycles.
            progress: Whether to display a tqdm progress bar.

        Returns:
            List of info dicts, one per update.
        """
        log: list[dict] = []
        iterator = tqdm(range(n_updates), desc="PPO Training") if progress else range(n_updates)
        for _ in iterator:
            buffer = self.collect_rollout()
            info = self.update(buffer)
            log.append(info)
        return log

    def save_checkpoint(self, path: str) -> None:
        """Save network and optimizer state to a file.

        Args:
            path: Path to the checkpoint file.
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load network and optimizer state from a file.

        Args:
            path: Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']
