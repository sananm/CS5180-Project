"""MCTS with PUCT selection for AlphaZero.

Implements Monte Carlo Tree Search with neural network policy/value guidance
following the AlphaZero algorithm (Silver et al., 2017).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from othello_rl.utils.tensor_utils import board_to_tensor

if TYPE_CHECKING:
    from othello_rl.game.othello_game import OthelloGame
    from othello_rl.models.shared_cnn import SharedCNN


class MCTS:
    """Monte Carlo Tree Search with PUCT selection.
    
    Uses a neural network for policy priors and value estimates.
    Stores tree information in dictionaries keyed by board string representation.
    
    Attributes:
        game: OthelloGame instance for state transitions and legal moves.
        network: SharedCNN for policy/value inference.
        num_sims: Number of MCTS simulations per move.
        cpuct: Exploration constant for PUCT formula.
        device: Device for neural network inference.
    """

    def __init__(
        self,
        game: OthelloGame,
        network: SharedCNN,
        num_sims: int = 100,
        cpuct: float = 1.0,
        device: str = 'cpu',
    ):
        self.game = game
        self.network = network
        self.num_sims = num_sims
        self.cpuct = cpuct
        self.device = device
        
        # Dictionary-based tree storage (per RESEARCH.md Pattern 1)
        self.Qsa: dict[tuple[bytes, int], float] = {}   # Q values for (s, a) pairs
        self.Nsa: dict[tuple[bytes, int], int] = {}     # Visit counts for (s, a) pairs
        self.Ns: dict[bytes, int] = {}      # Visit counts for state s
        self.Ps: dict[bytes, np.ndarray] = {}  # Policy prior from network
        self.Vs: dict[bytes, np.ndarray] = {}  # Valid moves cache
        self.Es: dict[bytes, int] = {}      # Game-ended cache

    def reset(self) -> None:
        """Clear all tree information for a new game."""
        self.Qsa.clear()
        self.Nsa.clear()
        self.Ns.clear()
        self.Ps.clear()
        self.Vs.clear()
        self.Es.clear()

    def get_action_prob(self, canonical_board: np.ndarray, temp: float = 1) -> np.ndarray:
        """Run MCTS simulations and return action probabilities.
        
        Args:
            canonical_board: Board from current player's perspective (8, 8).
            temp: Temperature for action selection.
                  temp=0: Greedy (max visit count).
                  temp=1: Proportional to visit counts.
        
        Returns:
            Action probability distribution (ACTION_SIZE,).
        """
        # Run simulations
        for _ in range(self.num_sims):
            self.search(canonical_board)
        
        # Get visit counts for this state
        s = self.game.stringRepresentation(canonical_board)
        action_size = self.game.getActionSize()
        counts = np.array([
            self.Nsa.get((s, a), 0) for a in range(action_size)
        ], dtype=np.float64)
        
        if temp == 0:
            # Greedy: select most-visited action(s)
            best_count = np.max(counts)
            best_actions = np.where(counts == best_count)[0]
            probs = np.zeros(action_size, dtype=np.float64)
            # Random tie-breaking among best actions
            selected = np.random.choice(best_actions)
            probs[selected] = 1.0
            return probs
        
        # Temperature-scaled probabilities: counts^(1/temp)
        # To avoid overflow with high counts, use log-space
        if counts.sum() == 0:
            # If no simulations completed, use uniform over valid
            probs = self.Vs.get(s, np.ones(action_size) / action_size)
            probs = probs / probs.sum()
            return probs
        
        # counts^(1/temp) with numerical stability
        counts_scaled = counts ** (1.0 / temp)
        probs = counts_scaled / counts_scaled.sum()
        return probs

    @torch.no_grad()
    def search(self, canonical_board: np.ndarray) -> float:
        """Recursive MCTS search with PUCT selection.
        
        Negamax-style: returns negative value for recursive calls.
        
        Args:
            canonical_board: Board from current player's perspective.
        
        Returns:
            Value of the position from current player's perspective.
        """
        s = self.game.stringRepresentation(canonical_board)
        
        # Check terminal state cache
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonical_board, 1)
        
        if self.Es[s] != 0:
            # Terminal state: return game result
            return -self.Es[s]  # Negative because we're returning to opponent's view
        
        # Check if this is a leaf node (not yet expanded)
        if s not in self.Ps:
            # Expand: get policy and value from neural network
            tensor = board_to_tensor(canonical_board).unsqueeze(0).to(self.device)
            policy_logits, value = self.network(tensor)
            policy = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
            v = value.item()
            
            # Get valid moves and mask policy
            valids = self.game.getValidMoves(canonical_board, 1)
            policy = policy * valids  # Mask invalid actions
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum  # Renormalize
            else:
                # If all valid actions were masked (shouldn't happen), uniform over valid
                policy = valids / valids.sum()
            
            # Store in tree
            self.Ps[s] = policy
            self.Vs[s] = valids
            self.Ns[s] = 0
            
            return -v  # Negative for negamax
        
        # Not a leaf: select action using PUCT
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        
        action_size = self.game.getActionSize()
        for a in range(action_size):
            if valids[a]:
                # PUCT formula: U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
                if (s, a) in self.Qsa:
                    q = self.Qsa[(s, a)]
                    n_sa = self.Nsa[(s, a)]
                else:
                    q = 0
                    n_sa = 0
                
                # Exploration bonus
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8) / (1 + n_sa)
                score = q + u
                
                if score > cur_best:
                    cur_best = score
                    best_act = a
        
        a = best_act
        
        # Take action and recurse
        next_board, next_player = self.game.getNextState(canonical_board, 1, a)
        # Get canonical form for next player
        next_canonical = self.game.getCanonicalForm(next_board, next_player)
        
        # Recursive search returns value from opponent's perspective
        v = self.search(next_canonical)
        
        # Backup: update Q and N
        if (s, a) in self.Qsa:
            # Incremental mean update
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        
        self.Ns[s] += 1
        
        return -v  # Negamax: return negative for parent
