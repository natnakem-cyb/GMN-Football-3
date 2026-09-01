"""
GMN-Football-3 — MAPPO (Multi-Agent PPO) Networks
Shared Actor with Centralized Critic Architecture for Cooperative Multi-Agent Play.

Architecture:
- SharedActor: Parameter-shared categorical policy across all controllable agents.
- CentralizedCritic: Global value function estimating team return from joint state.
Deliberately matches SB3's default net_arch=[64, 64] with Tanh activations for direct parity.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class SharedActor(nn.Module):
    def __init__(self, obs_dim: int = 127, action_dim: int = 19, hidden: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)


class CentralizedCritic(nn.Module):
    """
    Scalable Permutation-Invariant Centralized Critic (Deep Sets Architecture).
    Supports variable agent counts (3v1, 5v5, 11v11) with constant parameter count O(1).
    
    Architecture:
    1. Agent Encoder: MLP mapping each agent's observation vector (obs_dim=127) -> hidden representation (64).
    2. Deep Sets Pooling: Permutation-invariant dual (mean + max) pooling over agents -> (128).
    3. Joint Value Head: MLP mapping pooled representation -> scalar team state-value V(s).
    """
    def __init__(self, obs_dim: int = 127, hidden: int = 64, global_state_dim: int = None):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.global_state_dim = global_state_dim or (obs_dim * 3)

        # Scalable permutation-invariant set aggregation architecture (default)
        self.agent_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pooled_value_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # Legacy flat MLP net (retained for backward compatibility)
        self.flat_net = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Accepts:
        - 3D tensor of shape (batch, num_agents, obs_dim) -> runs Deep Sets invariant pooling
        - 2D tensor of shape (batch, num_agents * obs_dim) -> reshapes and runs invariant pooling
        - 2D tensor of shape (batch, global_state_dim) -> fallback to flat net if pooling shape differs
        """
        if x.dim() == 3:
            emb = self.agent_encoder(x)
            mean_pool = emb.mean(dim=1)
            max_pool = emb.max(dim=1)[0]
            joint = torch.cat([mean_pool, max_pool], dim=-1)
            return self.pooled_value_head(joint).squeeze(-1)

        # If 2D tensor divisible by obs_dim, reshape to (batch, num_agents, obs_dim) and use invariant pooling
        if x.dim() == 2 and x.shape[-1] % self.obs_dim == 0:
            b = x.shape[0]
            reshaped = x.view(b, -1, self.obs_dim)
            emb = self.agent_encoder(reshaped)
            mean_pool = emb.mean(dim=1)
            max_pool = emb.max(dim=1)[0]
            joint = torch.cat([mean_pool, max_pool], dim=-1)
            return self.pooled_value_head(joint).squeeze(-1)

        # Fallback to flat net
        return self.flat_net(x).squeeze(-1)
