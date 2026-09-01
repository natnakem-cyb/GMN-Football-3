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
    def __init__(self, obs_dim: int = 115, action_dim: int = 19, hidden: int = 64):
        super().__init__()
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
    Scalable Permutation-Invariant Centralized Critic.
    Supports both fixed concatenated global states and dynamic variable-agent pooling representations.
    
    Architecture:
    1. Direct Flat Mode: Standard MLP mapping fixed-dimension concatenated agent observations to team value.
    2. Deep Sets Pooling Mode: Permutation-invariant agent encoder + dual (mean, max) pooling head that
       scales seamlessly across arbitrary team sizes (3v1, 5v5, 11v11) without parameter explosion.
    """
    def __init__(self, global_state_dim: int = 345, obs_dim: int = 115, hidden: int = 64):
        super().__init__()
        self.global_state_dim = global_state_dim
        self.obs_dim = obs_dim
        self.hidden = hidden

        # Flat MLP net (direct SB3 parity & backward compatibility)
        self.flat_net = nn.Sequential(
            nn.Linear(global_state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # Scalable permutation-invariant set aggregation architecture
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

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        # If 3D tensor (batch, num_agents, obs_dim) -> use invariant pooling
        if global_state.dim() == 3:
            emb = self.agent_encoder(global_state)
            mean_pool = emb.mean(dim=1)
            max_pool = emb.max(dim=1)[0]
            joint = torch.cat([mean_pool, max_pool], dim=-1)
            return self.pooled_value_head(joint).squeeze(-1)

        # If 2D tensor matching exact flat dimension
        if global_state.shape[-1] == self.global_state_dim:
            return self.flat_net(global_state).squeeze(-1)

        # If 2D tensor of arbitrary size divisible by obs_dim, reshape and pool
        if global_state.shape[-1] % self.obs_dim == 0:
            b = global_state.shape[0]
            reshaped = global_state.view(b, -1, self.obs_dim)
            emb = self.agent_encoder(reshaped)
            mean_pool = emb.mean(dim=1)
            max_pool = emb.max(dim=1)[0]
            joint = torch.cat([mean_pool, max_pool], dim=-1)
            return self.pooled_value_head(joint).squeeze(-1)

        return self.flat_net(global_state).squeeze(-1)
