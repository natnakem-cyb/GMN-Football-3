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
    def __init__(
        self,
        obs_dim: int = 127,
        hidden: int = 64,
        mode: str = "pool",
        global_state_dim: int = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.mode = mode
        self.global_state_dim = global_state_dim or (obs_dim * 3)

        if self.mode == "pool":
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
            self.flat_net = None
        else:
            # Legacy flat MLP net (retained for explicit flat mode)
            self.agent_encoder = None
            self.pooled_value_head = None
            self.flat_net = nn.Sequential(
                nn.Linear(self.global_state_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        - In 'pool' mode (default):
            - 3D tensor of shape (batch, num_agents, obs_dim): invariant Deep Sets pooling across agents.
            - 2D tensor of shape (num_agents, obs_dim): unbatched input; unsqueezed to (1, num_agents, obs_dim).
            - 2D tensor of shape (batch, num_agents * obs_dim): reshapes to 3D and pools.
        - In 'flat' mode:
            - Passes through flat_net.
        """
        if self.mode == "flat":
            if self.flat_net is None:
                raise RuntimeError("CentralizedCritic initialized in pool mode cannot run flat forward pass")
            return self.flat_net(x).squeeze(-1)

        # 'pool' mode
        if x.dim() == 2:
            if x.shape[-1] == self.obs_dim:
                # Unbatched (num_agents, obs_dim) -> (1, num_agents, obs_dim)
                x = x.unsqueeze(0)
            elif x.shape[-1] % self.obs_dim == 0:
                # Batched flat representation (batch, num_agents * obs_dim) -> reshape to 3D
                b = x.shape[0]
                x = x.view(b, -1, self.obs_dim)
            else:
                raise ValueError(
                    f"Expected 2D input with last dim {self.obs_dim} or divisible by {self.obs_dim}, got shape {x.shape}"
                )

        emb = self.agent_encoder(x)  # (batch, num_agents, hidden)
        mean_pool = emb.mean(dim=1)  # (batch, hidden)
        max_pool = emb.max(dim=1)[0]  # (batch, hidden)
        joint = torch.cat([mean_pool, max_pool], dim=-1)  # (batch, hidden * 2)
        return self.pooled_value_head(joint).squeeze(-1)

