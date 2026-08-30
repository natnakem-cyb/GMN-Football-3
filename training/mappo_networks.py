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
    def __init__(self, global_state_dim: int = 345, hidden: int = 64):  # 345 = 115 * 3 agents
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)
