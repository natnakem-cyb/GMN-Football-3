"""
GMN-Football-3 — Modular Feature Encoders & Recurrent Memory (TiZero Architecture)
Decomposes the 127-dimensional simple115_v3_role observation into distinct entity channels:
- Ego Agent Sub-MLP (kinematics, relative ball vector, active index, assigned role)
- Ball State Sub-MLP (position, velocity, speed, possession)
- Teammates Permutation-Invariant Sub-MLP (Deep Sets mean+max pooling)
- Opponents Permutation-Invariant Sub-MLP (Deep Sets mean+max pooling)
- Match State Sub-MLP (game mode one-hot)
- Recurrent Core (GRU / LSTM) for temporal sequence reasoning
- Actor & Centralized Critic Heads
"""

from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
from torch.distributions import Categorical

from checkpoint_contract import (
    OBSERVATION_DIM,
    ACTION_SPACE_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    ROLE_DIM,
    BASE_OBSERVATION_DIM,
)

# Observation tensor slice offsets for simple115_v3_role (total 127 floats)
SLICE_LEFT_POS = slice(0, 22)        # 11 players x (x, y)
SLICE_LEFT_VEL = slice(22, 44)       # 11 players x (vx, vy)
SLICE_RIGHT_POS = slice(44, 66)      # 11 players x (x, y)
SLICE_RIGHT_VEL = slice(66, 88)      # 11 players x (vx, vy)
SLICE_BALL_POS = slice(88, 91)       # (x, y, z)
SLICE_BALL_VEL = slice(91, 94)       # (vx, vy, vz)
SLICE_BALL_OWNER = slice(94, 97)     # one-hot: [none, left, right]
SLICE_ACTIVE_IDX = slice(97, 108)    # one-hot over 11 players
SLICE_GAME_MODE = slice(108, 115)    # one-hot over 7 game modes
SLICE_ROLE = slice(115, 127)         # one-hot over 12 roles


class ModularFootballEncoder(nn.Module):
    """
    Modular Entity Feature Extractor for GMN-Football simple115_v3_role observation.
    Separates the flat 127-dim vector into entity streams and aggregates them with
    permutation-invariant pooling.
    """

    def __init__(
        self,
        ego_dim: int = 64,
        ball_dim: int = 32,
        team_dim: int = 64,
        opp_dim: int = 64,
        match_dim: int = 16,
        output_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = OBSERVATION_DIM
        self.output_dim = output_dim

        # 1. Ego Agent Encoder
        # Ego features: [pos_x, pos_y, vel_x, vel_y, rel_ball_x, rel_ball_y, rel_ball_dist, active_idx(11), role(12)] = 30
        self.ego_net = nn.Sequential(
            nn.Linear(30, ego_dim),
            nn.LayerNorm(ego_dim),
            nn.ReLU(),
            nn.Linear(ego_dim, ego_dim),
            nn.ReLU(),
        )

        # 2. Ball State Encoder
        # Ball features: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, speed, owner(3)] = 10
        self.ball_net = nn.Sequential(
            nn.Linear(10, ball_dim),
            nn.LayerNorm(ball_dim),
            nn.ReLU(),
            nn.Linear(ball_dim, ball_dim),
            nn.ReLU(),
        )

        # 3. Teammates Sub-MLP (per teammate)
        # Teammate features: [x, y, vx, vy, rel_x, rel_y, dist, is_ego, is_present] = 9
        self.teammate_mlp = nn.Sequential(
            nn.Linear(9, team_dim // 2),
            nn.LayerNorm(team_dim // 2),
            nn.ReLU(),
            nn.Linear(team_dim // 2, team_dim // 2),
            nn.ReLU(),
        )

        # 4. Opponents Sub-MLP (per opponent)
        # Opponent features: [x, y, vx, vy, rel_x, rel_y, dist, is_present] = 8
        self.opponent_mlp = nn.Sequential(
            nn.Linear(8, opp_dim // 2),
            nn.LayerNorm(opp_dim // 2),
            nn.ReLU(),
            nn.Linear(opp_dim // 2, opp_dim // 2),
            nn.ReLU(),
        )

        # 5. Match State Sub-MLP
        # Match features: game_mode one-hot = 7
        self.match_net = nn.Sequential(
            nn.Linear(7, match_dim),
            nn.LayerNorm(match_dim),
            nn.ReLU(),
        )

        # Total entity concatenation dimension: 64 + 32 + 64 + 64 + 16 = 240
        fused_in_dim = ego_dim + ball_dim + team_dim + opp_dim + match_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_in_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract modular entity features from raw observation tensor.
        Supports inputs of shape (batch, 127) or (batch, seq_len, 127).
        """
        orig_shape = obs.shape
        if obs.dim() > 2:
            # Flatten leading dimensions: (batch, seq_len, 127) -> (batch * seq_len, 127)
            batch_prefix = orig_shape[:-1]
            obs = obs.reshape(-1, self.obs_dim)
        else:
            batch_prefix = None

        batch_size = obs.shape[0]

        # Slicing the 127-float observation vector
        left_pos = obs[:, SLICE_LEFT_POS].view(batch_size, 11, 2)
        left_vel = obs[:, SLICE_LEFT_VEL].view(batch_size, 11, 2)
        right_pos = obs[:, SLICE_RIGHT_POS].view(batch_size, 11, 2)
        right_vel = obs[:, SLICE_RIGHT_VEL].view(batch_size, 11, 2)
        ball_pos = obs[:, SLICE_BALL_POS]      # (batch, 3)
        ball_vel = obs[:, SLICE_BALL_VEL]      # (batch, 3)
        ball_owner = obs[:, SLICE_BALL_OWNER]  # (batch, 3)
        active_idx = obs[:, SLICE_ACTIVE_IDX]  # (batch, 11)
        game_mode = obs[:, SLICE_GAME_MODE]    # (batch, 7)
        role = obs[:, SLICE_ROLE]              # (batch, 12)

        # --- 1. Ego Agent Feature Extraction ---
        # Ego position & velocity: weighted combination via one-hot active_idx
        active_weights = active_idx.unsqueeze(-1)  # (batch, 11, 1)
        ego_pos = (left_pos * active_weights).sum(dim=1)  # (batch, 2)
        ego_vel = (left_vel * active_weights).sum(dim=1)  # (batch, 2)

        # Relative ball kinematics from ego agent
        rel_ball_xy = ball_pos[:, :2] - ego_pos  # (batch, 2)
        rel_ball_dist = torch.norm(rel_ball_xy, dim=-1, keepdim=True)  # (batch, 1)

        ego_features = torch.cat(
            [ego_pos, ego_vel, rel_ball_xy, rel_ball_dist, active_idx, role], dim=-1
        )  # (batch, 30)
        ego_emb = self.ego_net(ego_features)  # (batch, ego_dim)

        # --- 2. Ball Feature Extraction ---
        ball_speed = torch.norm(ball_vel, dim=-1, keepdim=True)  # (batch, 1)
        ball_features = torch.cat([ball_pos, ball_vel, ball_speed, ball_owner], dim=-1)  # (batch, 10)
        ball_emb = self.ball_net(ball_features)  # (batch, ball_dim)

        # --- 3. Teammates Feature Extraction & Deep Sets Pooling ---
        # Mask inactive player slots (padded as -1.0, -1.0)
        teammate_present = (left_pos[:, :, 0] > -0.999).float().unsqueeze(-1)  # (batch, 11, 1)
        ego_pos_expanded = ego_pos.unsqueeze(1).expand(-1, 11, -1)  # (batch, 11, 2)

        rel_team_pos = left_pos - ego_pos_expanded  # (batch, 11, 2)
        team_dist = torch.norm(rel_team_pos, dim=-1, keepdim=True)  # (batch, 11, 1)
        is_ego = active_idx.unsqueeze(-1)  # (batch, 11, 1)

        teammate_raw = torch.cat(
            [left_pos, left_vel, rel_team_pos, team_dist, is_ego, teammate_present], dim=-1
        )  # (batch, 11, 9)

        # Only pass present teammates to pooling; zero-out ego from teammate pool
        teammate_embs = self.teammate_mlp(teammate_raw)  # (batch, 11, team_dim // 2)
        teammate_mask = teammate_present * (1.0 - is_ego)  # (batch, 11, 1)
        masked_teammate_embs = teammate_embs * teammate_mask

        # Masked mean and max pooling
        team_count = teammate_mask.sum(dim=1).clamp(min=1.0)  # (batch, 1)
        team_mean = masked_teammate_embs.sum(dim=1) / team_count  # (batch, team_dim // 2)
        
        # Max pool: replace masked elements with large negative before max
        neg_inf_mask = (1.0 - teammate_mask) * -1e9
        team_max = (teammate_embs + neg_inf_mask).max(dim=1)[0]
        team_max = torch.where(team_count > 0, team_max, torch.zeros_like(team_max))
        team_emb = torch.cat([team_mean, team_max], dim=-1)  # (batch, team_dim)

        # --- 4. Opponents Feature Extraction & Deep Sets Pooling ---
        opp_present = (right_pos[:, :, 0] > -0.999).float().unsqueeze(-1)  # (batch, 11, 1)
        rel_opp_pos = right_pos - ego_pos_expanded  # (batch, 11, 2)
        opp_dist = torch.norm(rel_opp_pos, dim=-1, keepdim=True)  # (batch, 11, 1)

        opp_raw = torch.cat(
            [right_pos, right_vel, rel_opp_pos, opp_dist, opp_present], dim=-1
        )  # (batch, 11, 8)
        opp_embs = self.opponent_mlp(opp_raw)  # (batch, 11, opp_dim // 2)
        masked_opp_embs = opp_embs * opp_present

        opp_count = opp_present.sum(dim=1).clamp(min=1.0)
        opp_mean = masked_opp_embs.sum(dim=1) / opp_count
        opp_neg_inf_mask = (1.0 - opp_present) * -1e9
        opp_max = (opp_embs + opp_neg_inf_mask).max(dim=1)[0]
        opp_max = torch.where(opp_count > 0, opp_max, torch.zeros_like(opp_max))
        opp_emb = torch.cat([opp_mean, opp_max], dim=-1)  # (batch, opp_dim)

        # --- 5. Match State Feature Extraction ---
        match_emb = self.match_net(game_mode)  # (batch, match_dim)

        # --- 6. Multimodal Entity Fusion ---
        fused = torch.cat([ego_emb, ball_emb, team_emb, opp_emb, match_emb], dim=-1)
        out = self.fusion(fused)  # (batch, output_dim)

        if batch_prefix is not None:
            out = out.view(*batch_prefix, self.output_dim)

        return out


class ActorCriticRNN(nn.Module):
    """
    Recurrent Actor-Critic Network with Modular Entity Encoding (TiZero Architecture).
    Integrates ModularFootballEncoder with a GRU temporal recurrent core for POMDP reasoning,
    producing categorical action distributions and value baselines.
    """

    def __init__(
        self,
        obs_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_SPACE_SIZE,
        hidden_dim: int = 128,
        rnn_type: str = "gru",
        num_rnn_layers: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()
        self.num_rnn_layers = num_rnn_layers

        # Modular entity feature extractor
        self.encoder = ModularFootballEncoder(
            ego_dim=64,
            ball_dim=32,
            team_dim=64,
            opp_dim=64,
            match_dim=16,
            output_dim=hidden_dim,
        )

        # Recurrent Core
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_rnn_layers,
                batch_first=True,
            )
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_rnn_layers,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Must be 'gru' or 'lstm'.")

        # Policy & Value Heads
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def init_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Initializes zero-filled recurrent state."""
        h0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim, device=device)
        if self.rnn_type == "lstm":
            c0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim, device=device)
            return (h0, c0)
        return h0

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[Categorical, torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        Args:
            obs: Observation tensor of shape (batch, 127) or (batch, seq_len, 127)
            hidden_state: Recurrent hidden state. If None, initialized to zeros.
        Returns:
            (action_distribution, value_estimate, next_hidden_state)
        """
        is_single_step = (obs.dim() == 2)
        if is_single_step:
            # (batch, 127) -> (batch, 1, 127)
            obs = obs.unsqueeze(1)

        batch_size, seq_len, _ = obs.shape
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, device=obs.device)

        # 1. Modular Entity Feature Extraction
        encoded = self.encoder(obs)  # (batch, seq_len, hidden_dim)

        # 2. Recurrent Temporal Reasoning
        rnn_out, next_hidden = self.rnn(encoded, hidden_state)  # (batch, seq_len, hidden_dim)

        # 3. Heads
        logits = self.actor_head(rnn_out)  # (batch, seq_len, 19)
        values = self.critic_head(rnn_out).squeeze(-1)  # (batch, seq_len)

        if is_single_step:
            logits = logits.squeeze(1)
            values = values.squeeze(1)

        dist = Categorical(logits=logits)
        return dist, values, next_hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Evaluates log-probabilities, state values, and entropy for PPO / JRPO updates.
        """
        dist, values, next_hidden = self.forward(obs, hidden_state)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy, next_hidden
