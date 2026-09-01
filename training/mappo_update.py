"""
GMN-Football-3 — MAPPO (Multi-Agent PPO) Update Step
Computes clipped surrogate policy loss, centralized value loss, and entropy regularization.
Performs mini-batch gradient descent for shared actor and centralized critic.
"""

from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn

from mappo_networks import SharedActor, CentralizedCritic


def ppo_update(
    actor: SharedActor,
    critic: CentralizedCritic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    buffer: Dict[str, np.ndarray],
    advantages: np.ndarray,
    returns: np.ndarray,
    clip_range: float = 0.2,
    n_epochs: int = 4,
    batch_size: int = 64,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> Dict[str, float]:
    """
    Performs PPO policy and value updates for MAPPO.

    Args:
        actor: Shared policy network
        critic: Centralized critic network
        actor_opt: Optimizer for actor
        critic_opt: Optimizer for critic
        buffer: Rollout trajectory dictionary
        advantages: GAE advantages array of shape (T,)
        returns: GAE returns array of shape (T,)
        clip_range: PPO clipping epsilon (default 0.2)
        n_epochs: Optimization epochs per rollout (default 4)
        batch_size: Mini-batch sample size (default 64)
        value_coef: Critic loss weight (default 0.5)
        entropy_coef: Policy entropy bonus weight (default 0.01)
        max_grad_norm: Maximum gradient norm clipping (default 0.5)

    Returns:
        Dictionary of mean loss and diagnostic metrics across mini-batches.
    """
    T, num_agents, obs_dim = buffer["local_obs"].shape

    # Flatten (T, num_agents, ...) -> (T*num_agents, ...), agent-major within
    # each timestep — matches collect_rollout's storage order exactly.
    flat_obs = buffer["local_obs"].reshape(T * num_agents, obs_dim)
    flat_actions = buffer["actions"].reshape(T * num_agents)
    flat_old_logprobs = buffer["logprobs"].reshape(T * num_agents)

    # Broadcast the shared per-timestep advantage/return to every agent at
    # that timestep — np.repeat's default ordering matches the reshape above.
    flat_advantages = np.repeat(advantages, num_agents)
    flat_returns = np.repeat(returns, num_agents)

    # Normalize advantages — standard PPO practice
    adv_std = flat_advantages.std()
    adv_mean = flat_advantages.mean()
    flat_advantages = (flat_advantages - adv_mean) / (adv_std + 1e-8)

    # Convert arrays to tensors (zero-copy when possible)
    obs_t = torch.from_numpy(flat_obs).float()
    actions_t = torch.from_numpy(flat_actions).long()
    old_logprobs_t = torch.from_numpy(flat_old_logprobs).float()
    advantages_t = torch.from_numpy(flat_advantages).float()
    returns_t = torch.from_numpy(flat_returns).float()
    
    # 3D joint observations for scalable permutation-invariant critic (Deep Sets)
    joint_obs_repeated = np.repeat(buffer["local_obs"], num_agents, axis=0)
    joint_obs_t = torch.from_numpy(joint_obs_repeated).float()

    n_samples = T * num_agents
    metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}

    for _ in range(n_epochs):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start : start + batch_size]

            dist = actor(obs_t[batch_idx])
            new_logprobs = dist.log_prob(actions_t[batch_idx])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logprobs - old_logprobs_t[batch_idx])
            surr1 = ratio * advantages_t[batch_idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages_t[batch_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            # Pass 3D tensor (batch_size, num_agents, obs_dim) to CentralizedCritic
            values_pred = critic(joint_obs_t[batch_idx])
            value_loss = ((values_pred - returns_t[batch_idx]) ** 2).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            actor_opt.zero_grad()
            critic_opt.zero_grad()
            loss.backward()

            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)

            actor_opt.step()
            critic_opt.step()

            with torch.no_grad():
                approx_kl = (old_logprobs_t[batch_idx] - new_logprobs).mean().item()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(entropy.item())
            metrics["approx_kl"].append(approx_kl)

    return {k: float(np.mean(v)) for k, v in metrics.items()}
