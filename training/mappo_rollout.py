"""
GMN-Football-3 — MAPPO Rollout Collection & Generalized Advantage Estimation (GAE)
Data pipeline for Multi-Agent PPO with Centralized Critic.

Collects trajectory buffers with local observations for shared actors and
concatenated joint observations for the centralized critic.
"""

from typing import Dict, Tuple, Any
import numpy as np
import torch

from mappo_networks import SharedActor, CentralizedCritic


def collect_rollout(
    env: Any,
    actor: SharedActor,
    critic: CentralizedCritic,
    num_steps: int = 256,
) -> Dict[str, Any]:
    """
    Collects a multi-agent rollout from the PettingZoo environment.

    Returns a dict of numpy arrays with explicit shapes:
    - local_obs: shape (num_steps, num_agents, 115) float32
    - global_state: shape (num_steps, 345) float32
    - actions: shape (num_steps, num_agents) int64
    - logprobs: shape (num_steps, num_agents) float32
    - values: shape (num_steps,) float32
    - rewards: shape (num_steps,) float32
    - dones: shape (num_steps,) bool
    - completed_episodes: list of dicts with {"reward": float, "length": int, "goal": int}
    """
    buffer: Dict[str, list] = {
        "local_obs": [],      # per-agent, shape (num_steps, num_agents, 115)
        "global_state": [],   # shape (num_steps, 345)
        "actions": [],        # shape (num_steps, num_agents)
        "logprobs": [],       # shape (num_steps, num_agents)
        "values": [],         # shape (num_steps,) — one shared value per step
        "rewards": [],        # shape (num_steps,) — shared team reward
        "dones": [],          # shape (num_steps,)
    }
    completed_episodes = []

    # Retrieve or initialize persistent rollout state on env
    if not hasattr(env, "_mappo_obs") or env._mappo_obs is None:
        obs_dict, _ = env.reset()
        env._mappo_obs = obs_dict
        env._mappo_ep_rew = 0.0
        env._mappo_ep_len = 0
    else:
        obs_dict = env._mappo_obs

    agent_order = list(env.agents if env.agents else env.possible_agents)
    num_agents = len(agent_order)

    for _ in range(num_steps):
        current_agents = list(env.agents if env.agents else env.possible_agents)
        local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
        global_state = local_obs.flatten().astype(np.float32)  # concat in agent_order

        # Verify concatenation integrity
        for idx, agent in enumerate(current_agents):
            start = idx * 115
            end = (idx + 1) * 115
            assert np.array_equal(global_state[start:end], local_obs[idx]), (
                f"Global state slice [{start}:{end}] does not match local_obs[{idx}] for agent {agent}"
            )

        with torch.no_grad():
            dist = actor(torch.tensor(local_obs, dtype=torch.float32))
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            value = critic(torch.tensor(global_state, dtype=torch.float32).unsqueeze(0))

        action_dict = {a: int(actions[i].item()) for i, a in enumerate(current_agents)}
        obs_dict, rewards, terminations, truncations, infos = env.step(action_dict)

        done = any(terminations.values()) or any(truncations.values())
        shared_reward = float(rewards[current_agents[0]])  # identical across agents

        env._mappo_ep_rew += shared_reward
        env._mappo_ep_len += 1

        buffer["local_obs"].append(local_obs)
        buffer["global_state"].append(global_state)
        buffer["actions"].append(actions.cpu().numpy())
        buffer["logprobs"].append(logprobs.cpu().numpy())
        buffer["values"].append(float(value.item()))
        buffer["rewards"].append(shared_reward)
        buffer["dones"].append(bool(done))

        if done:
            # Check if left team scored a goal in the terminal step
            goal_scored = 0
            for agent_info in infos.values():
                if agent_info.get("score", {}).get("left", 0) > 0:
                    goal_scored = 1
                    break

            completed_episodes.append({
                "reward": float(env._mappo_ep_rew),
                "length": int(env._mappo_ep_len),
                "goal": goal_scored,
            })
            env._mappo_ep_rew = 0.0
            env._mappo_ep_len = 0
            obs_dict, _ = env.reset()

    env._mappo_obs = obs_dict

    # Convert to structured numpy arrays
    res_buffer = {
        "local_obs": np.array(buffer["local_obs"], dtype=np.float32),
        "global_state": np.array(buffer["global_state"], dtype=np.float32),
        "actions": np.array(buffer["actions"], dtype=np.int64),
        "logprobs": np.array(buffer["logprobs"], dtype=np.float32),
        "values": np.array(buffer["values"], dtype=np.float32),
        "rewards": np.array(buffer["rewards"], dtype=np.float32),
        "dones": np.array(buffer["dones"], dtype=np.bool_),
        "completed_episodes": completed_episodes,
    }

    # Explicit shape assertions
    assert res_buffer["local_obs"].shape == (num_steps, num_agents, 115), (
        f"local_obs shape mismatch: expected {(num_steps, num_agents, 115)}, got {res_buffer['local_obs'].shape}"
    )
    assert res_buffer["global_state"].shape == (num_steps, 115 * num_agents), (
        f"global_state shape mismatch: expected {(num_steps, 115 * num_agents)}, got {res_buffer['global_state'].shape}"
    )
    assert res_buffer["actions"].shape == (num_steps, num_agents), (
        f"actions shape mismatch: expected {(num_steps, num_agents)}, got {res_buffer['actions'].shape}"
    )
    assert res_buffer["logprobs"].shape == (num_steps, num_agents), (
        f"logprobs shape mismatch: expected {(num_steps, num_agents)}, got {res_buffer['logprobs'].shape}"
    )
    assert res_buffer["values"].shape == (num_steps,), (
        f"values shape mismatch: expected {(num_steps,)}, got {res_buffer['values'].shape}"
    )
    assert res_buffer["rewards"].shape == (num_steps,), (
        f"rewards shape mismatch: expected {(num_steps,)}, got {res_buffer['rewards'].shape}"
    )
    assert res_buffer["dones"].shape == (num_steps,), (
        f"dones shape mismatch: expected {(num_steps,)}, got {res_buffer['dones'].shape}"
    )

    return res_buffer


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
    bootstrap_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Generalized Advantage Estimation (GAE) and Returns backwards over the rollout.

    Args:
        rewards: shape (T,)
        values: shape (T,)
        dones: shape (T,)
        gamma: discount factor (default 0.99)
        lam: GAE lambda parameter (default 0.95)
        bootstrap_value: value estimate of state at T (default 0.0)

    Returns:
        advantages: shape (T,)
        returns: shape (T,)
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = bootstrap_value if t == len(rewards) - 1 else values[t + 1]
        next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns
