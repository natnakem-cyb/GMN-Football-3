"""
GMN-Football-3 — Multi-Agent Policy Evaluation Script
Evaluates trained MAPPO model against the Multi-Agent environment.
"""

import argparse
import os
import sys
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor


def evaluate_mappo(
    checkpoint_path: str = "training/models/mappo_academy_3_vs_1_with_keeper_trained.pt",
    scenario: str = "academy_3_vs_1_with_keeper",
    num_episodes: int = 50,
    deterministic: bool = True,
):
    print("==================================================")
    print("GMN-FOOTBALL-3 — MAPPO EVALUATION RUNNER")
    print(f"Model Checkpoint: {checkpoint_path}")
    print(f"Scenario: {scenario} | Episodes: {num_episodes} | Deterministic: {deterministic}")
    print("==================================================")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    actor = SharedActor(obs_dim=115, action_dim=19, hidden=64)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = GMNMultiAgentEnv(scenario=scenario, auto_start_bridge=True)
    controllable_agents = list(env.possible_agents)

    rewards_list = []
    lengths_list = []
    goals_list = []

    for ep in range(1, num_episodes + 1):
        obs_dict, _ = env.reset()
        ep_reward = 0.0
        ep_length = 0
        goal_scored = 0

        while True:
            current_agents = list(env.agents if env.agents else controllable_agents)
            local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)

            with torch.no_grad():
                dist = actor(torch.from_numpy(local_obs).float())
                if deterministic:
                    # Argmax over action logits
                    actions = dist.logits.argmax(dim=-1)
                else:
                    actions = dist.sample()

            action_dict = {a: int(actions[i].item()) for i, a in enumerate(current_agents)}
            obs_dict, rewards, terminations, truncations, infos = env.step(action_dict)

            shared_rew = float(rewards[current_agents[0]])
            ep_reward += shared_rew
            ep_length += 1

            done = any(terminations.values()) or any(truncations.values())
            if done:
                for agent_info in infos.values():
                    if agent_info.get("score", {}).get("left", 0) > 0:
                        goal_scored = 1
                        break
                break

        rewards_list.append(ep_reward)
        lengths_list.append(ep_length)
        goals_list.append(goal_scored)

        if ep % 10 == 0 or ep == num_episodes:
            print(
                f"   [Episode {ep:3d}/{num_episodes}] Mean Reward: {np.mean(rewards_list):+.4f} | "
                f"Goal Rate: {np.mean(goals_list)*100.0:5.1f}% | "
                f"Mean Length: {np.mean(lengths_list):.1f}"
            )

    mean_rew = float(np.mean(rewards_list))
    std_rew = float(np.std(rewards_list))
    goal_rate = float(np.mean(goals_list)) * 100.0
    mean_len = float(np.mean(lengths_list))

    print("\n==================================================")
    print("MAPPO EVALUATION RESULTS SUMMARY")
    print("==================================================")
    print(f"Total Episodes Evaluated : {num_episodes}")
    print(f"Mean Episode Reward     : {mean_rew:+.4f} ± {std_rew:.4f}")
    print(f"Goal Conversion Rate    : {goal_rate:.1f}% ({sum(goals_list)}/{num_episodes} goals)")
    print(f"Mean Episode Length     : {mean_len:.1f} steps")
    print("==================================================")

    return {
        "mean_reward": mean_rew,
        "std_reward": std_rew,
        "goal_rate": goal_rate,
        "mean_length": mean_len,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="training/models/mappo_academy_3_vs_1_with_keeper_trained.pt")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    evaluate_mappo(
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
    )
