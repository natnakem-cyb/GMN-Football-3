import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_gym import GMNFootballEnv
from stable_baselines3 import PPO


def evaluate_checkpoint(model_filename: str = "ppo_academy_empty_goal_smoke.zip", num_episodes: int = 100):
    print("==================================================")
    print("6. PPO EVALUATION PROCEDURE")
    print(f"Model: {model_filename} | Episodes: {num_episodes}")
    print("==================================================")

    model_path = os.path.join(os.path.dirname(__file__), "models", model_filename)
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint {model_path} not found.")
        return False

    env = GMNFootballEnv(scenario="academy_empty_goal", port=5050, use_ws=True)
    model = PPO.load(model_path)

    rewards = []
    lengths = []
    goals = 0
    final_distances = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=2000 + ep)
        ep_rew = 0.0
        ep_len = 0
        done = False

        while not done and ep_len < 900:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            ep_len += 1
            done = terminated or truncated

        rewards.append(ep_rew)
        lengths.append(ep_len)
        final_distances.append(info.get("ballDistanceToGoal", 0.5))
        if info.get("score", {}).get("left", 0) > 0 or info.get("event") == "GOAL":
            goals += 1

    mean_rew = float(np.mean(rewards))
    std_rew = float(np.std(rewards))
    success_rate = (goals / num_episodes) * 100.0
    mean_len = float(np.mean(lengths))
    mean_dist = float(np.mean(final_distances))

    print(f"Results across {num_episodes} evaluation episodes (deterministic=True):")
    print(f"- Success Rate: {success_rate:.1f}% ({goals}/{num_episodes} goals)")
    print(f"- Mean Episode Reward: {mean_rew:.4f}")
    print(f"- Reward Std Deviation: {std_rew:.4f}")
    print(f"- Mean Episode Length: {mean_len:.1f} steps")
    print(f"- Mean Final Ball Distance: {mean_dist:.4f}")
    print("==================================================\n")

    env.close()
    return True


if __name__ == "__main__":
    fn = sys.argv[1] if len(sys.argv) > 1 else "ppo_academy_empty_goal_smoke.zip"
    evaluate_checkpoint(fn, 100)
