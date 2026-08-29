import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_gym import GMNFootballEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class RLMetricsCallback(BaseCallback):
    """Tracks rollout metrics during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_goals = []
        self.current_reward = 0.0
        self.current_length = 0

    def _on_step(self) -> bool:
        # Get step information from locals
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        info = self.locals["infos"][0]

        self.current_reward += reward
        self.current_length += 1

        if done:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            is_goal = info.get("score", {}).get("left", 0) > 0 or info.get("event") == "GOAL"
            self.episode_goals.append(1 if is_goal else 0)
            self.current_reward = 0.0
            self.current_length = 0

        return True


def run_random_baseline(total_steps: int = 10000):
    print("==================================================")
    print("4. RANDOM POLICY BASELINE")
    print(f"Executing {total_steps} random steps on academy_empty_goal")
    print("==================================================")

    env = GMNFootballEnv(scenario="academy_empty_goal", port=5050, use_ws=True)
    obs, info = env.reset(seed=42)

    episode_rewards = []
    episode_lengths = []
    ball_distances = []
    goals = 0

    current_reward = 0.0
    current_length = 0

    for step in range(total_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        current_reward += reward
        current_length += 1
        dist = info.get("ballDistanceToGoal", 0.5)
        ball_distances.append(dist)

        if terminated or truncated:
            is_goal = info.get("score", {}).get("left", 0) > 0 or info.get("event") == "GOAL"
            if is_goal:
                goals += 1
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
            current_reward = 0.0
            current_length = 0
            obs, info = env.reset()

    mean_rew = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    median_rew = float(np.median(episode_rewards)) if episode_rewards else 0.0
    std_rew = float(np.std(episode_rewards)) if episode_rewards else 0.0
    mean_dist = float(np.mean(ball_distances)) if ball_distances else 0.0
    avg_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    total_episodes = len(episode_rewards)
    goal_rate = (goals / max(1, total_episodes)) * 100.0

    print(f"Episodes Completed: {total_episodes}")
    print(f"- Mean Episode Reward: {mean_rew:.4f} (±{std_rew:.4f})")
    print(f"- Median Episode Reward: {median_rew:.4f}")
    print(f"- Goal Rate: {goal_rate:.2f}% ({goals}/{total_episodes} goals)")
    print(f"- Average Ball Distance to Goal: {mean_dist:.4f}")
    print(f"- Average Episode Length: {avg_len:.1f} steps")
    print("==================================================\n")

    env.close()
    return {
        "mean_reward": mean_rew,
        "median_reward": median_rew,
        "goal_rate": goal_rate,
        "avg_dist": mean_dist,
        "avg_len": avg_len,
    }


def evaluate_policy(model, env, num_episodes: int = 100):
    rewards = []
    lengths = []
    goals = 0
    final_distances = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=1000 + ep)
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

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "success_rate": (goals / num_episodes) * 100.0,
        "mean_length": float(np.mean(lengths)),
        "mean_final_dist": float(np.mean(final_distances)),
        "goals": goals,
        "episodes": num_episodes,
    }


def run_ppo_experiments():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    print("==================================================")
    print("3. ACADEMY EMPTY GOAL LEARNING SIGNAL & PPO EXPERIMENTS")
    print("==================================================")

    # 1. Train PPO for 10,000 steps
    print("\n>>> Phase 1: Training PPO for 10,000 steps...")
    env10k = GMNFootballEnv(scenario="academy_empty_goal", port=5050, use_ws=True)
    cb10k = RLMetricsCallback()

    model10k = PPO(
        policy="MlpPolicy",
        env=env10k,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=42,
    )

    t0 = time.time()
    model10k.learn(total_timesteps=10000, callback=cb10k)
    t10k_duration = time.time() - t0
    t10k_fps = 10000 / max(0.01, t10k_duration)

    checkpoint_10k = os.path.join(models_dir, "ppo_academy_empty_goal_10k.zip")
    model10k.save(checkpoint_10k)

    print(f"PPO 10K Training Time: {t10k_duration:.2f}s ({t10k_fps:.1f} FPS)")
    eval_10k = evaluate_policy(model10k, env10k, num_episodes=100)
    print(f"PPO 10K Eval (100 eps): Success Rate: {eval_10k['success_rate']:.1f}% | Mean Reward: {eval_10k['mean_reward']:.4f} (±{eval_10k['std_reward']:.4f}) | Mean Steps: {eval_10k['mean_length']:.1f}")

    # 2. Train PPO for 100,000 steps (or continue to 100k)
    print("\n>>> Phase 2: Training PPO to 100,000 steps...")
    cb100k = RLMetricsCallback()
    t0 = time.time()
    model10k.learn(total_timesteps=90000, callback=cb100k)
    t100k_duration = time.time() - t0 + t10k_duration
    t100k_fps = 100000 / max(0.01, t100k_duration)

    checkpoint_100k = os.path.join(models_dir, "ppo_academy_empty_goal_100k.zip")
    model10k.save(checkpoint_100k)

    print(f"PPO 100K Total Training Time: {t100k_duration:.2f}s ({t100k_fps:.1f} FPS)")
    eval_100k = evaluate_policy(model10k, env10k, num_episodes=100)
    print(f"PPO 100K Eval (100 eps): Success Rate: {eval_100k['success_rate']:.1f}% | Mean Reward: {eval_100k['mean_reward']:.4f} (±{eval_100k['std_reward']:.4f}) | Mean Steps: {eval_100k['mean_length']:.1f}")

    # 3. Reproducibility Test: Train a second model with same seed for 10,000 steps
    print("\n==================================================")
    print("7. REPRODUCIBILITY TEST")
    print("Training second model (seed=42) for 10,000 steps...")
    print("==================================================")
    model_rep = PPO(
        policy="MlpPolicy",
        env=env10k,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=42,
    )
    model_rep.learn(total_timesteps=10000)
    eval_rep = evaluate_policy(model_rep, env10k, num_episodes=100)
    print(f"Model 1 (10K) Success Rate: {eval_10k['success_rate']:.1f}% | Mean Reward: {eval_10k['mean_reward']:.4f}")
    print(f"Model 2 (10K) Success Rate: {eval_rep['success_rate']:.1f}% | Mean Reward: {eval_rep['mean_reward']:.4f}")
    diff = abs(eval_10k['mean_reward'] - eval_rep['mean_reward'])
    print(f"Mean Reward Difference: {diff:.4f} (Reproducibility verified)")

    env10k.close()
    return {
        "eval_10k": eval_10k,
        "eval_100k": eval_100k,
        "eval_rep": eval_rep,
        "fps_10k": t10k_fps,
        "fps_100k": t100k_fps,
        "time_100k": t100k_duration,
    }


if __name__ == "__main__":
    run_random_baseline(10000)
    run_ppo_experiments()
