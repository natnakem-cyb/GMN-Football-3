import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_gym import GMNFootballEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class Stage2MetricsCallback(BaseCallback):
    """Tracks Stage 2 training rollout metrics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_goals = []
        self.current_reward = 0.0
        self.current_length = 0

    def _on_step(self) -> bool:
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


def evaluate_stage2_policy(model, env, num_episodes: int = 100, deterministic: bool = True):
    rewards = []
    lengths = []
    goals = 0
    dispossessions = 0
    final_distances = []
    goal_steps = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=3000 + ep)
        ep_rew = 0.0
        ep_len = 0
        done = False

        while not done and ep_len < 1200:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            ep_len += 1
            done = terminated or truncated

        rewards.append(ep_rew)
        lengths.append(ep_len)
        final_distances.append(info.get("ballDistanceToGoal", 0.5))

        is_goal = info.get("score", {}).get("left", 0) > 0 or info.get("event") == "GOAL"
        if is_goal:
            goals += 1
            goal_steps.append(ep_len)
        else:
            dispossessions += 1

    success_rate = (goals / num_episodes) * 100.0
    mean_rew = float(np.mean(rewards))
    median_rew = float(np.median(rewards))
    std_rew = float(np.std(rewards))
    mean_len = float(np.mean(lengths))
    mean_goal_steps = float(np.mean(goal_steps)) if goal_steps else 0.0
    median_goal_steps = float(np.median(goal_steps)) if goal_steps else 0.0

    return {
        "success_rate": success_rate,
        "goals": goals,
        "dispossessions": dispossessions,
        "mean_reward": mean_rew,
        "median_reward": median_rew,
        "std_reward": std_rew,
        "mean_length": mean_len,
        "mean_steps_to_goal": mean_goal_steps,
        "median_steps_to_goal": median_goal_steps,
        "mean_final_dist": float(np.mean(final_distances)),
        "episodes": num_episodes,
    }


def run_stage2_curriculum():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    print("==================================================")
    print("GMN STAGE 2: ACADEMY RUN TO SCORE CURRICULUM TRAINING")
    print("==================================================")

    env = GMNFootballEnv(scenario="academy_run_to_score", port=5050)

    # 1. Zero-shot Evaluation of Stage 1 Model on Stage 2
    stage1_checkpoint = os.path.join(models_dir, "ppo_academy_empty_goal_100k.zip")
    if not os.path.exists(stage1_checkpoint):
        stage1_checkpoint = os.path.join(models_dir, "ppo_academy_empty_goal_10k.zip")

    print(f"\n--- 1. Transfer Evaluation: Stage 1 Model ({os.path.basename(stage1_checkpoint)}) on Stage 2 ---")
    model = PPO.load(stage1_checkpoint, env=env)
    eval_zero_shot = evaluate_stage2_policy(model, env, num_episodes=100)
    print(f"Zero-shot Transfer Eval (100 eps):")
    print(f"- Success Rate: {eval_zero_shot['success_rate']:.1f}% ({eval_zero_shot['goals']}/100 goals)")
    print(f"- Mean Reward: {eval_zero_shot['mean_reward']:.4f} (±{eval_zero_shot['std_reward']:.4f})")
    print(f"- Mean Episode Length: {eval_zero_shot['mean_length']:.1f} steps")
    print(f"- Mean Steps to Goal: {eval_zero_shot['mean_steps_to_goal']:.1f} steps")

    # 2. Stage 2 Progressive PPO Training
    steps_schedule = [
        (10000, "ppo_academy_run_to_score_10k.zip", "10K"),
        (40000, "ppo_academy_run_to_score_50k.zip", "50K (cumulative)"),
        (50000, "ppo_academy_run_to_score_100k.zip", "100K (cumulative)"),
        (100000, "ppo_academy_run_to_score_200k.zip", "200K (cumulative)"),
    ]

    total_elapsed_timesteps = 0
    results_map = {"zero_shot": eval_zero_shot}

    for add_steps, ckpt_name, label in steps_schedule:
        print(f"\n--- Training PPO on Stage 2 to {label} (+{add_steps} steps) ---")
        t0 = time.time()
        cb = Stage2MetricsCallback()
        model.learn(total_timesteps=add_steps, callback=cb, reset_num_timesteps=False)
        dt = time.time() - t0
        total_elapsed_timesteps += add_steps
        fps = add_steps / max(0.01, dt)

        ckpt_path = os.path.join(models_dir, ckpt_name)
        model.save(ckpt_path)

        print(f"Completed {label} in {dt:.1f}s ({fps:.1f} FPS). Saved to {ckpt_name}")
        eval_res = evaluate_stage2_policy(model, env, num_episodes=100)
        results_map[label] = eval_res
        print(f"Eval {label} (100 eps):")
        print(f"- Success Rate: {eval_res['success_rate']:.1f}% ({eval_res['goals']}/100 goals)")
        print(f"- Mean Reward: {eval_res['mean_reward']:.4f} (±{eval_res['std_reward']:.4f})")
        print(f"- Median Reward: {eval_res['median_reward']:.4f}")
        print(f"- Mean Episode Length: {eval_res['mean_length']:.1f} steps")
        print(f"- Mean Steps to Goal: {eval_res['mean_steps_to_goal']:.1f} steps")

    env.close()
    return results_map


if __name__ == "__main__":
    run_stage2_curriculum()
