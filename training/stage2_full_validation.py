import os
import sys
import json
import time
import shutil
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_gym import GMNFootballEnv
from stable_baselines3 import PPO

# Action Name and Grouping Definitions
# 0: IDLE
# 1..8: MOVE (Directional)
# 9: SHORT_PASS
# 10: HIGH_PASS
# 11: SHOT
# 12: SPRINT
# 13: TACKLE
# 14: DRIBBLE
ACTION_GROUP_NAMES = {
    0: "IDLE",
    1: "MOVE", 2: "MOVE", 3: "MOVE", 4: "MOVE", 5: "MOVE", 6: "MOVE", 7: "MOVE", 8: "MOVE",
    9: "SHORT_PASS",
    10: "HIGH_PASS",
    11: "SHOT",
    12: "SPRINT",
    13: "TACKLE",
    14: "DRIBBLE"
}

ACTION_CATEGORIES = ["IDLE", "MOVE", "SPRINT", "DRIBBLE", "SHORT_PASS", "HIGH_PASS", "SHOT", "TACKLE"]


def evaluate_checkpoint_deep(model, env, label: str, num_episodes: int = 100):
    print(f"\n==================================================")
    print(f"Deep Evaluation: {label} (100 episodes)")
    print(f"==================================================")

    rewards = []
    lengths = []
    goal_steps = []
    final_distances = []

    action_counts = {cat: 0 for cat in ACTION_CATEGORIES}
    total_actions = 0

    failure_modes = {
        "DEFENDER CATCHES PLAYER": 0,
        "GOALKEEPER BLOCKS SHOT": 0,
        "PLAYER SHOOTS TOO EARLY": 0,
        "PLAYER FAILS TO DRIBBLE": 0,
        "PLAYER MOVES WRONG DIRECTION": 0,
        "PLAYER LOSES POSSESSION": 0,
        "TIMEOUT / MAX STEPS": 0,
        "OTHER": 0,
    }

    goals = 0
    dispossessions = 0
    timeouts = 0

    for ep in range(num_episodes):
        obs, info = env.reset(seed=4000 + ep)
        ep_rew = 0.0
        ep_len = 0
        done = False
        shot_taken = False
        shot_x = 0.0
        min_dist_to_goal = 1.0
        player_x_history = []
        dribble_count = 0
        sprint_count = 0

        while not done and ep_len < 1200:
            action, _ = model.predict(obs, deterministic=True)
            act_idx = int(action)
            cat = ACTION_GROUP_NAMES.get(act_idx, "OTHER")
            if cat in action_counts:
                action_counts[cat] += 1
            total_actions += 1

            if cat == "SHOT":
                shot_taken = True
                # obs[0] is player_x, obs[88] is ball_x
                shot_x = float(obs[0])
            elif cat == "DRIBBLE":
                dribble_count += 1
            elif cat == "SPRINT":
                sprint_count += 1

            player_x_history.append(float(obs[0]))
            curr_dist = float(info.get("ballDistanceToGoal", 1.0))
            if curr_dist < min_dist_to_goal:
                min_dist_to_goal = curr_dist

            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            ep_len += 1
            done = terminated or truncated

        rewards.append(ep_rew)
        lengths.append(ep_len)
        final_distances.append(info.get("ballDistanceToGoal", 0.5))

        is_goal = info.get("score", {}).get("left", 0) > 0 or info.get("event") == "GOAL"
        is_opponent_possession = info.get("event") == "DISPOSSESSED" or (
            terminated and not is_goal and ep_len < 1200
        )

        if is_goal:
            goals += 1
            goal_steps.append(ep_len)
        elif truncated or ep_len >= 1200:
            timeouts += 1
            failure_modes["TIMEOUT / MAX STEPS"] += 1
        else:
            dispossessions += 1
            # Classify detailed failure reason
            # 1. Did defender catch from behind (early on, player_x < 0.5)?
            final_p_x = player_x_history[-1] if player_x_history else 0.0
            if final_p_x < 0.45:
                failure_modes["DEFENDER CATCHES PLAYER"] += 1
            # 2. Did goalkeeper save near goal (player_x >= 0.7 or min_dist < 0.3)?
            elif final_p_x >= 0.7:
                failure_modes["GOALKEEPER BLOCKS SHOT"] += 1
            # 3. Did player shoot too early and goalkeeper retrieved it?
            elif shot_taken and shot_x < 0.6:
                failure_modes["PLAYER SHOOTS TOO EARLY"] += 1
            # 4. Did player lose possession in midfield without dribbling/sprinting?
            elif dribble_count == 0 and sprint_count == 0:
                failure_modes["PLAYER FAILS TO DRIBBLE"] += 1
            # 5. Wrong direction (moving backward)
            elif len(player_x_history) > 10 and player_x_history[-1] < player_x_history[0]:
                failure_modes["PLAYER MOVES WRONG DIRECTION"] += 1
            else:
                failure_modes["PLAYER LOSES POSSESSION"] += 1

    success_rate = (goals / num_episodes) * 100.0
    mean_rew = float(np.mean(rewards))
    median_rew = float(np.median(rewards))
    std_rew = float(np.std(rewards))
    mean_len = float(np.mean(lengths))
    mean_goal_steps = float(np.mean(goal_steps)) if goal_steps else 0.0
    median_goal_steps = float(np.median(goal_steps)) if goal_steps else 0.0

    action_freq = {
        cat: (count / max(1, total_actions)) * 100.0 for cat, count in action_counts.items()
    }

    results = {
        "label": label,
        "success_rate": success_rate,
        "goals": goals,
        "dispossessions": dispossessions,
        "timeouts": timeouts,
        "mean_reward": mean_rew,
        "median_reward": median_rew,
        "std_reward": std_rew,
        "mean_length": mean_len,
        "mean_steps_to_goal": mean_goal_steps,
        "median_steps_to_goal": median_goal_steps,
        "mean_final_dist": float(np.mean(final_distances)),
        "action_frequency_pct": action_freq,
        "failure_modes": failure_modes,
        "episodes": num_episodes,
    }

    print(f"Results for {label}:")
    print(f"- Success Rate: {success_rate:.1f}% ({goals}/{num_episodes})")
    print(f"- Dispossessed Rate: {(dispossessions / num_episodes) * 100:.1f}%")
    print(f"- Timeout Rate: {(timeouts / num_episodes) * 100:.1f}%")
    print(f"- Mean Reward: {mean_rew:.4f} (±{std_rew:.4f}), Median: {median_rew:.4f}")
    print(f"- Mean Steps to Goal: {mean_goal_steps:.1f}, Median: {median_goal_steps:.1f}")
    print(f"- Action Usage (%): " + ", ".join([f"{k}: {v:.1f}%" for k, v in action_freq.items()]))
    print(f"- Failure Modes: " + ", ".join([f"{k}: {v}" for k, v in failure_modes.items() if v > 0]))

    return results


def run_full_validation():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    env = GMNFootballEnv(scenario="academy_run_to_score", port=5050, use_ws=True)

    checkpoints = [
        ("Stage 1 Zero-Shot", os.path.join(models_dir, "ppo_academy_empty_goal_100k.zip")),
        ("Stage 2 10K", os.path.join(models_dir, "ppo_academy_run_to_score_10k.zip")),
        ("Stage 2 50K", os.path.join(models_dir, "ppo_academy_run_to_score_50k.zip")),
        ("Stage 2 100K", os.path.join(models_dir, "ppo_academy_run_to_score_100k.zip")),
        ("Stage 2 200K", os.path.join(models_dir, "ppo_academy_run_to_score_200k.zip")),
    ]

    all_results = []

    for label, path in checkpoints:
        if not os.path.exists(path):
            print(f"[Warning] Checkpoint {path} not found yet. Skipping {label}.")
            continue
        model = PPO.load(path, env=env)
        res = evaluate_checkpoint_deep(model, env, label=label, num_episodes=100)
        all_results.append((label, path, res))

    env.close()

    # Determine Best Model
    # Primary: success_rate
    # Secondary: mean_steps_to_goal (lower is better, but must have goals)
    # Tertiary: mean_reward
    stage2_results = [r for r in all_results if r[0] != "Stage 1 Zero-Shot"]
    if stage2_results:
        best_tuple = max(
            stage2_results,
            key=lambda x: (
                x[2]["success_rate"],
                -x[2]["mean_steps_to_goal"] if x[2]["goals"] > 0 else -9999,
                x[2]["mean_reward"],
            ),
        )
        best_label, best_path, best_res = best_tuple
        print(f"\n==================================================")
        print(f"BEST STAGE 2 MODEL: {best_label} ({os.path.basename(best_path)})")
        print(f"Success Rate: {best_res['success_rate']:.1f}%, Mean Reward: {best_res['mean_reward']:.4f}, Mean Steps: {best_res['mean_steps_to_goal']:.1f}")
        print(f"==================================================")

        # Copy to best model path
        best_model_dest = os.path.join(models_dir, "ppo_academy_run_to_score_best.zip")
        shutil.copyfile(best_path, best_model_dest)
        print(f"Saved best model to {best_model_dest}")

        metadata = {
            "scenario": "academy_run_to_score",
            "training_timesteps": best_label,
            "source_checkpoint": os.path.basename(best_path),
            "success_rate": best_res["success_rate"],
            "goals": best_res["goals"],
            "dispossessions": best_res["dispossessions"],
            "timeouts": best_res["timeouts"],
            "mean_reward": best_res["mean_reward"],
            "median_reward": best_res["median_reward"],
            "std_reward": best_res["std_reward"],
            "mean_steps_to_goal": best_res["mean_steps_to_goal"],
            "median_steps_to_goal": best_res["median_steps_to_goal"],
            "evaluation_episodes": 100,
            "observation_dim": 115,
            "action_dim": 15,
            "training_seed": 42,
            "evaluation_seed": 4000,
            "action_frequency_pct": best_res["action_frequency_pct"],
            "failure_modes": best_res["failure_modes"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        meta_dest = os.path.join(models_dir, "ppo_academy_run_to_score_best.json")
        with open(meta_dest, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata JSON to {meta_dest}")

    # Output full summary JSON for report generation
    summary_dest = os.path.join(os.path.dirname(__file__), "stage2_validation_summary.json")
    with open(summary_dest, "w") as f:
        json.dump([r[2] for r in all_results], f, indent=2)
    print(f"Saved full validation summary to {summary_dest}")


if __name__ == "__main__":
    run_full_validation()
