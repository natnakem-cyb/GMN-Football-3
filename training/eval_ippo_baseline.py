import argparse
import os
import sys
import time
from typing import Any, Dict, List

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO
from training.gmn_pettingzoo import GMNMultiAgentEnv

# Reference Stage 2 Baseline Metrics for academy_3_vs_1_with_keeper
STAGE_2_BASELINE = {
    "agent_name": "RuleBasedAgent (Hand-coded)",
    "episodes": 50,
    "goal_rate": 20.0,  # 10 / 50 (18% - 20% range)
    "shot_attempts_per_ep": 1.04,  # 52 shots / 50 ep
    "turnover_rate": 78.0,  # 39 / 50
    "no_shot_turnovers": 12,
    "saved_or_missed_turnovers": 27,
    "timeouts": 1,
    "avg_steps": 142.6,
    "avg_sec": 2.38,
}


def evaluate_ippo_baseline(
    checkpoint_name: str = "ippo_academy_3_vs_1_with_keeper_trained.zip",
    num_episodes: int = 50,
    deterministic: bool = True,
    base_seed: int = 500000,
) -> Dict[str, Any]:
    print("========================================================================")
    print("GMN-FOOTBALL-3 — IPPO MULTI-AGENT EVALUATION vs. STAGE 2 BASELINE")
    print(f"Model Checkpoint: {checkpoint_name}")
    print(f"Evaluation Budget: {num_episodes} Episodes | Policy Mode: {'Deterministic' if deterministic else 'Stochastic'}")
    print("Scenario: academy_3_vs_1_with_keeper (3 Left IPPO vs. 1 CB + 1 GK Rule-Based)")
    print("========================================================================\n")

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    model_path = os.path.join(models_dir, checkpoint_name)
    if not os.path.exists(model_path):
        # Fallback to smoke model if trained is not available yet
        fallback_path = os.path.join(models_dir, "ippo_academy_3_vs_1_with_keeper_smoke.zip")
        if os.path.exists(fallback_path):
            print(f"[Notice] Checkpoint {model_path} not found. Falling back to smoke checkpoint: {fallback_path}")
            model_path = fallback_path
        else:
            raise FileNotFoundError(f"Neither {model_path} nor {fallback_path} exist.")

    print(f"1. Loading trained IPPO policy from: {model_path}...")
    model = PPO.load(model_path)
    print("   ✓ Policy loaded successfully.")

    print("\n2. Initializing Headless Multi-Agent PettingZoo Environment...")
    env = GMNMultiAgentEnv(scenario="academy_3_vs_1_with_keeper", auto_start_bridge=True)

    total_steps = 0
    total_time_sec = 0.0
    left_goals_total = 0
    right_goals_total = 0
    left_shot_attempts = 0
    turnover_count = 0
    saved_or_missed_turnovers = 0
    no_shot_turnovers = 0
    timeouts = 0
    crash_count = 0
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []

    print(f"\n3. Executing {num_episodes} Evaluation Episodes (Seed pattern: {base_seed} + ep * 1009)...")
    start_eval_time = time.time()

    for ep in range(num_episodes):
        seed = base_seed + ep * 1009
        try:
            obs_dict, info_dict = env.reset(seed=seed)
            ep_done = False
            step_count = 0
            ep_left_shots = 0
            ep_reward = 0.0
            last_terminated = False
            last_truncated = False
            last_info = {}

            # Max steps for 15s scenario (+ buffer)
            max_steps = 900 + 120

            while not ep_done and step_count < max_steps:
                step_count += 1

                # Select actions for all active left agents using the loaded policy
                actions = {}
                for agent_id in env.agents:
                    obs = obs_dict[agent_id]
                    action, _ = model.predict(obs, deterministic=deterministic)
                    act_int = int(action)
                    actions[agent_id] = act_int
                    if act_int == 12:  # ActionType.SHOT (action_shot)
                        left_shot_attempts += 1
                        ep_left_shots += 1

                obs_dict, rewards, terminations, truncations, infos = env.step(actions)

                # Record team reward from primary agent
                if env.possible_agents and env.possible_agents[0] in rewards:
                    ep_reward += rewards[env.possible_agents[0]]

                # Any agent termination/truncation represents episode termination
                if terminations:
                    last_terminated = any(terminations.values())
                if truncations:
                    last_truncated = any(truncations.values())

                if infos:
                    # Take info from first available agent
                    for agent_id, agent_info in infos.items():
                        last_info = agent_info
                        break

                if last_terminated or last_truncated or not env.agents:
                    ep_done = True

            total_steps += step_count
            total_time_sec += step_count / 60.0
            episode_rewards.append(ep_reward)
            episode_lengths.append(step_count)

            score_left = last_info.get("score", {}).get("left", 0)
            score_right = last_info.get("score", {}).get("right", 0)
            goal_scored = score_left > 0

            if ep < 10:
                print(
                    f"   [Ep {ep + 1:2d}] Seed: {seed} | Score: (Left {score_left} - Right {score_right}) | "
                    f"Outcome: {'GOAL' if goal_scored else ('SAVED/MISSED' if ep_left_shots > 0 else 'NO SHOT')} | "
                    f"Steps: {step_count:3d} ({step_count/60:.2f}s) | Shots: {ep_left_shots} | Reward: {ep_reward:+.4f}"
                )

            if goal_scored:
                left_goals_total += 1
            elif last_terminated:
                turnover_count += 1
                if ep_left_shots > 0:
                    saved_or_missed_turnovers += 1
                else:
                    no_shot_turnovers += 1
            elif last_truncated:
                timeouts += 1

        except Exception as e:
            crash_count += 1
            print(f"   [Error] Episode {ep + 1} crashed: {e}")

    env.close()
    eval_duration = time.time() - start_eval_time

    # Calculate Summary Statistics
    avg_steps = total_steps / max(1, num_episodes)
    avg_sec = total_time_sec / max(1, num_episodes)
    goal_rate = (left_goals_total / max(1, num_episodes)) * 100.0
    shots_per_ep = left_shot_attempts / max(1, num_episodes)
    turnover_rate = (turnover_count / max(1, num_episodes)) * 100.0
    mean_rew = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_rew = float(np.std(episode_rewards)) if episode_rewards else 0.0

    eval_result = {
        "checkpoint": checkpoint_name,
        "deterministic": deterministic,
        "episodes_run": num_episodes,
        "left_goals": left_goals_total,
        "goal_rate": goal_rate,
        "shot_attempts_total": left_shot_attempts,
        "shot_attempts_per_ep": shots_per_ep,
        "turnover_count": turnover_count,
        "turnover_rate": turnover_rate,
        "saved_or_missed_turnovers": saved_or_missed_turnovers,
        "no_shot_turnovers": no_shot_turnovers,
        "timeouts": timeouts,
        "crash_count": crash_count,
        "avg_steps": avg_steps,
        "avg_sec": avg_sec,
        "mean_reward": mean_rew,
        "std_reward": std_rew,
        "eval_duration_sec": eval_duration,
    }

    # Print Formatted Evaluation Report
    print("\n========================================================================")
    print("EVALUATION OUTCOMES: TRAINED IPPO vs. STAGE 2 RULE-BASED BASELINE")
    print("========================================================================")
    print(f"{'Metric':<32} | {'Stage 2 Baseline (RuleBased)':<28} | {'Trained IPPO (This Run)':<24}")
    print(f"{'-'*32}-+-{'-'*28}-+-{'-'*24}")
    print(
        f"{'Goal Rate (%)':<32} | {STAGE_2_BASELINE['goal_rate']:>25.1f}% | {eval_result['goal_rate']:>21.1f}% ({left_goals_total}/{num_episodes})"
    )
    print(
        f"{'Shot Attempts / Episode':<32} | {STAGE_2_BASELINE['shot_attempts_per_ep']:>28.2f} | {eval_result['shot_attempts_per_ep']:>24.2f} ({left_shot_attempts} total)"
    )
    print(
        f"{'Turnover Rate (%)':<32} | {STAGE_2_BASELINE['turnover_rate']:>25.1f}% | {eval_result['turnover_rate']:>21.1f}% ({turnover_count}/{num_episodes})"
    )
    print(
        f"{'  - Saved / Missed Shots':<32} | {STAGE_2_BASELINE['saved_or_missed_turnovers']:>28d} | {eval_result['saved_or_missed_turnovers']:>24d}"
    )
    print(
        f"{'  - No Shot Taken (Possession Loss)':<32} | {STAGE_2_BASELINE['no_shot_turnovers']:>28d} | {eval_result['no_shot_turnovers']:>24d}"
    )
    print(
        f"{'Timeouts (Truncations)':<32} | {STAGE_2_BASELINE['timeouts']:>28d} | {eval_result['timeouts']:>24d}"
    )
    print(
        f"{'Avg Episode Duration (Steps / Sec)':<32} | {STAGE_2_BASELINE['avg_steps']:>16.1f} st ({STAGE_2_BASELINE['avg_sec']:.2f}s) | {eval_result['avg_steps']:>12.1f} st ({eval_result['avg_sec']:.2f}s)"
    )
    print(
        f"{'Mean Cumulative Reward':<32} | {'N/A (heuristic)':>28} | {eval_result['mean_reward']:>21.4f} ± {eval_result['std_reward']:.4f}"
    )
    print(f"{'Crashes / Errors':<32} | {0:>28d} | {eval_result['crash_count']:>24d}")
    print("========================================================================\n")

    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained IPPO against Stage 2 Baseline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ippo_academy_3_vs_1_with_keeper_trained.zip",
        help="Model checkpoint filename in training/models/",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes (default: 50)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Run stochastic policy evaluation instead of deterministic",
    )
    parser.add_argument(
        "--compare-both",
        action="store_true",
        help="Run both deterministic and stochastic evaluations back-to-back",
    )

    args = parser.parse_args()

    if args.compare_both:
        print(">>> EVALUATION PASS 1: DETERMINISTIC ACTIONS <<<")
        evaluate_ippo_baseline(checkpoint_name=args.checkpoint, num_episodes=args.episodes, deterministic=True)
        print("\n>>> EVALUATION PASS 2: STOCHASTIC ACTIONS <<<")
        evaluate_ippo_baseline(checkpoint_name=args.checkpoint, num_episodes=args.episodes, deterministic=False)
    else:
        evaluate_ippo_baseline(
            checkpoint_name=args.checkpoint,
            num_episodes=args.episodes,
            deterministic=not args.stochastic,
        )
