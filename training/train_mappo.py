"""
GMN-Football-3 — Multi-Agent PPO (MAPPO) with Centralized Critic Training
Trains a parameter-shared policy network and centralized team critic on cooperative football scenarios.

Hyperparameters:
- Timesteps: 200,000 (apples-to-apples comparison against IPPO baseline)
- Rollout length (n_steps): 256
- Mini-batch size: 64
- PPO Epochs: 4
- Learning rate: 3e-4 (Adam)
- Discount (gamma): 0.99
- GAE lambda: 0.95
- PPO Clip range: 0.2
- Value coefficient: 0.5
- Entropy coefficient: 0.01
"""

import argparse
import os
import sys
import time
from typing import List, Tuple, Dict, Any

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor, CentralizedCritic
from training.mappo_rollout import collect_rollout, compute_gae
from training.mappo_update import ppo_update
from training.eval_progress import evaluate_checkpoint_progress, persist_trend_snapshots


def run_mappo_training(
    timesteps: int = 200000,
    checkpoint_name: str = None,
    scenario: str = "academy_3_vs_1_with_keeper",
    seed: int = 42,
) -> bool:
    is_smoke_test = timesteps < 50000
    if checkpoint_name is None:
        checkpoint_name = (
            "mappo_academy_3_vs_1_with_keeper_smoke.pt"
            if is_smoke_test
            else "mappo_academy_3_vs_1_with_keeper_trained.pt"
        )

    print("==================================================")
    print(f"GMN FOOTBALL — MULTI-AGENT PPO (MAPPO) {'SMOKE TEST' if is_smoke_test else 'REAL TRAINING RUN'}")
    print(f"Target Scenario: {scenario} | Timesteps: {timesteps}")
    print("Architecture: SharedActor (Mlp 64x64) + CentralizedCritic (Global State 345 -> 64x64 -> 1)")
    print(f"Checkpoint Output: models/{checkpoint_name}")
    print("==================================================")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("\n1. Initializing Multi-Agent Environment & MAPPO Networks...")
    env = GMNMultiAgentEnv(scenario=scenario, auto_start_bridge=True)
    num_agents = len(env.possible_agents)
    obs_dim = 115
    global_state_dim = obs_dim * num_agents
    action_dim = 19

    print(f"   Controllable Agents ({num_agents}): {env.possible_agents}")
    print(f"   Local Obs Dim: {obs_dim} | Global State Dim: {global_state_dim} | Action Dim: {action_dim}")

    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    critic = CentralizedCritic(global_state_dim=global_state_dim, hidden=64)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)

    n_steps = 256
    n_updates = timesteps // n_steps
    check_freq_steps = 1000 if is_smoke_test else 10000

    print(f"\n2. Configuration:")
    print(f"   Total Updates: {n_updates} ({n_steps} steps per rollout)")
    print(f"   Total Timesteps: {n_updates * n_steps}")
    print(f"   PPO Epochs: 4 | Mini-batch: 64 | LR: 3e-4 | Clip: 0.2")
    print(f"   GAE: gamma=0.99, lambda=0.95 | Value Coef: 0.5 | Entropy Coef: 0.01")

    # Metrics tracking
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_goals: List[int] = []
    trend_snapshots: List[Tuple[int, int, float, float]] = []  # (step, num_eps, mean_rew, goal_rate)
    loss_history: List[Dict[str, Any]] = []

    last_check_step = 0
    last_checkpoint_step = 0
    total_steps_elapsed = 0

    print(f"\n3. Starting MAPPO Training for {timesteps} steps...")
    start_time = time.time()

    for update_idx in range(1, n_updates + 1):
        # 1. Collect Rollout
        buffer = collect_rollout(env, actor, critic, num_steps=n_steps)
        total_steps_elapsed += n_steps

        # Record completed episodes
        for ep_info in buffer.get("completed_episodes", []):
            episode_rewards.append(ep_info["reward"])
            episode_lengths.append(ep_info["length"])
            episode_goals.append(ep_info["goal"])

        # 2. Compute GAE Advantages and Returns
        advantages, returns = compute_gae(
            rewards=buffer["rewards"],
            values=buffer["values"],
            dones=buffer["dones"],
            gamma=0.99,
            lam=0.95,
            bootstrap_value=0.0,
        )

        # 3. PPO Update Step
        metrics = ppo_update(
            actor=actor,
            critic=critic,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            buffer=buffer,
            advantages=advantages,
            returns=returns,
            clip_range=0.2,
            n_epochs=4,
            batch_size=64,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
        )
        metrics["step"] = total_steps_elapsed
        metrics["update"] = update_idx
        loss_history.append(metrics)

        # Diagnostics check for numerical instability
        if np.isnan(metrics["policy_loss"]) or np.isnan(metrics["value_loss"]):
            raise RuntimeError(f"NaN loss detected at update {update_idx}: {metrics}")
        if np.isnan(metrics["approx_kl"]) or abs(metrics["approx_kl"]) > 1.5:
            print(f"   [WARNING] High KL divergence at update {update_idx}: approx_kl = {metrics['approx_kl']:.4f}")

        # Periodic snapshot logging (matching train_ippo.py cadence)
        if total_steps_elapsed - last_check_step >= check_freq_steps:
            last_check_step = total_steps_elapsed
            recent_ep = episode_rewards[-50:] if episode_rewards else [0.0]
            recent_goals = episode_goals[-50:] if episode_goals else [0]
            mean_rew = float(np.mean(recent_ep))
            goal_pct = float(np.mean(recent_goals)) * 100.0
            trend_snapshots.append((total_steps_elapsed, len(episode_rewards), mean_rew, goal_pct))

            print(
                f"   [Step {total_steps_elapsed:7d} / {timesteps}] Update {update_idx:4d}/{n_updates} | "
                f"Completed Episodes: {len(episode_rewards):4d} | "
                f"Rolling Reward (last 50): {mean_rew:+.4f} | "
                f"Goal Rate: {goal_pct:5.1f}% | "
                f"Val Loss: {metrics['value_loss']:.5f} | "
                f"Entropy: {metrics['entropy']:.4f}",
                flush=True,
            )

        # 100k-interval checkpoint saving and milestone evaluation
        if total_steps_elapsed - last_checkpoint_step >= 100_000:
            last_checkpoint_step = total_steps_elapsed
            milestone_ckpt_name = f"mappo_{scenario}_{total_steps_elapsed}.pt"
            milestone_ckpt_path = os.path.join(models_dir, milestone_ckpt_name)
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "obs_dim": obs_dim,
                    "global_state_dim": global_state_dim,
                    "action_dim": action_dim,
                    "timesteps": total_steps_elapsed,
                },
                milestone_ckpt_path,
            )
            try:
                evaluate_checkpoint_progress(
                    checkpoint_path=milestone_ckpt_path,
                    scenario=scenario,
                    algorithm="MAPPO",
                    step=total_steps_elapsed,
                    learning_rate=3e-4,
                    num_episodes=50,
                    deterministic=True,
                )
            except Exception as e:
                print(f"[Notice] MAPPO milestone eval notice: {e}")

    duration = time.time() - start_time
    fps = total_steps_elapsed / max(0.001, duration)
    print(f"\n   ✓ MAPPO Training completed in {duration:.2f}s ({fps:.1f} steps/sec)", flush=True)

    # 4. Save model checkpoint
    checkpoint_path = os.path.join(models_dir, checkpoint_name)
    print(f"\n4. Saving MAPPO Model Checkpoint to: {checkpoint_path}...", flush=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "obs_dim": obs_dim,
            "global_state_dim": global_state_dim,
            "action_dim": action_dim,
            "timesteps": total_steps_elapsed,
        },
        checkpoint_path,
    )
    print(
        f"   ✓ Checkpoint saved successfully. File exists: {os.path.exists(checkpoint_path)} "
        f"(size: {os.path.getsize(checkpoint_path)} bytes)",
        flush=True,
    )

    # Persist trend snapshots
    if trend_snapshots:
        persist_trend_snapshots(trend_snapshots, algorithm="MAPPO", scenario=scenario)

    # End-of-run milestone evaluation
    try:
        evaluate_checkpoint_progress(
            checkpoint_path=checkpoint_path,
            scenario=scenario,
            algorithm="MAPPO",
            step=total_steps_elapsed,
            learning_rate=3e-4,
            num_episodes=50,
            deterministic=True,
        )
    except Exception as e:
        print(f"[Notice] End-of-run MAPPO eval notice: {e}")

    # 5. Print Training Reward & Performance Trend Summary
    print("\n5. Training Reward & Performance Trend Summary:", flush=True)
    if trend_snapshots:
        print(f"   {'Timestep':>9} | {'Episodes':>8} | {'Rolling Reward':>15} | {'Rolling Goal Rate':>18}", flush=True)
        print(f"   {'-'*9}-+-{'-'*8}-+-{'-'*15}-+-{'-'*18}", flush=True)
        for step, num_eps, rew, goal_rt in trend_snapshots:
            print(f"   {step:9d} | {num_eps:8d} | {rew:+15.4f} | {goal_rt:17.1f}%", flush=True)
    else:
        total_eps = len(episode_rewards)
        overall_rew = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        overall_goals = float(np.mean(episode_goals)) * 100.0 if episode_goals else 0.0
        print(f"   Total Episodes Completed: {total_eps}", flush=True)
        print(f"   Overall Mean Reward: {overall_rew:+.4f}", flush=True)
        print(f"   Overall Goal Rate: {overall_goals:.1f}%", flush=True)

    # 6. Sampled Loss Metrics Progression Across Training
    print("\n6. Sampled Loss & Diagnostic Metrics Progression Across Training:", flush=True)
    print(f"   {'Update':>7} | {'Timestep':>9} | {'Policy Loss':>12} | {'Value Loss':>12} | {'Entropy':>10} | {'Approx KL':>11}", flush=True)
    print(f"   {'-'*7}-+-{'-'*9}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*11}", flush=True)

    sample_indices = np.linspace(0, len(loss_history) - 1, num=min(10, len(loss_history)), dtype=int)
    for idx in sample_indices:
        m = loss_history[idx]
        print(
            f"   {m['update']:7d} | {m['step']:9d} | {m['policy_loss']:+12.6f} | "
            f"{m['value_loss']:12.6f} | {m['entropy']:10.4f} | {m['approx_kl']:+11.6f}",
            flush=True,
        )

    # 7. Checkpoint Reload Smoke Test
    print("\n7. Verifying Checkpoint Reload Smoke Test...", flush=True)
    loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    eval_actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    eval_critic = CentralizedCritic(global_state_dim=global_state_dim, hidden=64)

    eval_actor.load_state_dict(loaded_checkpoint["actor"])
    eval_critic.load_state_dict(loaded_checkpoint["critic"])
    eval_actor.eval()
    eval_critic.eval()

    # Test dummy forward pass
    dummy_obs = torch.zeros((num_agents, obs_dim), dtype=torch.float32)
    dummy_dist = eval_actor(dummy_obs)
    dummy_act = dummy_dist.sample()
    dummy_global = torch.zeros((1, global_state_dim), dtype=torch.float32)
    dummy_val = eval_critic(dummy_global)

    assert dummy_act.shape == (num_agents,), f"Dummy action shape mismatch: {dummy_act.shape}"
    assert dummy_val.shape == (1,), f"Dummy value shape mismatch: {dummy_val.shape}"
    print(f"   ✓ Checkpoint loaded and forward pass executed cleanly (action shape: {dummy_act.shape}, val: {dummy_val.item():.4f}).", flush=True)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAPPO on GMN-Football-3")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total environment timesteps to train")
    parser.add_argument("--checkpoint-name", type=str, default=None, help="Output checkpoint filename (.pt)")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper", help="Scenario name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_mappo_training(
        timesteps=args.timesteps,
        checkpoint_name=args.checkpoint_name,
        scenario=args.scenario,
        seed=args.seed,
    )
