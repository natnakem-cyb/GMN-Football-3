import argparse
import os
import sys
import time
from typing import List, Tuple

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.eval_progress import evaluate_checkpoint_progress, persist_trend_snapshots


class IPPORewardLoggingCallback(BaseCallback):
    """
    Logs reward trends, episode outcomes, and rolling performance across multi-agent rollouts,
    and runs evaluation at 100k step milestone checkpoints.
    """

    def __init__(
        self,
        check_freq_steps: int = 10000,
        milestone_freq_steps: int = 100000,
        scenario: str = "academy_3_vs_1_with_keeper",
        models_dir: str = "training/models",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.check_freq_steps = check_freq_steps
        self.milestone_freq_steps = milestone_freq_steps
        self.scenario = scenario
        self.models_dir = models_dir
        self.last_check_step = 0
        self.last_milestone_step = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_goals: List[int] = []
        self.current_rewards: List[float] = []
        self.current_lengths: List[int] = []
        self.trend_snapshots: List[Tuple[int, int, float, float]] = []  # (step, ep_count, mean_rew, goal_rate)

    def _on_training_start(self) -> None:
        num_envs = self.training_env.num_envs
        self.current_rewards = [0.0] * num_envs
        self.current_lengths = [0] * num_envs

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i in range(len(rewards)):
            self.current_rewards[i] += float(rewards[i])
            self.current_lengths[i] += 1
            if dones[i]:
                ep_rew = self.current_rewards[i]
                ep_len = self.current_lengths[i]
                info = infos[i] if i < len(infos) else {}
                goal_scored = 1 if info.get("score", {}).get("left", 0) > 0 else 0

                self.episode_rewards.append(ep_rew)
                self.episode_lengths.append(ep_len)
                self.episode_goals.append(goal_scored)

                self.current_rewards[i] = 0.0
                self.current_lengths[i] = 0

        # Periodic snapshot logging
        current_step = self.num_timesteps
        if current_step - self.last_check_step >= self.check_freq_steps:
            self.last_check_step = current_step
            recent_ep = self.episode_rewards[-50:] if self.episode_rewards else [0.0]
            recent_goals = self.episode_goals[-50:] if self.episode_goals else [0]
            mean_rew = float(np.mean(recent_ep))
            goal_pct = float(np.mean(recent_goals)) * 100.0
            self.trend_snapshots.append((current_step, len(self.episode_rewards), mean_rew, goal_pct))

            if self.verbose > 0:
                print(
                    f"   [Step {current_step:7d}] Completed Episodes: {len(self.episode_rewards):4d} | "
                    f"Rolling Mean Reward (last 50): {mean_rew:+.4f} | "
                    f"Rolling Goal Rate: {goal_pct:5.1f}%"
                )

        # 100k Milestone checkpoint save & persistent evaluation
        if current_step - self.last_milestone_step >= self.milestone_freq_steps:
            self.last_milestone_step = current_step
            ckpt_name = f"ippo_{self.scenario}_{current_step}_steps.zip"
            ckpt_path = os.path.join(self.models_dir, ckpt_name)
            self.model.save(ckpt_path)
            try:
                evaluate_checkpoint_progress(
                    checkpoint_path=ckpt_path,
                    scenario=self.scenario,
                    algorithm="IPPO",
                    step=current_step,
                    learning_rate=3e-4,
                    num_episodes=50,
                    deterministic=True,
                )
            except Exception as e:
                print(f"[IPPORewardLoggingCallback] Checkpoint eval notice: {e}")

        return True


def run_ippo_training(timesteps: int = 200000, checkpoint_name: str = None) -> bool:
    is_smoke_test = timesteps < 50000
    if checkpoint_name is None:
        checkpoint_name = (
            "ippo_academy_3_vs_1_with_keeper_smoke.zip"
            if is_smoke_test
            else "ippo_academy_3_vs_1_with_keeper_trained.zip"
        )

    print("==================================================")
    print(f"GMN FOOTBALL — INDEPENDENT PPO (IPPO) {'SMOKE TEST' if is_smoke_test else 'REAL TRAINING RUN'}")
    print(f"Target Scenario: academy_3_vs_1_with_keeper | Timesteps: {timesteps}")
    print("Architecture: Parameter-Sharing IPPO (SuperSuit + SB3)")
    print(f"Checkpoint Output: models/{checkpoint_name}")
    print("==================================================")

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("\n1. Initializing Multi-Agent PettingZoo Environment & SuperSuit Vectorization...")
    pz_env = GMNMultiAgentEnv(scenario="academy_3_vs_1_with_keeper", auto_start_bridge=True)
    print(f"   Controllable Agents: {pz_env.possible_agents}")

    # Vectorize the 3-agent ParallelEnv into an SB3-compatible VecEnv where each agent is 1 sub-env
    vec_env = ss.pettingzoo_env_to_vec_env_v1(pz_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    print(f"   SuperSuit VecEnv created: {vec_env.num_envs} vectorized sub-environments (sharing 1 policy)")

    # SB3 v2.x / SuperSuit 3.9.x compatibility adapter:
    # ConcatVecEnv adheres to Gymnasium (reset(seed=...)) and omits deprecated .seed() method;
    # attach adapter so SB3's set_random_seed() completes without AttributeError.
    if hasattr(vec_env, "venv") and not hasattr(vec_env.venv, "seed"):
        vec_env.venv.seed = lambda seed=None: [None] * vec_env.num_envs

    try:
        print("\n2. Configuring IPPO Model (MlpPolicy, gamma=0.99, n_steps=256, batch_size=64, lr=3e-4)...")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1 if is_smoke_test else 0,
            tensorboard_log=None,
            seed=42,
        )

        check_freq = 1000 if is_smoke_test else 10000
        callback = IPPORewardLoggingCallback(
            check_freq_steps=check_freq,
            milestone_freq_steps=100000,
            scenario="academy_3_vs_1_with_keeper",
            models_dir=models_dir,
            verbose=1,
        )
        checkpoint_cb = CheckpointCallback(
            save_freq=100_000,
            save_path=models_dir,
            name_prefix="ippo_academy_3_vs_1_with_keeper",
        )

        print(f"\n3. Starting Multi-Agent IPPO Training for {timesteps} steps...")
        start_time = time.time()
        model.learn(total_timesteps=timesteps, callback=[checkpoint_cb, callback])
        duration = time.time() - start_time
        fps = timesteps / max(0.001, duration)

        print(f"\n   ✓ IPPO Training completed in {duration:.2f}s ({fps:.1f} steps/sec)")

        # Save model checkpoint
        checkpoint_path = os.path.join(models_dir, checkpoint_name)
        print(f"\n4. Saving Multi-Agent Model Checkpoint to: {checkpoint_path}...")
        model.save(checkpoint_path)
        print(f"   ✓ Checkpoint saved successfully. File exists: {os.path.exists(checkpoint_path)} (size: {os.path.getsize(checkpoint_path)} bytes)")

        # Persist Trend Snapshots
        if callback.trend_snapshots:
            persist_trend_snapshots(
                callback.trend_snapshots,
                algorithm="IPPO",
                scenario="academy_3_vs_1_with_keeper",
            )

        # End of run milestone evaluation
        try:
            evaluate_checkpoint_progress(
                checkpoint_path=checkpoint_path,
                scenario="academy_3_vs_1_with_keeper",
                algorithm="IPPO",
                step=timesteps,
                learning_rate=3e-4,
                num_episodes=50,
                deterministic=True,
            )
        except Exception as e:
            print(f"[Notice] End of run IPPO eval notice: {e}")

        # Print Reward Progression Trend Summary
        print("\n5. Training Reward & Performance Trend Summary:")
        if callback.trend_snapshots:
            print(f"   {'Timestep':>9} | {'Episodes':>8} | {'Rolling Reward':>15} | {'Rolling Goal Rate':>18}")
            print(f"   {'-'*9}-+-{'-'*8}-+-{'-'*15}-+-{'-'*18}")
            for step, num_eps, rew, goal_rt in callback.trend_snapshots:
                print(f"   {step:9d} | {num_eps:8d} | {rew:+15.4f} | {goal_rt:17.1f}%")
        else:
            total_eps = len(callback.episode_rewards)
            overall_rew = float(np.mean(callback.episode_rewards)) if callback.episode_rewards else 0.0
            overall_goals = float(np.mean(callback.episode_goals)) * 100.0 if callback.episode_goals else 0.0
            print(f"   Total Episodes Completed: {total_eps}")
            print(f"   Overall Mean Reward: {overall_rew:+.4f}")
            print(f"   Overall Goal Rate: {overall_goals:.1f}%")

        # Verify loading model
        print("\n6. Testing Model Loading from Checkpoint...")
        loaded_model = PPO.load(checkpoint_path)
        print("   ✓ Checkpoint loaded successfully into memory.")

        # Run multi-agent PettingZoo sample evaluation
        eval_steps = 200 if is_smoke_test else 500
        print(f"\n7. Running Multi-Agent PettingZoo Evaluation Rollout ({eval_steps} steps)...")
        eval_pz_env = GMNMultiAgentEnv(scenario="academy_3_vs_1_with_keeper", auto_start_bridge=True)
        try:
            obs_dict, info_dict = eval_pz_env.reset(seed=100)
            total_team_reward = 0.0
            step_count = 0
            episodes_completed = 0
            goals_scored = 0

            for _ in range(eval_steps):
                if not eval_pz_env.agents:
                    episodes_completed += 1
                    obs_dict, info_dict = eval_pz_env.reset(seed=100 + episodes_completed)
                    if not eval_pz_env.agents:
                        break

                # Predict independently for each agent using the shared policy network
                actions = {
                    agent: int(loaded_model.predict(obs_dict[agent], deterministic=True)[0])
                    for agent in eval_pz_env.agents
                }

                obs_dict, rewards, terminations, truncations, infos = eval_pz_env.step(actions)
                step_count += 1

                if eval_pz_env.possible_agents and eval_pz_env.possible_agents[0] in rewards:
                    total_team_reward += rewards[eval_pz_env.possible_agents[0]]

                # Check goal in info
                for agent_id, agent_info in infos.items():
                    if agent_info.get("score", {}).get("left", 0) > 0 or agent_info.get("eventCode") == 1:
                        goals_scored += 1
                        break

            print(
                f"   ✓ Multi-Agent Evaluation Rollout completed: {step_count} steps across {episodes_completed + 1} episode(s)"
            )
            print(f"   ✓ Cumulative Team Reward: {total_team_reward:+.4f}")
            print(f"   ✓ Goals Scored during rollout: {goals_scored}")
            print("\n==================================================")
            print(f"RESULT: IPPO {'SMOKE TEST' if is_smoke_test else 'TRAINING RUN'} COMPLETED SUCCESSFULLY")
            print("==================================================")
            return True
        finally:
            eval_pz_env.close()

    finally:
        vec_env.close()
        pz_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Independent PPO (IPPO) on GMN Multi-Agent Football")
    parser.add_argument(
        "steps",
        type=int,
        nargs="?",
        default=None,
        help="Number of training timesteps (e.g. 3072 for smoke test, 200000 for real run)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Number of training timesteps (overrides positional arg if provided)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Custom output checkpoint filename (e.g. ippo_academy_3_vs_1_with_keeper_trained.zip)",
    )

    args = parser.parse_args()
    chosen_timesteps = args.timesteps if args.timesteps is not None else (args.steps if args.steps is not None else 200000)
    success = run_ippo_training(timesteps=chosen_timesteps, checkpoint_name=args.checkpoint)
    sys.exit(0 if success else 1)
