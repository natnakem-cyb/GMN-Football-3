import sys
import os
import time
import argparse
from typing import Callable

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_gym import GMNFootballEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule function that takes progress remaining (1.0 to 0.0) and returns current lr.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def run_ppo_training(
    scenario: str = "academy_empty_goal",
    timesteps: int = 1000,
    resume_path: str = None,
    checkpoint_name: str = None,
    lr_schedule: str = "constant",
    initial_lr: float = 3e-4,
    eval_episodes: int = 5,
):
    print("==================================================")
    print("GMN FOOTBALL — STABLE-BASELINES3 PPO TRAINING")
    print(f"Target Scenario: {scenario} | Timesteps: {timesteps}")
    print("==================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("\n1. Initializing Environment...")
    def make_env():
        return GMNFootballEnv(scenario=scenario, port=5050, use_ws=True)

    raw_env = make_env()
    env = DummyVecEnv([lambda: raw_env])

    vec_norm_path = resume_path.replace(".zip", "_vecnormalize.pkl") if resume_path else None
    if vec_norm_path and os.path.exists(vec_norm_path):
        print(f"Loading VecNormalize statistics from {vec_norm_path}...")
        env = VecNormalize.load(vec_norm_path, env)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0)

    try:
        lr = linear_schedule(initial_lr) if lr_schedule == "linear" else initial_lr

        if resume_path and os.path.exists(resume_path):
            print(f"\n2. Resuming PPO Model from Checkpoint: {resume_path}...")
            model = PPO.load(resume_path, env=env, learning_rate=lr)
        else:
            print(f"\n2. Initializing New PPO Model (MlpPolicy, gamma=0.99, n_steps=256, batch_size=64, lr_schedule={lr_schedule})...")
            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=lr,
                n_steps=256,
                batch_size=64,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                tensorboard_log=logs_dir,
                seed=42,
            )

        print(f"\n3. Starting PPO Training for {timesteps} steps...")
        start_time = time.time()
        model.learn(total_timesteps=timesteps, reset_num_timesteps=not bool(resume_path))
        duration = time.time() - start_time
        fps = timesteps / max(0.001, duration)

        print(f"\n   ✓ Training completed in {duration:.2f}s ({fps:.1f} steps/sec)")

        # Save model checkpoint
        out_name = checkpoint_name or f"ppo_{scenario}_{'smoke' if timesteps <= 5000 else 'trained'}.zip"
        checkpoint_path = os.path.join(models_dir, out_name)
        print(f"\n4. Saving Model Checkpoint to: {checkpoint_path}...")
        model.save(checkpoint_path)
        vec_save_path = checkpoint_path.replace(".zip", "_vecnormalize.pkl")
        env.save(vec_save_path)
        print("   ✓ Checkpoint and VecNormalize statistics saved successfully.")

        # Verify loading model
        print("\n5. Testing Model Loading from Checkpoint...")
        eval_vec_env = DummyVecEnv([lambda: raw_env])
        eval_vec_env = VecNormalize.load(vec_save_path, eval_vec_env)
        eval_vec_env.training = False
        eval_vec_env.norm_reward = False
        loaded_model = PPO.load(checkpoint_path, env=eval_vec_env)
        print("   ✓ Checkpoint loaded successfully into memory.")

        # Run evaluation rollouts
        print(f"\n6. Running Evaluation ({eval_episodes} episodes) with Loaded Policy...")
        eval_rewards = []
        eval_goals = 0

        for ep in range(eval_episodes):
            obs = eval_vec_env.reset()
            total_reward = 0.0
            steps = 0
            done = False

            while not done and steps < 300:
                action, _states = loaded_model.predict(obs, deterministic=True)
                obs, reward_arr, done_arr, info_list = eval_vec_env.step(action)
                reward = float(reward_arr[0])
                done = bool(done_arr[0])
                info = info_list[0]
                total_reward += reward
                steps += 1

            eval_rewards.append(total_reward)
            ev = info.get("event", {})
            if info.get("score", {}).get("left", 0) > 0 or (isinstance(ev, dict) and ev.get("type") == "goal"):
                eval_goals += 1

        avg_reward = sum(eval_rewards) / max(1, len(eval_rewards))
        print(f"   ✓ Evaluation finished: Avg Reward = {avg_reward:+.4f}, Goals = {eval_goals}/{eval_episodes}")
        print("\n==================================================")
        print("RESULT: PPO TRAINING PIPELINE SUCCESSFUL")
        print("==================================================")
        return True

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO policy for GMN Football")
    parser.add_argument("steps", type=int, nargs="?", default=None, help="Timesteps to train")
    parser.add_argument("--timesteps", type=int, default=None, help="Timesteps to train")
    parser.add_argument("--scenario", type=str, default="academy_empty_goal", help="Scenario ID")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint", type=str, default=None, help="Output checkpoint filename")
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "linear"], default="constant", help="Learning rate schedule")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of eval episodes")

    args = parser.parse_args()
    chosen_steps = args.timesteps if args.timesteps is not None else (args.steps if args.steps is not None else 1000)

    success = run_ppo_training(
        scenario=args.scenario,
        timesteps=chosen_steps,
        resume_path=args.resume,
        checkpoint_name=args.checkpoint,
        lr_schedule=args.lr_schedule,
        initial_lr=args.lr,
        eval_episodes=args.eval_episodes,
    )
    sys.exit(0 if success else 1)
