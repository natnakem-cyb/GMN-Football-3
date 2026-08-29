import sys
import os
import time

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_gym import GMNFootballEnv
from stable_baselines3 import PPO


def run_ppo_smoke_test(timesteps: int = 1000):
    print("==================================================")
    print("GMN FOOTBALL — STABLE-BASELINES3 PPO SMOKE TEST")
    print(f"Target Scenario: academy_empty_goal | Timesteps: {timesteps}")
    print("==================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("\n1. Initializing Environment...")
    env = GMNFootballEnv(scenario="academy_empty_goal", port=5050, use_ws=True)

    try:
        print("\n2. Configuring PPO Model (MlpPolicy, gamma=0.99, n_steps=256, batch_size=64)...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=None,
            seed=42,
        )

        print(f"\n3. Starting PPO Training for {timesteps} steps...")
        start_time = time.time()
        model.learn(total_timesteps=timesteps)
        duration = time.time() - start_time
        fps = timesteps / max(0.001, duration)

        print(f"\n   ✓ Training completed in {duration:.2f}s ({fps:.1f} steps/sec)")

        # Save model checkpoint
        checkpoint_path = os.path.join(models_dir, "ppo_academy_empty_goal_smoke.zip")
        print(f"\n4. Saving Model Checkpoint to: {checkpoint_path}...")
        model.save(checkpoint_path)
        print("   ✓ Checkpoint saved successfully.")

        # Verify loading model
        print("\n5. Testing Model Loading from Checkpoint...")
        loaded_model = PPO.load(checkpoint_path)
        print("   ✓ Checkpoint loaded successfully into memory.")

        # Run 1 evaluation rollout
        print("\n6. Running Evaluation Rollout with Loaded Policy...")
        obs, info = env.reset(seed=100)
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < 100:
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"   ✓ Rollout completed in {steps} steps | Cumulative Reward: {total_reward:+.4f}")
        print("\n==================================================")
        print("RESULT: PPO SMOKE TEST SUCCESSFUL")
        print("==================================================")
        return True

    finally:
        env.close()


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    success = run_ppo_smoke_test(steps)
    sys.exit(0 if success else 1)
