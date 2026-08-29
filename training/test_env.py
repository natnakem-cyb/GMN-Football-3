import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_gym import GMNFootballEnv
from stable_baselines3.common.env_checker import check_env


def test_gymnasium_environment():
    print("==================================================")
    print("GMN FOOTBALL — GYMNASIUM ENVIRONMENT VALIDATION")
    print("==================================================")

    print("\n1. Initializing GMNFootballEnv('academy_empty_goal')...")
    env = GMNFootballEnv(scenario="academy_empty_goal", port=5050, use_ws=True)

    try:
        print("\n2. Testing reset()...")
        obs, info = env.reset(seed=42)
        print(f"   ✓ Reset successful.")
        print(f"   ✓ Observation shape: {obs.shape}, dtype: {obs.dtype}")
        print(f"   ✓ Observation min: {obs.min():.4f}, max: {obs.max():.4f}")
        print(f"   ✓ Info keys: {list(info.keys())}")

        assert obs.shape == (115,), f"Expected shape (115,), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

        print("\n3. Testing 10 consecutive step() calls across discrete actions...")
        for a in range(10):
            action = a % 19
            next_obs, reward, term, trunc, step_info = env.step(action)
            print(f"   Step {a+1} | Action: {action:2d} | Reward: {reward:+.4f} | Term: {term} | Dist: {step_info.get('ballDistanceToGoal', 0):.3f}")
            assert next_obs.shape == (115,)
            assert isinstance(reward, float)
            assert isinstance(term, bool)
            assert isinstance(trunc, bool)

        print("\n4. Running Stable-Baselines3 check_env(env)...")
        check_env(env, warn=True)
        print("   ✓ Stable-Baselines3 check_env PASSED with 0 errors!")

        print("\n==================================================")
        print("RESULT: Gymnasium environment validation PASSED")
        print("==================================================")
        return True

    finally:
        env.close()


if __name__ == "__main__":
    success = test_gymnasium_environment()
    sys.exit(0 if success else 1)
