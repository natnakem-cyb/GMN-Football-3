#!/usr/bin/env python3
"""
End-to-End Determinism Test for GMN-Football-3.
Verifies complete trajectory reproducibility across the Python-to-TypeScript boundary.
"""

import sys
import numpy as np
from gmn_gym import GMNFootballEnv, OBSERVATION_DIM, ACTION_SPACE_SIZE


def run_episode(seed: int, action_seq: list) -> tuple:
    env = GMNFootballEnv(scenario="academy_empty_goal", auto_start_bridge=True)
    obs, info = env.reset(seed=seed)
    
    observations = [obs]
    rewards = []
    
    for action in action_seq:
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        if terminated or truncated:
            break
            
    env.close()
    return np.array(observations), np.array(rewards)


def main():
    print("=" * 60)
    print("GMN-FOOTBALL-3 — END-TO-END PYTHON/TYPESCRIPT DETERMINISM TEST")
    print("=" * 60)
    
    seed = 987654
    action_seq = [5, 5, 5, 5, 13, 13, 5, 5, 12, 0, 16, 17, 18, 1, 2, 3, 4, 9, 10, 11]
    
    print(f"\nRunning Episode 1 (Seed={seed}, Steps={len(action_seq)})...")
    obs1, rew1 = run_episode(seed, action_seq)
    
    print(f"Running Episode 2 (Seed={seed}, Steps={len(action_seq)})...")
    obs2, rew2 = run_episode(seed, action_seq)
    
    # Assert Exact Trajectory Equality
    obs_diff = np.max(np.abs(obs1 - obs2))
    rew_diff = np.max(np.abs(rew1 - rew2))
    
    print(f"\nMax Observation Difference: {obs_diff:.10f}")
    print(f"Max Reward Difference: {rew_diff:.10f}")
    
    if obs_diff > 1e-6 or rew_diff > 1e-6:
        print("✗ E2E DETERMINISM FAILED: Trajectories diverged!")
        sys.exit(1)
        
    print("✓ E2E DETERMINISM VERIFIED: Both Python Gymnasium episodes produced identical trajectories.")
    
    # Verify different seed or action produces valid changes
    print("\nRunning Episode 3 with different actions...")
    diff_actions = [1, 2, 3, 4, 5, 6, 7, 8]
    obs3, rew3 = run_episode(seed, diff_actions)
    
    print(f"Episode 3 Observation Shape: {obs3.shape}")
    assert obs3.shape[1] == OBSERVATION_DIM, f"Expected {OBSERVATION_DIM} dimensions."
    print("✓ Space dimensions confirmed.")
    
    print("\n" + "=" * 60)
    print("✓ ALL E2E DETERMINISM CHECKS PASSED")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
