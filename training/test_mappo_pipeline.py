"""
GMN-Football-3 — Verification Test for MAPPO Rollout Collection & Centralized Critic
Validates the data collection pipeline, tensor shapes, GAE advantage computation,
and observation concatenation invariants against live multi-agent simulation.
"""

import sys
import numpy as np
import torch

from gmn_pettingzoo import GMNMultiAgentEnv
from mappo_networks import SharedActor, CentralizedCritic
from mappo_rollout import collect_rollout, compute_gae


def run_mappo_pipeline_test():
    print("=" * 80)
    print("GMN-FOOTBALL-3 — MAPPO ROLLOUT & CENTRALIZED CRITIC PIPELINE VALIDATION")
    print("=" * 80)

    # 1. Initialize networks
    print("\n1. Initializing SharedActor and CentralizedCritic networks...")
    torch.manual_seed(42)
    np.random.seed(42)

    actor = SharedActor(obs_dim=115, action_dim=19, hidden=64)
    critic = CentralizedCritic(global_state_dim=345, hidden=64)

    print("   ✓ SharedActor:", actor)
    print("   ✓ CentralizedCritic:", critic)

    # 2. Connect to PettingZoo environment
    print("\n2. Initializing PettingZoo ParallelEnv on 'academy_3_vs_1_with_keeper'...")
    env = GMNMultiAgentEnv(scenario="academy_3_vs_1_with_keeper")
    print(f"   ✓ Discovered agents ({len(env.agents)}): {env.agents}")
    assert len(env.agents) == 3, f"Expected 3 agents, got {len(env.agents)}"
    assert env.agents == ["left_1", "left_2", "left_3"], f"Unexpected agent ordering: {env.agents}"

    # 3. Collect 256-step rollout
    num_steps = 256
    print(f"\n3. Collecting rollout ({num_steps} timesteps)...")
    buffer = collect_rollout(env, actor, critic, num_steps=num_steps)
    print("   ✓ Rollout collected successfully.")

    # 4. Validate buffer shapes and types
    print("\n4. Validating buffer shapes and types against specification:")
    shapes_spec = {
        "local_obs": ((num_steps, 3, 115), np.float32),
        "global_state": ((num_steps, 345), np.float32),
        "actions": ((num_steps, 3), np.int64),
        "logprobs": ((num_steps, 3), np.float32),
        "values": ((num_steps,), np.float32),
        "rewards": ((num_steps,), np.float32),
        "dones": ((num_steps,), np.bool_),
    }

    for key, (expected_shape, expected_dtype) in shapes_spec.items():
        arr = buffer[key]
        assert arr.shape == expected_shape, (
            f"Shape mismatch on '{key}': expected {expected_shape}, got {arr.shape}"
        )
        assert arr.dtype == expected_dtype, (
            f"Dtype mismatch on '{key}': expected {expected_dtype}, got {arr.dtype}"
        )
        has_nan = np.isnan(arr).any()
        has_inf = np.isinf(arr).any()
        assert not has_nan, f"NaN values detected in '{key}'"
        assert not has_inf, f"Inf values detected in '{key}'"
        print(f"   ✓ Key '{key:12s}': shape={str(arr.shape):18s} dtype={str(arr.dtype):10s} (NaN/Inf check passed)")

    # 5. Concatenation integrity verification
    print("\n5. Verifying Centralized State Concatenation Integrity...")
    for t in range(num_steps):
        local_obs_t = buffer["local_obs"][t]
        global_state_t = buffer["global_state"][t]

        # Check each agent slice matches exactly
        for a_idx in range(3):
            slice_start = a_idx * 115
            slice_end = (a_idx + 1) * 115
            agent_slice = global_state_t[slice_start:slice_end]
            assert np.array_equal(agent_slice, local_obs_t[a_idx]), (
                f"Concatenation mismatch at timestep {t}, agent {a_idx}"
            )
    print("   ✓ Verified all 256 timesteps: global_state exactly concatenates local_obs for all 3 agents in order.")

    # 6. Compute Generalized Advantage Estimation (GAE)
    print("\n6. Computing Generalized Advantage Estimation (GAE)...")
    advantages, returns = compute_gae(
        rewards=buffer["rewards"],
        values=buffer["values"],
        dones=buffer["dones"],
        gamma=0.99,
        lam=0.95,
        bootstrap_value=0.0,
    )

    assert advantages.shape == (num_steps,), f"Advantages shape mismatch: {advantages.shape}"
    assert returns.shape == (num_steps,), f"Returns shape mismatch: {returns.shape}"
    assert not np.isnan(advantages).any(), "NaN found in advantages"
    assert not np.isinf(advantages).any(), "Inf found in advantages"
    assert not np.isnan(returns).any(), "NaN found in returns"
    assert not np.isinf(returns).any(), "Inf found in returns"
    print(f"   ✓ Advantages shape={advantages.shape}, mean={advantages.mean():.5f}, std={advantages.std():.5f}")
    print(f"   ✓ Returns shape={returns.shape}, mean={returns.mean():.5f}, std={returns.std():.5f}")

    # 7. Print raw spot-check values for transparency and human inspection
    print("\n" + "=" * 80)
    print("SPOT-CHECK SAMPLE RAW VALUES (Steps 0, 1, 2)")
    print("=" * 80)

    for t in range(min(3, num_steps)):
        print(f"\n--- Timestep {t} ---")
        print(f"  Dones: {buffer['dones'][t]} | Shared Reward: {buffer['rewards'][t]:+.4f} | Value: {buffer['values'][t]:+.4f}")
        print(f"  Advantage: {advantages[t]:+.4f} | Return: {returns[t]:+.4f}")
        print(f"  Actions (agents 0, 1, 2): {buffer['actions'][t]}")
        print(f"  Logprobs (agents 0, 1, 2): {buffer['logprobs'][t]}")
        print(f"  Agent 0 local_obs sample [0..5]: {buffer['local_obs'][t, 0, :6]}")
        print(f"  Agent 1 local_obs sample [0..5]: {buffer['local_obs'][t, 1, :6]}")
        print(f"  Agent 2 local_obs sample [0..5]: {buffer['local_obs'][t, 2, :6]}")
        print(f"  Global state sample [0..6] (should match Agent 0): {buffer['global_state'][t, :6]}")
        print(f"  Global state sample [115..121] (should match Agent 1): {buffer['global_state'][t, 115:121]}")
        print(f"  Global state sample [230..236] (should match Agent 2): {buffer['global_state'][t, 230:236]}")

    print("\n" + "=" * 80)
    print("✓ ALL MAPPO ROLLOUT COLLECTION & CENTRALIZED CRITIC CHECKS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    run_mappo_pipeline_test()
