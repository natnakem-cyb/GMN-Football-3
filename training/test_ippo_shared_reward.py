"""
GMN-Football-3 — IPPO Shared Reward Behavior-Pinning Test
Asserts the CURRENT behavior: a single scalar reward from the bridge frame
is broadcast to all agents simultaneously.

This test serves as a documentation-of-intent test pinning that IPPO is deprecated
and its shared-reward broadcast is intentional legacy (superseded by MAPPO with centralized critic).
"""

import sys
import os
import struct
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv, OBSERVATION_DIM


def test_shared_reward_broadcast_unpack():
    print("========================================================================")
    print("GMN-FOOTBALL-3 — IPPO SHARED REWARD BROADCAST BEHAVIOR-PINNING TEST")
    print("========================================================================")

    # Instantiate env without connecting network
    env = GMNMultiAgentEnv(scenario="academy_3_vs_1_with_keeper", auto_start_bridge=False)
    env.agents = ["agent_0", "agent_1", "agent_2"]
    env.possible_agents = list(env.agents)
    env._step_count = 1

    # Synthesize binary step frame from bridge:
    # 17-byte header: <f (reward=0.75), ?? (term=False, trunc=False), BB (score_l=1, score_r=0), ff (cp_rew=0.1, dist=0.2), B (event=0)
    # followed by 3 agent observations (127 * 4 = 508 bytes each)
    test_reward_val = 0.75
    header = struct.pack(
        "<f??BBffB",
        test_reward_val,
        False,
        False,
        1,
        0,
        0.1,
        0.2,
        0,
    )

    obs_bytes = OBSERVATION_DIM * 4
    obs_payload = bytearray()
    for i in range(3):
        arr = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        arr[0] = float(i + 1)
        obs_payload.extend(arr.tobytes())

    frame_data = header + bytes(obs_payload)

    # Mock ws_client.recv to return frame_data and mock ws_client.send
    class MockWS:
        def send(self, msg):
            pass

        def recv(self):
            return frame_data

    env.ws_client = MockWS()

    # Step with dummy action dict
    actions = {"agent_0": 0, "agent_1": 0, "agent_2": 0}
    observations, rewards, terminations, truncations, infos = env.step(actions)

    print(f"Number of agents: {len(rewards)}")
    print(f"Unpacked rewards: {rewards}")

    # Assert behavior: single shared reward broadcast to EVERY agent
    assert len(rewards) == 3, f"Expected 3 rewards, got {len(rewards)}"
    for agent, r in rewards.items():
        assert r == test_reward_val, f"Agent {agent} expected reward {test_reward_val}, got {r}"

    # Confirm all rewards are identical (root cause of credit assignment issue in IPPO)
    reward_values = list(rewards.values())
    assert all(r == reward_values[0] for r in reward_values), "Expected all agent rewards to be identical scalar"

    print("✓ CONFIRMED: Shared reward broadcast is preserved as legacy behavior for IPPO.")
    print("✓ Pinned: rewards[agent_0] == rewards[agent_1] == rewards[agent_2] == scalar header reward.")
    print("========================================================================")


if __name__ == "__main__":
    test_shared_reward_broadcast_unpack()
