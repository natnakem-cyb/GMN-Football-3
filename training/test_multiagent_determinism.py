import sys
import os
import time
import json
import struct
import subprocess
import numpy as np
import websockets.sync.client

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_gym import OBSERVATION_DIM, ACTION_SPACE_SIZE, EVENT_CODE_MAP

PORT = int(os.environ.get("GMN_MULTI_TEST_PORT", "5066"))
SCENARIO = "academy_3_vs_1_with_keeper"
NUM_AGENTS = 3
TOTAL_STEPS = 250
SEED = 424242

EXPECTED_RESPONSE_BYTES = 17 + 460 * NUM_AGENTS  # 17 + 1380 = 1397 bytes


def start_bridge_server(port: int) -> subprocess.Popen:
    bridge_script = os.path.join(os.path.dirname(__file__), "bridge_server.ts")
    proc = subprocess.Popen(
        ["npx", "tsx", bridge_script],
        env=dict(os.environ, GMN_BRIDGE_PORT=str(port)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2.0)
    return proc


def run_multiagent_ws_trajectory(port: int, scenario: str, seed: int, steps: int):
    ws_url = f"ws://127.0.0.1:{port}"
    ws = websockets.sync.client.connect(ws_url, max_size=None)

    # 1. Reset
    reset_payload = {"type": "reset", "scenario": scenario, "seed": seed}
    ws.send(json.dumps(reset_payload))
    reset_resp = json.loads(ws.recv())

    controllable_ids = reset_resp.get("info", {}).get("controllableAgentIds", [])
    if len(controllable_ids) != NUM_AGENTS:
        raise ValueError(
            f"Expected {NUM_AGENTS} controllableAgentIds for {scenario}, got {len(controllable_ids)}: {controllable_ids}"
        )

    observations_per_agent = [[] for _ in range(NUM_AGENTS)]
    rewards = []
    terminations = []
    truncations = []
    scores = []
    response_byte_sizes = []

    # 2. Fixed action sequence for 3 agents
    # Cycle through distinct actions for each agent to exercise movement, passing, shooting, etc.
    for step_idx in range(steps):
        a0 = (step_idx * 3 + 1) % ACTION_SPACE_SIZE
        a1 = (step_idx * 5 + 3) % ACTION_SPACE_SIZE
        a2 = (step_idx * 7 + 5) % ACTION_SPACE_SIZE

        # 3-byte binary action frame
        action_frame = bytes([a0, a1, a2])
        ws.send(action_frame)

        resp_data = ws.recv()
        if not isinstance(resp_data, (bytes, bytearray)):
            raise RuntimeError(f"Expected binary WebSocket response, got {type(resp_data)}")

        response_byte_sizes.append(len(resp_data))
        if len(resp_data) != EXPECTED_RESPONSE_BYTES:
            raise ValueError(
                f"Byte size mismatch: expected exactly {EXPECTED_RESPONSE_BYTES} bytes, got {len(resp_data)}"
            )

        # Unpack 17-byte header
        reward, term, trunc, score_l, score_r, cp_reward, dist_goal, event_code = struct.unpack_from(
            "<f??BBffB", resp_data, 0
        )
        rewards.append(float(reward))
        terminations.append(bool(term))
        truncations.append(bool(trunc))
        scores.append((int(score_l), int(score_r)))

        # Unpack NUM_AGENTS observations (115 floats = 460 bytes each)
        for agent_idx in range(NUM_AGENTS):
            offset = 17 + agent_idx * 460
            obs = np.frombuffer(resp_data, dtype="<f4", count=OBSERVATION_DIM, offset=offset).copy()
            observations_per_agent[agent_idx].append(obs)

        if term or trunc:
            # Re-seed next episode deterministically
            ws.send(json.dumps({"type": "reset", "scenario": scenario, "seed": seed + step_idx + 1}))
            reset_resp = json.loads(ws.recv())

    ws.close()
    return {
        "controllable_ids": controllable_ids,
        "observations_per_agent": observations_per_agent,
        "rewards": rewards,
        "terminations": terminations,
        "truncations": truncations,
        "scores": scores,
        "response_byte_sizes": response_byte_sizes,
    }


def test_multiagent_determinism():
    print("========================================================================")
    print("GMN-FOOTBALL-3 — MULTI-AGENT WS BRIDGE DETERMINISM & PROTOCOL TEST")
    print(f"Scenario: {SCENARIO} ({NUM_AGENTS} controllable left agents)")
    print(f"Steps: {TOTAL_STEPS} | Seed: {SEED} | Port: {PORT}")
    print("========================================================================")

    proc = start_bridge_server(PORT)

    try:
        print("[1/2] Running Multi-Agent Run 1 (WebSocket binary)...")
        run1 = run_multiagent_ws_trajectory(PORT, SCENARIO, SEED, TOTAL_STEPS)

        print("[2/2] Running Multi-Agent Run 2 (WebSocket binary)...")
        run2 = run_multiagent_ws_trajectory(PORT, SCENARIO, SEED, TOTAL_STEPS)

        # 1. Verify byte sizes
        byte_sizes = run1["response_byte_sizes"]
        unique_byte_sizes = set(byte_sizes)
        print(f"\nResponse Frame Verification:")
        print(f"- Total steps: {len(byte_sizes)}")
        print(f"- Unique byte sizes observed: {unique_byte_sizes}")
        print(f"- Expected size: {EXPECTED_RESPONSE_BYTES} bytes (17B header + 460B * {NUM_AGENTS} agents)")
        assert unique_byte_sizes == {EXPECTED_RESPONSE_BYTES}, (
            f"Expected all responses to be {EXPECTED_RESPONSE_BYTES} bytes, got {unique_byte_sizes}"
        )
        print(f"✓ Exact frame length verified: {EXPECTED_RESPONSE_BYTES} bytes")

        # 2. Verify controllableAgentIds
        print(f"- Controllable Agent IDs: {run1['controllable_ids']}")
        assert run1["controllable_ids"] == run2["controllable_ids"]

        # 3. Verify Bitwise Determinism across all agents
        print(f"\nComparing Trajectories across Run 1 and Run 2 ({TOTAL_STEPS} steps):")

        for agent_idx in range(NUM_AGENTS):
            obs1 = run1["observations_per_agent"][agent_idx]
            obs2 = run2["observations_per_agent"][agent_idx]
            assert len(obs1) == len(obs2) == TOTAL_STEPS

            max_obs_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(obs1, obs2))
            print(f"- Agent {agent_idx} ({run1['controllable_ids'][agent_idx]}) Max Obs Float32 Diff: {max_obs_diff:.8e}")
            assert max_obs_diff < 1e-6, f"Agent {agent_idx} observation diff {max_obs_diff} exceeds tolerance"

        max_rew_diff = max(abs(a - b) for a, b in zip(run1["rewards"], run2["rewards"]))
        term_matches = all(a == b for a, b in zip(run1["terminations"], run2["terminations"]))
        trunc_matches = all(a == b for a, b in zip(run1["truncations"], run2["truncations"]))
        score_matches = all(a == b for a, b in zip(run1["scores"], run2["scores"]))

        print(f"- Max Reward Difference: {max_rew_diff:.8e}")
        print(f"- Terminations Match: {term_matches}")
        print(f"- Truncations Match: {trunc_matches}")
        print(f"- Scores Match: {score_matches}")

        assert max_rew_diff < 1e-6, f"Reward diff {max_rew_diff} exceeds tolerance"
        assert term_matches, "Termination flags mismatch"
        assert trunc_matches, "Truncation flags mismatch"
        assert score_matches, "Scores mismatch"

        print("\n✓ MULTI-AGENT PROTOCOL & BIT-IDENTICAL DETERMINISM VERIFIED CLEANLY!")

    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    test_multiagent_determinism()
