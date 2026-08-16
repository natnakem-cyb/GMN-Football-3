import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_gym import GMNFootballEnv, ACTION_SPACE_SIZE, OBSERVATION_DIM


def test_transport_parity(steps: int = 500, seed: int = 424242, scenario: str = "academy_empty_goal"):
    print("========================================================================")
    print("GMN-FOOTBALL-3 — HTTP vs. BINARY WEBSOCKET TRANSPORT PARITY TEST")
    print(f"Scenario: {scenario} | Seed: {seed} | Steps: {steps}")
    print("========================================================================")

    # 1. Initialize HTTP-based environment
    print("[Test] Initializing HTTP transport environment (use_ws=False)...")
    env_http = GMNFootballEnv(scenario=scenario, port=5050, use_ws=False)
    obs_http, info_http = env_http.reset(seed=seed)

    # 2. Initialize WebSocket-based environment
    print("[Test] Initializing Binary WebSocket transport environment (use_ws=True)...")
    env_ws = GMNFootballEnv(scenario=scenario, port=5050, use_ws=True)
    obs_ws, info_ws = env_ws.reset(seed=seed)

    # Initial reset observation comparison
    max_init_obs_diff = np.max(np.abs(obs_http - obs_ws))
    print(f"[Reset Check] Initial observation max absolute difference: {max_init_obs_diff:.8e}")
    assert max_init_obs_diff < 1e-6, f"Reset observation mismatch! Diff: {max_init_obs_diff}"
    assert info_http.get("score") == info_ws.get("score"), "Reset score mismatch!"

    mismatches = 0
    max_obs_diff_overall = 0.0

    print(f"[Stepping] Running {steps} synchronized actions across both transports...")
    for step_idx in range(steps):
        action = step_idx % ACTION_SPACE_SIZE

        obs_h, rew_h, term_h, trunc_h, info_h = env_http.step(action)
        obs_w, rew_w, term_w, trunc_w, info_w = env_ws.step(action)

        obs_diff = float(np.max(np.abs(obs_h - obs_w)))
        max_obs_diff_overall = max(max_obs_diff_overall, obs_diff)

        # Assert parity
        if obs_diff >= 1e-6 or abs(rew_h - rew_w) >= 1e-6 or term_h != term_w or trunc_h != trunc_w:
            mismatches += 1
            print(
                f"[ERROR Step {step_idx}] Action={action} Mismatch:\n"
                f"  Obs Max Diff: {obs_diff:.8e}\n"
                f"  Reward: HTTP={rew_h} vs WS={rew_w}\n"
                f"  Terminated: HTTP={term_h} vs WS={term_w}\n"
                f"  Truncated: HTTP={trunc_h} vs WS={trunc_w}\n"
                f"  Info HTTP={info_h}\n"
                f"  Info WS={info_w}"
            )
            if mismatches > 5:
                break

        # Check scores
        score_h = info_h.get("score", {})
        score_w = info_w.get("score", {})
        if score_h.get("left") != score_w.get("left") or score_h.get("right") != score_w.get("right"):
            mismatches += 1
            print(f"[ERROR Step {step_idx}] Score mismatch: HTTP={score_h} vs WS={score_w}")
            break

        # If terminal/truncated on either, reset both with next seed
        if term_h or trunc_h or term_w or trunc_w:
            next_seed = seed + step_idx + 1
            obs_h, info_h = env_http.reset(seed=next_seed)
            obs_w, info_w = env_ws.reset(seed=next_seed)
            reset_diff = float(np.max(np.abs(obs_h - obs_w)))
            assert reset_diff < 1e-6, f"Episode reset mismatch at step {step_idx}: {reset_diff}"

    env_http.close()
    env_ws.close()

    print("------------------------------------------------------------------------")
    print(f"Results over {steps} steps:")
    print(f"- Total Mismatches: {mismatches}")
    print(f"- Max Observation Float32 Difference: {max_obs_diff_overall:.8e}")
    print(f"- Exact Trajectory Bit-Parity: {'PASS' if mismatches == 0 else 'FAIL'}")
    print("------------------------------------------------------------------------")

    if mismatches > 0:
        print("✗ TRANSPORT PARITY FAILED")
        sys.exit(1)
    else:
        print("✓ TRANSPORT PARITY VERIFIED CLEANLY — HTTP AND BINARY WS ARE BIT-IDENTICAL")


if __name__ == "__main__":
    test_transport_parity(500, 424242, "academy_empty_goal")
