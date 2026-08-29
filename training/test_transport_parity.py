import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_gym import GMNFootballEnv, ACTION_SPACE_SIZE, OBSERVATION_DIM


def run_trajectory(use_ws: bool, port: int, steps: int = 500, seed: int = 424242, scenario: str = "academy_empty_goal"):
    env = GMNFootballEnv(scenario=scenario, port=port, use_ws=use_ws)
    obs, info = env.reset(seed=seed)
    trajectory = [obs.copy()]
    rewards = []
    terminations = []
    truncations = []
    scores = [info.get("score", {"left": 0, "right": 0})]

    for step_idx in range(steps):
        action = step_idx % ACTION_SPACE_SIZE
        obs, rew, term, trunc, info = env.step(action)
        trajectory.append(obs.copy())
        rewards.append(rew)
        terminations.append(term)
        truncations.append(trunc)
        scores.append(info.get("score", {"left": 0, "right": 0}))

        if term or trunc:
            obs, info = env.reset(seed=seed + step_idx + 1)
            trajectory.append(obs.copy())
            scores.append(info.get("score", {"left": 0, "right": 0}))

    env.close()
    return trajectory, rewards, terminations, truncations, scores


def test_transport_parity(steps: int = 500, seed: int = 424242, scenario: str = "academy_empty_goal"):
    print("========================================================================")
    print("GMN-FOOTBALL-3 — HTTP vs. BINARY WEBSOCKET TRANSPORT PARITY TEST")
    print(f"Scenario: {scenario} | Seed: {seed} | Steps: {steps}")
    print("========================================================================")

    # 1. Run HTTP trajectory on isolated port
    print("[1/2] Recording HTTP transport trajectory on port 5062...")
    traj_http, rew_http, term_http, trunc_http, score_http = run_trajectory(
        use_ws=False, port=5062, steps=steps, seed=seed, scenario=scenario
    )

    # 2. Run WebSocket trajectory on isolated port
    print("[2/2] Recording Binary WebSocket transport trajectory on port 5063...")
    traj_ws, rew_ws, term_ws, trunc_ws, score_ws = run_trajectory(
        use_ws=True, port=5063, steps=steps, seed=seed, scenario=scenario
    )

    # 3. Compare trajectories
    assert len(traj_http) == len(traj_ws), f"Trajectory length mismatch: HTTP={len(traj_http)}, WS={len(traj_ws)}"
    assert len(rew_http) == len(rew_ws), f"Reward list length mismatch: HTTP={len(rew_http)}, WS={len(rew_ws)}"

    max_obs_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(traj_http, traj_ws))
    max_rew_diff = max(abs(a - b) for a, b in zip(rew_http, rew_ws))
    term_matches = all(a == b for a, b in zip(term_http, term_ws))
    trunc_matches = all(a == b for a, b in zip(trunc_http, trunc_ws))
    score_matches = all(a == b for a, b in zip(score_http, score_ws))

    print("------------------------------------------------------------------------")
    print(f"Results over {steps} steps ({len(traj_http)} states):")
    print(f"- Max Observation Float32 Difference: {max_obs_diff:.8e}")
    print(f"- Max Reward Difference: {max_rew_diff:.8e}")
    print(f"- Terminations Match: {term_matches}")
    print(f"- Truncations Match: {trunc_matches}")
    print(f"- Scores Match: {score_matches}")
    print("------------------------------------------------------------------------")

    assert max_obs_diff < 1e-6, f"Observation diff {max_obs_diff} exceeds tolerance"
    assert max_rew_diff < 1e-6, f"Reward diff {max_rew_diff} exceeds tolerance"
    assert term_matches, "Termination flags mismatch"
    assert trunc_matches, "Truncation flags mismatch"
    assert score_matches, "Scores mismatch"

    print("✓ TRANSPORT PARITY VERIFIED CLEANLY — HTTP AND BINARY WS ARE BIT-IDENTICAL")


if __name__ == "__main__":
    test_transport_parity(500, 424242, "academy_empty_goal")
