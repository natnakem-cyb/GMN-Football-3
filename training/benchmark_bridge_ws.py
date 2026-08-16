import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_gym import GMNFootballEnv, ACTION_SPACE_SIZE, OBSERVATION_DIM


def benchmark_ws_bridge(total_steps: int = 2000):
    print("==================================================")
    print("GMN-FOOTBALL-3 — BINARY WEBSOCKET BRIDGE BENCHMARK")
    print(f"Steps: {total_steps} | Path: Python -> Binary WS (uint8 action) -> Node Bridge -> GameEngine -> Binary Frame (477B) -> Python")
    print("==================================================")

    env = GMNFootballEnv(scenario="academy_empty_goal", port=5050, use_ws=True)
    obs, info = env.reset(seed=42)

    # Warmup 50 steps
    for w in range(50):
        obs, reward, term, trunc, info = env.step(w % ACTION_SPACE_SIZE)
        if term or trunc:
            obs, info = env.reset()

    start_time = time.perf_counter()

    for i in range(total_steps):
        action = i % ACTION_SPACE_SIZE
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset()

    duration = time.perf_counter() - start_time
    env_steps_per_sec = total_steps / max(1e-6, duration)
    latency_ms = (duration / total_steps) * 1000

    print(f"\nResults:")
    print(f"- Total Time: {duration:.3f}s")
    print(f"- WebSocket Binary Throughput: {env_steps_per_sec:.1f} steps/sec")
    print(f"- Round-trip Latency per Step: {latency_ms:.3f} ms")
    print("==================================================\n")

    env.close()
    return env_steps_per_sec


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    benchmark_ws_bridge(steps)
