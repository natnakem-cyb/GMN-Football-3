import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_gym import GMNFootballEnv, ACTION_SPACE_SIZE, OBSERVATION_DIM


def benchmark_python_bridge(total_steps: int = 1000):
    print("==================================================")
    print("GMN-FOOTBALL-3 — PYTHON / HTTP BRIDGE BENCHMARK")
    print(f"Steps: {total_steps} | Path: Python -> HTTP Session -> Node Bridge -> GameEngine -> Python")
    print("==================================================")

    env = GMNFootballEnv(scenario="academy_empty_goal", port=5050, use_ws=False)
    obs, info = env.reset(seed=42)

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
    print(f"- Environment Throughput: {env_steps_per_sec:.1f} steps/sec")
    print(f"- Round-trip Latency per Step: {latency_ms:.2f} ms")
    print("==================================================\n")

    env.close()


if __name__ == "__main__":
    benchmark_python_bridge(1000)
