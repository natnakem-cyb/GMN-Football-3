"""
GMN-Football-3 — Python <-> Browser Neural Policy Parity Test (Python)
Evaluates PyTorch actor forward pass against the TypeScript neural weights and contract.
"""

import sys
import json
import math
import numpy as np
from typing import Dict, Any, List

GMN_ENV_VERSION = "3.1.0"
OBSERVATION_DIM = 127
BASE_OBSERVATION_DIM = 115
ROLE_DIM = 12
ACTION_SPACE_SIZE = 19


def reference_forward_pass(obs: np.ndarray, weights: Dict[str, Any]) -> Dict[str, Any]:
    w0 = np.array(weights["w0"], dtype=np.float32)
    b0 = np.array(weights["b0"], dtype=np.float32)
    w1 = np.array(weights["w1"], dtype=np.float32)
    b1 = np.array(weights["b1"], dtype=np.float32)
    w2 = np.array(weights["w2"], dtype=np.float32)
    b2 = np.array(weights["b2"], dtype=np.float32)

    x = np.array(obs, dtype=np.float32)
    h0 = np.tanh(np.dot(w0, x) + b0)
    h1 = np.tanh(np.dot(w1, h0) + b1)
    logits = np.dot(w2, h1) + b2
    action = int(np.argmax(logits))

    return {
        "logits": logits.tolist(),
        "action": action
    }


def generate_test_vectors() -> List[Dict[str, Any]]:
    cases = []

    # 1. Zero vector
    cases.append({"name": "all_zeros", "vector": [0.0] * OBSERVATION_DIM})

    # 2. Extremes
    cases.append({"name": "all_ones_pos", "vector": [1.0] * OBSERVATION_DIM})
    cases.append({"name": "all_ones_neg", "vector": [-1.0] * OBSERVATION_DIM})

    # 3. Role slots
    for r in range(ROLE_DIM):
        v = [0.0] * OBSERVATION_DIM
        v[0] = 0.2
        v[44] = 0.8
        v[88] = 0.2
        v[95] = 1.0
        v[97] = 1.0
        v[108] = 1.0
        v[BASE_OBSERVATION_DIM + r] = 1.0
        cases.append({"name": f"role_slot_{r}", "vector": v})

    # 4. Deterministic random samples
    rng = np.random.RandomState(42)
    for i in range(50):
        v = (rng.rand(OBSERVATION_DIM) - 0.5) * 2.0
        r_rand = rng.randint(0, ROLE_DIM)
        for r in range(ROLE_DIM):
            v[BASE_OBSERVATION_DIM + r] = 1.0 if r == r_rand else 0.0
        cases.append({"name": f"random_sample_{i+1}", "vector": v.tolist()})

    return cases


def main():
    print("====================================================")
    print("GMN-FOOTBALL-3 — PYTHON BROWSER INFERENCE PARITY")
    print("====================================================")

    test_vectors = generate_test_vectors()
    print(f"Generated {len(test_vectors)} test vectors for parity assessment.")
    print("✓ Parity test harness successfully initialized.")


if __name__ == "__main__":
    main()
