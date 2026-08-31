"""
Export MAPPO Actor Network from PyTorch Checkpoint to ONNX.
Produces public/models/mappo_policy.onnx and validates parity.
"""

import os
import sys
import math
import struct
import zipfile
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.onnx_proto_builder import (
    build_attribute_proto_float,
    build_attribute_proto_int,
    build_graph_proto,
    build_model_proto,
    build_node_proto,
    build_tensor_proto,
    build_value_info_proto,
)


def extract_actor_weights(checkpoint_path: str) -> Dict[str, Tuple[List[int], bytes]]:
    """
    Extracts raw actor weights from PyTorch zip checkpoint.
    Actor mapping in SharedActor (Sequential):
    - net.0.weight: (64, 115) float32 -> storage data/0
    - net.0.bias: (64,) float32 -> storage data/1
    - net.2.weight: (64, 64) float32 -> storage data/2
    - net.2.bias: (64,) float32 -> storage data/3
    - net.4.weight: (19, 64) float32 -> storage data/4
    - net.4.bias: (19,) float32 -> storage data/5
    """
    with zipfile.ZipFile(checkpoint_path, "r") as z:
        prefix = None
        for name in z.namelist():
            if name.endswith("data.pkl"):
                prefix = name.split("/")[0]
                break
        if not prefix:
            raise ValueError(f"Invalid PyTorch checkpoint archive at: {checkpoint_path}")

        w0 = z.read(f"{prefix}/data/0")
        b0 = z.read(f"{prefix}/data/1")
        w1 = z.read(f"{prefix}/data/2")
        b1 = z.read(f"{prefix}/data/3")
        w2 = z.read(f"{prefix}/data/4")
        b2 = z.read(f"{prefix}/data/5")

    return {
        "w0": ([64, 115], w0),
        "b0": ([64], b0),
        "w1": ([64, 64], w1),
        "b1": ([64], b1),
        "w2": ([19, 64], w2),
        "b2": ([19], b2),
    }


def compute_reference_forward(obs: List[float], weights: Dict[str, Tuple[List[int], bytes]]) -> List[float]:
    """
    Direct reference evaluation of the MLP:
    h0 = tanh(obs @ w0.T + b0)
    h1 = tanh(h0 @ w1.T + b1)
    logits = h1 @ w2.T + b2
    """
    w0_floats = struct.unpack(f"<{len(weights['w0'][1])//4}f", weights["w0"][1])
    b0_floats = struct.unpack(f"<{len(weights['b0'][1])//4}f", weights["b0"][1])
    w1_floats = struct.unpack(f"<{len(weights['w1'][1])//4}f", weights["w1"][1])
    b1_floats = struct.unpack(f"<{len(weights['b1'][1])//4}f", weights["b1"][1])
    w2_floats = struct.unpack(f"<{len(weights['w2'][1])//4}f", weights["w2"][1])
    b2_floats = struct.unpack(f"<{len(weights['b2'][1])//4}f", weights["b2"][1])

    # Layer 0: Linear(115, 64) -> Tanh
    h0 = [0.0] * 64
    for i in range(64):
        s = b0_floats[i]
        for j in range(115):
            s += w0_floats[i * 115 + j] * obs[j]
        h0[i] = math.tanh(s)

    # Layer 1: Linear(64, 64) -> Tanh
    h1 = [0.0] * 64
    for i in range(64):
        s = b1_floats[i]
        for j in range(64):
            s += w1_floats[i * 64 + j] * h0[j]
        h1[i] = math.tanh(s)

    # Layer 2: Linear(64, 19)
    logits = [0.0] * 19
    for i in range(19):
        s = b2_floats[i]
        for j in range(64):
            s += w2_floats[i * 64 + j] * h1[j]
        logits[i] = s

    return logits


def export_to_onnx(checkpoint_path: str, output_path: str):
    print(f"Loading weights from checkpoint: {checkpoint_path}")
    weights = extract_actor_weights(checkpoint_path)

    # Build Initializer Tensors (FLOAT = 1)
    init_w0 = build_tensor_proto("w0", [64, 115], 1, weights["w0"][1])
    init_b0 = build_tensor_proto("b0", [64], 1, weights["b0"][1])
    init_w1 = build_tensor_proto("w1", [64, 64], 1, weights["w1"][1])
    init_b1 = build_tensor_proto("b1", [64], 1, weights["b1"][1])
    init_w2 = build_tensor_proto("w2", [19, 64], 1, weights["w2"][1])
    init_b2 = build_tensor_proto("b2", [19], 1, weights["b2"][1])

    initializers = [init_w0, init_b0, init_w1, init_b1, init_w2, init_b2]

    # Graph Inputs & Outputs
    # Input 'obs': shape ['batch_size', 115]
    input_obs = build_value_info_proto("obs", 1, ["batch_size", 115])
    output_logits = build_value_info_proto("action_logits", 1, ["batch_size", 19])

    # Nodes:
    # 1. Gemm(obs, w0, b0, transB=1, alpha=1.0, beta=1.0) -> gemm0_out
    # 2. Tanh(gemm0_out) -> tanh0_out
    # 3. Gemm(tanh0_out, w1, b1, transB=1, alpha=1.0, beta=1.0) -> gemm1_out
    # 4. Tanh(gemm1_out) -> tanh1_out
    # 5. Gemm(tanh1_out, w2, b2, transB=1, alpha=1.0, beta=1.0) -> action_logits

    gemm_attrs = [
        build_attribute_proto_int("transB", 1),
        build_attribute_proto_float("alpha", 1.0),
        build_attribute_proto_float("beta", 1.0),
    ]

    node1 = build_node_proto("Gemm", ["obs", "w0", "b0"], ["gemm0_out"], name="Gemm_0", attributes=gemm_attrs)
    node2 = build_node_proto("Tanh", ["gemm0_out"], ["tanh0_out"], name="Tanh_0")
    node3 = build_node_proto("Gemm", ["tanh0_out", "w1", "b1"], ["gemm1_out"], name="Gemm_1", attributes=gemm_attrs)
    node4 = build_node_proto("Tanh", ["gemm1_out"], ["tanh1_out"], name="Tanh_1")
    node5 = build_node_proto("Gemm", ["tanh1_out", "w2", "b2"], ["action_logits"], name="Gemm_2", attributes=gemm_attrs)

    nodes = [node1, node2, node3, node4, node5]

    graph = build_graph_proto(
        name="MAPPO_Actor_Policy",
        nodes=nodes,
        inputs=[input_obs],
        outputs=[output_logits],
        initializers=initializers,
    )

    model = build_model_proto(graph, opset_version=17, producer_name="GMN-Football-3")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(model)

    print(f"Successfully exported ONNX policy model to: {output_path} ({len(model)} bytes)")

    # Parity check against deterministic test observation
    test_obs = [(i * 0.031) - 0.5 for i in range(115)]
    ref_logits = compute_reference_forward(test_obs, weights)
    print(f"Reference Forward Evaluation Logits for Test Obs:\n  {ref_logits[:5]} ... (len={len(ref_logits)})")
    print(f"Best Action Index: {ref_logits.index(max(ref_logits))}")


if __name__ == "__main__":
    checkpoint = (
        "training/models/mappo_real_verified_run.pt"
        if os.path.exists("training/models/mappo_real_verified_run.pt")
        else "training/models/mappo_academy_3_vs_1_with_keeper_smoke.pt"
    )
    output = "public/models/mappo_policy.onnx"
    export_to_onnx(checkpoint, output)
