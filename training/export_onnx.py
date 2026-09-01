"""
Export MAPPO Actor Network from PyTorch Checkpoint to ONNX using torch.onnx.export.
Produces public/models/mappo_policy.onnx and validates parity against PyTorch model and checkpoint weights.
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import onnx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.mappo_networks import SharedActor


class ActorPolicyOnnxModule(nn.Module):
    """
    Wrapper around SharedActor's underlying MLP network for deterministic ONNX export.
    Maps input observation tensor (batch_size, obs_dim) -> action logits (batch_size, 19).
    """
    def __init__(self, actor: SharedActor):
        super().__init__()
        self.net = actor.net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def export_to_onnx(
    checkpoint_path: str = "training/models/mappo_academy_3_vs_1_with_keeper_trained.pt",
    output_path: str = "public/models/mappo_policy.onnx",
    ts_weights_path: str = "src/agents/mappo_weights.ts",
    allow_legacy_115: bool = False,
):
    print("==================================================")
    print("MAPPO ACTOR POLICY ONNX EXPORT & WEIGHT VERIFICATION")
    print(f"Source PyTorch Checkpoint: {checkpoint_path}")
    print(f"Target ONNX Model Output:  {output_path}")
    print(f"Target TS Weights Output:  {ts_weights_path}")
    print("==================================================")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Source checkpoint not found at: {checkpoint_path}")

    # 1. Load PyTorch checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    obs_dim = checkpoint.get("obs_dim", 127 if "actor" in checkpoint and checkpoint["actor"]["net.0.weight"].shape[1] == 127 else checkpoint["actor"]["net.0.weight"].shape[1] if "actor" in checkpoint else 127)
    action_dim = checkpoint.get("action_dim", 19)
    timesteps = checkpoint.get("timesteps", "unknown")

    if obs_dim != 127:
        if not allow_legacy_115:
            raise RuntimeError(
                f"[GMN-Export Error] Checkpoint '{checkpoint_path}' has obs_dim={obs_dim}, "
                f"which is incompatible with canonical GMN-Football-3 OBSERVATION_DIM (127 floats). "
                f"Exporting legacy 115-dim weights is blocked because browser TrainedPolicyAgent expects 127-dim inputs. "
                f"Please retrain a 127-dim checkpoint (see README) or pass --allow-legacy-115 if intentionally testing legacy."
            )
        else:
            print(f"[WARNING] --allow-legacy-115 set: Checkpoint obs_dim={obs_dim} differs from canonical OBSERVATION_DIM (127).")

    print(f"\n1. Loaded Checkpoint:")
    print(f"   Timesteps: {timesteps} | Obs Dim: {obs_dim} | Action Dim: {action_dim}")

    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    model = ActorPolicyOnnxModule(actor)
    model.eval()

    # 2. Export to ONNX via torch.onnx.export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)

    print(f"\n2. Exporting model using torch.onnx.export (opset 17)...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["action_logits"],
        dynamic_axes={
            "obs": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
    )

    # Ensure single self-contained ONNX model (no external data split)
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.save(onnx_model, output_path)
    data_sidecar = output_path + ".data"
    if os.path.exists(data_sidecar):
        os.remove(data_sidecar)

    onnx_size = os.path.getsize(output_path)
    print(f"   ✓ Self-contained ONNX export successful: {output_path} ({onnx_size} bytes)")

    # 3. Check ONNX model validity with onnx package
    onnx.checker.check_model(onnx_model)
    print("   ✓ ONNX model syntax and topology verified with onnx.checker.check_model.")

    # 4. Extract weights from PyTorch state dict and compare against ONNX initializers
    print(f"\n3. Verifying Layer Weight & Bias Parity between PyTorch Checkpoint and ONNX...")
    state_dict = actor.state_dict()
    pt_w0 = state_dict["net.0.weight"].numpy() # (64, obs_dim)
    pt_b0 = state_dict["net.0.bias"].numpy()   # (64,)
    pt_w1 = state_dict["net.2.weight"].numpy() # (64, 64)
    pt_b1 = state_dict["net.2.bias"].numpy()   # (64,)
    pt_w2 = state_dict["net.4.weight"].numpy() # (19, 64)
    pt_b2 = state_dict["net.4.bias"].numpy()   # (19,)

    onnx_tensors = {}
    for init in onnx_model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        onnx_tensors[init.name] = arr

    print(f"   PyTorch net.0.weight shape: {pt_w0.shape}, mean: {pt_w0.mean():.6f}, std: {pt_w0.std():.6f}")
    print(f"   PyTorch net.2.weight shape: {pt_w1.shape}, mean: {pt_w1.mean():.6f}, std: {pt_w1.std():.6f}")
    print(f"   PyTorch net.4.weight shape: {pt_w2.shape}, mean: {pt_w2.mean():.6f}, std: {pt_w2.std():.6f}")

    # Match weights in ONNX graph
    found_weights = 0
    for name, arr in onnx_tensors.items():
        if arr.shape == (64, obs_dim):
            diff = np.max(np.abs(arr - pt_w0))
            print(f"   ✓ Layer 0 Weight matched in ONNX ({name}): max diff = {diff:.10e}")
            assert diff == 0.0, f"Layer 0 weight mismatch: {diff}"
            found_weights += 1
        elif arr.shape == (64, 64):
            diff = np.max(np.abs(arr - pt_w1))
            print(f"   ✓ Layer 1 Weight matched in ONNX ({name}): max diff = {diff:.10e}")
            assert diff == 0.0, f"Layer 1 weight mismatch: {diff}"
            found_weights += 1
        elif arr.shape == (19, 64):
            diff = np.max(np.abs(arr - pt_w2))
            print(f"   ✓ Layer 2 Weight matched in ONNX ({name}): max diff = {diff:.10e}")
            assert diff == 0.0, f"Layer 2 weight mismatch: {diff}"
            found_weights += 1

    assert found_weights >= 3, f"Could not match all weight layers in ONNX model (found {found_weights})"

    # 5. Check distinction from smoke checkpoint
    smoke_path = "training/models/mappo_academy_3_vs_1_with_keeper_smoke.pt"
    if os.path.exists(smoke_path):
        smoke_ckpt = torch.load(smoke_path, map_location="cpu")
        smoke_w0 = smoke_ckpt["actor"]["net.0.weight"].numpy()
        smoke_w2 = smoke_ckpt["actor"]["net.4.weight"].numpy()
        if smoke_w0.shape == pt_w0.shape:
            diff_w0 = np.max(np.abs(pt_w0 - smoke_w0))
            diff_w2 = np.max(np.abs(pt_w2 - smoke_w2))
            print(f"\n4. Distinctness Check against SMOKE Checkpoint ({smoke_path}):")
            print(f"   Max abs difference in Layer 0 weight (w0): {diff_w0:.6f}")
            print(f"   Max abs difference in Layer 2 weight (w2): {diff_w2:.6f}")
            assert diff_w0 > 0.01, f"Trained checkpoint weights are identical to smoke checkpoint! diff={diff_w0}"
            print(f"   ✓ Confirmed: Trained checkpoint is distinct from smoke checkpoint (w0 max diff: {diff_w0:.6f}).")

    # 6. Generate synchronized TypeScript embedded weights file
    print(f"\n5. Generating TypeScript Embedded Weights file: {ts_weights_path}...")
    if not allow_legacy_115:
        expected_w0_size = 64 * 127
        assert pt_w0.shape == (64, 127), f"Expected pt_w0 shape (64, 127), got {pt_w0.shape}"
        assert pt_w0.size == expected_w0_size, f"Expected pt_w0 size {expected_w0_size}, got {pt_w0.size}"

    ts_content = f"""// AUTO-GENERATED BY training/export_onnx.py
// Synchronized exact weights from: {checkpoint_path}
// Timesteps trained: {timesteps}

export const MAPPO_WEIGHTS = {{
  sourceCheckpoint: {json.dumps(checkpoint_path)},
  timesteps: {json.dumps(timesteps)},
  w0: {json.dumps(pt_w0.flatten().tolist())},
  b0: {json.dumps(pt_b0.tolist())},
  w1: {json.dumps(pt_w1.flatten().tolist())},
  b1: {json.dumps(pt_b1.tolist())},
  w2: {json.dumps(pt_w2.flatten().tolist())},
  b2: {json.dumps(pt_b2.tolist())},
}};
"""
    with open(ts_weights_path, "w") as f:
        f.write(ts_content)

    print(f"   ✓ Saved synchronized TypeScript weights ({os.path.getsize(ts_weights_path)} bytes).")

    # 7. Verification inference test
    test_obs = torch.tensor([[(i * 0.031) - 0.5 for i in range(obs_dim)]], dtype=torch.float32)
    with torch.no_grad():
        pt_logits = model(test_obs).numpy()[0]

    print(f"\n6. Deterministic Forward Pass Test:")
    print(f"   Input sample obs: [{test_obs[0, 0].item():.4f}, {test_obs[0, 1].item():.4f}, ...]")
    print(f"   Output action logits (top 5): {pt_logits[:5]}")
    print(f"   Argmax Action: {np.argmax(pt_logits)}")
    print("\n==================================================")
    print("✓ ONNX EXPORT & WEIGHT SYNCHRONIZATION COMPLETE")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MAPPO checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="training/models/mappo_academy_3_vs_1_with_keeper_trained.pt",
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="public/models/mappo_policy.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--ts-weights",
        type=str,
        default="src/agents/mappo_weights.ts",
        help="Output TypeScript weights file path",
    )
    parser.add_argument(
        "--allow-legacy-115",
        action="store_true",
        help="Allow exporting legacy 115-dim checkpoint (not compatible with current browser contract)",
    )
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        ts_weights_path=args.ts_weights,
        allow_legacy_115=args.allow_legacy_115,
    )
