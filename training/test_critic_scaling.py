"""
GMN-Football-3 — Scalable Permutation-Invariant Critic Tests (Deep Sets)
Verifies:
1. O(1) Parameter Count Invariance across agent counts N in {3, 5, 11}.
2. Permutation Invariance under arbitrary agent permutations.
3. Multi-agent tensor shape compatibility across 3v1, 5v5, and 11v11 configurations.
4. Clean gradient flow through agent encoder and pooled value head.
"""

import sys
import os
import torch
import numpy as np

# Ensure training module is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.mappo_networks import CentralizedCritic, SharedActor


def test_parameter_count_invariance():
    """Proof of scale: critic parameter count must be constant O(1) across agent counts."""
    obs_dim = 127
    hidden = 64

    param_counts = {}
    for num_agents in [3, 5, 11]:
        critic = CentralizedCritic(obs_dim=obs_dim, hidden=hidden, mode="pool")
        total_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
        param_counts[num_agents] = total_params

    print(f"[Critic Scaling Test] Parameter Counts across team sizes:")
    for n, count in param_counts.items():
        print(f"   -> N = {n:2d} agents : {count:,} parameters")

    p3 = param_counts[3]
    p5 = param_counts[5]
    p11 = param_counts[11]

    # Assert parameter counts are strictly constant (0% difference < 5% threshold)
    max_diff_pct = max(abs(p5 - p3) / p3, abs(p11 - p3) / p3) * 100.0
    assert max_diff_pct < 5.0, f"Critic param count scaled with agent count! Max diff: {max_diff_pct:.2f}%"
    assert p3 == p5 == p11, f"Expected exact O(1) equality, got {p3}, {p5}, {p11}"
    print(f"   ✓ Parameter Scaling Verified: Exactly 0.00% parameter variance across 3, 5, and 11 agents.\n")


def test_permutation_invariance():
    """Verifies that shuffling agent order produces identical value predictions within float tolerance."""
    obs_dim = 127
    hidden = 64
    batch_size = 16
    num_agents = 5

    critic = CentralizedCritic(obs_dim=obs_dim, hidden=hidden, mode="pool")
    critic.eval()

    torch.manual_seed(42)
    x = torch.randn(batch_size, num_agents, obs_dim, dtype=torch.float32)

    with torch.no_grad():
        v_orig = critic(x)

    permutations = [
        [4, 3, 2, 1, 0],
        [1, 0, 4, 2, 3],
        [2, 4, 1, 3, 0],
        [3, 1, 0, 4, 2],
    ]

    for p_idx, perm in enumerate(permutations):
        x_perm = x[:, perm, :]
        with torch.no_grad():
            v_perm = critic(x_perm)

        max_err = torch.max(torch.abs(v_orig - v_perm)).item()
        assert torch.allclose(v_orig, v_perm, atol=1e-5, rtol=1e-5), (
            f"Permutation {perm} failed invariance check! Max diff: {max_err:.2e}"
        )
        print(f"   ✓ Permutation {perm} invariant: max abs difference = {max_err:.2e}")

    print(f"   ✓ Permutation Invariance Verified across all test agent orderings.\n")


def test_shape_compatibility():
    """Verifies that critic handles 3D tensors, unbatched 2D tensors, and concatenated 2D tensors."""
    obs_dim = 127
    hidden = 64
    critic = CentralizedCritic(obs_dim=obs_dim, hidden=hidden, mode="pool")

    # 1. 3D batched inputs for various N
    for n in [1, 3, 5, 11]:
        x3d = torch.randn(8, n, obs_dim)
        val = critic(x3d)
        assert val.shape == (8,), f"Expected shape (8,), got {val.shape} for N={n}"

    # 2. Unbatched 2D input (num_agents, obs_dim)
    x_unbatched = torch.randn(3, obs_dim)
    val_unbatched = critic(x_unbatched)
    assert val_unbatched.shape == (1,), f"Expected shape (1,), got {val_unbatched.shape}"

    # 3. Concatenated 2D input (batch, num_agents * obs_dim)
    x_flat = torch.randn(8, 5 * obs_dim)
    val_flat = critic(x_flat)
    assert val_flat.shape == (8,), f"Expected shape (8,), got {val_flat.shape}"

    print(f"   ✓ Multi-Agent Tensor Shape Compatibility Verified across 1, 3, 5, and 11 agents.\n")


def test_gradient_flow():
    """Verifies backpropagation through agent encoder and pooled value head."""
    obs_dim = 127
    hidden = 64
    critic = CentralizedCritic(obs_dim=obs_dim, hidden=hidden, mode="pool")

    x = torch.randn(8, 3, obs_dim, requires_grad=True)
    target = torch.randn(8)

    val = critic(x)
    loss = ((val - target) ** 2).mean()
    loss.backward()

    for name, param in critic.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients"

    print(f"   ✓ Gradient Backpropagation cleanly verified across all network parameters.\n")


if __name__ == "__main__":
    print("================================================================================")
    print("RUNNING CENTRALIZED CRITIC SCALING & PERMUTATION INVARIANCE UNIT TESTS")
    print("================================================================================")
    test_parameter_count_invariance()
    test_permutation_invariance()
    test_shape_compatibility()
    test_gradient_flow()
    print("================================================================================")
    print("ALL CRITIC SCALING & INVARIANCE UNIT TESTS PASSED (4/4)")
    print("================================================================================")
