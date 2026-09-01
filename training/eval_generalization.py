"""
GMN-Football-3 — Multi-Agent Generalization Evaluation Harness (Step 1)
Evaluates whether trained MAPPO policies generalize to varied opponent defensive formations
and goalkeeper arrangements, or whether they have overfitted / memorized fixed positions.

NOTE: This script requires a running local TypeScript engine (npx tsx training/bridge_server.ts)
and must be executed on the user's local machine with active WebSocket bridge connection.
"""

import argparse
import csv
import datetime
import getpass
import os
import platform
import sys
from typing import Dict, List, Any, Tuple
import numpy as np
import torch

# Ensure repo root is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor


VARIATION_SCENARIO_MAP = {
    "baseline": "academy_3_vs_1_with_keeper",
    "defender_2": "academy_3_vs_1_defender_2",
    "defender_3": "academy_3_vs_1_defender_3",
    "keeper_aggressive": "academy_3_vs_1_keeper_aggressive",
    "shifted": "academy_3_vs_1_shifted",
    "randomized": "academy_3_vs_1_randomized",
}


def evaluate_variation(
    actor: SharedActor,
    variation_name: str,
    scenario_code: str,
    num_episodes: int = 50,
    base_seed: int = 500000,
) -> Tuple[Dict[str, Any], int]:
    """Runs deterministic evaluation of frozen policy on a specific scenario variation."""
    env = GMNMultiAgentEnv(scenario=scenario_code, auto_start_bridge=True)
    controllable_agents = list(env.possible_agents)

    rewards_list = []
    lengths_list = []
    goals_list = []
    total_steps_executed = 0

    for ep in range(1, num_episodes + 1):
        ep_seed = base_seed + ep
        obs_dict, _ = env.reset(seed=ep_seed)
        ep_reward = 0.0
        ep_length = 0
        goal_scored = 0

        while True:
            current_agents = list(env.agents if env.agents else controllable_agents)
            local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)

            with torch.no_grad():
                dist = actor(torch.from_numpy(local_obs).float())
                actions = dist.logits.argmax(dim=-1)

            action_dict = {a: int(actions[i].item()) for i, a in enumerate(current_agents)}
            obs_dict, rewards, terminations, truncations, infos = env.step(action_dict)
            total_steps_executed += 1

            shared_rew = float(rewards[current_agents[0]]) if current_agents and current_agents[0] in rewards else 0.0
            ep_reward += shared_rew
            ep_length += 1

            term = any(terminations.values()) if terminations else False
            trunc = any(truncations.values()) if truncations else False
            done = term or trunc

            if done:
                step_score = {"left": 0, "right": 0}
                if infos:
                    for agent_info in infos.values():
                        if "score" in agent_info:
                            step_score = agent_info["score"]
                            break
                if step_score.get("left", 0) > 0:
                    goal_scored = 1
                break

        rewards_list.append(ep_reward)
        lengths_list.append(ep_length)
        goals_list.append(goal_scored)

    env.close()

    mean_rew = float(np.mean(rewards_list))
    std_rew = float(np.std(rewards_list))
    goal_rate = float(np.mean(goals_list)) * 100.0
    mean_len = float(np.mean(lengths_list))

    return {
        "variation": variation_name,
        "scenario": scenario_code,
        "goal_rate": goal_rate,
        "mean_reward": mean_rew,
        "std_reward": std_rew,
        "mean_length": mean_len,
        "goals_count": int(sum(goals_list)),
        "episodes": num_episodes,
    }, total_steps_executed


def run_generalization_suite(
    checkpoint_path: str = "training/models/mappo_academy_3_vs_1_with_keeper_trained.pt",
    variations: List[str] = None,
    num_episodes: int = 50,
    base_seed: int = 500000,
    output_csv: str = "training/results/generalization.csv",
) -> List[Dict[str, Any]]:
    if variations is None:
        variations = ["baseline", "defender_2", "defender_3", "keeper_aggressive", "shifted", "randomized"]

    provenance_host = platform.node()
    provenance_user = getpass.getuser()
    provenance_date = datetime.datetime.now().isoformat()
    provenance_cmd = " ".join(sys.argv)
    provenance_str = f"host={provenance_host}|user={provenance_user}|date={provenance_date}|cmd={provenance_cmd}"

    print("================================================================================")
    print("GMN-FOOTBALL-3 — MAPPO GENERALIZATION BENCHMARK SUITE")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Variations : {', '.join(variations)}")
    print(f"Episodes   : {num_episodes} per variation (Deterministic Seeds {base_seed + 1}..{base_seed + num_episodes})")
    print(f"Provenance : {provenance_str}")
    print("================================================================================")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Inspect checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    obs_dim = checkpoint.get("obs_dim", 115)
    action_dim = checkpoint.get("action_dim", 19)
    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    results = []
    baseline_goal_rate = None
    suite_total_steps = 0

    for var_name in variations:
        scenario_code = VARIATION_SCENARIO_MAP.get(var_name, var_name)
        print(f"\n>> Evaluating variation: '{var_name}' (Scenario: {scenario_code})...")
        res, var_steps = evaluate_variation(actor, var_name, scenario_code, num_episodes, base_seed)
        suite_total_steps += var_steps
        
        if var_name == "baseline" or baseline_goal_rate is None:
            baseline_goal_rate = res["goal_rate"]

        drop = baseline_goal_rate - res["goal_rate"]
        res["generalization_drop"] = drop
        res["checkpoint"] = os.path.basename(checkpoint_path)
        res["provenance"] = provenance_str
        results.append(res)

        print(
            f"   Result: Goal Rate = {res['goal_rate']:5.1f}% | "
            f"Reward = {res['mean_reward']:+.3f} ± {res['std_reward']:.3f} | "
            f"Length = {res['mean_length']:.1f} | "
            f"Gen Drop = {drop:+5.1f}% | Steps: {var_steps}"
        )

    # Runtime Guard: Refuse to write output if zero steps were executed
    if suite_total_steps == 0:
        raise RuntimeError(
            "[Runtime Guard Error] Zero environment steps were executed during generalization evaluation. "
            "Refusing to write fake or unmeasured rows. Ensure the local TypeScript bridge is actively serving."
        )

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "checkpoint",
        "variation",
        "scenario",
        "episodes",
        "goal_rate",
        "generalization_drop",
        "mean_reward",
        "std_reward",
        "mean_length",
        "goals_count",
        "provenance",
    ]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print("\n================================================================================")
    print("GENERALIZATION EVALUATION SUMMARY TABLE")
    print("================================================================================")
    print(f"{'Variation':<20} | {'Goal Rate':<10} | {'Gen Drop':<10} | {'Mean Reward':<16} | {'Avg Steps':<10}")
    print("-" * 76)
    for r in results:
        drop_str = f"{r['generalization_drop']:+.1f}%" if r["variation"] != "baseline" else "0.0% (ref)"
        print(
            f"{r['variation']:<20} | {r['goal_rate']:>8.1f}% | {drop_str:>10} | "
            f"{r['mean_reward']:>+7.3f} ± {r['std_reward']:<5.3f} | {r['mean_length']:>8.1f}"
        )
    print("================================================================================")
    print(f"Results written with provenance to: {output_csv}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MAPPO generalization across opponent variations. "
                    "NOTE: Requires local TypeScript engine execution (npx tsx training/bridge_server.ts)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="training/models/mappo_academy_3_vs_1_with_keeper_trained.pt",
        help="Path to trained MAPPO checkpoint (.pt)",
    )
    parser.add_argument(
        "--variations",
        type=str,
        nargs="+",
        default=["baseline", "defender_2", "defender_3", "keeper_aggressive", "shifted", "randomized"],
        help="List of variations to evaluate",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per variation")
    parser.add_argument("--base-seed", type=int, default=500000, help="Base seed for reproducibility")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="training/results/generalization.csv",
        help="CSV file path to save generalization results",
    )
    args = parser.parse_args()

    run_generalization_suite(
        checkpoint_path=args.checkpoint,
        variations=args.variations,
        num_episodes=args.episodes,
        base_seed=args.base_seed,
        output_csv=args.output_csv,
    )
