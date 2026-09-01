"""
GMN-Football-3 — Comparison Table Generator
Generates Markdown and HTML comparison tables for MAPPO vs. IPPO / PPO
benchmarking win-rate (goal conversion rate) and learning rate progression across 100k intervals.
"""

import os
import sys
import csv
from typing import Dict, Any, List, Set

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
CSV_PATH = os.path.join(RESULTS_DIR, "win_rate_progress.csv")
MD_OUT_PATH = os.path.join(RESULTS_DIR, "comparison_table.md")
HTML_OUT_PATH = os.path.join(RESULTS_DIR, "comparison_table.html")

STANDARD_STEPS = [100000 * i for i in range(1, 11)]  # 100k to 1M


def load_win_rate_data(csv_path: str = CSV_PATH) -> List[Dict[str, Any]]:
    """Loads records from win_rate_progress.csv."""
    if not os.path.exists(csv_path):
        return []
    records = []
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append({
                    "scenario": row.get("scenario", "").strip(),
                    "algorithm": row.get("algorithm", "").strip().upper(),
                    "step": int(float(row.get("step", 0))),
                    "learning_rate": row.get("learning_rate", "--").strip(),
                    "goal_rate_pct": float(row.get("goal_rate_pct", 0.0)) if row.get("goal_rate_pct") else None,
                    "mean_reward": float(row.get("mean_reward", 0.0)) if row.get("mean_reward") else None,
                    "shots_per_ep": float(row.get("shots_per_ep", 0.0)) if row.get("shots_per_ep") else None,
                    "turnover_rate": float(row.get("turnover_rate", 0.0)) if row.get("turnover_rate") else None,
                    "episodes": int(float(row.get("episodes", 0))) if row.get("episodes") else None,
                    "checkpoint_path": row.get("checkpoint_path", ""),
                })
            except Exception as e:
                print(f"[generate_comparison_table] Warning skipping invalid row {row}: {e}")
    return records


def build_comparison_matrices(records: List[Dict[str, Any]]):
    """
    Builds structured lookup tables:
    data[scenario][step][algorithm] = record
    """
    scenarios: Set[str] = set()
    algorithms: Set[str] = set()
    all_steps_set = set(STANDARD_STEPS)

    matrix: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = {}

    for r in records:
        sc = r["scenario"]
        algo = r["algorithm"]
        st = r["step"]
        scenarios.add(sc)
        algorithms.add(algo)
        all_steps_set.add(st)

        if sc not in matrix:
            matrix[sc] = {}
        if st not in matrix[sc]:
            matrix[sc][st] = {}
        matrix[sc][st][algo] = r

    sorted_steps = sorted(list(all_steps_set))
    sorted_scenarios = sorted(list(scenarios)) if scenarios else ["academy_3_vs_1_with_keeper"]
    # Ensure preferred algorithm ordering
    preferred_algos = ["MAPPO", "IPPO", "PPO"]
    ordered_algos = [a for a in preferred_algos if a in algorithms] + [a for a in sorted(algorithms) if a not in preferred_algos]
    if not ordered_algos:
        ordered_algos = ["MAPPO", "IPPO"]

    return matrix, sorted_scenarios, ordered_algos, sorted_steps


def generate_markdown(matrix, scenarios, algorithms, steps) -> str:
    md_lines = [
        "# GMN-Football-3 — Reinforcement Learning Benchmark Comparison Table",
        "",
        "> **Methodology**: Win-Rate is defined strictly as Goal Conversion Rate (% of evaluation episodes where left team scores $\\ge 1$ goal) evaluated deterministically over fixed evaluation seeds. Missing checkpoints are indicated with `--`.",
        "",
    ]

    for sc in scenarios:
        md_lines.append(f"## Scenario: `{sc}`")
        md_lines.append("")

        # Header 1
        h1 = ["| Steps | LR "]
        h2 = ["| :--- | :---: "]
        for algo in algorithms:
            h1.append(f"| {algo} Win-Rate | {algo} Mean Rew | {algo} Shots/Ep ")
            h2.append("| :---: | :---: | :---: ")
        h1.append("|")
        h2.append("|")

        md_lines.append("".join(h1))
        md_lines.append("".join(h2))

        for st in steps:
            step_str = f"{st // 1000}k" if st % 1000 == 0 else f"{st}"
            row_cells = [f"| **{step_str}**"]

            # Common or default LR
            first_lr = "--"
            for algo in algorithms:
                rec = matrix.get(sc, {}).get(st, {}).get(algo)
                if rec and rec.get("learning_rate") != "--":
                    first_lr = rec.get("learning_rate")
                    break
            row_cells.append(f"| `{first_lr}` ")

            for algo in algorithms:
                rec = matrix.get(sc, {}).get(st, {}).get(algo)
                if rec and rec.get("goal_rate_pct") is not None:
                    gr = f"**{rec['goal_rate_pct']:.1f}%**"
                    mr = f"{rec['mean_reward']:+.3f}" if rec.get("mean_reward") is not None else "--"
                    sh = f"{rec['shots_per_ep']:.2f}" if rec.get("shots_per_ep") is not None else "--"
                else:
                    gr = "--"
                    mr = "--"
                    sh = "--"
                row_cells.append(f"| {gr} | {mr} | {sh} ")

            row_cells.append("|")
            md_lines.append("".join(row_cells))

        md_lines.append("")

    return "\n".join(md_lines)


def generate_html(matrix, scenarios, algorithms, steps) -> str:
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "  <title>GMN-Football-3 Benchmark Comparison</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; color: #f8fafc; padding: 2rem; margin: 0; line-height: 1.5; }",
        "    .container { max-width: 1200px; margin: 0 auto; }",
        "    h1 { color: #38bdf8; font-size: 1.8rem; margin-bottom: 0.5rem; }",
        "    p.subtitle { color: #94a3b8; font-size: 0.95rem; margin-bottom: 2rem; }",
        "    .scenario-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; margin-bottom: 2.5rem; overflow-x: auto; }",
        "    .scenario-title { font-size: 1.25rem; font-weight: 600; color: #f1f5f9; margin-top: 0; margin-bottom: 1rem; }",
        "    table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }",
        "    th { background: #0f172a; color: #cbd5e1; font-weight: 600; padding: 0.75rem 1rem; border-bottom: 2px solid #475569; }",
        "    td { padding: 0.75rem 1rem; border-bottom: 1px solid #334155; }",
        "    tr:hover { background: #283548; }",
        "    .step-col { font-weight: bold; color: #e2e8f0; font-family: monospace; }",
        "    .lr-col { font-family: monospace; color: #94a3b8; }",
        "    .win-rate-mappo { color: #4ade80; font-weight: bold; }",
        "    .win-rate-ippo { color: #60a5fa; font-weight: bold; }",
        "    .win-rate-ppo { color: #facc15; font-weight: bold; }",
        "    .empty-cell { color: #64748b; }",
        "    .badge { display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-left: 0.25rem; }",
        "    .badge-mappo { background: rgba(74, 222, 128, 0.15); color: #4ade80; }",
        "    .badge-ippo { background: rgba(96, 165, 250, 0.15); color: #60a5fa; }",
        "    .badge-ppo { background: rgba(250, 204, 21, 0.15); color: #facc15; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class='container'>",
        "    <h1>⚽ GMN-Football-3 Benchmark Comparison</h1>",
        "    <p class='subtitle'>Deterministic Checkpoint Evaluation Across 100k Timestep Intervals (Goal Conversion Rate & Policy Metrics)</p>",
    ]

    for sc in scenarios:
        html_lines.append("    <div class='scenario-card'>")
        html_lines.append(f"      <div class='scenario-title'>Scenario: <code>{sc}</code></div>")
        html_lines.append("      <table>")
        html_lines.append("        <thead>")
        html_lines.append("          <tr>")
        html_lines.append("            <th>Step</th>")
        html_lines.append("            <th>LR</th>")

        for algo in algorithms:
            badge_cls = f"badge-{algo.lower()}"
            html_lines.append(f"            <th colspan='3'>{algo} <span class='badge {badge_cls}'>{algo}</span></th>")
        html_lines.append("          </tr>")

        html_lines.append("          <tr>")
        html_lines.append("            <th></th>")
        html_lines.append("            <th></th>")
        for algo in algorithms:
            html_lines.append("            <th>Win-Rate</th>")
            html_lines.append("            <th>Mean Reward</th>")
            html_lines.append("            <th>Shots/Ep</th>")
        html_lines.append("          </tr>")
        html_lines.append("        </thead>")
        html_lines.append("        <tbody>")

        for st in steps:
            step_str = f"{st // 1000}k" if st % 1000 == 0 else f"{st}"
            html_lines.append("          <tr>")
            html_lines.append(f"            <td class='step-col'>{step_str}</td>")

            # LR
            first_lr = "--"
            for algo in algorithms:
                rec = matrix.get(sc, {}).get(st, {}).get(algo)
                if rec and rec.get("learning_rate") != "--":
                    first_lr = rec.get("learning_rate")
                    break
            html_lines.append(f"            <td class='lr-col'>{first_lr}</td>")

            for algo in algorithms:
                rec = matrix.get(sc, {}).get(st, {}).get(algo)
                cls_name = f"win-rate-{algo.lower()}"
                if rec and rec.get("goal_rate_pct") is not None:
                    html_lines.append(f"            <td class='{cls_name}'>{rec['goal_rate_pct']:.1f}%</td>")
                    html_lines.append(f"            <td>{rec['mean_reward']:+.3f}</td>")
                    html_lines.append(f"            <td>{rec['shots_per_ep']:.2f}</td>")
                else:
                    html_lines.append("            <td class='empty-cell'>--</td>")
                    html_lines.append("            <td class='empty-cell'>--</td>")
                    html_lines.append("            <td class='empty-cell'>--</td>")

            html_lines.append("          </tr>")

        html_lines.append("        </tbody>")
        html_lines.append("      </table>")
        html_lines.append("    </div>")

    html_lines.extend([
        "  </div>",
        "</body>",
        "</html>",
    ])
    return "\n".join(html_lines)


def generate_tables(csv_path: str = CSV_PATH) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = load_win_rate_data(csv_path)
    matrix, scenarios, algorithms, steps = build_comparison_matrices(records)

    md_content = generate_markdown(matrix, scenarios, algorithms, steps)
    html_content = generate_html(matrix, scenarios, algorithms, steps)

    with open(MD_OUT_PATH, mode="w", encoding="utf-8") as f:
        f.write(md_content)

    with open(HTML_OUT_PATH, mode="w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[generate_comparison_table] ✓ Generated Markdown Table -> {MD_OUT_PATH}")
    print(f"[generate_comparison_table] ✓ Generated HTML Table     -> {HTML_OUT_PATH}")


if __name__ == "__main__":
    generate_tables()
