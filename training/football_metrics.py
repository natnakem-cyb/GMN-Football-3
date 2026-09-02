"""
GMN-Football-3 — Traceable Football Metrics Engine (Python)
Collects structured, traceable metrics across match episodes.
Calculates outcome, possession, passing, shooting, ball progression, defensive, and behavioral stats.
Records explicit numerators and denominators for complete scientific reproducibility.
"""

import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class EpisodeMetrics:
    episode_index: int
    seed: int
    scenario: str
    duration_seconds: float
    duration_ticks: int

    # Outcome metrics
    goals_scored: int
    goals_conceded: int
    goal_difference: int
    is_win: bool
    is_draw: bool
    is_loss: bool
    is_success: bool

    # Possession metrics
    possession_ticks_left: float
    possession_ticks_right: float
    possession_rate_left: float
    time_to_first_possession_seconds: float
    possession_changes: int

    # Passing metrics
    passes_attempted: int
    passes_completed: int
    pass_completion_rate: float

    # Shooting metrics
    shots_total: int
    shots_on_target: int
    shot_accuracy: float
    goals_from_shots: int

    # Ball progression metrics
    initial_ball_dist_to_opp_goal: float
    final_ball_dist_to_opp_goal: float
    min_ball_dist_to_opp_goal: float
    forward_progress_total: float
    max_ball_progress_x: float

    # Defensive metrics
    tackles_attempted: int
    tackles_successful: int
    interceptions: int
    fouls_committed: int
    yellow_cards: int
    red_cards: int
    turnovers_forced: int
    turnovers_conceded: int

    # Behavioral action distributions
    action_counts: List[int]
    action_frequencies: List[float]
    movement_direction_counts: List[int]
    idle_action_ratio: float
    pass_ratio: float
    shot_ratio: float
    dribble_ratio: float
    sprint_ratio: float

    # Rewards
    cumulative_reward: float
    mean_step_reward: float


def compute_distribution(
    values: List[float],
    numerator_total: Optional[int] = None,
    denominator_total: Optional[int] = None
) -> Dict[str, Any]:
    n = len(values)
    if n == 0:
        return {
            "mean": 0.0, "std": 0.0, "median": 0.0,
            "min": 0.0, "max": 0.0, "ci95_low": 0.0, "ci95_high": 0.0
        }

    mean_val = sum(values) / n
    variance = sum((v - mean_val) ** 2 for v in values) / (n - 1 if n > 1 else 1)
    std_val = math.sqrt(max(0.0, variance))

    sorted_vals = sorted(values)
    median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0 if n % 2 == 0 else sorted_vals[n // 2]
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]

    sem = std_val / math.sqrt(n)
    ci95_low = mean_val - 1.96 * sem
    ci95_high = mean_val + 1.96 * sem

    res = {
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
    }
    if numerator_total is not None:
        res["numerator_total"] = numerator_total
    if denominator_total is not None:
        res["denominator_total"] = denominator_total
    return res


class FootballMetricsTracker:
    def __init__(self):
        self.episodes: List[EpisodeMetrics] = []
        self.action_space_size = 19
        self.current_episode_index = 0
        self._reset_current_episode()

    def _reset_current_episode(self):
        self.current_scenario = ""
        self.current_seed = 0
        self.current_ticks = 0
        self.current_reward = 0.0
        self.action_counts = [0] * self.action_space_size
        self.direction_counts = [0] * 8
        self.initial_ball_dist = 0.0
        self.min_ball_dist = float("inf")
        self.time_to_first_possession = None
        self.previous_possession_team = None
        self.possession_changes = 0
        self.turnovers_conceded = 0
        self.turnovers_forced = 0

    def start_episode(self, scenario: str, seed: int, ball_pos: Dict[str, float]):
        self._reset_current_episode()
        self.current_scenario = scenario
        self.current_seed = seed
        bx = ball_pos.get("x", 0.0)
        by = ball_pos.get("y", 0.0)
        self.initial_ball_dist = math.hypot(1.0 - bx, by)
        self.min_ball_dist = self.initial_ball_dist

    def record_tick(
        self,
        left_action_indices: List[int],
        step_reward: float,
        ball_pos: Dict[str, float],
        owner_team: Optional[str]
    ):
        self.current_ticks += 1
        self.current_reward += step_reward

        for a_idx in left_action_indices:
            if 0 <= a_idx < self.action_space_size:
                self.action_counts[a_idx] += 1
                if 1 <= a_idx <= 8:
                    self.direction_counts[a_idx - 1] += 1

        bx = ball_pos.get("x", 0.0)
        by = ball_pos.get("y", 0.0)
        dist = math.hypot(1.0 - bx, by)
        if dist < self.min_ball_dist:
            self.min_ball_dist = dist

        if owner_team == "left" and self.time_to_first_possession is None:
            self.time_to_first_possession = self.current_ticks / 60.0

        if owner_team and owner_team != self.previous_possession_team:
            self.possession_changes += 1
            if self.previous_possession_team == "left" and owner_team == "right":
                self.turnovers_conceded += 1
            elif self.previous_possession_team == "right" and owner_team == "left":
                self.turnovers_forced += 1
            self.previous_possession_team = owner_team

    def end_episode(
        self,
        final_score: Dict[str, int],
        final_stats: Dict[str, Any],
        final_ball_pos: Dict[str, float],
        max_ball_progress_x: float = 0.0
    ) -> EpisodeMetrics:
        total_ticks = max(1, self.current_ticks)
        duration_seconds = total_ticks / 60.0

        goals_scored = final_score.get("left", 0)
        goals_conceded = final_score.get("right", 0)
        goal_diff = goals_scored - goalsConceded if "goalsConceded" in locals() else goals_scored - goals_conceded
        is_win = goals_scored > goals_conceded
        is_draw = goals_scored == goals_conceded
        is_loss = goals_scored < goals_conceded
        is_success = goals_scored > 0

        bx = final_ball_pos.get("x", 0.0)
        by = final_ball_pos.get("y", 0.0)
        final_dist = math.hypot(1.0 - bx, by)
        forward_progress = self.initial_ball_dist - final_dist

        total_actions = sum(self.action_counts)
        freqs = [c / total_actions if total_actions > 0 else 0.0 for c in self.action_counts]

        idle_count = self.action_counts[0]
        pass_count = sum(self.action_counts[9:12])
        shot_count = self.action_counts[12]
        sprint_count = self.action_counts[13]
        dribble_count = self.action_counts[17]

        passes_attempted = final_stats.get("passes", {}).get("left", 0) if isinstance(final_stats.get("passes"), dict) else 0
        passes_completed = final_stats.get("completedPasses", {}).get("left", 0) if isinstance(final_stats.get("completedPasses"), dict) else 0
        pass_comp_rate = (passes_completed / passes_attempted * 100.0) if passes_attempted > 0 else 0.0

        shots_total = final_stats.get("shots", {}).get("left", 0) if isinstance(final_stats.get("shots"), dict) else 0
        shots_on_target = final_stats.get("shotsOnTarget", {}).get("left", 0) if isinstance(final_stats.get("shotsOnTarget"), dict) else 0
        shot_acc = (shots_on_target / shots_total * 100.0) if shots_total > 0 else 0.0

        possession_left = final_stats.get("possession", {}).get("left", 50.0) if isinstance(final_stats.get("possession"), dict) else 50.0

        ep = EpisodeMetrics(
            episode_index=self.current_episode_index,
            seed=self.current_seed,
            scenario=self.current_scenario,
            duration_seconds=duration_seconds,
            duration_ticks=total_ticks,
            goals_scored=goals_scored,
            goals_conceded=goals_conceded,
            goal_difference=goal_diff,
            is_win=is_win,
            is_draw=is_draw,
            is_loss=is_loss,
            is_success=is_success,
            possession_ticks_left=possession_left,
            possession_ticks_right=100.0 - possession_left,
            possession_rate_left=possession_left,
            time_to_first_possession_seconds=self.time_to_first_possession if self.time_to_first_possession is not None else duration_seconds,
            possession_changes=self.possession_changes,
            passes_attempted=passes_attempted,
            passes_completed=passes_completed,
            pass_completion_rate=pass_comp_rate,
            shots_total=shots_total,
            shots_on_target=shots_on_target,
            shot_accuracy=shot_acc,
            goals_from_shots=goals_scored,
            initial_ball_dist_to_opp_goal=self.initial_ball_dist,
            final_ball_dist_to_opp_goal=final_dist,
            min_ball_dist_to_opp_goal=self.min_ball_dist,
            forward_progress_total=forward_progress,
            max_ball_progress_x=max_ball_progress_x,
            tackles_attempted=final_stats.get("tackles", {}).get("left", 0) if isinstance(final_stats.get("tackles"), dict) else 0,
            tackles_successful=final_stats.get("tackles", {}).get("left", 0) if isinstance(final_stats.get("tackles"), dict) else 0,
            interceptions=final_stats.get("interceptions", {}).get("left", 0) if isinstance(final_stats.get("interceptions"), dict) else 0,
            fouls_committed=final_stats.get("fouls", {}).get("left", 0) if isinstance(final_stats.get("fouls"), dict) else 0,
            yellow_cards=final_stats.get("yellowCards", {}).get("left", 0) if isinstance(final_stats.get("yellowCards"), dict) else 0,
            red_cards=final_stats.get("redCards", {}).get("left", 0) if isinstance(final_stats.get("redCards"), dict) else 0,
            turnovers_forced=self.turnovers_forced,
            turnovers_conceded=self.turnovers_conceded,
            action_counts=list(self.action_counts),
            action_frequencies=freqs,
            movement_direction_counts=list(self.direction_counts),
            idle_action_ratio=(idle_count / total_actions * 100.0) if total_actions > 0 else 0.0,
            pass_ratio=(pass_count / total_actions * 100.0) if total_actions > 0 else 0.0,
            shot_ratio=(shot_count / total_actions * 100.0) if total_actions > 0 else 0.0,
            dribble_ratio=(dribble_count / total_actions * 100.0) if total_actions > 0 else 0.0,
            sprint_ratio=(sprint_count / total_actions * 100.0) if total_actions > 0 else 0.0,
            cumulative_reward=self.current_reward,
            mean_step_reward=self.current_reward / total_ticks if total_ticks > 0 else 0.0,
        )

        self.current_episode_index += 1
        self.episodes.append(ep)
        return ep

    def aggregate(self, policy_name: str, scenario: str) -> Dict[str, Any]:
        episodes = self.episodes
        n = len(episodes)
        if n == 0:
            raise ValueError("[FootballMetricsTracker] No episodes to aggregate.")

        seeds = sorted(list(set(e.seed for e in episodes)))
        total_steps = sum(e.duration_ticks for e in episodes)

        win_vals = [100.0 if e.is_win else 0.0 for e in episodes]
        draw_vals = [100.0 if e.is_draw else 0.0 for e in episodes]
        loss_vals = [100.0 if e.is_loss else 0.0 for e in episodes]
        success_vals = [100.0 if e.is_success else 0.0 for e in episodes]

        tot_passes_att = sum(e.passes_attempted for e in episodes)
        tot_passes_comp = sum(e.passes_completed for e in episodes)
        tot_shots = sum(e.shots_total for e in episodes)
        tot_shots_on_target = sum(e.shots_on_target for e in episodes)

        # Action distribution
        combined_action_counts = [0] * self.action_space_size
        for ep in episodes:
            for i in range(self.action_space_size):
                combined_action_counts[i] += ep.action_counts[i]
        tot_act = sum(combined_action_counts)
        action_dist = [c / tot_act if tot_act > 0 else 0.0 for c in combined_action_counts]

        return {
            "total_episodes": n,
            "total_steps": total_steps,
            "scenario": scenario,
            "policy_name": policy_name,
            "seeds": seeds,
            "win_rate_pct": compute_distribution(win_vals, sum(1 for e in episodes if e.is_win), n),
            "draw_rate_pct": compute_distribution(draw_vals, sum(1 for e in episodes if e.is_draw), n),
            "loss_rate_pct": compute_distribution(loss_vals, sum(1 for e in episodes if e.is_loss), n),
            "success_rate_pct": compute_distribution(success_vals, sum(1 for e in episodes if e.is_success), n),

            "goals_scored_per_episode": compute_distribution([float(e.goals_scored) for e in episodes]),
            "goals_conceded_per_episode": compute_distribution([float(e.goals_conceded) for e in episodes]),
            "goal_difference_per_episode": compute_distribution([float(e.goal_difference) for e in episodes]),

            "possession_rate_pct": compute_distribution([e.possession_rate_left for e in episodes]),
            "time_to_first_possession_sec": compute_distribution([e.time_to_first_possession_seconds for e in episodes]),
            "possession_changes_per_episode": compute_distribution([float(e.possession_changes) for e in episodes]),

            "passes_attempted_per_episode": compute_distribution([float(e.passes_attempted) for e in episodes]),
            "passes_completed_per_episode": compute_distribution([float(e.passes_completed) for e in episodes]),
            "pass_completion_rate_pct": compute_distribution(
                [e.pass_completion_rate for e in episodes], tot_passes_comp, tot_passes_att
            ),

            "shots_per_episode": compute_distribution([float(e.shots_total) for e in episodes]),
            "shots_on_target_per_episode": compute_distribution([float(e.shots_on_target) for e in episodes]),
            "shot_accuracy_pct": compute_distribution(
                [e.shot_accuracy for e in episodes], tot_shots_on_target, tot_shots
            ),

            "forward_progress_per_episode": compute_distribution([e.forward_progress_total for e in episodes]),
            "min_distance_to_goal": compute_distribution([e.min_ball_dist_to_opp_goal for e in episodes]),
            "max_ball_progress_x": compute_distribution([e.max_ball_progress_x for e in episodes]),

            "tackles_per_episode": compute_distribution([float(e.tackles_attempted) for e in episodes]),
            "interceptions_per_episode": compute_distribution([float(e.interceptions) for e in episodes]),
            "turnovers_conceded_per_episode": compute_distribution([float(e.turnovers_conceded) for e in episodes]),

            "cumulative_reward": compute_distribution([e.cumulative_reward for e in episodes]),
            "action_distribution": action_dist,
            "idle_action_ratio_pct": compute_distribution([e.idle_action_ratio for e in episodes]),
            "pass_ratio_pct": compute_distribution([e.pass_ratio for e in episodes]),
            "shot_ratio_pct": compute_distribution([e.shot_ratio for e in episodes]),
            "dribble_ratio_pct": compute_distribution([e.dribble_ratio for e in episodes]),
            "sprint_ratio_pct": compute_distribution([e.sprint_ratio for e in episodes]),
        }
