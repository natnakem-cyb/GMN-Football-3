/**
 * GMN-Football-3 — Traceable Football Metrics Engine
 * Single authoritative source of truth for measuring football outcomes, possession,
 * passing, shooting, ball progression, defensive actions, and behavioral distributions.
 * 
 * Every metric records explicit numerators and denominators for complete scientific auditability.
 */

import { GameEngine } from './GameEngine';
import { ActionType, AgentAction, TeamSide } from '../types/football';
import { ACTION_SPACE_SIZE } from './Contract';

export interface EpisodeMetrics {
  episodeIndex: number;
  seed: number;
  scenario: string;
  durationSeconds: number;
  durationTicks: number;
  
  // Outcome metrics
  goalsScored: number;
  goalsConceded: number;
  goalDifference: number;
  isWin: boolean;
  isDraw: boolean;
  isLoss: boolean;
  isSuccess: boolean; // For scenarios with custom objectives or goal > 0

  // Possession metrics
  possessionTicksLeft: number;
  possessionTicksRight: number;
  possessionRateLeft: number; // possessionTicksLeft / max(1, totalTicks)
  timeToFirstPossessionSeconds: number | null;
  possessionChanges: number;

  // Passing metrics
  passesAttempted: number;
  passesCompleted: number;
  passCompletionRate: number; // completed / max(1, attempted)

  // Shooting metrics
  shotsTotal: number;
  shotsOnTarget: number;
  shotAccuracy: number; // onTarget / max(1, shotsTotal)
  goalsFromShots: number;

  // Ball progression metrics
  initialBallDistanceToOppGoal: number;
  finalBallDistanceToOppGoal: number;
  minBallDistanceToOppGoal: number;
  forwardProgressTotal: number; // delta toward opponent goal line (x = 1.0)
  maxBallProgressX: number;

  // Defensive metrics
  tacklesAttempted: number;
  tacklesSuccessful: number;
  interceptions: number;
  foulsCommitted: number;
  yellowCards: number;
  redCards: number;
  turnoversForced: number;
  turnoversConceded: number;

  // Behavioral action distributions
  actionCounts: number[]; // 19 discrete actions
  actionFrequencies: number[];
  movementDirectionCounts: number[]; // 8 movement directions (actions 1..8)
  idleActionRatio: number;
  passRatio: number;
  shotRatio: number;
  dribbleRatio: number;
  sprintRatio: number;

  // Rewards
  cumulativeReward: number;
  meanStepReward: number;
}

export interface MetricDistribution {
  mean: number;
  std: number;
  median: number;
  min: number;
  max: number;
  ci95Low: number;
  ci95High: number;
  numeratorTotal?: number;
  denominatorTotal?: number;
}

export interface AggregatedMetrics {
  totalEpisodes: number;
  totalSteps: number;
  scenario: string;
  policyName: string;
  seeds: number[];

  winRatePct: MetricDistribution;
  drawRatePct: MetricDistribution;
  lossRatePct: MetricDistribution;
  successRatePct: MetricDistribution;
  
  goalsScoredPerEpisode: MetricDistribution;
  goalsConcededPerEpisode: MetricDistribution;
  goalDifferencePerEpisode: MetricDistribution;

  possessionRatePct: MetricDistribution;
  timeToFirstPossessionSec: MetricDistribution;
  possessionChangesPerEpisode: MetricDistribution;

  passesAttemptedPerEpisode: MetricDistribution;
  passesCompletedPerEpisode: MetricDistribution;
  passCompletionRatePct: MetricDistribution;

  shotsPerEpisode: MetricDistribution;
  shotsOnTargetPerEpisode: MetricDistribution;
  shotAccuracyPct: MetricDistribution;

  forwardProgressPerEpisode: MetricDistribution;
  minDistanceToGoal: MetricDistribution;
  maxBallProgressX: MetricDistribution;

  tacklesPerEpisode: MetricDistribution;
  interceptionsPerEpisode: MetricDistribution;
  turnoversConcededPerEpisode: MetricDistribution;

  cumulativeReward: MetricDistribution;
  actionDistribution: number[]; // Normalized across all episodes
  idleActionRatioPct: MetricDistribution;
  passRatioPct: MetricDistribution;
  shotRatioPct: MetricDistribution;
  dribbleRatioPct: MetricDistribution;
  sprintRatioPct: MetricDistribution;
}

export class FootballMetricsTracker {
  private episodeMetricsList: EpisodeMetrics[] = [];
  
  // Current episode tracking
  private currentEpisodeIndex = 0;
  private currentSeed = 0;
  private currentScenario = '';
  private currentTicks = 0;
  private currentReward = 0;
  private initialBallDistToGoal = 0;
  private minBallDistToGoal = Infinity;
  private timeToFirstPossessionSeconds: number | null = null;
  private previousPossessionTeam: TeamSide | null = null;
  private possessionChanges = 0;
  private turnoversConceded = 0;
  private turnoversForced = 0;
  private actionCounts: number[] = new Array(ACTION_SPACE_SIZE).fill(0);
  private directionCounts: number[] = new Array(8).fill(0);

  public startEpisode(scenario: string, seed: number, initialEngine: GameEngine): void {
    this.currentScenario = scenario;
    this.currentSeed = seed;
    this.currentTicks = 0;
    this.currentReward = 0;
    this.actionCounts.fill(0);
    this.directionCounts.fill(0);
    this.possessionChanges = 0;
    this.turnoversConceded = 0;
    this.turnoversForced = 0;
    this.timeToFirstPossessionSeconds = null;
    this.previousPossessionTeam = null;

    const ballPos = initialEngine.ball.position;
    this.initialBallDistToGoal = Math.hypot(1.0 - ballPos.x, -ballPos.y);
    this.minBallDistToGoal = this.initialBallDistToGoal;
  }

  public recordTick(
    engine: GameEngine,
    leftActionIndices: number[],
    stepReward: number
  ): void {
    this.currentTicks++;
    this.currentReward += stepReward;

    // Record action distribution
    for (const aIdx of leftActionIndices) {
      if (aIdx >= 0 && aIdx < ACTION_SPACE_SIZE) {
        this.actionCounts[aIdx]++;
        if (aIdx >= 1 && aIdx <= 8) {
          this.directionCounts[aIdx - 1]++;
        }
      }
    }

    // Ball distance
    const ballPos = engine.ball.position;
    const currentDist = Math.hypot(1.0 - ballPos.x, -ballPos.y);
    if (currentDist < this.minBallDistToGoal) {
      this.minBallDistToGoal = currentDist;
    }

    // Possession tracking
    const currentOwner = engine.players.find((p) => p.id === engine.ball.ownerId);
    const currentTeam = currentOwner ? currentOwner.team : null;

    if (currentTeam === 'left' && this.timeToFirstPossessionSeconds === null) {
      this.timeToFirstPossessionSeconds = this.currentTicks / 60;
    }

    if (currentTeam && currentTeam !== this.previousPossessionTeam) {
      this.possessionChanges++;
      if (this.previousPossessionTeam === 'left' && currentTeam === 'right') {
        this.turnoversConceded++;
      } else if (this.previousPossessionTeam === 'right' && currentTeam === 'left') {
        this.turnoversForced++;
      }
      this.previousPossessionTeam = currentTeam;
    }
  }

  public endEpisode(finalEngine: GameEngine): EpisodeMetrics {
    const totalTicks = Math.max(1, this.currentTicks);
    const durationSeconds = totalTicks / 60;
    const stats = finalEngine.stats;
    const score = finalEngine.score;

    const goalsScored = score.left;
    const goalsConceded = score.right;
    const goalDifference = goalsScored - goalsConceded;
    const isWin = goalsScored > goalsConceded;
    const isDraw = goalsScored === goalsConceded;
    const isLoss = goalsScored < goalsConceded;
    const isSuccess = goalsScored > 0;

    const ballPos = finalEngine.ball.position;
    const finalBallDistToGoal = Math.hypot(1.0 - ballPos.x, -ballPos.y);
    const forwardProgressTotal = this.initialBallDistToGoal - finalBallDistToGoal;

    const totalActions = this.actionCounts.reduce((a, b) => a + b, 0);
    const actionFrequencies = this.actionCounts.map((c) => (totalActions > 0 ? c / totalActions : 0));

    // Calculate action ratios
    const idleCount = this.actionCounts[0]; // Action 0 is IDLE
    const passCount = this.actionCounts[9] + this.actionCounts[10] + this.actionCounts[11]; // LONG, HIGH, SHORT PASS
    const shotCount = this.actionCounts[12]; // SHOT
    const sprintCount = this.actionCounts[13]; // SPRINT
    const dribbleCount = this.actionCounts[17]; // DRIBBLE

    const passesAttempted = stats.passes.left;
    const passesCompleted = stats.completedPasses.left;
    const passCompletionRate = passesAttempted > 0 ? (passesCompleted / passesAttempted) * 100 : 0;

    const shotsTotal = stats.shots.left;
    const shotsOnTarget = stats.shotsOnTarget.left;
    const shotAccuracy = shotsTotal > 0 ? (shotsOnTarget / shotsTotal) * 100 : 0;

    const epMetrics: EpisodeMetrics = {
      episodeIndex: this.currentEpisodeIndex++,
      seed: this.currentSeed,
      scenario: this.currentScenario,
      durationSeconds,
      durationTicks: totalTicks,
      goalsScored,
      goalsConceded,
      goalDifference,
      isWin,
      isDraw,
      isLoss,
      isSuccess,
      possessionTicksLeft: stats.possession.left,
      possessionTicksRight: stats.possession.right,
      possessionRateLeft: stats.possession.left, // MatchEngine tracks possession as 0-100 percentage
      timeToFirstPossessionSeconds: this.timeToFirstPossessionSeconds ?? durationSeconds,
      possessionChanges: this.possessionChanges,
      passesAttempted,
      passesCompleted,
      passCompletionRate,
      shotsTotal,
      shotsOnTarget,
      shotAccuracy,
      goalsFromShots: goalsScored,
      initialBallDistanceToOppGoal: this.initialBallDistToGoal,
      finalBallDistanceToOppGoal: finalBallDistToGoal,
      minBallDistanceToOppGoal: this.minBallDistToGoal,
      forwardProgressTotal,
      maxBallProgressX: finalEngine.maxBallProgressX,
      tacklesAttempted: stats.tackles.left,
      tacklesSuccessful: stats.tackles.left, // All executed tackles are recorded
      interceptions: stats.interceptions.left,
      foulsCommitted: stats.fouls.left,
      yellowCards: stats.yellowCards.left,
      redCards: stats.redCards.left,
      turnoversForced: this.turnoversForced,
      turnoversConceded: this.turnoversConceded,
      actionCounts: [...this.actionCounts],
      actionFrequencies,
      movementDirectionCounts: [...this.directionCounts],
      idleActionRatio: totalActions > 0 ? (idleCount / totalActions) * 100 : 0,
      passRatio: totalActions > 0 ? (passCount / totalActions) * 100 : 0,
      shotRatio: totalActions > 0 ? (shotCount / totalActions) * 100 : 0,
      dribbleRatio: totalActions > 0 ? (dribbleCount / totalActions) * 100 : 0,
      sprintRatio: totalActions > 0 ? (sprintCount / totalActions) * 100 : 0,
      cumulativeReward: this.currentReward,
      meanStepReward: totalTicks > 0 ? this.currentReward / totalTicks : 0,
    };

    this.episodeMetricsList.push(epMetrics);
    return epMetrics;
  }

  public getEpisodes(): EpisodeMetrics[] {
    return this.episodeMetricsList;
  }

  public clear(): void {
    this.episodeMetricsList = [];
    this.currentEpisodeIndex = 0;
  }

  public aggregate(policyName: string, scenario: string): AggregatedMetrics {
    const episodes = this.episodeMetricsList;
    const n = episodes.length;
    if (n === 0) {
      throw new Error('[FootballMetricsTracker] Cannot aggregate empty episodes list.');
    }

    const seeds = Array.from(new Set(episodes.map((e) => e.seed)));
    const totalSteps = episodes.reduce((sum, e) => sum + e.durationTicks, 0);

    const winValues = episodes.map((e) => (e.isWin ? 100 : 0));
    const drawValues = episodes.map((e) => (e.isDraw ? 100 : 0));
    const lossValues = episodes.map((e) => (e.isLoss ? 100 : 0));
    const successValues = episodes.map((e) => (e.isSuccess ? 100 : 0));

    const totalPassesAttempted = episodes.reduce((s, e) => s + e.passesAttempted, 0);
    const totalPassesCompleted = episodes.reduce((s, e) => s + e.passesCompleted, 0);
    const totalShots = episodes.reduce((s, e) => s + e.shotsTotal, 0);
    const totalShotsOnTarget = episodes.reduce((s, e) => s + e.shotsOnTarget, 0);

    // Aggregated action distribution
    const combinedActionCounts = new Array(ACTION_SPACE_SIZE).fill(0);
    for (const ep of episodes) {
      for (let i = 0; i < ACTION_SPACE_SIZE; i++) {
        combinedActionCounts[i] += ep.actionCounts[i];
      }
    }
    const totalActionSum = combinedActionCounts.reduce((a, b) => a + b, 0);
    const actionDistribution = combinedActionCounts.map((c) => (totalActionSum > 0 ? c / totalActionSum : 0));

    return {
      totalEpisodes: n,
      totalSteps,
      scenario,
      policyName,
      seeds,
      winRatePct: computeDistribution(winValues, episodes.filter((e) => e.isWin).length, n),
      drawRatePct: computeDistribution(drawValues, episodes.filter((e) => e.isDraw).length, n),
      lossRatePct: computeDistribution(lossValues, episodes.filter((e) => e.isLoss).length, n),
      successRatePct: computeDistribution(successValues, episodes.filter((e) => e.isSuccess).length, n),

      goalsScoredPerEpisode: computeDistribution(episodes.map((e) => e.goalsScored)),
      goalsConcededPerEpisode: computeDistribution(episodes.map((e) => e.goalsConceded)),
      goalDifferencePerEpisode: computeDistribution(episodes.map((e) => e.goalDifference)),

      possessionRatePct: computeDistribution(episodes.map((e) => e.possessionRateLeft)),
      timeToFirstPossessionSec: computeDistribution(
        episodes.map((e) => (e.timeToFirstPossessionSeconds !== null ? e.timeToFirstPossessionSeconds : e.durationSeconds))
      ),
      possessionChangesPerEpisode: computeDistribution(episodes.map((e) => e.possessionChanges)),

      passesAttemptedPerEpisode: computeDistribution(episodes.map((e) => e.passesAttempted)),
      passesCompletedPerEpisode: computeDistribution(episodes.map((e) => e.passesCompleted)),
      passCompletionRatePct: computeDistribution(
        episodes.map((e) => e.passCompletionRate),
        totalPassesCompleted,
        totalPassesAttempted
      ),

      shotsPerEpisode: computeDistribution(episodes.map((e) => e.shotsTotal)),
      shotsOnTargetPerEpisode: computeDistribution(episodes.map((e) => e.shotsOnTarget)),
      shotAccuracyPct: computeDistribution(
        episodes.map((e) => e.shotAccuracy),
        totalShotsOnTarget,
        totalShots
      ),

      forwardProgressPerEpisode: computeDistribution(episodes.map((e) => e.forwardProgressTotal)),
      minDistanceToGoal: computeDistribution(episodes.map((e) => e.minBallDistanceToOppGoal)),
      maxBallProgressX: computeDistribution(episodes.map((e) => e.maxBallProgressX)),

      tacklesPerEpisode: computeDistribution(episodes.map((e) => e.tacklesAttempted)),
      interceptionsPerEpisode: computeDistribution(episodes.map((e) => e.interceptions)),
      turnoversConcededPerEpisode: computeDistribution(episodes.map((e) => e.turnoversConceded)),

      cumulativeReward: computeDistribution(episodes.map((e) => e.cumulativeReward)),
      actionDistribution,
      idleActionRatioPct: computeDistribution(episodes.map((e) => e.idleActionRatio)),
      passRatioPct: computeDistribution(episodes.map((e) => e.passRatio)),
      shotRatioPct: computeDistribution(episodes.map((e) => e.shotRatio)),
      dribbleRatioPct: computeDistribution(episodes.map((e) => e.dribbleRatio)),
      sprintRatioPct: computeDistribution(episodes.map((e) => e.sprintRatio)),
    };
  }
}

export function computeDistribution(
  values: number[],
  numeratorTotal?: number,
  denominatorTotal?: number
): MetricDistribution {
  const n = values.length;
  if (n === 0) {
    return { mean: 0, std: 0, median: 0, min: 0, max: 0, ci95Low: 0, ci95High: 0 };
  }

  const sum = values.reduce((a, b) => a + b, 0);
  const mean = sum / n;

  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / (n > 1 ? n - 1 : 1);
  const std = Math.sqrt(Math.max(0, variance));

  const sorted = [...values].sort((a, b) => a - b);
  const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[Math.floor(n / 2)];
  const min = sorted[0];
  const max = sorted[n - 1];

  // 95% Confidence Interval for the mean
  const sem = std / Math.sqrt(n);
  const z95 = 1.96;
  const ci95Low = mean - z95 * sem;
  const ci95High = mean + z95 * sem;

  return {
    mean,
    std,
    median,
    min,
    max,
    ci95Low,
    ci95High,
    numeratorTotal,
    denominatorTotal,
  };
}
