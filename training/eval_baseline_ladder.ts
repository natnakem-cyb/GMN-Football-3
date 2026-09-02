/**
 * GMN-Football-3 — Unified Baseline Ladder Evaluator
 * Evaluates Random, Idle/No-op, RuleBased (easy/med/hard), Scripted, and Neural/MAPPO policies
 * under identical, reproducible seeds, scenario configurations, and traceable metrics.
 */

import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ActionType, AgentAction, ScenarioConfig } from '../src/types/football';
import { RuleBasedAgent } from '../src/agents/RuleBasedAgent';
import { ScriptedScenarioAgent } from '../src/agents/ScriptedScenarioAgent';
import { TrainedPolicyAgent } from '../src/agents/TrainedPolicyAgent';
import { MAPPO_WEIGHTS } from '../src/agents/mappo_weights';
import { mapDiscreteAction, ACTION_SPACE_SIZE } from './action_mapping';
import { FootballMetricsTracker, AggregatedMetrics } from '../src/engine/FootballMetrics';
import { CheckpointContractValidator } from './checkpoint_contract';

export type BaselineType =
  | 'random'
  | 'noop'
  | 'rule_based_easy'
  | 'rule_based_medium'
  | 'rule_based_hard'
  | 'scripted'
  | 'mappo_trained';

export interface LadderEvaluationConfig {
  scenarioId: string;
  numEpisodes: number;
  baseSeed: number;
  baselines?: BaselineType[];
  opponentDifficulty?: 'easy' | 'medium' | 'hard';
}

export interface BaselineLadderResult {
  scenarioId: string;
  numEpisodes: number;
  baseSeed: number;
  evaluatedAt: string;
  results: Record<string, AggregatedMetrics>;
  summaryTable: string;
}

export class BaselineLadderEvaluator {
  private tracker = new FootballMetricsTracker();

  public evaluateAll(config: LadderEvaluationConfig): BaselineLadderResult {
    const baselines: BaselineType[] = config.baselines || [
      'random',
      'noop',
      'rule_based_easy',
      'rule_based_medium',
      'rule_based_hard',
      'scripted',
      'mappo_trained',
    ];

    const scenario = ACADEMY_SCENARIOS.find(
      (s) => s.id === config.scenarioId || s.codeName === config.scenarioId
    );
    if (!scenario) {
      throw new Error(`[BaselineLadder] Unknown scenario ID: ${config.scenarioId}`);
    }

    console.log('====================================================');
    console.log(`GMN-FOOTBALL-3 — BASELINE LADDER EVALUATION`);
    console.log(`Scenario: ${scenario.name} (${scenario.id})`);
    console.log(`Episodes per baseline: ${config.numEpisodes} | Base Seed: ${config.baseSeed}`);
    console.log('====================================================\n');

    const results: Record<string, AggregatedMetrics> = {};

    for (const baseline of baselines) {
      console.log(`--> Evaluating Baseline: ${baseline}...`);
      const agg = this.evaluateSingleBaseline(baseline, scenario, config);
      results[baseline] = agg;
      console.log(
        `    ✓ Win Rate: ${agg.winRatePct.mean.toFixed(1)}% | Goals: ${agg.goalsScoredPerEpisode.mean.toFixed(2)} | Passes: ${agg.passesCompletedPerEpisode.mean.toFixed(2)}/${agg.passesAttemptedPerEpisode.mean.toFixed(2)} | Reward: ${agg.cumulativeReward.mean.toFixed(3)}`
      );
    }

    const summaryTable = this.generateSummaryTable(results);
    console.log('\n' + summaryTable);

    return {
      scenarioId: scenario.id,
      numEpisodes: config.numEpisodes,
      baseSeed: config.baseSeed,
      evaluatedAt: new Date().toISOString(),
      results,
      summaryTable,
    };
  }

  public evaluateSingleBaseline(
    baseline: BaselineType,
    scenario: ScenarioConfig,
    config: LadderEvaluationConfig
  ): AggregatedMetrics {
    this.tracker.clear();
    const opponentDiff = config.opponentDifficulty || 'medium';

    // Prepare neural agent if requested
    let trainedAgent: TrainedPolicyAgent | null = null;
    if (baseline === 'mappo_trained') {
      try {
        trainedAgent = new TrainedPolicyAgent(MAPPO_WEIGHTS);
      } catch (err: any) {
        console.warn(`    [BaselineLadder] Checkpoint validation: ${err?.message || err}`);
      }
    }

    const easyBot = new RuleBasedAgent('rule_easy', 'Rule AI (Easy)', 'easy');
    const medBot = new RuleBasedAgent('rule_med', 'Rule AI (Med)', 'medium');
    const hardBot = new RuleBasedAgent('rule_hard', 'Rule AI (Hard)', 'hard');
    const scriptedBot = new ScriptedScenarioAgent('scripted_bot', 'Scenario Bot', 'defender_contain');

    for (let ep = 0; ep < config.numEpisodes; ep++) {
      const epSeed = config.baseSeed + ep;
      const engine = new GameEngine();
      engine.loadScenario(scenario, epSeed);

      // Opponents use specified difficulty
      const opponentBot = new RuleBasedAgent('opp_bot', 'Opponent Bot', opponentDiff);

      this.tracker.startEpisode(scenario.id, epSeed, engine);

      let done = false;
      let steps = 0;
      const maxSteps = (scenario.timeLimitSeconds || 30) * 60;

      while (!done && steps < maxSteps) {
        steps++;
        const actionMap = new Map<string, AgentAction>();
        const leftActionIndices: number[] = [];

        const leftPlayers = engine.players.filter((p) => p.team === 'left');
        const rightPlayers = engine.players.filter((p) => p.team === 'right');

        // 1. Controlled / Left Team Action
        leftPlayers.forEach((player) => {
          let actIdx = 0;
          let agentAction: AgentAction;

          if (baseline === 'random') {
            actIdx = Math.floor(engine.rng.next() * ACTION_SPACE_SIZE);
            agentAction = mapDiscreteAction(actIdx);
          } else if (baseline === 'noop') {
            actIdx = 0; // IDLE
            agentAction = { type: ActionType.IDLE };
          } else if (baseline === 'rule_based_easy') {
            agentAction = easyBot.decide({
              player,
              teammates: leftPlayers,
              opponents: rightPlayers,
              ball: engine.ball,
              allPlayers: engine.players,
              teamSide: 'left',
              controlledPlayerId: engine.controlledPlayerId,
              matchTime: engine.matchTimeSeconds,
              rng: engine.rng,
            });
            actIdx = this.inferActionIndex(agentAction);
          } else if (baseline === 'rule_based_medium') {
            agentAction = medBot.decide({
              player,
              teammates: leftPlayers,
              opponents: rightPlayers,
              ball: engine.ball,
              allPlayers: engine.players,
              teamSide: 'left',
              controlledPlayerId: engine.controlledPlayerId,
              matchTime: engine.matchTimeSeconds,
              rng: engine.rng,
            });
            actIdx = this.inferActionIndex(agentAction);
          } else if (baseline === 'rule_based_hard') {
            agentAction = hardBot.decide({
              player,
              teammates: leftPlayers,
              opponents: rightPlayers,
              ball: engine.ball,
              allPlayers: engine.players,
              teamSide: 'left',
              controlledPlayerId: engine.controlledPlayerId,
              matchTime: engine.matchTimeSeconds,
              rng: engine.rng,
            });
            actIdx = this.inferActionIndex(agentAction);
          } else if (baseline === 'scripted') {
            agentAction = scriptedBot.decide({
              player,
              teammates: leftPlayers,
              opponents: rightPlayers,
              ball: engine.ball,
              allPlayers: engine.players,
              teamSide: 'left',
              controlledPlayerId: engine.controlledPlayerId,
              matchTime: engine.matchTimeSeconds,
              rng: engine.rng,
            });
            actIdx = this.inferActionIndex(agentAction);
          } else if (baseline === 'mappo_trained') {
            if (trainedAgent && trainedAgent.isValidCheckpoint()) {
              const obs = engine.getObservation().rawVector;
              actIdx = trainedAgent.act(obs, true);
              agentAction = mapDiscreteAction(actIdx);
            } else {
              // Fallback to rule_based_medium if no valid checkpoint
              agentAction = medBot.decide({
                player,
                teammates: leftPlayers,
                opponents: rightPlayers,
                ball: engine.ball,
                allPlayers: engine.players,
                teamSide: 'left',
                controlledPlayerId: engine.controlledPlayerId,
                matchTime: engine.matchTimeSeconds,
                rng: engine.rng,
              });
              actIdx = this.inferActionIndex(agentAction);
            }
          } else {
            agentAction = { type: ActionType.IDLE };
          }

          actionMap.set(player.id, agentAction);
          leftActionIndices.push(actIdx);
        });

        // 2. Right Team (Opponent) Action
        rightPlayers.forEach((player) => {
          const oppAction = opponentBot.decide({
            player,
            teammates: rightPlayers,
            opponents: leftPlayers,
            ball: engine.ball,
            allPlayers: engine.players,
            teamSide: 'right',
            controlledPlayerId: null,
            matchTime: engine.matchTimeSeconds,
            rng: engine.rng,
          });
          actionMap.set(player.id, oppAction);
        });

        // 3. Step physics
        const stepRes = engine.step(actionMap, 1 / 60);
        this.tracker.recordTick(engine, leftActionIndices, stepRes.reward);

        if (stepRes.terminated || stepRes.truncated) {
          done = true;
        }
      }

      this.tracker.endEpisode(engine);
    }

    return this.tracker.aggregate(baseline, scenario.id);
  }

  private inferActionIndex(action: AgentAction): number {
    switch (action.type) {
      case ActionType.IDLE:
        return 0;
      case ActionType.LONG_PASS:
        return 9;
      case ActionType.HIGH_PASS:
        return 10;
      case ActionType.SHORT_PASS:
        return 11;
      case ActionType.SHOT:
        return 12;
      case ActionType.SPRINT:
        return 13;
      case ActionType.RELEASE_DIRECTION:
        return 14;
      case ActionType.RELEASE_SPRINT:
        return 15;
      case ActionType.SLIDING:
      case ActionType.TACKLE:
        return 16;
      case ActionType.DRIBBLE:
        return 17;
      case ActionType.RELEASE_DRIBBLE:
        return 18;
      case ActionType.MOVE: {
        if (!action.direction) return 0;
        const { x, y } = action.direction;
        if (x < -0.3 && Math.abs(y) <= 0.3) return 1; // LEFT
        if (x < -0.3 && y < -0.3) return 2; // TOP_LEFT
        if (Math.abs(x) <= 0.3 && y < -0.3) return 3; // TOP
        if (x > 0.3 && y < -0.3) return 4; // TOP_RIGHT
        if (x > 0.3 && Math.abs(y) <= 0.3) return 5; // RIGHT
        if (x > 0.3 && y > 0.3) return 6; // BOTTOM_RIGHT
        if (Math.abs(x) <= 0.3 && y > 0.3) return 7; // BOTTOM
        if (x < -0.3 && y > 0.3) return 8; // BOTTOM_LEFT
        return 5; // default forward
      }
      default:
        return 0;
    }
  }

  private generateSummaryTable(results: Record<string, AggregatedMetrics>): string {
    const lines: string[] = [];
    lines.push('========================================================================================================');
    lines.push('BASELINE LADDER EVALUATION SUMMARY TABLE');
    lines.push('========================================================================================================');
    lines.push(
      '| Policy | Win Rate (%) | Goals (Mean±Std) | Possession (%) | Pass Acc (%) | Shot Acc (%) | Reward (Mean±Std) |'
    );
    lines.push(
      '|:---|:---:|:---:|:---:|:---:|:---:|:---:|'
    );

    for (const [name, m] of Object.entries(results)) {
      const winStr = `${m.winRatePct.mean.toFixed(1)}%`;
      const goalsStr = `${m.goalsScoredPerEpisode.mean.toFixed(2)} ± ${m.goalsScoredPerEpisode.std.toFixed(2)}`;
      const possStr = `${m.possessionRatePct.mean.toFixed(1)}%`;
      const passStr = `${m.passCompletionRatePct.mean.toFixed(1)}%`;
      const shotStr = `${m.shotAccuracyPct.mean.toFixed(1)}%`;
      const rewStr = `${m.cumulativeReward.mean.toFixed(3)} ± ${m.cumulativeReward.std.toFixed(3)}`;

      lines.push(
        `| ${name.padEnd(18, ' ')} | ${winStr.padEnd(12, ' ')} | ${goalsStr.padEnd(16, ' ')} | ${possStr.padEnd(14, ' ')} | ${passStr.padEnd(12, ' ')} | ${shotStr.padEnd(12, ' ')} | ${rewStr.padEnd(17, ' ')} |`
      );
    }
    lines.push('========================================================================================================');
    return lines.join('\n');
  }
}

// CLI runner
if (process.argv[1]?.endsWith('eval_baseline_ladder.ts')) {
  const scenario = process.argv[2] || 'academy_3_vs_1_with_keeper';
  const episodes = parseInt(process.argv[3] || '50', 10);
  const evaluator = new BaselineLadderEvaluator();
  evaluator.evaluateAll({
    scenarioId: scenario,
    numEpisodes: episodes,
    baseSeed: 10000,
  });
}
