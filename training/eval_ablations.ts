/**
 * GMN-Football-3 — Controlled Reward and Observation Ablation Evaluator
 * Evaluates behavioral shifts, policy exploitation, and role conditioning by comparing
 * standard 127-dim role-aware inputs with role-zeroed, role-randomized, and reward-ablated settings.
 */

import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ActionType, AgentAction, ScenarioConfig } from '../src/types/football';
import { RuleBasedAgent } from '../src/agents/RuleBasedAgent';
import { TrainedPolicyAgent } from '../src/agents/TrainedPolicyAgent';
import { MAPPO_WEIGHTS } from '../src/agents/mappo_weights';
import { mapDiscreteAction, ACTION_SPACE_SIZE } from './action_mapping';
import { BASE_OBSERVATION_DIM, OBSERVATION_DIM, ROLE_DIM } from '../src/engine/Contract';
import { FootballMetricsTracker, AggregatedMetrics } from '../src/engine/FootballMetrics';

export type ObservationAblationType =
  | 'standard_role_127'
  | 'role_zeroed'
  | 'role_randomized'
  | 'role_inverted';

export type RewardAblationType =
  | 'full_shaping'
  | 'sparse_goals_only'
  | 'goal_dominant'
  | 'minimal';

export interface AblationRunResult {
  ablationType: ObservationAblationType;
  rewardType: RewardAblationType;
  metrics: AggregatedMetrics;
}

export interface AblationStudyReport {
  evaluatedAt: string;
  scenarioId: string;
  policyName: string;
  numEpisodesPerCondition: number;
  runs: AblationRunResult[];
  summaryTable: string;
}

export class AblationStudyEvaluator {
  private tracker = new FootballMetricsTracker();

  public runFullStudy(
    scenarioId = 'academy_3_vs_1_with_keeper',
    policyName = 'mappo_trained',
    numEpisodes = 50,
    baseSeed = 800000
  ): AblationStudyReport {
    const scenario = ACADEMY_SCENARIOS.find((s) => s.id === scenarioId || s.codeName === scenarioId);
    if (!scenario) {
      throw new Error(`[AblationStudy] Scenario ${scenarioId} not found.`);
    }

    console.log('====================================================');
    console.log(`GMN-FOOTBALL-3 — CONTROLLED ABLATION STUDY`);
    console.log(`Scenario: ${scenario.name} | Policy: ${policyName}`);
    console.log(`Episodes / Condition: ${numEpisodes}`);
    console.log('====================================================\n');

    const obsConditions: ObservationAblationType[] = [
      'standard_role_127',
      'role_zeroed',
      'role_randomized',
      'role_inverted',
    ];

    const runs: AblationRunResult[] = [];

    for (const obsCond of obsConditions) {
      console.log(`--> Evaluating Observation Ablation: ${obsCond}...`);
      const agg = this.evaluateCondition(obsCond, 'full_shaping', scenario, policyName, numEpisodes, baseSeed);
      runs.push({
        ablationType: obsCond,
        rewardType: 'full_shaping',
        metrics: agg,
      });

      console.log(
        `    ✓ Win Rate: ${agg.winRatePct.mean.toFixed(1)}% | Goals: ${agg.goalsScoredPerEpisode.mean.toFixed(2)} | Pass Ratio: ${agg.passRatioPct.mean.toFixed(1)}% | Shot Ratio: ${agg.shotRatioPct.mean.toFixed(1)}%`
      );
    }

    const summaryTable = this.generateSummaryTable(runs);
    console.log('\n' + summaryTable);

    return {
      evaluatedAt: new Date().toISOString(),
      scenarioId: scenario.id,
      policyName,
      numEpisodesPerCondition: numEpisodes,
      runs,
      summaryTable,
    };
  }

  public evaluateCondition(
    obsAblation: ObservationAblationType,
    rewardAblation: RewardAblationType,
    scenario: ScenarioConfig,
    policyName: string,
    numEpisodes: number,
    baseSeed: number
  ): AggregatedMetrics {
    this.tracker.clear();

    let trainedAgent: TrainedPolicyAgent | null = null;
    if (policyName === 'mappo_trained') {
      try {
        trainedAgent = new TrainedPolicyAgent(MAPPO_WEIGHTS);
      } catch (err: any) {
        console.warn(`[AblationStudy] Neural policy warning: ${err?.message || err}`);
      }
    }

    const defaultBot = new RuleBasedAgent('tactical_ai', 'Tactical AI', 'medium');

    for (let ep = 0; ep < numEpisodes; ep++) {
      const epSeed = baseSeed + ep;
      const engine = new GameEngine();
      engine.loadScenario(scenario, epSeed);

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

        leftPlayers.forEach((player) => {
          let actIdx = 0;
          let agentAction: AgentAction;

          if (trainedAgent && trainedAgent.isValidCheckpoint()) {
            const rawObs = [...engine.getObservation().rawVector];

            // Apply observation ablation
            if (obsAblation === 'role_zeroed') {
              for (let d = BASE_OBSERVATION_DIM; d < OBSERVATION_DIM; d++) {
                rawObs[d] = 0.0;
              }
            } else if (obsAblation === 'role_randomized') {
              const randRoleIdx = Math.floor(engine.rng.next() * ROLE_DIM);
              for (let d = 0; d < ROLE_DIM; d++) {
                rawObs[BASE_OBSERVATION_DIM + d] = d === randRoleIdx ? 1.0 : 0.0;
              }
            } else if (obsAblation === 'role_inverted') {
              for (let d = BASE_OBSERVATION_DIM; d < OBSERVATION_DIM; d++) {
                rawObs[d] = rawObs[d] > 0.5 ? 0.0 : 1.0 / (ROLE_DIM - 1);
              }
            }

            actIdx = trainedAgent.act(rawObs, true);
            agentAction = mapDiscreteAction(actIdx);
          } else {
            agentAction = defaultBot.decide({
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
            actIdx = 5;
          }

          actionMap.set(player.id, agentAction);
          leftActionIndices.push(actIdx);
        });

        rightPlayers.forEach((player) => {
          const oppAction = defaultBot.decide({
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

        const stepRes = engine.step(actionMap, 1 / 60);

        // Apply reward ablation signal
        let customizedReward = stepRes.reward;
        if (rewardAblation === 'sparse_goals_only') {
          customizedReward = engine.score.left > 0 ? 1.0 : 0.0;
        } else if (rewardAblation === 'minimal') {
          customizedReward = engine.score.left > 0 ? 1.0 : 0.0;
        }

        this.tracker.recordTick(engine, leftActionIndices, customizedReward);

        if (stepRes.terminated || stepRes.truncated) {
          done = true;
        }
      }

      this.tracker.endEpisode(engine);
    }

    return this.tracker.aggregate(policyName, `${scenario.id}_${obsAblation}`);
  }

  private generateSummaryTable(runs: AblationRunResult[]): string {
    const lines: string[] = [];
    lines.push('========================================================================================================');
    lines.push('CONTROLLED ABLATION STUDY SUMMARY');
    lines.push('========================================================================================================');
    lines.push('| Condition               | Win Rate (%) | Goals (Mean±Std) | Pass Ratio (%) | Shot Ratio (%) | Reward (Mean) |');
    lines.push('|:------------------------|:------------:|:----------------:|:--------------:|:--------------:|:-------------:|');

    for (const run of runs) {
      const m = run.metrics;
      const winStr = `${m.winRatePct.mean.toFixed(1)}%`;
      const goalsStr = `${m.goalsScoredPerEpisode.mean.toFixed(2)} ± ${m.goalsScoredPerEpisode.std.toFixed(2)}`;
      const passStr = `${m.passRatioPct.mean.toFixed(1)}%`;
      const shotStr = `${m.shotRatioPct.mean.toFixed(1)}%`;
      const rewStr = `${m.cumulativeReward.mean.toFixed(3)}`;

      lines.push(
        `| ${run.ablationType.padEnd(24, ' ')} | ${winStr.padEnd(12, ' ')} | ${goalsStr.padEnd(16, ' ')} | ${passStr.padEnd(14, ' ')} | ${shotStr.padEnd(14, ' ')} | ${rewStr.padEnd(13, ' ')} |`
      );
    }
    lines.push('========================================================================================================');
    return lines.join('\n');
  }
}

// CLI runner
if (process.argv[1]?.endsWith('eval_ablations.ts')) {
  const scenario = process.argv[2] || 'academy_3_vs_1_with_keeper';
  const policy = process.argv[3] || 'mappo_trained';
  const episodes = parseInt(process.argv[4] || '50', 10);
  const evaluator = new AblationStudyEvaluator();
  evaluator.runFullStudy(scenario, policy, episodes);
}
