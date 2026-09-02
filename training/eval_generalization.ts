/**
 * GMN-Football-3 — Formal Held-Out Generalization Evaluation Suite
 * Evaluates policies across formal Train / Validation / Test / Scale-up scenario partitions.
 * Quantifies generalization gap, transfer retention, and scenario-specific degradation.
 */

import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ActionType, AgentAction, ScenarioConfig } from '../src/types/football';
import { RuleBasedAgent } from '../src/agents/RuleBasedAgent';
import { TrainedPolicyAgent } from '../src/agents/TrainedPolicyAgent';
import { MAPPO_WEIGHTS } from '../src/agents/mappo_weights';
import { mapDiscreteAction, ACTION_SPACE_SIZE } from './action_mapping';
import { FootballMetricsTracker, AggregatedMetrics } from '../src/engine/FootballMetrics';

export interface GeneralizationPartition {
  partition: 'TRAIN' | 'VALIDATION' | 'TEST' | 'SCALE_UP';
  scenarioIds: string[];
  description: string;
}

export const FORMAL_PARTITIONS: GeneralizationPartition[] = [
  {
    partition: 'TRAIN',
    scenarioIds: ['academy_3_vs_1_with_keeper'],
    description: 'Reference training drill arrangement (1 central defender + goalkeeper)',
  },
  {
    partition: 'VALIDATION',
    scenarioIds: ['academy_3_vs_1_defender_2', 'academy_3_vs_1_defender_3'],
    description: 'Increased defender density variations (2 and 3 defenders)',
  },
  {
    partition: 'TEST',
    scenarioIds: ['academy_3_vs_1_keeper_aggressive', 'academy_3_vs_1_shifted', 'academy_3_vs_1_randomized'],
    description: 'Unseen positional, sweeping goalkeeper, and randomized arrangements',
  },
];

export interface GeneralizationReport {
  evaluatedAt: string;
  policyName: string;
  numEpisodesPerScenario: number;
  baseSeed: number;
  scenarioResults: Record<string, AggregatedMetrics>;
  partitionSummary: {
    trainWinRate: number;
    validationWinRate: number;
    testWinRate: number;
    generalizationTransferRate: number; // test / max(0.001, train)
    overfittingGap: number; // train - test
  };
  summaryTable: string;
}

export class GeneralizationEvaluator {
  private tracker = new FootballMetricsTracker();

  public evaluatePolicy(
    policyName = 'mappo_trained',
    numEpisodes = 50,
    baseSeed = 500000
  ): GeneralizationReport {
    console.log('====================================================');
    console.log(`GMN-FOOTBALL-3 — HELD-OUT GENERALIZATION EVALUATION`);
    console.log(`Policy: ${policyName} | Episodes / scenario: ${numEpisodes}`);
    console.log('====================================================\n');

    let trainedAgent: TrainedPolicyAgent | null = null;
    if (policyName === 'mappo_trained') {
      try {
        trainedAgent = new TrainedPolicyAgent(MAPPO_WEIGHTS);
      } catch (err: any) {
        console.warn(`[Generalization] Neural policy warning: ${err?.message || err}`);
      }
    }

    const defaultBot = new RuleBasedAgent('bot_agent', 'Tactical AI', 'medium');
    const allScenarios = ACADEMY_SCENARIOS;
    const scenarioResults: Record<string, AggregatedMetrics> = {};

    for (const part of FORMAL_PARTITIONS) {
      console.log(`--- PARTITION: ${part.partition} (${part.description}) ---`);

      for (const scId of part.scenarioIds) {
        const scenario = allScenarios.find((s) => s.id === scId || s.codeName === scId);
        if (!scenario) {
          console.warn(`Skipping missing scenario: ${scId}`);
          continue;
        }

        this.tracker.clear();

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
                const obs = engine.getObservation().rawVector;
                actIdx = trainedAgent.act(obs, true);
                agentAction = mapDiscreteAction(actIdx);
              } else {
                // Rule-based fallback
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
                actIdx = 5; // Forward movement approximation
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
            this.tracker.recordTick(engine, leftActionIndices, stepRes.reward);

            if (stepRes.terminated || stepRes.truncated) {
              done = true;
            }
          }

          this.tracker.endEpisode(engine);
        }

        const agg = this.tracker.aggregate(policyName, scenario.id);
        scenarioResults[scenario.id] = agg;
        console.log(
          `  ✓ [${part.partition.padEnd(10, ' ')}] ${scenario.name.padEnd(45, ' ')} | Win Rate: ${agg.winRatePct.mean.toFixed(1)}% | Goals: ${agg.goalsScoredPerEpisode.mean.toFixed(2)} | Pass Acc: ${agg.passCompletionRatePct.mean.toFixed(1)}%`
        );
      }
    }

    // Partition aggregates
    const trainScenarios = FORMAL_PARTITIONS.find((p) => p.partition === 'TRAIN')?.scenarioIds || [];
    const valScenarios = FORMAL_PARTITIONS.find((p) => p.partition === 'VALIDATION')?.scenarioIds || [];
    const testScenarios = FORMAL_PARTITIONS.find((p) => p.partition === 'TEST')?.scenarioIds || [];

    const trainWins = trainScenarios.map((id) => scenarioResults[id]?.winRatePct.mean ?? 0);
    const valWins = valScenarios.map((id) => scenarioResults[id]?.winRatePct.mean ?? 0);
    const testWins = testScenarios.map((id) => scenarioResults[id]?.winRatePct.mean ?? 0);

    const trainWinRate = trainWins.reduce((a, b) => a + b, 0) / Math.max(1, trainWins.length);
    const valWinRate = valWins.reduce((a, b) => a + b, 0) / Math.max(1, valWins.length);
    const testWinRate = testWins.reduce((a, b) => a + b, 0) / Math.max(1, testWins.length);

    const generalizationTransferRate = trainWinRate > 0 ? (testWinRate / trainWinRate) * 100 : 0;
    const overfittingGap = trainWinRate - testWinRate;

    const summaryTable = this.generateSummaryTable(scenarioResults, {
      trainWinRate,
      validationWinRate: valWinRate,
      testWinRate,
      generalizationTransferRate,
      overfittingGap,
    });

    console.log('\n' + summaryTable);

    return {
      evaluatedAt: new Date().toISOString(),
      policyName,
      numEpisodesPerScenario: numEpisodes,
      baseSeed,
      scenarioResults,
      partitionSummary: {
        trainWinRate,
        validationWinRate: valWinRate,
        testWinRate,
        generalizationTransferRate,
        overfittingGap,
      },
      summaryTable,
    };
  }

  private generateSummaryTable(
    scenarioResults: Record<string, AggregatedMetrics>,
    summary: {
      trainWinRate: number;
      validationWinRate: number;
      testWinRate: number;
      generalizationTransferRate: number;
      overfittingGap: number;
    }
  ): string {
    const lines: string[] = [];
    lines.push('========================================================================================================');
    lines.push('HELD-OUT GENERALIZATION SUMMARY');
    lines.push('========================================================================================================');
    lines.push('| Partition   | Scenario                                       | Win Rate (%) | Goals Scored   | Pass Acc (%) |');
    lines.push('|:------------|:-----------------------------------------------|:------------:|:--------------:|:------------:|');

    for (const part of FORMAL_PARTITIONS) {
      for (const scId of part.scenarioIds) {
        const m = scenarioResults[scId];
        if (!m) continue;
        const winStr = `${m.winRatePct.mean.toFixed(1)}%`;
        const goalsStr = `${m.goalsScoredPerEpisode.mean.toFixed(2)} ± ${m.goalsScoredPerEpisode.std.toFixed(2)}`;
        const passStr = `${m.passCompletionRatePct.mean.toFixed(1)}%`;

        lines.push(
          `| ${part.partition.padEnd(11, ' ')} | ${scId.padEnd(46, ' ')} | ${winStr.padEnd(12, ' ')} | ${goalsStr.padEnd(14, ' ')} | ${passStr.padEnd(12, ' ')} |`
        );
      }
    }

    lines.push('--------------------------------------------------------------------------------------------------------');
    lines.push(`Overall Train Partition Win Rate:      ${summary.trainWinRate.toFixed(1)}%`);
    lines.push(`Overall Validation Partition Win Rate: ${summary.validationWinRate.toFixed(1)}%`);
    lines.push(`Overall Held-out Test Win Rate:        ${summary.testWinRate.toFixed(1)}%`);
    lines.push(`Generalization Transfer Retention:     ${summary.generalizationTransferRate.toFixed(1)}%`);
    lines.push(`Overfitting Gap (Train - Test):        ${summary.overfittingGap.toFixed(1)}%`);
    lines.push('========================================================================================================');

    return lines.join('\n');
  }
}

// CLI runner
if (process.argv[1]?.endsWith('eval_generalization.ts')) {
  const policy = process.argv[2] || 'mappo_trained';
  const episodes = parseInt(process.argv[3] || '50', 10);
  const evaluator = new GeneralizationEvaluator();
  evaluator.evaluatePolicy(policy, episodes);
}
