/**
 * GMN-Football-3 — Opponent Robustness Evaluation Matrix
 * Evaluates policy transfer and resilience across opponent difficulty levels (easy, medium, hard, master)
 * and scripted defensive behaviors to detect opponent-specific policy exploitation.
 */

import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ActionType, AgentAction } from '../src/types/football';
import { RuleBasedAgent } from '../src/agents/RuleBasedAgent';
import { ScriptedScenarioAgent } from '../src/agents/ScriptedScenarioAgent';
import { TrainedPolicyAgent } from '../src/agents/TrainedPolicyAgent';
import { MAPPO_WEIGHTS } from '../src/agents/mappo_weights';
import { mapDiscreteAction } from './action_mapping';
import { FootballMetricsTracker, AggregatedMetrics } from '../src/engine/FootballMetrics';

export interface OpponentConfig {
  id: string;
  name: string;
  type: 'rule_based' | 'scripted';
  difficulty?: 'easy' | 'medium' | 'hard' | 'master';
  scriptedType?: 'defender_contain' | 'keeper_dive' | 'static_obstacle';
}

export const OPPONENT_MATRIX: OpponentConfig[] = [
  { id: 'opp_rule_easy', name: 'Weak / Easy Tactical Bot', type: 'rule_based', difficulty: 'easy' },
  { id: 'opp_rule_medium', name: 'Default / Medium Tactical Bot', type: 'rule_based', difficulty: 'medium' },
  { id: 'opp_rule_hard', name: 'Strong / Aggressive Tactical Bot', type: 'rule_based', difficulty: 'hard' },
  { id: 'opp_rule_master', name: 'Master / Elite Tactical Bot', type: 'rule_based', difficulty: 'master' },
  { id: 'opp_scripted_contain', name: 'Scripted Defender Contain Bot', type: 'scripted', scriptedType: 'defender_contain' },
];

export interface OpponentRobustnessReport {
  evaluatedAt: string;
  policyName: string;
  scenarioId: string;
  numEpisodesPerOpponent: number;
  results: Record<string, AggregatedMetrics>;
  summaryTable: string;
}

export class OpponentRobustnessEvaluator {
  private tracker = new FootballMetricsTracker();

  public evaluateMatrix(
    scenarioId = 'academy_3_vs_1_with_keeper',
    policyName = 'mappo_trained',
    numEpisodes = 50,
    baseSeed = 700000
  ): OpponentRobustnessReport {
    const scenario = ACADEMY_SCENARIOS.find((s) => s.id === scenarioId || s.codeName === scenarioId);
    if (!scenario) {
      throw new Error(`[OpponentRobustness] Scenario ${scenarioId} not found.`);
    }

    console.log('====================================================');
    console.log(`GMN-FOOTBALL-3 — OPPONENT ROBUSTNESS EVALUATION`);
    console.log(`Scenario: ${scenario.name} | Policy: ${policyName}`);
    console.log(`Episodes / Opponent: ${numEpisodes}`);
    console.log('====================================================\n');

    let trainedAgent: TrainedPolicyAgent | null = null;
    if (policyName === 'mappo_trained') {
      try {
        trainedAgent = new TrainedPolicyAgent(MAPPO_WEIGHTS);
      } catch (err: any) {
        console.warn(`[OpponentRobustness] Neural policy warning: ${err?.message || err}`);
      }
    }

    const defaultLeftBot = new RuleBasedAgent('left_rule', 'Left AI', 'medium');
    const results: Record<string, AggregatedMetrics> = {};

    for (const opp of OPPONENT_MATRIX) {
      this.tracker.clear();

      for (let ep = 0; ep < numEpisodes; ep++) {
        const epSeed = baseSeed + ep;
        const engine = new GameEngine();
        engine.loadScenario(scenario, epSeed);

        // Instantiate opponent agent
        const rightBot =
          opp.type === 'rule_based'
            ? new RuleBasedAgent(`opp_${opp.id}`, opp.name, opp.difficulty || 'medium')
            : new ScriptedScenarioAgent(`opp_${opp.id}`, opp.name, opp.scriptedType || 'defender_contain');

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

          // Left team actions
          leftPlayers.forEach((player) => {
            let actIdx = 0;
            let agentAction: AgentAction;

            if (trainedAgent && trainedAgent.isValidCheckpoint()) {
              const obs = engine.getObservation().rawVector;
              actIdx = trainedAgent.act(obs, true);
              agentAction = mapDiscreteAction(actIdx);
            } else {
              agentAction = defaultLeftBot.decide({
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

          // Right team (Opponent) actions
          rightPlayers.forEach((player) => {
            const oppAction = rightBot.decide({
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

      const agg = this.tracker.aggregate(policyName, `${scenario.id}_${opp.id}`);
      results[opp.id] = agg;
      console.log(
        `  ✓ Opponent: ${opp.name.padEnd(35, ' ')} | Win Rate: ${agg.winRatePct.mean.toFixed(1)}% | Goals: ${agg.goalsScoredPerEpisode.mean.toFixed(2)} | Turnovers: ${agg.turnoversConcededPerEpisode.mean.toFixed(2)}`
      );
    }

    const summaryTable = this.generateSummaryTable(results);
    console.log('\n' + summaryTable);

    return {
      evaluatedAt: new Date().toISOString(),
      policyName,
      scenarioId: scenario.id,
      numEpisodesPerOpponent: numEpisodes,
      results,
      summaryTable,
    };
  }

  private generateSummaryTable(results: Record<string, AggregatedMetrics>): string {
    const lines: string[] = [];
    lines.push('========================================================================================================');
    lines.push('OPPONENT ROBUSTNESS SUMMARY TABLE');
    lines.push('========================================================================================================');
    lines.push('| Opponent Configuration              | Win Rate (%) | Goals Scored   | Turnovers Conceded | Pass Acc (%) |');
    lines.push('|:------------------------------------|:------------:|:--------------:|:------------------:|:------------:|');

    for (const opp of OPPONENT_MATRIX) {
      const m = results[opp.id];
      if (!m) continue;
      const winStr = `${m.winRatePct.mean.toFixed(1)}%`;
      const goalsStr = `${m.goalsScoredPerEpisode.mean.toFixed(2)} ± ${m.goalsScoredPerEpisode.std.toFixed(2)}`;
      const turnStr = `${m.turnoversConcededPerEpisode.mean.toFixed(2)}`;
      const passStr = `${m.passCompletionRatePct.mean.toFixed(1)}%`;

      lines.push(
        `| ${opp.name.padEnd(35, ' ')} | ${winStr.padEnd(12, ' ')} | ${goalsStr.padEnd(14, ' ')} | ${turnStr.padEnd(18, ' ')} | ${passStr.padEnd(12, ' ')} |`
      );
    }
    lines.push('========================================================================================================');
    return lines.join('\n');
  }
}

// CLI runner
if (process.argv[1]?.endsWith('eval_opponent_robustness.ts')) {
  const scenario = process.argv[2] || 'academy_3_vs_1_with_keeper';
  const policy = process.argv[3] || 'mappo_trained';
  const episodes = parseInt(process.argv[4] || '50', 10);
  const evaluator = new OpponentRobustnessEvaluator();
  evaluator.evaluateMatrix(scenario, policy, episodes);
}
