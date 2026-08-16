import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { RuleBasedAgent } from '../src/agents/RuleBasedAgent';
import { AgentAction, AgentDecisionContext } from '../src/types/football';

interface ScenarioPlayabilityResult {
  scenarioId: string;
  scenarioName: string;
  episodesRun: number;
  crashCount: number;
  avgEpisodeLengthSteps: number;
  avgEpisodeLengthSec: number;
  leftGoalsTotal: number;
  rightGoalsTotal: number;
  leftGoalRate: number;
  objectiveCompletions: Record<string, { text: string; completedCount: number; rate: number }>;
}

export function runPlayabilitySuite(episodesPerScenario = 50): ScenarioPlayabilityResult[] {
  console.log('========================================================================');
  console.log(`GMN-FOOTBALL-3 — SCENARIO PLAYABILITY & OUTCOME VALIDATION GATE`);
  console.log(`Evaluating ${ACADEMY_SCENARIOS.length} scenarios across ${episodesPerScenario} episodes each`);
  console.log('========================================================================\n');

  const results: ScenarioPlayabilityResult[] = [];
  let emptyGoalPassed = false;

  for (const scenario of ACADEMY_SCENARIOS) {
    console.log(`------------------------------------------------------------------------`);
    console.log(`Running Scenario: ${scenario.name} (${scenario.id}) [${scenario.difficulty}]`);
    console.log(`Time Limit: ${scenario.timeLimitSeconds}s | Team: ${scenario.teamLeftPlayers}v${scenario.teamRightPlayers}`);
    console.log(`------------------------------------------------------------------------`);

    let crashCount = 0;
    let totalSteps = 0;
    let totalTimeSec = 0;
    let leftGoalsTotal = 0;
    let rightGoalsTotal = 0;

    const objectiveTracker: Record<string, { text: string; count: number }> = {};
    scenario.objectives.forEach((obj) => {
      objectiveTracker[obj.id] = { text: obj.text, count: 0 };
    });

    const leftAgent = new RuleBasedAgent('rule_left', 'Rule-Based Left AI', 'medium');
    const rightAgent = new RuleBasedAgent('rule_right', 'Rule-Based Right AI', 'medium');

    for (let ep = 0; ep < episodesPerScenario; ep++) {
      const seed = 500000 + ep * 1009;
      const engine = new GameEngine();

      try {
        engine.loadScenario(scenario, seed);

        const maxSteps = Math.ceil(scenario.timeLimitSeconds * 60) + 120;
        let epDone = false;
        let stepCount = 0;

        while (!epDone && stepCount < maxSteps) {
          stepCount++;

          const actionMap = new Map<string, AgentAction>();

          for (const player of engine.players) {
            const teammates = engine.players.filter((p) => p.team === player.team);
            const opponents = engine.players.filter((p) => p.team !== player.team);

            const context: AgentDecisionContext = {
              player,
              teammates,
              opponents,
              ball: engine.ball,
              allPlayers: engine.players,
              teamSide: player.team,
              controlledPlayerId: engine.controlledPlayerId,
              matchTime: engine.matchTimeSeconds,
              rng: engine.rng,
            };

            const agent = player.team === 'left' ? leftAgent : rightAgent;
            actionMap.set(player.id, agent.decide(context));
          }

          const stepRes = engine.step(actionMap, 1 / 60);
          if (stepRes.terminated || stepRes.truncated) {
            epDone = true;
          }
        }

        totalSteps += stepCount;
        totalTimeSec += engine.matchTimeSeconds;
        leftGoalsTotal += engine.score.left;
        rightGoalsTotal += engine.score.right;

        if (engine.activeScenario) {
          for (const obj of engine.activeScenario.objectives) {
            if (obj.isCompleted && objectiveTracker[obj.id]) {
              objectiveTracker[obj.id].count++;
            }
          }
        }
      } catch (err) {
        crashCount++;
        console.error(`  [CRASH] Episode ${ep + 1} failed with exception:`, err);
      }
    }

    const avgSteps = totalSteps / episodesPerScenario;
    const avgSec = totalTimeSec / episodesPerScenario;
    const leftGoalRate = (leftGoalsTotal / episodesPerScenario) * 100;

    const objectiveSummary: Record<string, { text: string; completedCount: number; rate: number }> = {};
    for (const [id, data] of Object.entries(objectiveTracker)) {
      objectiveSummary[id] = {
        text: data.text,
        completedCount: data.count,
        rate: (data.count / episodesPerScenario) * 100,
      };
    }

    const summary: ScenarioPlayabilityResult = {
      scenarioId: scenario.id,
      scenarioName: scenario.name,
      episodesRun: episodesPerScenario,
      crashCount,
      avgEpisodeLengthSteps: Math.round(avgSteps * 10) / 10,
      avgEpisodeLengthSec: Math.round(avgSec * 100) / 100,
      leftGoalsTotal,
      rightGoalsTotal,
      leftGoalRate: Math.round(leftGoalRate * 10) / 10,
      objectiveCompletions: objectiveSummary,
    };

    results.push(summary);

    console.log(`Episodes Run: ${summary.episodesRun} | Crashes: ${summary.crashCount}`);
    console.log(`Avg Duration: ${summary.avgEpisodeLengthSec}s (${summary.avgEpisodeLengthSteps} steps)`);
    console.log(`Goals Scored: Left: ${summary.leftGoalsTotal} (${summary.leftGoalRate}%), Right: ${summary.rightGoalsTotal}`);
    console.log(`Objectives:`);
    for (const [id, obj] of Object.entries(summary.objectiveCompletions)) {
      console.log(`  - [${id}] "${obj.text}": ${obj.completedCount}/${summary.episodesRun} (${obj.rate.toFixed(1)}%)`);
    }
    console.log('');

    if (scenario.id === 'academy_empty_goal') {
      if (summary.leftGoalsTotal > 0) {
        emptyGoalPassed = true;
      }
    }
  }

  console.log('========================================================================');
  console.log('PLAYABILITY SUMMARY TABLE');
  console.log('========================================================================');
  console.log(
    '| SCENARIO ID                        | EPISODES | CRASHES | AVG SEC | L-GOALS | R-GOALS | GOAL RATE |'
  );
  console.log(
    '|------------------------------------|----------|---------|---------|---------|---------|-----------|'
  );
  for (const r of results) {
    const idPad = r.scenarioId.padEnd(34, ' ');
    const epPad = String(r.episodesRun).padStart(8, ' ');
    const crPad = String(r.crashCount).padStart(7, ' ');
    const secPad = (r.avgEpisodeLengthSec.toFixed(2) + 's').padStart(7, ' ');
    const lgPad = String(r.leftGoalsTotal).padStart(7, ' ');
    const rgPad = String(r.rightGoalsTotal).padStart(7, ' ');
    const ratePad = (r.leftGoalRate.toFixed(1) + '%').padStart(9, ' ');
    console.log(`| ${idPad} | ${epPad} | ${crPad} | ${secPad} | ${lgPad} | ${rgPad} | ${ratePad} |`);
  }
  console.log('========================================================================\n');

  if (!emptyGoalPassed) {
    console.error('FATAL GATE FAILURE: academy_empty_goal scored 0 goals across all episodes!');
    process.exit(1);
  }

  console.log('✓ Playability gate passed: academy_empty_goal achieved non-zero goal outcomes.');
  return results;
}

if (import.meta.url.endsWith(process.argv[1]) || process.argv[1]?.includes('verify_scenario_playability')) {
  const episodes = parseInt(process.argv[2], 10) || 50;
  runPlayabilitySuite(episodes);
}
