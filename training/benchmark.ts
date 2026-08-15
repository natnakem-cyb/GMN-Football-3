import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ActionType, AgentAction } from '../src/types/football';
import { mapDiscreteAction, ACTION_SPACE_SIZE } from './action_mapping';

function benchmarkRawEngine(totalSteps = 10000) {
  console.log('==================================================');
  console.log('GMN FOOTBALL — RAW TYPESCRIPT ENGINE BENCHMARK');
  console.log(`Steps: ${totalSteps} | Scenario: academy_empty_goal`);
  console.log('==================================================');

  const engine = new GameEngine();
  const scenario = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_empty_goal')!;
  engine.loadScenario(scenario);

  const startTime = performance.now();

  for (let i = 0; i < totalSteps; i++) {
    const actionMap = new Map<string, AgentAction>();
    const controlledPlayer = engine.players[0];
    if (controlledPlayer) {
      actionMap.set(controlledPlayer.id, mapDiscreteAction(i % ACTION_SPACE_SIZE, controlledPlayer.heading));
    }

    const res = engine.step(actionMap, 1 / 60);

    // Auto-reset when episode ends during benchmark
    if (res.terminated || res.truncated) {
      engine.loadScenario(scenario);
    }
  }

  const durationMs = performance.now() - startTime;
  const durationSec = durationMs / 1000;
  const stepsPerSec = totalSteps / durationSec;

  console.log(`\nResults:`);
  console.log(`- Total Time: ${durationSec.toFixed(3)}s`);
  console.log(`- Throughput: ${Math.round(stepsPerSec).toLocaleString()} steps/sec`);
  console.log(`- Latency per step: ${(durationMs / totalSteps * 1000).toFixed(2)} µs`);
  console.log('==================================================\n');
}

benchmarkRawEngine(10000);
