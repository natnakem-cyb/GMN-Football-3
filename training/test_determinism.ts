import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { AgentAction } from '../src/types/football';
import { mapDiscreteAction } from './action_mapping';
import { OBSERVATION_DIM } from '../src/engine/Contract';

function runSeededSimulation(scenarioId: string, seed: number, actionSequence: number[]): {
  observations: number[][];
  rewards: number[];
  finalBallX: number;
} {
  const engine = new GameEngine();
  const scenario = ACADEMY_SCENARIOS.find((s) => s.id === scenarioId || s.codeName === scenarioId)!;
  if (!scenario) {
    throw new Error(`Scenario ${scenarioId} not found`);
  }

  engine.loadScenario(scenario, seed);

  const initialObs = engine.getObservation();
  if (initialObs.rawVector.length !== OBSERVATION_DIM) {
    throw new Error(`Initial observation dimension mismatch: expected ${OBSERVATION_DIM}, got ${initialObs.rawVector.length}`);
  }

  const observations: number[][] = [initialObs.rawVector];
  const rewards: number[] = [];

  for (let t = 0; t < actionSequence.length; t++) {
    const actionIdx = actionSequence[t];
    const player = engine.players.find((p) => p.id === engine.controlledPlayerId) || engine.players[0];
    const action = mapDiscreteAction(actionIdx, player.heading);

    const actionMap = new Map<string, AgentAction>();
    actionMap.set(player.id, action);

    const res = engine.step(actionMap, 1 / 60);
    observations.push(res.observation.rawVector);
    rewards.push(res.reward);
  }

  return { observations, rewards, finalBallX: engine.ball.position.x };
}

console.log('====================================================');
console.log('GMN-FOOTBALL-3 — DETERMINISM & SEED REPRODUCIBILITY TEST');
console.log('====================================================');

const actions = [5, 5, 5, 5, 5, 13, 13, 13, 13, 13, 5, 5, 12, 0, 0, 16, 17, 18, 1, 2, 3, 4, 9, 10, 11];

let allPassed = true;

for (const sc of ['academy_empty_goal', 'academy_run_to_score', 'academy_pass_and_shoot_with_keeper']) {
  console.log(`\nTesting scenario: ${sc} (Seed 424242)...`);
  const run1 = runSeededSimulation(sc, 424242, actions);
  const run2 = runSeededSimulation(sc, 424242, actions);

  let scenarioMatch = true;
  let maxDiff = 0;

  for (let i = 0; i < run1.observations.length; i++) {
    const obs1 = run1.observations[i];
    const obs2 = run2.observations[i];

    for (let d = 0; d < obs1.length; d++) {
      const diff = Math.abs(obs1[d] - obs2[d]);
      if (diff > maxDiff) maxDiff = diff;
      if (diff > 1e-7) {
        scenarioMatch = false;
        console.error(`  [MISMATCH] Step ${i}, Dim ${d}: ${obs1[d]} vs ${obs2[d]}`);
        break;
      }
    }
  }

  for (let i = 0; i < run1.rewards.length; i++) {
    if (Math.abs(run1.rewards[i] - run2.rewards[i]) > 1e-7) {
      scenarioMatch = false;
      console.error(`  [REWARD MISMATCH] Step ${i}: ${run1.rewards[i]} vs ${run2.rewards[i]}`);
    }
  }

  if (scenarioMatch && maxDiff === 0) {
    console.log(`  ✓ ${sc}: 100% bitwise trajectory determinism confirmed (max diff: ${maxDiff}).`);
  } else {
    console.error(`  ✗ ${sc}: Determinism check failed.`);
    allPassed = false;
  }
}

if (allPassed) {
  console.log('\n====================================================');
  console.log('✓ ALL DETERMINISM SUITES PASSED CLEANLY');
  console.log('====================================================');
  process.exit(0);
} else {
  console.error('\n✗ DETERMINISM TEST SUITE FAILED');
  process.exit(1);
}
