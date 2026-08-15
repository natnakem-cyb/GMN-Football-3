import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { OBSERVATION_DIM, ACTION_SPACE_SIZE } from '../src/engine/Contract';
import { mapDiscreteAction } from './action_mapping';
import { AgentAction } from '../src/types/football';

console.log('====================================================');
console.log('GMN-FOOTBALL-3 — SCENARIO VALIDATION SUITE');
console.log('====================================================');

let totalTests = 0;
let passedTests = 0;

for (const scenario of ACADEMY_SCENARIOS) {
  totalTests++;
  console.log(`\nValidating Scenario: ${scenario.name} (${scenario.id} / ${scenario.codeName})...`);

  try {
    const engine = new GameEngine();
    engine.loadScenario(scenario, 12345);

    // 1. Initial State Checks
    if (!engine.controlledPlayerId) {
      throw new Error(`Scenario ${scenario.id} has null controlledPlayerId.`);
    }

    if (engine.players.length === 0) {
      throw new Error(`Scenario ${scenario.id} loaded 0 players.`);
    }

    // 2. Initial Observation Verification
    const initialObs = engine.getObservation();
    if (initialObs.rawVector.length !== OBSERVATION_DIM) {
      throw new Error(
        `Initial observation vector length mismatch: expected ${OBSERVATION_DIM}, got ${initialObs.rawVector.length}`
      );
    }

    // Check for NaN or Inf
    for (let i = 0; i < initialObs.rawVector.length; i++) {
      const val = initialObs.rawVector[i];
      if (Number.isNaN(val) || !Number.isFinite(val)) {
        throw new Error(`Invalid float in initial observation at offset ${i}: ${val}`);
      }
    }

    // 3. Step validation across various actions
    const testActions = [0, 1, 5, 12, 13, 14, 16, 17, 18, 9, 10, 11];
    for (let step = 0; step < testActions.length; step++) {
      const actionIdx = testActions[step % ACTION_SPACE_SIZE];
      const player = engine.players.find((p) => p.id === engine.controlledPlayerId) || engine.players[0];
      const mapped = mapDiscreteAction(actionIdx, player.heading);

      const actionMap = new Map<string, AgentAction>();
      actionMap.set(player.id, mapped);

      const res = engine.step(actionMap, 1 / 60);

      if (res.observation.rawVector.length !== OBSERVATION_DIM) {
        throw new Error(`Step ${step} observation length mismatch: ${res.observation.rawVector.length}`);
      }

      if (Number.isNaN(res.reward) || !Number.isFinite(res.reward)) {
        throw new Error(`Step ${step} invalid reward: ${res.reward}`);
      }
    }

    console.log(`  ✓ Successfully verified setup, observations, and stepping for ${scenario.id}`);
    passedTests++;
  } catch (err: any) {
    console.error(`  ✗ FAILED scenario ${scenario.id}:`, err.message);
  }
}

console.log('\n====================================================');
console.log(`Scenario Validation Summary: ${passedTests}/${totalTests} Scenarios Passed`);
console.log('====================================================');

if (passedTests === totalTests) {
  process.exit(0);
} else {
  process.exit(1);
}
