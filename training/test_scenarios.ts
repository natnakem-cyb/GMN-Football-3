import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { OBSERVATION_DIM, ACTION_SPACE_SIZE } from '../src/engine/Contract';
import { ObservationEncoder } from '../src/engine/ObservationEncoder';
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
      const mapped = mapDiscreteAction(actionIdx);

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

    // 4. Per-agent role differentiation check for multi-agent scenarios
    const leftAgents = engine.players.filter((p) => p.team === 'left');
    if (leftAgents.length > 1) {
      const distinctRoles = new Set(leftAgents.map((p) => p.role));
      if (distinctRoles.size > 1) {
        const agentObsSlices = leftAgents.map((p) => {
          const obs = ObservationEncoder.encode(
            engine.players,
            engine.ball,
            p.id,
            engine.score,
            engine.tickCount,
            3600,
            engine.gameMode
          );
          return obs.rawVector.slice(115, 127);
        });
        // Check that not all role slices are identical
        const firstSliceStr = JSON.stringify(agentObsSlices[0]);
        const hasDivergence = agentObsSlices.some((sl) => JSON.stringify(sl) !== firstSliceStr);
        if (!hasDivergence) {
          throw new Error(`Role slices failed to differentiate across agents with distinct roles in ${scenario.id}`);
        }
      }
    }

    passedTests++;
  } catch (err: any) {
    console.error(`  ✗ FAILED scenario ${scenario.id}:`, err.message);
  }
}

// Regression test for Issue #2: resetToKickoff(resetScore = false, seed?: number)
// Verifies: unrecognized scenario / kickoff reset + seed -> score unchanged, seed correctly applied
totalTests++;
console.log('\nValidating Regression: resetToKickoff with seed and score preservation...');
try {
  const engine = new GameEngine();
  engine.score = { left: 3, right: 2 };
  const targetSeed = 987654;
  engine.resetToKickoff(false, targetSeed);

  if (engine.score.left !== 3 || engine.score.right !== 2) {
    throw new Error(`Score was corrupted by resetToKickoff with seed! Expected 3-2, got ${engine.score.left}-${engine.score.right}`);
  }

  // Verify RNG state matches fresh SeededRNG with targetSeed
  const testVal1 = engine.rng.next();
  const testVal2 = engine.rng.next();

  const referenceRng = new (engine.rng.constructor as any)(targetSeed);
  const refVal1 = referenceRng.next();
  const refVal2 = referenceRng.next();

  if (testVal1 !== refVal1 || testVal2 !== refVal2) {
    throw new Error(`Seed was not correctly applied to RNG in resetToKickoff!`);
  }

  console.log('  ✓ Successfully verified resetToKickoff score preservation and seed application.');
  passedTests++;
} catch (err: any) {
  console.error('  ✗ FAILED regression test for resetToKickoff:', err.message);
}

console.log('\n====================================================');
console.log(`Scenario Validation Summary: ${passedTests}/${totalTests} Scenarios Passed`);
console.log('====================================================');

if (passedTests === totalTests) {
  process.exit(0);
} else {
  process.exit(1);
}
