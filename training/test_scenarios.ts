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

// Regression test for A1: Replay frame event field sticky bug
// Verifies: run past a goal with no further events; only goal tick has event: 'goal', later ticks have no event
totalTests++;
console.log('\nValidating Regression A1: Replay frame sticky event isolation...');
try {
  const engine = new GameEngine();
  engine.resetToKickoff(true, 12345);
  engine.replayBuffer = [];
  engine.events = [];

  const emptyActionMap = new Map<string, AgentAction>();
  engine.step(emptyActionMap, 1 / 60);
  const frame0 = engine.replayBuffer[engine.replayBuffer.length - 1];
  if (frame0.event !== undefined) {
    throw new Error(`Expected frame 0 to have no event, got ${JSON.stringify(frame0.event)}`);
  }

  // Trigger goal event
  engine.ball.position = { x: 1.05, y: 0, z: 0 };
  engine.ball.velocity = { x: 0, y: 0, z: 0 };
  engine.ball.ownerId = null;
  engine.step(emptyActionMap, 1 / 60);
  const goalFrameIdx = engine.replayBuffer.length - 1;
  const goalFrame = engine.replayBuffer[goalFrameIdx];
  if (!goalFrame.event || goalFrame.event.type !== 'goal') {
    throw new Error(`Expected goal frame to have event.type === 'goal', got ${JSON.stringify(goalFrame.event)}`);
  }

  // Run 5 further ticks without any events
  for (let i = 0; i < 5; i++) {
    engine.ball.position = { x: 0, y: 0, z: 0 };
    engine.ball.velocity = { x: 0, y: 0, z: 0 };
    engine.ball.ownerId = null;
    engine.step(emptyActionMap, 1 / 60);
    const postGoalFrame = engine.replayBuffer[engine.replayBuffer.length - 1];
    if (postGoalFrame.event !== undefined) {
      throw new Error(
        `Sticky event bug detected at tick +${i + 1}! Event was re-stamped: ${JSON.stringify(postGoalFrame.event)}`
      );
    }
  }

  console.log('  ✓ Successfully verified replay frames only capture events on the exact tick they occur.');
  passedTests++;
} catch (err: any) {
  console.error('  ✗ FAILED regression test for replay sticky event:', err.message);
}

// Regression test for A2: Throw-in touchline 0.05 inset
// Verifies: forcing ball out over touchline triggers exactly 1 out_of_bounds event without immediate re-trigger
totalTests++;
console.log('\nValidating Regression A2: Touchline throw-in rapid re-trigger prevention...');
try {
  const engine = new GameEngine();
  engine.resetToKickoff(true, 54321);
  engine.events = [];

  const emptyActionMap = new Map<string, AgentAction>();

  // Force ball over maxY touchline (e.g. y = 0.45, PITCH.maxY = 0.42)
  engine.ball.position = { x: 0, y: 0.45, z: 0 };
  engine.ball.velocity = { x: 0, y: 0, z: 0 };

  // Step 1: Triggers throw-in and repositioning with 0.05 inset
  engine.step(emptyActionMap, 1 / 60);

  const initialOobEvents = engine.events.filter((e) => e.type === 'out_of_bounds');
  if (initialOobEvents.length !== 1) {
    throw new Error(`Expected exactly 1 out_of_bounds event, got ${initialOobEvents.length}`);
  }

  // Step next 10 ticks with zero velocity
  for (let i = 0; i < 10; i++) {
    engine.ball.velocity = { x: 0, y: 0, z: 0 };
    engine.step(emptyActionMap, 1 / 60);
  }

  const finalOobEvents = engine.events.filter((e) => e.type === 'out_of_bounds');
  if (finalOobEvents.length !== 1) {
    throw new Error(
      `Throw-in re-triggered! Expected 1 out_of_bounds event over 10 ticks, got ${finalOobEvents.length}`
    );
  }

  console.log('  ✓ Successfully verified throw-in ball position inset prevents rapid re-trigger.');
  passedTests++;
} catch (err: any) {
  console.error('  ✗ FAILED regression test for throw-in rapid re-trigger:', err.message);
}

console.log('\n====================================================');
console.log(`Scenario Validation Summary: ${passedTests}/${totalTests} Scenarios Passed`);
console.log('====================================================');

if (passedTests === totalTests) {
  process.exit(0);
} else {
  process.exit(1);
}
