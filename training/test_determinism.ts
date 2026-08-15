import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ActionType, AgentAction } from '../src/types/football';
import { mapDiscreteAction } from './action_mapping';

function runDeterministicRun(runId: number, numSteps = 100): { observations: number[][]; rewards: number[]; finalDist: number } {
  const engine = new GameEngine();
  const scenario = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_empty_goal')!;
  engine.loadScenario(scenario);

  const observations: number[][] = [];
  const rewards: number[] = [];

  for (let t = 0; t < numSteps; t++) {
    // Action pattern: Move right (5), sprint (13), shoot (12)
    const actionIdx = t < 60 ? 5 : t < 80 ? 13 : 12;
    const player = engine.players.find((p) => p.id === engine.controlledPlayerId) || engine.players[0];
    const action = mapDiscreteAction(actionIdx, player.heading);

    const actionMap = new Map<string, AgentAction>();
    actionMap.set(player.id, action);

    const res = engine.step(actionMap, 1 / 60);
    observations.push(res.observation.rawVector);
    rewards.push(res.reward);
  }

  const finalDist = engine.ball.position.x;
  return { observations, rewards, finalDist };
}

console.log('==================================================');
console.log('GMN FOOTBALL — DETERMINISM TEST');
console.log('==================================================');

console.log('\nRunning Run 1 (100 steps)...');
const run1 = runDeterministicRun(1, 100);

console.log('Running Run 2 (100 steps)...');
const run2 = runDeterministicRun(2, 100);

let isIdentical = true;
let maxDiff = 0;

for (let i = 0; i < run1.observations.length; i++) {
  const obs1 = run1.observations[i];
  const obs2 = run2.observations[i];
  const rew1 = run1.rewards[i];
  const rew2 = run2.rewards[i];

  if (Math.abs(rew1 - rew2) > 1e-6) {
    isIdentical = false;
    console.error(`Reward mismatch at step ${i}: ${rew1} vs ${rew2}`);
  }

  for (let d = 0; d < obs1.length; d++) {
    const diff = Math.abs(obs1[d] - obs2[d]);
    if (diff > maxDiff) maxDiff = diff;
    if (diff > 1e-6) {
      isIdentical = false;
      console.error(`Observation mismatch at step ${i}, dim ${d}: ${obs1[d]} vs ${obs2[d]}`);
      break;
    }
  }
}

console.log(`\nMax numerical difference: ${maxDiff}`);
if (isIdentical && maxDiff === 0) {
  console.log('✓ DETERMINISM VERIFIED: Both runs produced 100% identical states, observations, and rewards.');
  process.exit(0);
} else {
  console.error('✗ DETERMINISM FAILED');
  process.exit(1);
}
