import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ObservationEncoder } from '../src/engine/ObservationEncoder';
import { ROLE_VOCABULARY, ROLE_DIM, BASE_OBSERVATION_DIM, OBSERVATION_DIM } from '../src/engine/Contract';

console.log('====================================================');
console.log('GMN-FOOTBALL-3 — PER-AGENT ROLE DIFFERENTIATION TEST');
console.log('====================================================');

const scenario = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_3_vs_1_with_keeper');
if (!scenario) {
  throw new Error('academy_3_vs_1_with_keeper scenario not found');
}

const engine = new GameEngine();
engine.loadScenario(scenario, 42);

const leftPlayers = engine.players.filter((p) => p.team === 'left');
console.log(`Found ${leftPlayers.length} controllable left agents:`);
leftPlayers.forEach((p, idx) => {
  console.log(`  Agent [${idx}]: id=${p.id}, name=${p.name}, role=${p.role}`);
});

if (leftPlayers.length < 3) {
  throw new Error(`Expected at least 3 left agents, got ${leftPlayers.length}`);
}

// Encode observations individually from each agent's viewpoint
const obsVectors = leftPlayers.map((p) => {
  const obs = ObservationEncoder.encode(
    engine.players,
    engine.ball,
    p.id,
    engine.score,
    engine.tickCount,
    3600,
    engine.gameMode
  );
  return {
    agentId: p.id,
    role: p.role,
    rawVector: obs.rawVector,
  };
});

// Verify each observation vector length
obsVectors.forEach(({ agentId, rawVector }) => {
  if (rawVector.length !== OBSERVATION_DIM) {
    throw new Error(`Agent ${agentId} rawVector length is ${rawVector.length}, expected ${OBSERVATION_DIM}`);
  }
});

// Extract role one-hot slices (offset 115..126)
const roleSlices = obsVectors.map(({ agentId, role, rawVector }) => {
  const slice = rawVector.slice(BASE_OBSERVATION_DIM, BASE_OBSERVATION_DIM + ROLE_DIM);
  const activeRoleIdx = slice.indexOf(1.0);
  const activeRoleName = activeRoleIdx >= 0 ? ROLE_VOCABULARY[activeRoleIdx] : 'UNKNOWN';
  return {
    agentId,
    expectedRole: role,
    slice,
    activeRoleIdx,
    activeRoleName,
  };
});

console.log('\nRole Slice Verification:');
roleSlices.forEach((r) => {
  console.log(`  Agent ${r.agentId} (expected: ${r.expectedRole}): activeRole=${r.activeRoleName}, slice=[${r.slice.join(', ')}]`);
  if (r.activeRoleName !== r.expectedRole) {
    throw new Error(`Role mismatch for agent ${r.agentId}: expected ${r.expectedRole}, got ${r.activeRoleName}`);
  }
  const onesCount = r.slice.filter((v) => v === 1.0).length;
  if (onesCount !== 1) {
    throw new Error(`Expected exactly 1 one-hot active bit in role slice, got ${onesCount}`);
  }
});

// Assert that the role slices DIFFERS across the CAM, LW, and RW agents
const camSliceStr = JSON.stringify(roleSlices[0].slice);
const lwSliceStr = JSON.stringify(roleSlices[1].slice);
const rwSliceStr = JSON.stringify(roleSlices[2].slice);

if (camSliceStr === lwSliceStr || camSliceStr === rwSliceStr || lwSliceStr === rwSliceStr) {
  throw new Error('FAILED: Role slices across agents are identical! Expected distinct role slices.');
}

console.log('\n✓ CONFIRMED: Each agent observation has a DISTINCT, per-agent self-role one-hot vector matching its own role.');
console.log('====================================================');
process.exit(0);
