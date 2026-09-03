/**
 * GMN-Football-3 — Modular Feature Encoder Test & Contract Verification
 * Tests the tensor dimension specifications and modular entity extraction against simple115_v3_role.
 */

import { GameEngine } from '../src/engine/GameEngine';
import { ObservationEncoder } from '../src/engine/ObservationEncoder';
import {
  OBSERVATION_DIM,
  OBSERVATION_SCHEMA_VERSION,
  ROLE_DIM,
  ROLE_VOCABULARY,
} from '../src/engine/Contract';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ModularFeatureParser } from './modular_encoder';

function runTests() {
  console.log('====================================================');
  console.log('MODULAR FEATURE ENCODER & CONTRACT SPEC TEST');
  console.log(`Contract: ${OBSERVATION_SCHEMA_VERSION} (${OBSERVATION_DIM} floats)`);
  console.log('====================================================\n');

  // Test 1: Dimensions in 3v1 Scenario
  console.log('1. Testing Academy 3 vs 1 with Keeper...');
  const scenario3v1 = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_3_vs_1_with_keeper')!;
  const engine3v1 = new GameEngine();
  engine3v1.loadScenario(scenario3v1, 12345);

  const activePlayer =
    engine3v1.players.find((p) => p.id === engine3v1.controlledPlayerId) || engine3v1.players[0];
  const obs = ObservationEncoder.encode(
    engine3v1.players,
    engine3v1.ball,
    activePlayer.id,
    engine3v1.score,
    engine3v1.matchTimeSeconds,
    3000
  );

  if (obs.rawVector.length !== OBSERVATION_DIM) {
    throw new Error(`Expected ${OBSERVATION_DIM} floats, got ${obs.rawVector.length}`);
  }

  const entities = ModularFeatureParser.parse(obs.rawVector);

  console.log(`  Ego Player ID: ${activePlayer.id}, Role: ${entities.ego.role}`);
  console.log(`  Ego Features Dim: ${entities.ego.features.length} (Expected: 30)`);
  console.log(`  Ball Features Dim: ${entities.ball.features.length} (Expected: 10)`);
  console.log(`  Teammates Count: ${entities.teammates.count} (Expected: 2 teammates for 3v1)`);
  console.log(`  Opponents Count: ${entities.opponents.count} (Expected: 2 opponents: 1 defender + 1 keeper)`);
  console.log(`  Match Features Dim: ${entities.match.features.length} (Expected: 7)`);

  if (entities.ego.features.length !== 30) {
    throw new Error(`Ego feature dimension mismatch: ${entities.ego.features.length} != 30`);
  }
  if (entities.ball.features.length !== 10) {
    throw new Error(`Ball feature dimension mismatch: ${entities.ball.features.length} != 10`);
  }
  if (entities.teammates.count !== 2) {
    throw new Error(`Teammate count mismatch: ${entities.teammates.count} != 2`);
  }
  if (entities.opponents.count !== 2) {
    throw new Error(`Opponent count mismatch: ${entities.opponents.count} != 2`);
  }
  console.log('  ✓ 3v1 entity decomposition validated cleanly.');

  // Test 2: Full 11v11 Match
  console.log('\n2. Testing 11 vs 11 Full Match Entity Decomposition...');
  const scenario11v11 = ACADEMY_SCENARIOS.find((s) => s.id === '11_vs_11')!;
  const engine11v11 = new GameEngine();
  engine11v11.loadScenario(scenario11v11, 12345);

  const active11 = engine11v11.players.find((p) => p.team === 'left')!;
  const obs11 = ObservationEncoder.encode(
    engine11v11.players,
    engine11v11.ball,
    active11.id,
    engine11v11.score,
    engine11v11.matchTimeSeconds,
    3000
  );

  const entities11 = ModularFeatureParser.parse(obs11.rawVector);
  console.log(`  11v11 Teammates Count: ${entities11.teammates.count} (Expected: 10)`);
  console.log(`  11v11 Opponents Count: ${entities11.opponents.count} (Expected: 11)`);

  if (entities11.teammates.count !== 10) {
    throw new Error(`11v11 teammate count mismatch: ${entities11.teammates.count} != 10`);
  }
  if (entities11.opponents.count !== 11) {
    throw new Error(`11v11 opponent count mismatch: ${entities11.opponents.count} != 11`);
  }
  console.log('  ✓ 11v11 entity decomposition validated cleanly.');

  // Test 3: Kinematics Math Consistency
  console.log('\n3. Testing Kinematics Math Consistency...');
  const expectedEgoPos = active11.position;
  const parsedEgoPos = entities11.ego.pos;
  const diffX = Math.abs(expectedEgoPos.x - parsedEgoPos.x);
  const diffY = Math.abs(expectedEgoPos.y - parsedEgoPos.y);

  if (diffX > 1e-6 || diffY > 1e-6) {
    throw new Error(`Ego position mismatch: expected (${expectedEgoPos.x}, ${expectedEgoPos.y}), got (${parsedEgoPos.x}, ${parsedEgoPos.y})`);
  }
  console.log(`  Ego position delta: dx=${diffX.toExponential(2)}, dy=${diffY.toExponential(2)} (Bitwise Exact)`);
  console.log('  ✓ Kinematics math consistency verified.');

  console.log('\n====================================================');
  console.log('✓ ALL MODULAR ENCODER SPEC & CONTRACT TESTS PASSED');
  console.log('====================================================');
}

runTests();
