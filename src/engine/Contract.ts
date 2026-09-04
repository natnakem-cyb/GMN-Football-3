/**
 * Versioned Environment & Schema Contracts for GMN-Football-3.
 * Authoritative single source of truth for RL environments, bridges, and neural networks.
 */

export const GMN_ENV_VERSION = '3.1.0';
export const OBSERVATION_SCHEMA_VERSION = 'simple115_v3_role';
export const ACTION_SCHEMA_VERSION = 'discrete19_v1';

export const ROLE_VOCABULARY = [
  'GK',
  'CB',
  'LB',
  'RB',
  'CDM',
  'CM',
  'LM',
  'RM',
  'LW',
  'RW',
  'CAM',
  'ST',
] as const;

export const ROLE_DIM = 12;
export const BASE_OBSERVATION_DIM = 115;
export const OBSERVATION_DIM = BASE_OBSERVATION_DIM + ROLE_DIM; // 127
export const ACTION_SPACE_SIZE = 19;

export const EVENT_CODE_MAP = [
  undefined,
  'goal',
  'shot',
  'shot_saved',
  'shot_missed',
  'pass',
  'interception',
  'tackle',
  'foul',
  'kickoff',
  'out_of_bounds',
  'scenario_complete',
  'scenario_failed',
] as const;

export function getEventCode(eventType?: string): number {
  if (!eventType) return 0;
  const idx = (EVENT_CODE_MAP as readonly (string | undefined)[]).indexOf(eventType);
  return idx >= 0 ? idx : 0;
}

/**
 * Dynamically resolves or infers a player's tactical role from match state
 * to ensure role features (indices 115-126) are never left zero-padded.
 */
export function inferPlayerRole(
  player?: { role?: string; isGoalkeeper?: boolean; position?: { x: number; y: number } } | null
): typeof ROLE_VOCABULARY[number] {
  if (!player) return 'ST';
  if (player.role && (ROLE_VOCABULARY as readonly string[]).includes(player.role)) {
    return player.role as typeof ROLE_VOCABULARY[number];
  }
  if (player.isGoalkeeper) return 'GK';

  // Dynamic tactical inference based on pitch positioning
  const x = player.position?.x ?? 0;
  const y = player.position?.y ?? 0;

  if (x < -0.35) {
    if (Math.abs(y) > 0.22) return y < 0 ? 'LB' : 'RB';
    return 'CB';
  } else if (x < 0.1) {
    if (Math.abs(y) > 0.22) return y < 0 ? 'LM' : 'RM';
    return x < -0.15 ? 'CDM' : 'CM';
  } else {
    if (Math.abs(y) > 0.22) return y < 0 ? 'LW' : 'RW';
    return x > 0.35 ? 'ST' : 'CAM';
  }
}

/**
 * Validates that an observation vector matches the strict 127-dimensional schema
 * and that the role features (indices 115-126) are populated with a valid one-hot.
 */
export function validateObservationVector(
  rawVector: number[] | Float32Array
): { valid: boolean; reason?: string } {
  if (rawVector.length !== OBSERVATION_DIM) {
    return {
      valid: false,
      reason: `Expected observation vector of length ${OBSERVATION_DIM}, got ${rawVector.length}`,
    };
  }

  // Check role feature slice: indices 115 to 126
  let roleSum = 0;
  for (let i = BASE_OBSERVATION_DIM; i < OBSERVATION_DIM; i++) {
    const val = rawVector[i];
    if (isNaN(val)) {
      return { valid: false, reason: `NaN detected at role index ${i}` };
    }
    roleSum += val;
  }

  if (Math.abs(roleSum - 1.0) > 1e-4) {
    return {
      valid: false,
      reason: `Observation vector has unpopulated or zero-padded role features (indices 115-126 sum to ${roleSum}, expected 1.0)`,
    };
  }

  return { valid: true };
}

export interface EnvironmentContractSpec {
  environment: string;
  environment_version: string;
  observation_dim: number;
  observation_schema_version: string;
  action_space_size: number;
  action_schema_version: string;
  scenarios: string[];
}
