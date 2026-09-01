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

export interface EnvironmentContractSpec {
  environment: string;
  environment_version: string;
  observation_dim: number;
  observation_schema_version: string;
  action_space_size: number;
  action_schema_version: string;
  scenarios: string[];
}
