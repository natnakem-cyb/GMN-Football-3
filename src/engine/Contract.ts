/**
 * Versioned Environment & Schema Contracts for GMN-Football-3.
 * Authoritative single source of truth for RL environments, bridges, and neural networks.
 */

export const GMN_ENV_VERSION = '3.0.0';
export const OBSERVATION_SCHEMA_VERSION = 'simple115_v2';
export const ACTION_SCHEMA_VERSION = 'discrete19_v1';

export const OBSERVATION_DIM = 115;
export const ACTION_SPACE_SIZE = 19;

export interface EnvironmentContractSpec {
  environment: string;
  environment_version: string;
  observation_dim: number;
  observation_schema_version: string;
  action_space_size: number;
  action_schema_version: string;
  scenarios: string[];
}
