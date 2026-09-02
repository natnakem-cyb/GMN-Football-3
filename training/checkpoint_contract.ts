/**
 * GMN-Football-3 — Checkpoint Contract & Metadata Validation System
 * Ensures all evaluated and trained checkpoints satisfy the authoritative environment contracts,
 * valid schema dimensions, non-padded weight criteria, and full metadata provenance.
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import {
  GMN_ENV_VERSION,
  OBSERVATION_SCHEMA_VERSION,
  ACTION_SCHEMA_VERSION,
  OBSERVATION_DIM,
  BASE_OBSERVATION_DIM,
  ROLE_DIM,
  ACTION_SPACE_SIZE,
} from '../src/engine/Contract';

export interface CheckpointMetadata {
  algorithm: 'mappo' | 'ippo' | 'ppo' | 'rule_based' | 'scripted' | 'random' | 'noop';
  scenario: string;
  trainingSteps: number;
  seed: number;
  envVersion: string;
  observationSchema: string;
  observationDim: number;
  actionSchema: string;
  actionSpaceSize: number;
  gitCommit?: string;
  checkpointPath: string;
  checkpointHash: string;
  isRoleAware: boolean;
  roleWeightsNonZero: boolean;
  isValid: boolean;
  rejectionReason?: string;
  hyperparameters?: Record<string, any>;
  createdAt?: string;
}

export interface WeightTensors {
  w0: number[] | number[][];
  b0: number[];
  w1: number[] | number[][];
  b1: number[];
  w2: number[] | number[][];
  b2: number[];
}

export class CheckpointContractValidator {
  /**
   * Computes SHA256 hash of a file.
   */
  public static computeFileHash(filePath: string): string {
    if (!fs.existsSync(filePath)) {
      return 'FILE_NOT_FOUND';
    }
    const buffer = fs.readFileSync(filePath);
    return crypto.createHash('sha256').update(buffer).digest('hex');
  }

  /**
   * Validates raw actor weight tensors against the 127-dim role-aware schema contract.
   */
  public static validateWeightTensors(weights: WeightTensors): {
    isValid: boolean;
    isRoleAware: boolean;
    reason?: string;
  } {
    const { w0, b0, w1, b1, w2, b2 } = weights;

    if (!w0 || !Array.isArray(w0) || w0.length === 0) {
      return { isValid: false, isRoleAware: false, reason: 'Layer 0 weight matrix (w0) is missing or empty.' };
    }

    const is1D = typeof w0[0] === 'number';
    const hiddenDim = b0.length || 64;
    const inDim = is1D ? w0.length / hiddenDim : (w0 as number[][])[0].length;

    if (inDim !== OBSERVATION_DIM) {
      return {
        isValid: false,
        isRoleAware: false,
        reason: `Input dimension mismatch: expected ${OBSERVATION_DIM} (schema: ${OBSERVATION_SCHEMA_VERSION}), got ${inDim}.`,
      };
    }

    if (b0.length !== hiddenDim || b1.length !== hiddenDim) {
      return { isValid: false, isRoleAware: false, reason: `Hidden dimension mismatch: expected ${hiddenDim}.` };
    }

    if (b2.length !== ACTION_SPACE_SIZE) {
      return {
        isValid: false,
        isRoleAware: false,
        reason: `Output dimension mismatch: expected ${ACTION_SPACE_SIZE} (schema: ${ACTION_SCHEMA_VERSION}), got ${b2.length}.`,
      };
    }

    // Role-feature slice validation (indices BASE_OBSERVATION_DIM .. OBSERVATION_DIM - 1)
    let nonZeroRoleWeights = 0;
    let totalRoleWeights = 0;

    for (let neuron = 0; neuron < hiddenDim; neuron++) {
      for (let d = BASE_OBSERVATION_DIM; d < OBSERVATION_DIM; d++) {
        totalRoleWeights++;
        const weightVal = is1D
          ? (w0 as number[])[neuron * OBSERVATION_DIM + d]
          : (w0 as number[][])[neuron][d];
        if (Math.abs(weightVal) > 1e-7) {
          nonZeroRoleWeights++;
        }
      }
    }

    if (nonZeroRoleWeights === 0) {
      return {
        isValid: false,
        isRoleAware: false,
        reason: `CHECKPOINT REJECTED: All ${totalRoleWeights} role-feature weights (indices ${BASE_OBSERVATION_DIM}..${OBSERVATION_DIM - 1}) are exact zeros. This indicates a zero-padded placeholder/smoke-test model rather than a genuinely trained role-aware policy.`,
      };
    }

    return { isValid: true, isRoleAware: true };
  }

  /**
   * Validates metadata object.
   */
  public static validateMetadata(meta: CheckpointMetadata): { isValid: boolean; reason?: string } {
    if (meta.envVersion !== GMN_ENV_VERSION) {
      return {
        isValid: false,
        reason: `Environment version mismatch: expected ${GMN_ENV_VERSION}, got ${meta.envVersion}.`,
      };
    }

    if (meta.observationSchema !== OBSERVATION_SCHEMA_VERSION || meta.observationDim !== OBSERVATION_DIM) {
      return {
        isValid: false,
        reason: `Observation schema mismatch: expected ${OBSERVATION_SCHEMA_VERSION} (${OBSERVATION_DIM}-dim), got ${meta.observationSchema} (${meta.observationDim}-dim).`,
      };
    }

    if (meta.actionSchema !== ACTION_SCHEMA_VERSION || meta.actionSpaceSize !== ACTION_SPACE_SIZE) {
      return {
        isValid: false,
        reason: `Action schema mismatch: expected ${ACTION_SCHEMA_VERSION} (${ACTION_SPACE_SIZE} actions), got ${meta.actionSchema} (${meta.actionSpaceSize} actions).`,
      };
    }

    if (!meta.roleWeightsNonZero && meta.algorithm === 'mappo') {
      return {
        isValid: false,
        reason: `MAPPO checkpoint must have non-zero role-feature weights under schema ${OBSERVATION_SCHEMA_VERSION}.`,
      };
    }

    return { isValid: true };
  }

  /**
   * Creates a standardized experiment manifest.
   */
  public static createExperimentManifest(config: {
    algorithm: 'mappo' | 'ippo' | 'ppo' | 'rule_based' | 'scripted' | 'random' | 'noop';
    scenario: string;
    trainingSteps: number;
    seed: number;
    checkpointPath: string;
    weights?: WeightTensors;
    hyperparameters?: Record<string, any>;
  }): CheckpointMetadata {
    const hash = CheckpointContractValidator.computeFileHash(config.checkpointPath);
    let roleWeightsNonZero = false;
    let isValid = true;
    let rejectionReason: string | undefined;

    if (config.weights) {
      const weightCheck = CheckpointContractValidator.validateWeightTensors(config.weights);
      roleWeightsNonZero = weightCheck.isRoleAware;
      isValid = weightCheck.isValid;
      rejectionReason = weightCheck.reason;
    } else if (config.algorithm === 'rule_based' || config.algorithm === 'scripted' || config.algorithm === 'random' || config.algorithm === 'noop') {
      roleWeightsNonZero = true;
      isValid = true;
    }

    const metadata: CheckpointMetadata = {
      algorithm: config.algorithm,
      scenario: config.scenario,
      trainingSteps: config.trainingSteps,
      seed: config.seed,
      envVersion: GMN_ENV_VERSION,
      observationSchema: OBSERVATION_SCHEMA_VERSION,
      observationDim: OBSERVATION_DIM,
      actionSchema: ACTION_SCHEMA_VERSION,
      actionSpaceSize: ACTION_SPACE_SIZE,
      checkpointPath: config.checkpointPath,
      checkpointHash: hash,
      isRoleAware: roleWeightsNonZero,
      roleWeightsNonZero,
      isValid,
      rejectionReason,
      hyperparameters: config.hyperparameters,
      createdAt: new Date().toISOString(),
    };

    return metadata;
  }
}
