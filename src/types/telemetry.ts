import { Vector2D } from './football';

export interface TrainingMetricsSnapshot {
  step: number;
  update: number;
  policyLoss: number;
  valueLoss: number;
  entropy: number;
  approxKl: number;
  clipFraction: number;
  learningRate: number;
  gradNorm: number;
  rollingReward: number;
  goalRate: number;
  timestamp: number;
}

export interface HardwareMetrics {
  sps: number; // steps per second
  fps: number; // visual frame rate
  gpuVramUsedMb: number;
  gpuVramTotalMb: number;
  gpuUtilizationPct: number;
  cpuUtilizationPct: number;
  workerCount: number;
  bufferSize: number;
  bufferCapacity: number;
  ipcLatencyMs: number;
  activeDevice: 'CUDA (RTX 4090 / Cloud T4)' | 'Apple MPS' | 'WebGPU / WASM Vectorized';
}

export interface AgentCreditMetrics {
  playerId: string;
  playerName: string;
  role: string;
  counterfactualAdvantage: number;
  rewardContribution: number;
  passCompletionRate: number;
  keyPasses: number;
  distanceCovered: number;
  spaceCreationScore: number;
  defensiveInterceptions: number;
  positionalDiscipline: number;
}

export interface ActionProbabilityEntry {
  index: number;
  name: string;
  shortLabel: string;
  probability: number;
  logit: number;
  category: 'move' | 'pass' | 'shot' | 'defense' | 'sticky';
}

export interface PolicyActionDistribution {
  playerId: string;
  role: string;
  valueEstimate: number; // V(s) baseline in [-1, +1]
  actions: ActionProbabilityEntry[];
  bestActionIndex: number;
  bestActionName: string;
  confidence: number;
  attention?: {
    targetPlayerId?: string;
    targetPos?: Vector2D;
    passClearanceProb?: number;
    shotAngleClearance?: number;
  };
}

export interface TrainingHyperparameters {
  learningRate: number;
  clipRange: number;
  entropyCoef: number;
  valueCoef: number;
  miniBatchSize: number;
  nEpochs: number;
  gamma: number;
  gaeLambda: number;
  targetTimesteps: number;
  maxGradNorm: number;
}
