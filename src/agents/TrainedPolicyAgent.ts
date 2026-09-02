import { ActionType, AgentAction, GameMode } from '../types/football';
import { AgentDecisionContext, IAgent } from './BaseAgent';
import { ObservationEncoder } from '../engine/ObservationEncoder';
import { OBSERVATION_DIM, ACTION_SPACE_SIZE } from '../engine/Contract';
import { mapDiscreteAction } from '../engine/ActionMapping';
import { MAPPO_WEIGHTS } from './mappo_weights';

/**
 * Assert that MAPPO_WEIGHTS matches the current OBSERVATION_DIM (127 floats) and network architecture.
 * Throws a clear runtime error if the embedded weights were exported from a legacy checkpoint or corrupted.
 */
export function assertMappoWeightsValid(): void {
  if (!MAPPO_WEIGHTS) {
    throw new Error('[TrainedPolicyAgent] MAPPO_WEIGHTS is missing or undefined.');
  }

  const expectedW0Len = 64 * OBSERVATION_DIM;
  if (MAPPO_WEIGHTS.w0.length !== expectedW0Len) {
    throw new Error(
      `[TrainedPolicyAgent] Weight/OBSERVATION_DIM mismatch: w0.length=${MAPPO_WEIGHTS.w0.length}, expected ${expectedW0Len}. Re-export weights from a 127-dim checkpoint (see training/export_onnx.py).`
    );
  }

  if (MAPPO_WEIGHTS.b0.length !== 64) {
    throw new Error(`[TrainedPolicyAgent] Bias dimension mismatch: b0.length=${MAPPO_WEIGHTS.b0.length}, expected 64.`);
  }

  if (MAPPO_WEIGHTS.w1.length !== 64 * 64) {
    throw new Error(`[TrainedPolicyAgent] Hidden weight mismatch: w1.length=${MAPPO_WEIGHTS.w1.length}, expected 4096.`);
  }

  if (MAPPO_WEIGHTS.b1.length !== 64) {
    throw new Error(`[TrainedPolicyAgent] Bias dimension mismatch: b1.length=${MAPPO_WEIGHTS.b1.length}, expected 64.`);
  }

  if (MAPPO_WEIGHTS.w2.length !== ACTION_SPACE_SIZE * 64) {
    throw new Error(`[TrainedPolicyAgent] Output weight mismatch: w2.length=${MAPPO_WEIGHTS.w2.length}, expected ${ACTION_SPACE_SIZE * 64}.`);
  }

  if (MAPPO_WEIGHTS.b2.length !== ACTION_SPACE_SIZE) {
    throw new Error(`[TrainedPolicyAgent] Bias dimension mismatch: b2.length=${MAPPO_WEIGHTS.b2.length}, expected ${ACTION_SPACE_SIZE}.`);
  }
}

export class TrainedPolicyAgent implements IAgent {
  id: string;
  name = 'PPO Neural Policy Agent (Trained, MAPPO)';
  type: 'neural' = 'neural';

  private lastAction: AgentAction = { type: ActionType.IDLE };

  private constructor(id: string) {
    assertMappoWeightsValid();
    this.id = id;
  }

  /**
   * Async factory: loads and initializes the verified MAPPO trained policy.
   */
  static async create(
    _modelSource: string | ArrayBuffer | Uint8Array = '/models/mappo_policy.onnx',
    id = 'trained_ppo'
  ): Promise<TrainedPolicyAgent> {
    assertMappoWeightsValid();
    const agent = new TrainedPolicyAgent(id);
    return agent;
  }

  /**
   * Synchronous decide function per IAgent contract.
   * Evaluates the observation vector through the trained multi-layer perceptron (MLP).
   */
  decide(context: AgentDecisionContext): AgentAction {
    // Encode standard OBSERVATION_DIM-float (127) GRF observation vector using the shared ObservationEncoder
    const obs = ObservationEncoder.encode(
      context.allPlayers,
      context.ball,
      context.player.id,
      { left: 0, right: 0 },
      0,
      3600,
      context.gameMode ?? GameMode.Normal
    );

    const logits = this.computeForwardMath(obs.rawVector);

    let bestIdx = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > bestVal) {
        bestVal = logits[i];
        bestIdx = i;
      }
    }

    this.lastAction = mapDiscreteAction(bestIdx);
    return this.lastAction;
  }

  reset(): void {
    this.lastAction = { type: ActionType.IDLE };
  }

  /**
   * Direct forward-pass MLP evaluation of the trained actor network:
   * Layer 0: Linear(OBSERVATION_DIM, 64) -> Tanh (where OBSERVATION_DIM = 127)
   * Layer 1: Linear(64, 64) -> Tanh
   * Layer 2: Linear(64, 19) -> Logits
   */
  private computeForwardMath(obs: number[]): number[] {
    assertMappoWeightsValid();
    const { w0, b0, w1, b1, w2, b2 } = MAPPO_WEIGHTS;

    // Layer 0: Linear(OBSERVATION_DIM, 64) -> Tanh
    const h0 = new Float32Array(64);
    for (let i = 0; i < 64; i++) {
      let sum = b0[i];
      const offset = i * OBSERVATION_DIM;
      for (let j = 0; j < OBSERVATION_DIM; j++) {
        sum += w0[offset + j] * obs[j];
      }
      h0[i] = Math.tanh(sum);
    }

    // Layer 1: Linear(64, 64) -> Tanh
    const h1 = new Float32Array(64);
    for (let i = 0; i < 64; i++) {
      let sum = b1[i];
      const offset = i * 64;
      for (let j = 0; j < 64; j++) {
        sum += w1[offset + j] * h0[j];
      }
      h1[i] = Math.tanh(sum);
    }

    // Layer 2: Linear(64, 19) -> Logits
    const logits = new Float32Array(ACTION_SPACE_SIZE);
    for (let i = 0; i < ACTION_SPACE_SIZE; i++) {
      let sum = b2[i];
      const offset = i * 64;
      for (let j = 0; j < 64; j++) {
        sum += w2[offset + j] * h1[j];
      }
      logits[i] = sum;
    }

    return Array.from(logits);
  }
}
