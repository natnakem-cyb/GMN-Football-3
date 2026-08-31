import { ActionType, AgentAction, GameMode } from '../types/football';
import { AgentDecisionContext, IAgent } from './BaseAgent';
import { ObservationEncoder } from '../engine/ObservationEncoder';
import { MAPPO_WEIGHTS } from './mappo_weights';

const SQRT_HALF = 0.7071067811865476;

export class TrainedPolicyAgent implements IAgent {
  id: string;
  name = 'PPO Neural Policy Agent (Trained, MAPPO)';
  type: 'neural' = 'neural';

  private lastAction: AgentAction = { type: ActionType.IDLE };

  private constructor(id: string) {
    this.id = id;
  }

  /**
   * Async factory: loads and initializes the verified MAPPO trained policy.
   */
  static async create(
    _modelSource: string | ArrayBuffer | Uint8Array = '/models/mappo_policy.onnx',
    id = 'trained_ppo'
  ): Promise<TrainedPolicyAgent> {
    const agent = new TrainedPolicyAgent(id);
    return agent;
  }

  /**
   * Synchronous decide function per IAgent contract.
   * Evaluates the observation vector through the trained multi-layer perceptron (MLP).
   */
  decide(context: AgentDecisionContext): AgentAction {
    // Encode standard 115-float GRF observation vector using the shared ObservationEncoder
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

    this.lastAction = this.mapActionIndex(bestIdx, context.player.heading);
    return this.lastAction;
  }

  reset(): void {
    this.lastAction = { type: ActionType.IDLE };
  }

  /**
   * Direct forward-pass MLP evaluation of the trained actor network:
   * Layer 0: Linear(115, 64) -> Tanh
   * Layer 1: Linear(64, 64) -> Tanh
   * Layer 2: Linear(64, 19) -> Logits
   */
  private computeForwardMath(obs: number[]): number[] {
    const { w0, b0, w1, b1, w2, b2 } = MAPPO_WEIGHTS;

    // Layer 0: Linear(115, 64) -> Tanh
    const h0 = new Float32Array(64);
    for (let i = 0; i < 64; i++) {
      let sum = b0[i];
      const offset = i * 115;
      for (let j = 0; j < 115; j++) {
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
    const logits = new Float32Array(19);
    for (let i = 0; i < 19; i++) {
      let sum = b2[i];
      const offset = i * 64;
      for (let j = 0; j < 64; j++) {
        sum += w2[offset + j] * h1[j];
      }
      logits[i] = sum;
    }

    return Array.from(logits);
  }

  /**
   * Authoritative 19-action discrete mapping matching GMN-Football-3 specifications.
   */
  private mapActionIndex(idx: number, _currentHeading = 0): AgentAction {
    switch (idx) {
      case 0:
        return { type: ActionType.IDLE };
      case 1: // LEFT
        return { type: ActionType.MOVE, direction: { x: -1.0, y: 0.0 } };
      case 2: // TOP_LEFT
        return { type: ActionType.MOVE, direction: { x: -SQRT_HALF, y: -SQRT_HALF } };
      case 3: // TOP
        return { type: ActionType.MOVE, direction: { x: 0.0, y: -1.0 } };
      case 4: // TOP_RIGHT
        return { type: ActionType.MOVE, direction: { x: SQRT_HALF, y: -SQRT_HALF } };
      case 5: // RIGHT
        return { type: ActionType.MOVE, direction: { x: 1.0, y: 0.0 } };
      case 6: // BOTTOM_RIGHT
        return { type: ActionType.MOVE, direction: { x: SQRT_HALF, y: SQRT_HALF } };
      case 7: // BOTTOM
        return { type: ActionType.MOVE, direction: { x: 0.0, y: 1.0 } };
      case 8: // BOTTOM_LEFT
        return { type: ActionType.MOVE, direction: { x: -SQRT_HALF, y: SQRT_HALF } };
      case 9: // LONG_PASS
        return { type: ActionType.LONG_PASS, power: 1.0 };
      case 10: // HIGH_PASS
        return { type: ActionType.HIGH_PASS, power: 0.85 };
      case 11: // SHORT_PASS
        return { type: ActionType.SHORT_PASS, power: 0.75 };
      case 12: // SHOT
        return { type: ActionType.SHOT, power: 0.95 };
      case 13: // SPRINT
        return { type: ActionType.SPRINT };
      case 14: // RELEASE_DIRECTION
        return { type: ActionType.RELEASE_DIRECTION };
      case 15: // RELEASE_SPRINT
        return { type: ActionType.RELEASE_SPRINT };
      case 16: // SLIDING / TACKLE
        return { type: ActionType.TACKLE };
      case 17: // DRIBBLE
        return { type: ActionType.DRIBBLE };
      case 18: // RELEASE_DRIBBLE
        return { type: ActionType.RELEASE_DRIBBLE };
      default:
        return { type: ActionType.IDLE };
    }
  }
}
