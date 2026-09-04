import { ActionType, AgentAction, Vector2D } from '../types/football';
import { AgentDecisionContext, IAgent } from './BaseAgent';
import { Vec2 } from '../engine/Vector';
import { SeededRNG } from '../engine/SeededRNG';

export class NeuralHeuristicAgent implements IAgent {
  id: string;
  name: string;
  type: 'neural' = 'neural';
  private rng: SeededRNG;

  // Observable neural layer activations for UI inspection
  public lastActivations: {
    inputs: number[];
    hidden1: number[];
    hidden2: number[];
    actionProbabilities: { action: ActionType; prob: number }[];
  } = {
    inputs: [],
    hidden1: [],
    hidden2: [],
    actionProbabilities: [],
  };

  private weights: {
    w1: number[][];
    w2: number[][];
    w3: number[][];
  };

  constructor(id = 'neural_heuristic_v1', name = 'Heuristic Bot (Untrained Baseline)', seed = 1337) {
    this.id = id;
    this.name = name;
    this.rng = new SeededRNG(seed);

    // Deterministic weights generated via SeededRNG
    const initRng = new SeededRNG(seed);
    this.weights = {
      // 8 inputs -> 12 hidden1 -> 6 hidden2 -> 5 action outputs
      w1: Array.from({ length: 8 }, () => Array.from({ length: 12 }, () => (initRng.next() - 0.5) * 0.8)),
      w2: Array.from({ length: 12 }, () => Array.from({ length: 6 }, () => (initRng.next() - 0.5) * 0.8)),
      w3: Array.from({ length: 6 }, () => Array.from({ length: 5 }, () => (initRng.next() - 0.5) * 0.8)),
    };
  }

  public setSeed(seed: number): void {
    this.rng.setSeed(seed);
  }

  decide(context: AgentDecisionContext): AgentAction {
    const { player, ball, opponents, teammates, teamSide } = context;
    const opponentGoalX = teamSide === 'left' ? 1.0 : -1.0;

    // Feature extraction (8 normalized inputs)
    const relBallX = ball.position.x - player.position.x;
    const relBallY = ball.position.y - player.position.y;
    const distToBall = Vec2.distance(player.position, { x: ball.position.x, y: ball.position.y });
    const distToGoal = Math.abs(opponentGoalX - player.position.x);
    const hasBall = player.hasBall || ball.ownerId === player.id ? 1 : 0;
    const playerStamina = player.stamina / 100;
    const ballVelX = ball.velocity.x * 50;
    const ballVelY = ball.velocity.y * 50;

    const input = [relBallX, relBallY, distToBall, distToGoal, hasBall, playerStamina, ballVelX, ballVelY];

    // Forward pass: Hidden Layer 1 (ReLU)
    const hidden1 = this.weights.w1[0].map((_, col) => {
      let sum = 0;
      for (let i = 0; i < input.length; i++) {
        sum += input[i] * this.weights.w1[i][col];
      }
      return Math.max(0, sum); // ReLU
    });

    // Forward pass: Hidden Layer 2 (Tanh)
    const hidden2 = this.weights.w2[0].map((_, col) => {
      let sum = 0;
      for (let i = 0; i < hidden1.length; i++) {
        sum += hidden1[i] * this.weights.w2[i][col];
      }
      return Math.tanh(sum);
    });

    // Output Layer (Softmax logits over 5 core actions)
    const rawOutputs = this.weights.w3[0].map((_, col) => {
      let sum = 0;
      for (let i = 0; i < hidden2.length; i++) {
        sum += hidden2[i] * this.weights.w3[i][col];
      }
      return sum;
    });

    // Bias towards goal & ball control based on trained heuristics
    if (hasBall) {
      if (distToGoal < 0.6) {
        rawOutputs[3] += 1.8; // Boost Shot
      } else {
        rawOutputs[1] += 1.2; // Boost Sprint towards goal
        rawOutputs[2] += 0.8; // Boost Pass
      }
    } else {
      if (distToBall > 0.1) {
        rawOutputs[0] += 1.5; // Boost Move to ball
        rawOutputs[1] += 1.0; // Boost Sprint to ball
      } else {
        rawOutputs[4] += 1.6; // Boost Tackle
      }
    }

    const expScores = rawOutputs.map((v) => Math.exp(Math.min(5, Math.max(-5, v))));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    const probs = expScores.map((v) => v / sumExp);

    const actionCategories: ActionType[] = [
      ActionType.MOVE,
      ActionType.SPRINT,
      ActionType.SHORT_PASS,
      ActionType.SHOT,
      ActionType.TACKLE,
    ];

    this.lastActivations = {
      inputs: input,
      hidden1,
      hidden2,
      actionProbabilities: actionCategories.map((act, idx) => ({ action: act, prob: probs[idx] })),
    };

    // Pick action with highest probability or sampled
    let maxProbIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxProbIdx]) {
        maxProbIdx = i;
      }
    }

    const chosenAction = actionCategories[maxProbIdx];

    if (chosenAction === ActionType.SHOT) {
      const rngVal = context.rng ? context.rng.next() : this.rng.next();
      const shootDir = Vec2.sub({ x: opponentGoalX, y: (rngVal - 0.5) * 0.08 }, player.position);
      return {
        type: ActionType.SHOT,
        direction: Vec2.normalize(shootDir),
        power: 0.9,
      };
    }

    if (chosenAction === ActionType.SHORT_PASS) {
      const openTeammate = teammates.find((t) => t.id !== player.id);
      if (openTeammate) {
        return {
          type: ActionType.SHORT_PASS,
          direction: Vec2.normalize(Vec2.sub(openTeammate.position, player.position)),
          power: 0.75,
          targetPlayerId: openTeammate.id,
        };
      }
    }

    if (chosenAction === ActionType.TACKLE) {
      return { type: ActionType.TACKLE };
    }

    if (chosenAction === ActionType.SPRINT || chosenAction === ActionType.MOVE) {
      const target = hasBall
        ? { x: opponentGoalX, y: 0 }
        : { x: ball.position.x, y: ball.position.y };

      return {
        type: chosenAction,
        direction: Vec2.normalize(Vec2.sub(target, player.position)),
      };
    }

    return { type: ActionType.IDLE };
  }
}
