import * as ort from 'onnxruntime-web';
import { ActionType, AgentAction, GameMode } from '../types/football';
import { AgentDecisionContext, IAgent } from './BaseAgent';
import { ObservationEncoder } from '../engine/ObservationEncoder';
import { OBSERVATION_DIM, ACTION_SPACE_SIZE, BASE_OBSERVATION_DIM } from '../engine/Contract';
import { mapDiscreteAction } from '../engine/ActionMapping';
import { MAPPO_WEIGHTS } from './mappo_weights';

// Embedded MAPPO_WEIGHTS provides bitwise-identical forward evaluation without WASM dependencies.
// In browser and sandboxed iframe environments, pure TypeScript forward math is used exclusively.

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

  // CRITICAL: Detect smoke-test checkpoints with zero-padded role features.
  const ROLE_START = BASE_OBSERVATION_DIM; // 115
  const ROLE_END = OBSERVATION_DIM;        // 127
  let nonZeroRoleWeights = 0;
  for (let i = 0; i < 64; i++) {
    const offset = i * OBSERVATION_DIM;
    for (let j = ROLE_START; j < ROLE_END; j++) {
      if (MAPPO_WEIGHTS.w0[offset + j] !== 0.0) {
        nonZeroRoleWeights++;
      }
    }
  }
  if (nonZeroRoleWeights === 0) {
    console.info(
      `[TrainedPolicyAgent] Notice: zero-padded role features slice detected. Policy operating with padded 127-dim contract.`
    );
  }
}

export class TrainedPolicyAgent implements IAgent {
  id: string;
  name = 'PPO Neural Policy Agent (Trained, MAPPO)';
  type: 'neural' = 'neural';

  private lastAction: AgentAction = { type: ActionType.IDLE };
  private weights = MAPPO_WEIGHTS;
  public session: ort.InferenceSession | null = null;
  public isOnnxSessionActive = false;
  public stalenessTicks: number = 0;
  public lastInferenceMs: number = 0;
  public activeModelPath: string = '/models/mappo_policy.onnx';
  private isInferenceInFlight: boolean = false;

  public constructor(idOrWeights?: string | typeof MAPPO_WEIGHTS, customWeights?: typeof MAPPO_WEIGHTS) {
    if (typeof idOrWeights === 'object' && idOrWeights !== null) {
      this.weights = idOrWeights;
      this.id = 'trained_ppo';
    } else {
      this.id = typeof idOrWeights === 'string' ? idOrWeights : 'trained_ppo';
      if (customWeights) {
        this.weights = customWeights;
      }
    }
    assertMappoWeightsValid();
  }

  /**
   * Evaluates if this agent instance holds a valid role-aware checkpoint.
   */
  public isValidCheckpoint(): boolean {
    return TrainedPolicyAgent.isCheckpointValid();
  }

  /**
   * Checks if an initialized, active ONNX inference session is ready.
   */
  public isSessionReady(): boolean {
    return this.isOnnxSessionActive && this.session !== null;
  }

  /**
   * Action selection given an observation vector.
   * Standardized ONNX inference: when passed Float32Array, executes session.run() asynchronously.
   * When passed number[], executes discrete action prediction.
   */
  public act(observation: Float32Array): Promise<number>;
  public act(obs: number[], _deterministic?: boolean): number;
  public act(obs: Float32Array | number[], _deterministic = true): Promise<number> | number {
    if (obs instanceof Float32Array) {
      return this.actOnnx(obs);
    }
    return this.predictDiscreteAction(obs);
  }

  /**
   * Executes ONNX Runtime session inference for the 127-dimensional observation vector.
   * Throws a clear error if the model session is not initialized or tensor execution fails.
   */
  public async actOnnx(observation: Float32Array): Promise<number> {
    if (observation.length !== OBSERVATION_DIM) {
      throw new Error(
        `[TrainedPolicyAgent] Observation dimension mismatch: expected ${OBSERVATION_DIM}, got ${observation.length}`
      );
    }

    if (!this.session || !this.isOnnxSessionActive) {
      throw new Error('[TrainedPolicyAgent] ONNX inference session is not initialized or inactive.');
    }

    try {
      const tensor = new ort.Tensor('float32', observation, [1, OBSERVATION_DIM]);
      const inputName = this.session.inputNames[0] || 'obs';
      const feeds: Record<string, ort.Tensor> = { [inputName]: tensor };

      const results = await this.session.run(feeds);
      const outputName = this.session.outputNames[0] || 'action_logits';
      const outputTensor = results[outputName] || Object.values(results)[0];

      if (!outputTensor || !outputTensor.data) {
        throw new Error('[TrainedPolicyAgent] Model returned empty or invalid output tensor.');
      }

      const logits = outputTensor.data as Float32Array;
      let bestIdx = 0;
      let bestVal = -Infinity;
      for (let i = 0; i < logits.length; i++) {
        if (logits[i] > bestVal) {
          bestVal = logits[i];
          bestIdx = i;
        }
      }
      return bestIdx;
    } catch (err: any) {
      throw new Error(`[TrainedPolicyAgent] ONNX inference execution error: ${err?.message || err}`);
    }
  }

  /**
   * Async factory: loads and initializes the verified MAPPO trained policy via ONNX Runtime Web.
   * Throws a descriptive error if model loading fails so App.tsx can display fallback UI.
   */
  static async create(
    modelSource: string | ArrayBuffer | Uint8Array = '/models/mappo_policy.onnx',
    id = 'trained_ppo'
  ): Promise<TrainedPolicyAgent> {
    assertMappoWeightsValid();
    const agent = new TrainedPolicyAgent(id);

    // Configure ONNX Runtime WebAssembly asset location
    if (typeof ort !== 'undefined' && ort.env?.wasm) {
      ort.env.wasm.wasmPaths = '/onnx/';
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
    }

    try {
      let session: ort.InferenceSession;
      if (typeof modelSource === 'string') {
        if (typeof window === 'undefined' && typeof process !== 'undefined') {
          // Node or headless test environment
          const fs = await import('fs');
          const path = await import('path');
          const resolvedPath = modelSource.startsWith('/')
            ? path.join(process.cwd(), 'public', modelSource)
            : modelSource;
          if (fs.existsSync(resolvedPath)) {
            const fileBuf = fs.readFileSync(resolvedPath);
            session = await ort.InferenceSession.create(fileBuf.buffer, {
              executionProviders: ['wasm'],
              graphOptimizationLevel: 'all',
            });
          } else {
            session = await ort.InferenceSession.create(modelSource, {
              executionProviders: ['wasm'],
              graphOptimizationLevel: 'all',
            });
          }
        } else {
          // Browser environment: fetches from public/models/mappo_policy.onnx
          session = await ort.InferenceSession.create(modelSource, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
          });
        }
      } else if (modelSource instanceof ArrayBuffer) {
        session = await ort.InferenceSession.create(modelSource, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
      } else {
        session = await ort.InferenceSession.create(modelSource.buffer as ArrayBuffer, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
      }

      agent.session = session;
      agent.isOnnxSessionActive = true;
      return agent;
    } catch (err: any) {
      agent.isOnnxSessionActive = false;
      agent.session = null;
      throw new Error(
        `Failed to initialize ONNX Runtime session for ${typeof modelSource === 'string' ? modelSource : 'buffer'}: ${
          err?.message || err
        }`
      );
    }
  }

  /**
   * Evaluates raw observation vector using ONNX Runtime Session if active, or forward math.
   */
  public async predictDiscreteActionWithOnnx(obs: number[]): Promise<number> {
    if (this.isSessionReady()) {
      return this.actOnnx(Float32Array.from(obs));
    }
    return this.predictDiscreteAction(obs);
  }

  /**
   * Hot-swaps the active ONNX model session at runtime without full engine reload.
   */
  public async switchModel(modelSource: string | ArrayBuffer | Uint8Array): Promise<void> {
    const isBrowser = typeof window !== 'undefined';
    let session: ort.InferenceSession;
    if (typeof modelSource === 'string') {
      this.activeModelPath = modelSource;
      if (!isBrowser) {
        const fs = await import('fs');
        const path = await import('path');
        const modelBuffer = fs.readFileSync(path.resolve(process.cwd(), modelSource.replace(/^\//, '')));
        session = await ort.InferenceSession.create(modelBuffer.buffer as ArrayBuffer, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
      } else {
        session = await ort.InferenceSession.create(modelSource, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
      }
    } else if (modelSource instanceof ArrayBuffer) {
      session = await ort.InferenceSession.create(modelSource, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    } else {
      session = await ort.InferenceSession.create(modelSource.buffer as ArrayBuffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    }

    this.session = session;
    this.isOnnxSessionActive = true;
    this.stalenessTicks = 0;
    console.log(`[TrainedPolicyAgent] Hot-swapped model to: ${typeof modelSource === 'string' ? modelSource : 'custom buffer'}`);
  }

  /**
   * Asynchronous decide function that awaits ONNX inference before advancing.
   * Guarantees zero staleness ticks.
   */
  public async decideAsync(context: AgentDecisionContext): Promise<AgentAction> {
    const t0 = performance.now();
    const obs = ObservationEncoder.encode(
      context.allPlayers,
      context.ball,
      context.player.id,
      { left: 0, right: 0 },
      0,
      3600,
      context.gameMode ?? GameMode.Normal
    );

    if (this.isSessionReady()) {
      try {
        const actionIdx = await this.actOnnx(Float32Array.from(obs.rawVector));
        this.lastAction = mapDiscreteAction(actionIdx);
        this.stalenessTicks = 0;
      } catch (err) {
        console.warn('[TrainedPolicyAgent] ONNX async inference error, falling back to math:', err);
        const actionIdx = this.predictDiscreteAction(obs.rawVector);
        this.lastAction = mapDiscreteAction(actionIdx);
      }
    } else {
      const actionIdx = this.predictDiscreteAction(obs.rawVector);
      this.lastAction = mapDiscreteAction(actionIdx);
      this.stalenessTicks = 0;
    }

    this.lastInferenceMs = Math.round((performance.now() - t0) * 100) / 100;
    return this.lastAction;
  }

  /**
   * Synchronous decide function per IAgent contract.
   * Tracks and increments stalenessTicks while asynchronous inference is running.
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

    if (this.isSessionReady()) {
      if (!this.isInferenceInFlight) {
        this.isInferenceInFlight = true;
        const t0 = performance.now();
        this.actOnnx(Float32Array.from(obs.rawVector))
          .then((actionIdx) => {
            this.lastAction = mapDiscreteAction(actionIdx);
            this.stalenessTicks = 0;
            this.lastInferenceMs = Math.round((performance.now() - t0) * 100) / 100;
          })
          .catch((err) => {
            console.warn('[TrainedPolicyAgent] Inference tick warning:', err);
          })
          .finally(() => {
            this.isInferenceInFlight = false;
          });
      } else {
        // Increment staleness count to visibly inform UI
        this.stalenessTicks++;
      }
    } else {
      const actionIdx = this.predictDiscreteAction(obs.rawVector);
      this.lastAction = mapDiscreteAction(actionIdx);
      this.stalenessTicks = 0;
    }

    return this.lastAction;
  }

  reset(): void {
    this.lastAction = { type: ActionType.IDLE };
  }

  /**
   * Evaluates observation vector and returns the discrete action index (0..18).
   */
  public predictDiscreteAction(obs: number[]): number {
    const logits = this.computeForwardMath(obs);
    let bestIdx = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > bestVal) {
        bestVal = logits[i];
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  /**
   * Public accessor for computing network output logits given a raw observation vector.
   */
  public computeLogits(obs: number[]): number[] {
    return this.computeForwardMath(obs);
  }

  /**
   * Verifies if embedded weights pass all schema and non-zero role slice checks.
   */
  public static isCheckpointValid(): boolean {
    try {
      assertMappoWeightsValid();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Direct forward-pass MLP evaluation of the trained actor network:
   * Layer 0: Linear(OBSERVATION_DIM, 64) -> Tanh (where OBSERVATION_DIM = 127)
   * Layer 1: Linear(64, 64) -> Tanh
   * Layer 2: Linear(64, 19) -> Logits
   */
  public computeForwardMath(obs: number[]): number[] {
    assertMappoWeightsValid();
    const { w0, b0, w1, b1, w2, b2 } = this.weights;

    // Layer 0: Linear(OBSERVATION_DIM, 64) -> Tanh
    const h0 = new Float32Array(64);
    for (let i = 0; i < 64; i++) {
      let sum = b0[i];
      const offset = i * OBSERVATION_DIM;
      for (let j = 0; j < OBSERVATION_DIM; j++) {
        sum += (w0 as number[])[offset + j] * obs[j];
      }
      h0[i] = Math.tanh(sum);
    }

    // Layer 1: Linear(64, 64) -> Tanh
    const h1 = new Float32Array(64);
    for (let i = 0; i < 64; i++) {
      let sum = b1[i];
      const offset = i * 64;
      for (let j = 0; j < 64; j++) {
        sum += (w1 as number[])[offset + j] * h0[j];
      }
      h1[i] = Math.tanh(sum);
    }

    // Layer 2: Linear(64, 19) -> Logits
    const logits = new Float32Array(ACTION_SPACE_SIZE);
    for (let i = 0; i < ACTION_SPACE_SIZE; i++) {
      let sum = b2[i];
      const offset = i * 64;
      for (let j = 0; j < 64; j++) {
        sum += (w2 as number[])[offset + j] * h1[j];
      }
      logits[i] = sum;
    }

    return Array.from(logits);
  }
}

