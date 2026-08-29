import http from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ActionType, AgentAction, ScenarioConfig } from '../src/types/football';
import { mapDiscreteAction } from './action_mapping';
import {
  GMN_ENV_VERSION,
  OBSERVATION_SCHEMA_VERSION,
  ACTION_SCHEMA_VERSION,
  OBSERVATION_DIM,
  ACTION_SPACE_SIZE,
  getEventCode,
} from '../src/engine/Contract';
import { Vec2 } from '../src/engine/Vector';
import { RuleBasedAgent } from '../src/agents/RuleBasedAgent';
import { ObservationEncoder } from '../src/engine/ObservationEncoder';

const PORT = parseInt(process.env.GMN_BRIDGE_PORT || '5050', 10);
const HOST = '127.0.0.1';

export class GMNBridgeService {
  private engine: GameEngine;
  private botAgents: Map<string, RuleBasedAgent>;
  private scenarioMap: Map<string, ScenarioConfig>;

  constructor() {
    this.engine = new GameEngine();
    this.botAgents = new Map();
    this.scenarioMap = new Map();

    ACADEMY_SCENARIOS.forEach((sc) => {
      this.scenarioMap.set(sc.id, sc);
      this.scenarioMap.set(sc.codeName, sc);
    });

    // Default to academy_empty_goal
    const defaultScenario = this.scenarioMap.get('academy_empty_goal');
    if (defaultScenario) {
      this.engine.loadScenario(defaultScenario);
    }
  }

  public reset(scenarioName = 'academy_empty_goal', seed?: number) {
    const sc = this.scenarioMap.get(scenarioName) || this.scenarioMap.get('academy_empty_goal');
    if (sc) {
      this.engine.loadScenario(sc, seed);
    } else {
      this.engine.resetToKickoff(seed);
    }

    // Reset bot states
    this.botAgents.clear();

    // Pure initial observation without stepping physics
    const initialObs = this.engine.getObservation();

    const controllableAgentIds = this.engine.players
      .filter((p) => p.team === 'left')
      .map((p) => p.id);

    const perAgentObservations = controllableAgentIds.map((id) =>
      ObservationEncoder.encode(
        this.engine.players,
        this.engine.ball,
        id,
        this.engine.score,
        this.engine.tickCount,
        this.engine.activeScenario ? this.engine.activeScenario.timeLimitSeconds * 60 : 3600,
        this.engine.gameMode
      ).rawVector
    );

    return {
      observation: initialObs.rawVector,
      observations: perAgentObservations,
      info: {
        score: { ...this.engine.score },
        ballDistanceToGoal: Vec2.distance(
          { x: this.engine.ball.position.x, y: this.engine.ball.position.y },
          { x: 1.0, y: 0 }
        ),
        scenario: sc?.codeName || 'free_play',
        controlledPlayerId: this.engine.controlledPlayerId,
        controllableAgentIds,
      },
    };
  }

  public step(actionIdx: number) {
    const actionMap = new Map<string, AgentAction>();

    // 1. Controlled player action from RL agent
    const controlledPlayer = this.engine.players.find(
      (p) => p.id === this.engine.controlledPlayerId
    ) || this.engine.players.find((p) => p.team === 'left');

    if (controlledPlayer) {
      const mappedAction = mapDiscreteAction(actionIdx, controlledPlayer.heading);
      actionMap.set(controlledPlayer.id, mappedAction);
    }

    // 2. Automated bots for other players (if any)
    this.engine.players.forEach((player) => {
      if (player.id === controlledPlayer?.id) return;

      if (!this.botAgents.has(player.id)) {
        this.botAgents.set(
          player.id,
          new RuleBasedAgent(`bot_${player.id}`, player.name, 'medium')
        );
      }
      const bot = this.botAgents.get(player.id)!;
      const context = {
        player,
        teammates: this.engine.players.filter((p) => p.team === player.team),
        opponents: this.engine.players.filter((p) => p.team !== player.team),
        ball: this.engine.ball,
        allPlayers: this.engine.players,
        teamSide: player.team,
        controlledPlayerId: this.engine.controlledPlayerId,
        matchTime: this.engine.matchTimeSeconds,
        rng: this.engine.rng,
      };
      actionMap.set(player.id, bot.decide(context));
    });

    // 3. Execute deterministic physics tick (1/60s)
    const result = this.engine.step(actionMap, 1 / 60);

    return {
      observation: result.observation.rawVector,
      reward: result.reward,
      terminated: result.terminated,
      truncated: result.truncated,
      info: {
        score: result.info.score,
        event: result.info.event,
        checkpointReward: result.info.checkpointReward,
        ballDistanceToGoal: result.info.ballDistanceToGoal,
      },
    };
  }

  public stepMulti(actionIndices: number[]) {
    const controllableIds = this.engine.players
      .filter((p) => p.team === 'left')
      .map((p) => p.id);

    if (actionIndices.length !== controllableIds.length) {
      throw new Error(
        `[GMN Multi-Agent] Expected ${controllableIds.length} actions, got ${actionIndices.length}`
      );
    }

    const actionMap = new Map<string, AgentAction>();

    // 1. Controlled agents (left team), in fixed order
    controllableIds.forEach((id, i) => {
      const player = this.engine.players.find((p) => p.id === id)!;
      actionMap.set(id, mapDiscreteAction(actionIndices[i], player.heading));
    });

    // 2. Automated bots for other players (if any)
    this.engine.players.forEach((player) => {
      if (controllableIds.includes(player.id)) return;
      if (!this.botAgents.has(player.id)) {
        this.botAgents.set(
          player.id,
          new RuleBasedAgent(`bot_${player.id}`, player.name, 'medium')
        );
      }
      const bot = this.botAgents.get(player.id)!;
      actionMap.set(
        player.id,
        bot.decide({
          player,
          teammates: this.engine.players.filter((p) => p.team === player.team),
          opponents: this.engine.players.filter((p) => p.team !== player.team),
          ball: this.engine.ball,
          allPlayers: this.engine.players,
          teamSide: player.team,
          controlledPlayerId: this.engine.controlledPlayerId,
          matchTime: this.engine.matchTimeSeconds,
          rng: this.engine.rng,
        })
      );
    });

    // 3. Execute deterministic physics tick (1/60s)
    const result = this.engine.step(actionMap, 1 / 60);

    // 4. Re-encode one observation per controlled agent from the
    // already-updated post-step state — do not step the engine again
    const observations = controllableIds.map((id) =>
      ObservationEncoder.encode(
        this.engine.players,
        this.engine.ball,
        id,
        this.engine.score,
        this.engine.tickCount,
        this.engine.activeScenario ? this.engine.activeScenario.timeLimitSeconds * 60 : 3600,
        this.engine.gameMode
      ).rawVector
    );

    return {
      reward: result.reward,
      terminated: result.terminated,
      truncated: result.truncated,
      info: {
        score: result.info.score,
        event: result.info.event,
        checkpointReward: result.info.checkpointReward,
        ballDistanceToGoal: result.info.ballDistanceToGoal,
      },
      observations, // array, same order as controllableIds
    };
  }

  public getInfo() {
    return {
      status: 'ok',
      environment: 'GMN-Football-3',
      environment_version: GMN_ENV_VERSION,
      observation_dim: OBSERVATION_DIM,
      observation_schema_version: OBSERVATION_SCHEMA_VERSION,
      action_space_size: ACTION_SPACE_SIZE,
      action_schema_version: ACTION_SCHEMA_VERSION,
      scenario: this.engine.activeScenario?.codeName || 'none',
      controlledPlayerId: this.engine.controlledPlayerId,
      scenarios: Array.from(new Set(ACADEMY_SCENARIOS.map((s) => s.codeName))),
    };
  }
}

// Instantiate Service
const bridge = new GMNBridgeService();

// Create Lightweight HTTP Server
const server = http.createServer((req, res) => {
  // CORS & JSON Headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Content-Type', 'application/json');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  let body = '';
  req.on('data', (chunk) => {
    body += chunk;
  });

  req.on('end', () => {
    try {
      const parsedBody = body ? JSON.parse(body) : {};

      if (req.method === 'GET' && (req.url === '/' || req.url === '/info' || req.url === '/health')) {
        res.writeHead(200);
        res.end(JSON.stringify(bridge.getInfo()));
        return;
      }

      if (req.method === 'POST' && req.url === '/reset') {
        const resetResult = bridge.reset(parsedBody.scenario, parsedBody.seed);
        res.writeHead(200);
        res.end(JSON.stringify(resetResult));
        return;
      }

      if (req.method === 'POST' && req.url === '/step') {
        if (typeof parsedBody.action !== 'number' || !Number.isInteger(parsedBody.action)) {
          res.writeHead(400);
          res.end(JSON.stringify({ error: `Invalid action: ${parsedBody.action}. Must be integer in [0, ${ACTION_SPACE_SIZE - 1}].` }));
          return;
        }
        const stepResult = bridge.step(parsedBody.action);
        res.writeHead(200);
        res.end(JSON.stringify(stepResult));
        return;
      }

      if (req.method === 'POST' && req.url === '/step_multi') {
        if (!Array.isArray(parsedBody.actions)) {
          res.writeHead(400);
          res.end(JSON.stringify({ error: 'actions must be an array of integers' }));
          return;
        }
        const multiResult = bridge.stepMulti(parsedBody.actions);
        res.writeHead(200);
        res.end(JSON.stringify(multiResult));
        return;
      }

      if (req.method === 'POST' && req.url === '/close') {
        res.writeHead(200);
        res.end(JSON.stringify({ status: 'closing' }));
        server.close(() => {
          process.exit(0);
        });
        return;
      }

      res.writeHead(404);
      res.end(JSON.stringify({ error: 'Endpoint not found' }));
    } catch (err: any) {
      res.writeHead(500);
      res.end(JSON.stringify({ error: err.message || 'Internal error' }));
    }
  });
});

// Binary step-response layout: 477 bytes total, all little-endian
// Offset 0 (4B float32): reward
// Offset 4 (1B uint8): terminated (0/1)
// Offset 5 (1B uint8): truncated (0/1)
// Offset 6 (1B uint8): scoreLeft
// Offset 7 (1B uint8): scoreRight
// Offset 8 (4B float32): checkpointReward
// Offset 12 (4B float32): ballDistanceToGoal
// Offset 16 (1B uint8): eventCode
// Offset 17 (460B): 115 * float32 observation
export function encodeStepBinary(stepResult: ReturnType<typeof bridge.step>): Buffer {
  const buf = Buffer.allocUnsafe(477);
  buf.writeFloatLE(stepResult.reward || 0.0, 0);
  buf.writeUInt8(stepResult.terminated ? 1 : 0, 4);
  buf.writeUInt8(stepResult.truncated ? 1 : 0, 5);
  buf.writeUInt8(Math.max(0, Math.min(255, stepResult.info.score?.left ?? 0)), 6);
  buf.writeUInt8(Math.max(0, Math.min(255, stepResult.info.score?.right ?? 0)), 7);
  buf.writeFloatLE(stepResult.info.checkpointReward ?? 0.0, 8);
  buf.writeFloatLE(stepResult.info.ballDistanceToGoal ?? 0.0, 12);

  const eventType = stepResult.info.event?.type;
  const eventCode = getEventCode(eventType);
  buf.writeUInt8(eventCode, 16);

  const obs = stepResult.observation;
  for (let i = 0; i < OBSERVATION_DIM; i++) {
    buf.writeFloatLE(obs[i] ?? 0.0, 17 + i * 4);
  }

  return buf;
}

// Multi-Agent Binary step-response layout: 17 + 460 * N bytes total, all little-endian
// Offset 0 (4B float32): reward (shared team reward)
// Offset 4 (1B uint8): terminated (0/1)
// Offset 5 (1B uint8): truncated (0/1)
// Offset 6 (1B uint8): scoreLeft
// Offset 7 (1B uint8): scoreRight
// Offset 8 (4B float32): checkpointReward
// Offset 12 (4B float32): ballDistanceToGoal
// Offset 16 (1B uint8): eventCode
// Offset 17 (460 * N B): N observations, 115 * float32 each, in controllableAgentIds order
export function encodeMultiStepBinary(multiResult: ReturnType<typeof bridge.stepMulti>): Buffer {
  const N = multiResult.observations.length;
  const buf = Buffer.allocUnsafe(17 + 460 * N);
  buf.writeFloatLE(multiResult.reward || 0.0, 0);
  buf.writeUInt8(multiResult.terminated ? 1 : 0, 4);
  buf.writeUInt8(multiResult.truncated ? 1 : 0, 5);
  buf.writeUInt8(Math.max(0, Math.min(255, multiResult.info.score?.left ?? 0)), 6);
  buf.writeUInt8(Math.max(0, Math.min(255, multiResult.info.score?.right ?? 0)), 7);
  buf.writeFloatLE(multiResult.info.checkpointReward ?? 0.0, 8);
  buf.writeFloatLE(multiResult.info.ballDistanceToGoal ?? 0.0, 12);

  const eventType = multiResult.info.event?.type;
  const eventCode = getEventCode(eventType);
  buf.writeUInt8(eventCode, 16);

  for (let agentIdx = 0; agentIdx < N; agentIdx++) {
    const obs = multiResult.observations[agentIdx];
    const baseOffset = 17 + agentIdx * 460;
    for (let i = 0; i < OBSERVATION_DIM; i++) {
      buf.writeFloatLE(obs[i] ?? 0.0, baseOffset + i * 4);
    }
  }

  return buf;
}

// Attach WebSocket Server to the same HTTP Server instance
const wss = new WebSocketServer({ server });

wss.on('connection', (ws: WebSocket) => {
  ws.on('message', (data: Buffer | ArrayBuffer | Buffer[], isBinary: boolean) => {
    try {
      if (isBinary) {
        const buf = Buffer.isBuffer(data) ? data : Buffer.from(data as any);
        if (buf.length === 1) {
          // existing single-agent path — unchanged
          const actionIdx = buf.readUInt8(0);
          if (actionIdx >= ACTION_SPACE_SIZE) return;
          const stepResult = bridge.step(actionIdx);
          ws.send(encodeStepBinary(stepResult), { binary: true });
        } else if (buf.length > 1) {
          // new multi-agent path
          const actionIndices = Array.from(buf); // one uint8 per controlled agent, in controllableAgentIds order
          const multiResult = bridge.stepMulti(actionIndices);
          ws.send(encodeMultiStepBinary(multiResult), { binary: true });
        }
      } else {
        const text = data.toString('utf8');
        const parsed = JSON.parse(text);
        if (parsed.type === 'reset') {
          const resetResult = bridge.reset(parsed.scenario, parsed.seed);
          ws.send(JSON.stringify(resetResult));
        } else if (parsed.type === 'close') {
          ws.close();
        } else if (parsed.type === 'info') {
          ws.send(JSON.stringify(bridge.getInfo()));
        } else if (parsed.type === 'step') {
          const stepResult = bridge.step(parsed.action);
          ws.send(JSON.stringify(stepResult));
        } else if (parsed.type === 'step_multi') {
          const multiResult = bridge.stepMulti(parsed.actions);
          ws.send(JSON.stringify(multiResult));
        }
      }
    } catch (err: any) {
      console.error('[WS Error]', err);
    }
  });
});

server.keepAliveTimeout = 120000;
server.headersTimeout = 125000;
server.on('error', (err) => {
  console.error('[GMN Bridge Server] Socket error:', err);
});

server.listen(PORT, HOST, () => {
  console.log(`[GMN Headless Bridge] Server listening on http://${HOST}:${PORT} (HTTP + Binary WebSocket)`);
});

process.on('SIGINT', () => {
  server.close(() => process.exit(0));
});

process.on('SIGTERM', () => {
  server.close(() => process.exit(0));
});
