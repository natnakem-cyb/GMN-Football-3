import http from 'http';
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
} from '../src/engine/Contract';
import { Vec2 } from '../src/engine/Vector';
import { RuleBasedAgent } from '../src/agents/RuleBasedAgent';

const PORT = parseInt(process.env.GMN_BRIDGE_PORT || '5050', 10);
const HOST = '127.0.0.1';

class GMNBridgeService {
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
    return {
      observation: initialObs.rawVector,
      info: {
        score: { ...this.engine.score },
        ballDistanceToGoal: Vec2.distance(
          { x: this.engine.ball.position.x, y: this.engine.ball.position.y },
          { x: 1.0, y: 0 }
        ),
        scenario: sc?.codeName || 'free_play',
        controlledPlayerId: this.engine.controlledPlayerId,
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

server.keepAliveTimeout = 120000;
server.headersTimeout = 125000;
server.on('error', (err) => {
  console.error('[GMN Bridge Server] Socket error:', err);
});

server.listen(PORT, HOST, () => {
  console.log(`[GMN Headless Bridge] Server listening on http://${HOST}:${PORT}`);
});

process.on('SIGINT', () => {
  server.close(() => process.exit(0));
});

process.on('SIGTERM', () => {
  server.close(() => process.exit(0));
});
