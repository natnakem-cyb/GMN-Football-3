import { MatchEvent, MatchScore, ReplayFrame, TeamSide, Vector2D, Vector3D } from '../types/football';
import { GMN_ENV_VERSION, OBSERVATION_SCHEMA_VERSION } from '../engine/Contract';

export interface TraceMetadata {
  type: 'metadata';
  scenario: string;
  seed: number | null;
  agent_ids: string[];
  checkpoint?: string | null;
  recorded_at: string;
  env_version?: string;
  schema_version?: string;
}

export interface TraceStepRecord {
  tick: number;
  actions?: Record<string, number>;
  observations?: Record<string, number[]>;
  reward?: number;
  terminated?: boolean;
  truncated?: boolean;
  score?: MatchScore;
  event?: string | null;
  // Full frame payload if serialized directly from game engine
  frame?: ReplayFrame;
}

/**
 * Serializes an in-memory replay session to JSON Lines (.jsonl) format.
 */
export function exportReplayToJsonl(
  frames: ReplayFrame[],
  events: MatchEvent[],
  scenarioName: string,
  seed: number | null = null,
  checkpoint: string | null = null
): string {
  const metadata: TraceMetadata = {
    type: 'metadata',
    scenario: scenarioName,
    seed,
    agent_ids: frames.length > 0 ? frames[0].players.filter((p) => p.team === 'left').map((p) => p.id) : [],
    checkpoint,
    recorded_at: new Date().toISOString(),
    env_version: GMN_ENV_VERSION,
    schema_version: OBSERVATION_SCHEMA_VERSION,
  };

  const lines: string[] = [JSON.stringify(metadata)];

  for (const frame of frames) {
    const record: TraceStepRecord = {
      tick: frame.tick,
      score: frame.score,
      event: frame.event?.type || null,
      frame,
    };
    lines.push(JSON.stringify(record));
  }

  return lines.join('\n');
}

/**
 * Parses a .jsonl trace file into structured ReplayFrames and MatchEvents.
 * Automatically decodes both full-frame traces and raw RL observation traces.
 */
export function parseJsonlTrace(content: string): {
  metadata: TraceMetadata;
  frames: ReplayFrame[];
  events: MatchEvent[];
} {
  const rawLines = content.split('\n').map((l) => l.trim()).filter(Boolean);
  if (rawLines.length === 0) {
    throw new Error('Trace file is empty');
  }

  const firstObj = JSON.parse(rawLines[0]);
  let metadata: TraceMetadata;
  let stepLines: string[];

  if (firstObj.type === 'metadata') {
    metadata = firstObj as TraceMetadata;
    stepLines = rawLines.slice(1);
  } else {
    metadata = {
      type: 'metadata',
      scenario: 'custom_trace',
      seed: null,
      agent_ids: ['left_1'],
      checkpoint: null,
      recorded_at: new Date().toISOString(),
    };
    stepLines = rawLines;
  }

  const frames: ReplayFrame[] = [];
  const events: MatchEvent[] = [];

  for (let i = 0; i < stepLines.length; i++) {
    const parsed = JSON.parse(stepLines[i]);

    if (parsed.frame) {
      // Direct full frame
      const frame = parsed.frame as ReplayFrame;
      frames.push(frame);
      if (frame.event) {
        events.push(frame.event);
      }
    } else if (parsed.observations) {
      // Reconstruct frame from 115/127-float observation vector
      const agentKeys = Object.keys(parsed.observations);
      const firstObs = parsed.observations[agentKeys[0]] as number[];
      const frame = reconstructFrameFromObs(parsed.tick || i, firstObs, parsed.score, parsed.event);
      frames.push(frame);
      if (frame.event) {
        events.push(frame.event);
      }
    }
  }

  return { metadata, frames, events };
}

/**
 * Reconstructs 2D/3D entities from canonical simple115_v2 / simple115_v3_role observation array.
 */
function reconstructFrameFromObs(
  tick: number,
  obs: number[],
  score?: MatchScore,
  eventStr?: string | null
): ReplayFrame {
  const players: ReplayFrame['players'] = [];

  // Left team positions: 0..21 (11 players x 2)
  // Left team velocities: 22..43 (11 players x 2)
  for (let i = 0; i < 11; i++) {
    const px = obs[i * 2];
    const py = obs[i * 2 + 1];
    if (px !== -1.0 || py !== -1.0) {
      const vx = (obs[22 + i * 2] || 0) / 50;
      const vy = (obs[22 + i * 2 + 1] || 0) / 50;
      players.push({
        id: `left_${i + 1}`,
        team: 'left',
        position: { x: px, y: py },
        velocity: { x: vx, y: vy },
        heading: Math.atan2(vy, vx) || 0,
        stamina: 100,
        isTackling: false,
        hasBall: false,
      });
    }
  }

  // Right team positions: 44..65 (11 players x 2)
  // Right team velocities: 66..87 (11 players x 2)
  for (let i = 0; i < 11; i++) {
    const px = obs[44 + i * 2];
    const py = obs[44 + i * 2 + 1];
    if (px !== -1.0 || py !== -1.0) {
      const vx = (obs[66 + i * 2] || 0) / 50;
      const vy = (obs[66 + i * 2 + 1] || 0) / 50;
      players.push({
        id: `right_${i + 1}`,
        team: 'right',
        position: { x: px, y: py },
        velocity: { x: vx, y: vy },
        heading: Math.atan2(vy, vx) || Math.PI,
        stamina: 100,
        isTackling: false,
        hasBall: false,
      });
    }
  }

  // Ball pos: 88..90, Ball vel: 91..93
  const bx = obs[88] ?? 0;
  const by = obs[89] ?? 0;
  const bz = obs[90] ?? 0;
  const bvx = (obs[91] ?? 0) / 50;
  const bvy = (obs[92] ?? 0) / 50;
  const bvz = (obs[93] ?? 0) / 50;

  let matchEvent: MatchEvent | undefined = undefined;
  if (eventStr) {
    matchEvent = {
      id: `evt_${tick}`,
      timeSeconds: tick / 60,
      type: eventStr as any,
      description: `Event: ${eventStr}`,
      position: { x: bx, y: by },
    };
  }

  return {
    tick,
    timestamp: Date.now(),
    matchTimeSeconds: tick / 60,
    ball: {
      position: { x: bx, y: by, z: bz },
      velocity: { x: bvx, y: bvy, z: bvz },
      ownerId: null,
    },
    players,
    score: score || { left: 0, right: 0 },
    event: matchEvent,
  };
}
