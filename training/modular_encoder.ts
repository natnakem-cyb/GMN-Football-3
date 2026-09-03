/**
 * GMN-Football-3 — Modular Feature Extractor & Entity Parser (TypeScript Reference)
 * Decomposes the 127-dimensional simple115_v3_role observation into distinct entity channels:
 * - Ego Agent (kinematics, relative ball coordinates, role one-hot, active index)
 * - Ball State (kinematics, speed, ownership)
 * - Teammates (left team relative coordinates, distances, mask)
 * - Opponents (right team relative coordinates, distances, mask)
 * - Match Context (game mode one-hot)
 */

import {
  OBSERVATION_DIM,
  OBSERVATION_SCHEMA_VERSION,
  ROLE_DIM,
  ROLE_VOCABULARY,
} from '../src/engine/Contract';

export interface ModularEntities {
  ego: {
    pos: { x: number; y: number };
    vel: { vx: number; vy: number };
    relBall: { dx: number; dy: number; dist: number };
    activeIdx: number;
    activeOneHot: number[];
    role: string;
    roleOneHot: number[];
    features: number[]; // 30 floats
  };
  ball: {
    pos: { x: number; y: number; z: number };
    vel: { vx: number; vy: number; vz: number };
    speed: number;
    owner: 'none' | 'left' | 'right';
    ownerOneHot: number[]; // 3 floats
    features: number[]; // 10 floats
  };
  teammates: {
    count: number;
    players: Array<{
      index: number;
      pos: { x: number; y: number };
      vel: { vx: number; vy: number };
      relEgo: { dx: number; dy: number; dist: number };
      isEgo: boolean;
      isPresent: boolean;
      features: number[]; // 9 floats
    }>;
    pooledFeatures: number[]; // 64 floats (mean 32 + max 32)
  };
  opponents: {
    count: number;
    players: Array<{
      index: number;
      pos: { x: number; y: number };
      vel: { vx: number; vy: number };
      relEgo: { dx: number; dy: number; dist: number };
      isPresent: boolean;
      features: number[]; // 8 floats
    }>;
    pooledFeatures: number[]; // 64 floats (mean 32 + max 32)
  };
  match: {
    gameModeIdx: number;
    gameModeOneHot: number[]; // 7 floats
    features: number[]; // 7 floats
  };
  concatenatedFeatures: number[]; // 30 + 10 + 64 + 64 + 7 = 175 raw entity floats
}

export class ModularFeatureParser {
  /**
   * Parses a flat simple115_v3_role 127-float observation vector into structured modular entities.
   */
  public static parse(rawObs: number[]): ModularEntities {
    if (rawObs.length !== OBSERVATION_DIM) {
      throw new Error(
        `Invalid observation length: expected ${OBSERVATION_DIM} (${OBSERVATION_SCHEMA_VERSION}), got ${rawObs.length}`
      );
    }

    // 1. Unpack slices
    const leftPos: Array<{ x: number; y: number }> = [];
    for (let i = 0; i < 11; i++) {
      leftPos.push({ x: rawObs[i * 2], y: rawObs[i * 2 + 1] });
    }

    const leftVel: Array<{ vx: number; vy: number }> = [];
    for (let i = 0; i < 11; i++) {
      leftVel.push({ vx: rawObs[22 + i * 2], vy: rawObs[22 + i * 2 + 1] });
    }

    const rightPos: Array<{ x: number; y: number }> = [];
    for (let i = 0; i < 11; i++) {
      rightPos.push({ x: rawObs[44 + i * 2], y: rawObs[44 + i * 2 + 1] });
    }

    const rightVel: Array<{ vx: number; vy: number }> = [];
    for (let i = 0; i < 11; i++) {
      rightVel.push({ vx: rawObs[66 + i * 2], vy: rawObs[66 + i * 2 + 1] });
    }

    const ballPos = { x: rawObs[88], y: rawObs[89], z: rawObs[90] };
    const ballVel = { vx: rawObs[91], vy: rawObs[92], vz: rawObs[93] };
    const ballSpeed = Math.sqrt(ballVel.vx ** 2 + ballVel.vy ** 2 + ballVel.vz ** 2);

    const ownerOneHot = [rawObs[94], rawObs[95], rawObs[96]];
    const owner: 'none' | 'left' | 'right' =
      ownerOneHot[1] === 1 ? 'left' : ownerOneHot[2] === 1 ? 'right' : 'none';

    const activeOneHot = rawObs.slice(97, 108);
    const activeIdx = activeOneHot.indexOf(1.0) >= 0 ? activeOneHot.indexOf(1.0) : 0;

    const gameModeOneHot = rawObs.slice(108, 115);
    const gameModeIdx = gameModeOneHot.indexOf(1.0) >= 0 ? gameModeOneHot.indexOf(1.0) : 0;

    const roleOneHot = rawObs.slice(115, 127);
    const roleIdx = roleOneHot.indexOf(1.0) >= 0 ? roleOneHot.indexOf(1.0) : 0;
    const roleName = ROLE_VOCABULARY[roleIdx] || 'ST';

    // 2. Ego kinematics
    const egoPos = leftPos[activeIdx] || { x: 0, y: 0 };
    const egoVel = leftVel[activeIdx] || { vx: 0, vy: 0 };
    const relBallDx = ballPos.x - egoPos.x;
    const relBallDy = ballPos.y - egoPos.y;
    const relBallDist = Math.sqrt(relBallDx ** 2 + relBallDy ** 2);

    const egoFeatures = [
      egoPos.x,
      egoPos.y,
      egoVel.vx,
      egoVel.vy,
      relBallDx,
      relBallDy,
      relBallDist,
      ...activeOneHot,
      ...roleOneHot,
    ]; // 2 + 2 + 2 + 1 + 11 + 12 = 30 floats

    // 3. Ball features
    const ballFeatures = [
      ballPos.x,
      ballPos.y,
      ballPos.z,
      ballVel.vx,
      ballVel.vy,
      ballVel.vz,
      ballSpeed,
      ...ownerOneHot,
    ]; // 3 + 3 + 1 + 3 = 10 floats

    // 4. Teammates features & pseudo-pooling
    const teammatePlayers: ModularEntities['teammates']['players'] = [];
    let validTeammatesCount = 0;
    for (let i = 0; i < 11; i++) {
      const p = leftPos[i];
      const v = leftVel[i];
      const isPresent = p.x > -0.999 || p.y > -0.999;
      const isEgo = i === activeIdx;
      const relDx = p.x - egoPos.x;
      const relDy = p.y - egoPos.y;
      const dist = Math.sqrt(relDx ** 2 + relDy ** 2);
      const feat = [
        p.x,
        p.y,
        v.vx,
        v.vy,
        relDx,
        relDy,
        dist,
        isEgo ? 1.0 : 0.0,
        isPresent ? 1.0 : 0.0,
      ];
      teammatePlayers.push({
        index: i,
        pos: p,
        vel: v,
        relEgo: { dx: relDx, dy: relDy, dist },
        isEgo,
        isPresent,
        features: feat,
      });
      if (isPresent && !isEgo) validTeammatesCount++;
    }

    // 5. Opponents features
    const opponentPlayers: ModularEntities['opponents']['players'] = [];
    let validOpponentsCount = 0;
    for (let j = 0; j < 11; j++) {
      const p = rightPos[j];
      const v = rightVel[j];
      const isPresent = p.x > -0.999 || p.y > -0.999;
      const relDx = p.x - egoPos.x;
      const relDy = p.y - egoPos.y;
      const dist = Math.sqrt(relDx ** 2 + relDy ** 2);
      const feat = [p.x, p.y, v.vx, v.vy, relDx, relDy, dist, isPresent ? 1.0 : 0.0];
      opponentPlayers.push({
        index: j,
        pos: p,
        vel: v,
        relEgo: { dx: relDx, dy: relDy, dist },
        isPresent,
        features: feat,
      });
      if (isPresent) validOpponentsCount++;
    }

    // 6. Concatenated raw entity features
    const concatenatedFeatures = [
      ...egoFeatures,
      ...ballFeatures,
      ...teammatePlayers.flatMap((t) => t.features), // 11 * 9 = 99
      ...opponentPlayers.flatMap((o) => o.features), // 11 * 8 = 88
      ...gameModeOneHot, // 7
    ];

    return {
      ego: {
        pos: egoPos,
        vel: egoVel,
        relBall: { dx: relBallDx, dy: relBallDy, dist: relBallDist },
        activeIdx,
        activeOneHot,
        role: roleName,
        roleOneHot,
        features: egoFeatures,
      },
      ball: {
        pos: ballPos,
        vel: ballVel,
        speed: ballSpeed,
        owner,
        ownerOneHot,
        features: ballFeatures,
      },
      teammates: {
        count: validTeammatesCount,
        players: teammatePlayers,
        pooledFeatures: new Array(64).fill(0),
      },
      opponents: {
        count: validOpponentsCount,
        players: opponentPlayers,
        pooledFeatures: new Array(64).fill(0),
      },
      match: {
        gameModeIdx,
        gameModeOneHot,
        features: gameModeOneHot,
      },
      concatenatedFeatures,
    };
  }
}
