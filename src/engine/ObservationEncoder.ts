import { Ball, GameMode, MatchScore, Player, RLObservation, TeamSide } from '../types/football';
import { PITCH } from './Rules';
import { Vec2 } from './Vector';
import {
  OBSERVATION_DIM,
  OBSERVATION_SCHEMA_VERSION,
  BASE_OBSERVATION_DIM,
  ROLE_DIM,
  ROLE_VOCABULARY,
} from './Contract';

export { OBSERVATION_DIM, OBSERVATION_SCHEMA_VERSION, BASE_OBSERVATION_DIM, ROLE_DIM };

export class ObservationEncoder {
  /**
   * Generates a Google Research Football compatible SMM / Feature vector observation
   * with role differentiation (127 floats total):
   * - Offset 0 (len 22): Left team player (x, y) positions, 11 players
   * - Offset 22 (len 22): Left team player (x, y) movement direction
   * - Offset 44 (len 22): Right team player (x, y) positions
   * - Offset 66 (len 22): Right team player (x, y) movement direction
   * - Offset 88 (len 3): Ball (x, y, z) position
   * - Offset 91 (len 3): Ball (x, y, z) movement direction
   * - Offset 94 (len 3): Ball ownership, one-hot: [no-one, left, right]
   * - Offset 97 (len 11): Active player, one-hot over 11 players
   * - Offset 108 (len 7): game_mode, one-hot: [Normal, KickOff, GoalKick, FreeKick, Corner, ThrowIn, Penalty]
   * - Offset 115 (len 12): Agent's assigned role one-hot over ROLE_VOCABULARY:
   *   [GK, CB, LB, RB, CDM, CM, LM, RM, LW, RW, CAM, ST]
   * Total: 127 floats. Inactive player slots are set to -1.
   */
  static encode(
    players: Player[],
    ball: Ball,
    viewpointPlayerId: string | null,
    score: MatchScore,
    stepCount: number,
    maxSteps: number,
    gameMode: GameMode = GameMode.Normal
  ): RLObservation {
    const leftPlayers = players.filter((p) => p.team === 'left');
    const rightPlayers = players.filter((p) => p.team === 'right');

    const leftPositions = leftPlayers.map((p) => [p.position.x, p.position.y]);
    const leftVelocities = leftPlayers.map((p) => [p.velocity.x * 50, p.velocity.y * 50]);
    const rightPositions = rightPlayers.map((p) => [p.position.x, p.position.y]);
    const rightVelocities = rightPlayers.map((p) => [p.velocity.x * 50, p.velocity.y * 50]);

    let ballOwnedTeam: -1 | 0 | 1 = -1;
    let ballOwnedPlayer = -1;

    if (ball.ownerId) {
      const owner = players.find((p) => p.id === ball.ownerId);
      if (owner) {
        if (owner.team === 'left') {
          ballOwnedTeam = 0;
          ballOwnedPlayer = leftPlayers.findIndex((p) => p.id === owner.id);
        } else {
          ballOwnedTeam = 1;
          ballOwnedPlayer = rightPlayers.findIndex((p) => p.id === owner.id);
        }
      }
    }

    const activeIndex = viewpointPlayerId
      ? leftPlayers.findIndex((p) => p.id === viewpointPlayerId)
      : (leftPlayers.length > 0 ? 0 : -1);

    // Construct flat rawVector with exactly OBSERVATION_DIM (127) floats
    const rawVector: number[] = [];

    // 0..21 (Length 22): Left team player (x, y) positions, 11 players
    for (let i = 0; i < 11; i++) {
      if (i < leftPlayers.length) {
        rawVector.push(leftPlayers[i].position.x, leftPlayers[i].position.y);
      } else {
        rawVector.push(-1.0, -1.0);
      }
    }

    // 22..43 (Length 22): Left team player (x, y) movement direction, 11 players
    for (let i = 0; i < 11; i++) {
      if (i < leftPlayers.length) {
        rawVector.push(leftPlayers[i].velocity.x * 50, leftPlayers[i].velocity.y * 50);
      } else {
        rawVector.push(-1.0, -1.0);
      }
    }

    // 44..65 (Length 22): Right team player (x, y) positions, 11 players
    for (let i = 0; i < 11; i++) {
      if (i < rightPlayers.length) {
        rawVector.push(rightPlayers[i].position.x, rightPlayers[i].position.y);
      } else {
        rawVector.push(-1.0, -1.0);
      }
    }

    // 66..87 (Length 22): Right team player (x, y) movement direction, 11 players
    for (let i = 0; i < 11; i++) {
      if (i < rightPlayers.length) {
        rawVector.push(rightPlayers[i].velocity.x * 50, rightPlayers[i].velocity.y * 50);
      } else {
        rawVector.push(-1.0, -1.0);
      }
    }

    // 88..90 (Length 3): Ball (x, y, z) position
    rawVector.push(ball.position.x, ball.position.y, ball.position.z);

    // 91..93 (Length 3): Ball (x, y, z) movement direction
    rawVector.push(ball.velocity.x * 50, ball.velocity.y * 50, ball.velocity.z * 50);

    // 94..96 (Length 3): Ball ownership, one-hot: [no-one, left, right]
    rawVector.push(
      ballOwnedTeam === -1 ? 1.0 : 0.0,
      ballOwnedTeam === 0 ? 1.0 : 0.0,
      ballOwnedTeam === 1 ? 1.0 : 0.0
    );

    // 97..107 (Length 11): Viewpoint / Controlled player, one-hot over 11 players
    for (let i = 0; i < 11; i++) {
      rawVector.push(activeIndex === i ? 1.0 : 0.0);
    }

    // 108..114 (Length 7): game_mode, one-hot: [Normal, KickOff, GoalKick, FreeKick, Corner, ThrowIn, Penalty]
    const modeIndices: GameMode[] = [
      GameMode.Normal,
      GameMode.KickOff,
      GameMode.GoalKick,
      GameMode.FreeKick,
      GameMode.Corner,
      GameMode.ThrowIn,
      GameMode.Penalty,
    ];
    for (const mode of modeIndices) {
      rawVector.push(gameMode === mode ? 1.0 : 0.0);
    }

    // 115..126 (Length 12): Self Agent Role One-Hot over ROLE_VOCABULARY
    const viewpointPlayer = viewpointPlayerId ? players.find((p) => p.id === viewpointPlayerId) : leftPlayers[0];
    const role = viewpointPlayer?.role;
    for (let r = 0; r < ROLE_VOCABULARY.length; r++) {
      rawVector.push(role === ROLE_VOCABULARY[r] ? 1.0 : 0.0);
    }

    if (rawVector.length !== OBSERVATION_DIM) {
      throw new Error(
        `[ObservationEncoder Contract Violation] Encoded vector length mismatch: expected ${OBSERVATION_DIM}, got ${rawVector.length}`
      );
    }

    return {
      leftTeamPositions: leftPositions,
      leftTeamVelocities: leftVelocities,
      rightTeamPositions: rightPositions,
      rightTeamVelocities: rightVelocities,
      ballPosition: [ball.position.x, ball.position.y, ball.position.z],
      ballVelocity: [ball.velocity.x, ball.velocity.y, ball.velocity.z],
      ballOwnedTeam,
      ballOwnedPlayer,
      activePlayerIndex: Math.max(0, activeIndex),
      gameMode: gameMode as number,
      score: [score.left, score.right],
      stepsLeft: Math.max(0, maxSteps - stepCount),
      rawVector,
    };
  }

  /**
   * Reward shaping computation:
   * +1.0 for scoring a goal
   * -1.0 for conceding a goal
   * Checkpoint reward for monotonically advancing ball closer to opponent goal (up to +0.05)
   * +0.03 shot-attempt shaping bonus
   */
  static computeReward(
    prevBallX: number,
    currBallX: number,
    goalScoredTeam: TeamSide | null,
    targetTeam: TeamSide = 'left',
    shotTakenByTargetTeam = false,
    maxBallProgressX?: number
  ): { reward: number; checkpoint: number; newMaxBallProgressX: number } {
    let reward = 0;
    let checkpoint = 0;
    let newMaxBallProgressX = maxBallProgressX !== undefined ? maxBallProgressX : prevBallX;

    if (goalScoredTeam === targetTeam) {
      reward += 1.0;
    } else if (goalScoredTeam && goalScoredTeam !== targetTeam) {
      reward -= 1.0;
    }

    // Monotonic checkpoint reward: only pays when exceeding the episode high-water mark
    // (for left team, progress is positive X; for right team, progress is negative X)
    if (targetTeam === 'left') {
      if (currBallX > newMaxBallProgressX) {
        const deltaX = currBallX - newMaxBallProgressX;
        if (deltaX > 0.005) {
          checkpoint = Math.min(0.05, deltaX * 0.5);
          reward += checkpoint;
        }
        newMaxBallProgressX = currBallX;
      }
    } else if (targetTeam === 'right') {
      if (currBallX < newMaxBallProgressX) {
        const deltaX = newMaxBallProgressX - currBallX;
        if (deltaX > 0.005) {
          checkpoint = Math.min(0.05, deltaX * 0.5);
          reward += checkpoint;
        }
        newMaxBallProgressX = currBallX;
      }
    }

    // Shot-attempt shaping bonus — encourages discovering the act of
    // shooting, distinct from and much smaller than the goal reward itself.
    const SHOT_ATTEMPT_BONUS = 0.03;
    if (shotTakenByTargetTeam) {
      reward += SHOT_ATTEMPT_BONUS;
    }

    return { reward, checkpoint, newMaxBallProgressX };
  }
}
