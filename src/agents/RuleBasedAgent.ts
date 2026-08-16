import { ActionType, AgentAction, Ball, Player, TeamSide, Vector2D } from '../types/football';
import { AgentDecisionContext, IAgent } from './BaseAgent';
import { PITCH } from '../engine/Rules';
import { Vec2 } from '../engine/Vector';

export class RuleBasedAgent implements IAgent {
  id: string;
  name: string;
  type: 'rule_based' = 'rule_based';
  difficulty: 'easy' | 'medium' | 'hard' | 'master';

  constructor(id = 'rule_ai', name = 'Tactical Rule AI', difficulty: 'easy' | 'medium' | 'hard' | 'master' = 'medium') {
    this.id = id;
    this.name = name;
    this.difficulty = difficulty;
  }

  decide(context: AgentDecisionContext): AgentAction {
    const { player, ball, teammates, opponents, teamSide, rng } = context;
    const rnd = rng ? () => rng.next() : Math.random;
    const opponentGoalX = teamSide === 'left' ? 1.0 : -1.0;
    const ownGoalX = teamSide === 'left' ? -1.0 : 1.0;

    const ballPos2D: Vector2D = { x: ball.position.x, y: ball.position.y };
    const distToBall = Vec2.distance(player.position, ballPos2D);
    const distToGoal = Math.abs(opponentGoalX - player.position.x);

    // 1. Goalkeeper Specific Heuristic
    if (player.isGoalkeeper) {
      return this.handleGoalkeeper(player, ball, teammates, ownGoalX, rnd);
    }

    // 2. If this player currently HAS the ball
    if (player.hasBall || ball.ownerId === player.id) {
      // Shooting heuristic: within shooting range and reasonable angle
      const SHOT_RANGE = PITCH.penaltyBoxLength + 0.05;
      const CLOSE_RANGE_EXEMPTION = PITCH.penaltyBoxLength * 0.4;
      const ANGLE_MULTIPLIER = 1.1;
      const goalCenterPoint: Vector2D = { x: opponentGoalX, y: 0 };
      const trueDistToGoal = Vec2.distance(player.position, goalCenterPoint);
      const xDistToGoal = Math.abs(opponentGoalX - player.position.x);
      const lateralOffset = Math.abs(player.position.y);
      const hasReasonableAngle =
        trueDistToGoal <= CLOSE_RANGE_EXEMPTION ||
        lateralOffset <= xDistToGoal * ANGLE_MULTIPLIER + PITCH.goalWidth;

      if (trueDistToGoal < SHOT_RANGE && hasReasonableAngle) {
        const keeper = opponents.find((o) => o.isGoalkeeper);
        let targetY: number;

        if (keeper) {
          const keeperY = keeper.position.y;
          // Aim for whichever post has more room relative to the keeper's current position
          const preferredSide = keeperY >= 0 ? -1 : 1;
          const postMargin = 0.015; // stay just inside the frame, not right at the edge
          const postY = preferredSide * (PITCH.goalWidth / 2 - postMargin); // ±0.055
          targetY = postY + (rnd() - 0.5) * 0.02; // small jitter around the chosen post
        } else {
          // No goalkeeper in this scenario — no one to beat, wider variance is fine
          targetY = (rnd() - 0.5) * PITCH.goalWidth * 0.8;
        }

        const goalCenter: Vector2D = { x: opponentGoalX, y: targetY };
        const shootDir = Vec2.sub(goalCenter, player.position);
        
        // Random shot variance based on difficulty
        const variance = this.difficulty === 'master' ? 0.01 : this.difficulty === 'hard' ? 0.03 : 0.07;
        shootDir.y += (rnd() - 0.5) * variance;

        return {
          type: ActionType.SHOT,
          direction: Vec2.normalize(shootDir),
          power: 0.85 + rnd() * 0.15,
        };
      }

      // Check for pressure from nearby opponent
      const nearestOpponent = this.getNearestPlayer(player.position, opponents);
      const isUnderPressure = nearestOpponent && Vec2.distance(player.position, nearestOpponent.position) < 0.12;

      // Passing heuristic: find open teammate further up the pitch
      if (isUnderPressure || rnd() < (this.difficulty === 'master' ? 0.25 : 0.12)) {
        const passTarget = this.findBestPassTarget(player, teammates, opponents, teamSide);
        if (passTarget) {
          const passDir = Vec2.sub(passTarget.position, player.position);
          const isLongPass = Vec2.distance(player.position, passTarget.position) > 0.45;

          return {
            type: isLongPass ? ActionType.HIGH_PASS : ActionType.SHORT_PASS,
            direction: Vec2.normalize(passDir),
            power: Math.min(1.0, Vec2.distance(player.position, passTarget.position) * 2.2 + 0.25),
            targetPlayerId: passTarget.id,
          };
        }
      }

      // Dribble towards opponent goal / open space
      const forwardDir = teamSide === 'left' ? 1 : -1;
      let targetY = player.position.y * 0.8; // tend toward center
      if (nearestOpponent && Math.abs(nearestOpponent.position.x - player.position.x) < 0.15) {
        // Evade opponent by moving sideways
        targetY += nearestOpponent.position.y > player.position.y ? -0.12 : 0.12;
      }

      const moveTarget: Vector2D = {
        x: player.position.x + forwardDir * 0.25,
        y: Math.max(PITCH.minY + 0.05, Math.min(PITCH.maxY - 0.05, targetY)),
      };

      return {
        type: this.difficulty === 'master' || this.difficulty === 'hard' ? ActionType.SPRINT : ActionType.MOVE,
        direction: Vec2.normalize(Vec2.sub(moveTarget, player.position)),
      };
    }

    // 3. If teammate has the ball: support attacking runs
    const ballOwner = teammates.find((t) => t.hasBall || t.id === ball.ownerId);
    if (ballOwner) {
      // Advance into space according to role
      let advanceX = player.targetPosition.x;
      if (teamSide === 'left') {
        advanceX = Math.min(PITCH.maxX - 0.1, advanceX + 0.15);
      } else {
        advanceX = Math.max(PITCH.minX + 0.1, advanceX - 0.15);
      }

      return {
        type: ActionType.MOVE,
        direction: Vec2.normalize(Vec2.sub({ x: advanceX, y: player.targetPosition.y }, player.position)),
      };
    }

    // 4. If opponent has the ball or loose ball
    // Determine if this player is the closest defender to the ball
    const isClosestToBall = this.getNearestPlayer(ballPos2D, teammates)?.id === player.id;

    if (isClosestToBall || distToBall < 0.2) {
      // Press the ball aggressively
      if (distToBall < 0.055 && ball.ownerId) {
        // In tackle range!
        return { type: ActionType.TACKLE };
      }

      // Chase ball
      return {
        type: ActionType.SPRINT,
        direction: Vec2.normalize(Vec2.sub(ballPos2D, player.position)),
      };
    }

    // Default: hold positional shape with tactical balance
    const dirToShape = Vec2.sub(player.targetPosition, player.position);
    if (Vec2.length(dirToShape) > 0.04) {
      return {
        type: ActionType.MOVE,
        direction: Vec2.normalize(dirToShape),
      };
    }

    return { type: ActionType.IDLE };
  }

  private handleGoalkeeper(
    keeper: Player,
    ball: Ball,
    teammates: Player[],
    ownGoalX: number,
    rnd: () => number = Math.random
  ): AgentAction {
    const ballPos2D: Vector2D = { x: ball.position.x, y: ball.position.y };

    if (keeper.hasBall || ball.ownerId === keeper.id) {
      // Distribute to outfield player
      const outfielders = teammates.filter((t) => !t.isGoalkeeper);
      const target = outfielders[Math.floor(rnd() * outfielders.length)] || outfielders[0];
      if (target) {
        return {
          type: ActionType.HIGH_PASS,
          direction: Vec2.normalize(Vec2.sub(target.position, keeper.position)),
          power: 0.8,
          targetPlayerId: target.id,
        };
      }
    }

    // Position on goal line mirroring ball Y coordinate within goal posts
    const targetX = ownGoalX + (keeper.team === 'left' ? 0.04 : -0.04);
    const clampedY = Math.max(PITCH.goalMinY * 1.2, Math.min(PITCH.goalMaxY * 1.2, ball.position.y * 0.7));

    const targetPos: Vector2D = { x: targetX, y: clampedY };
    const distToTarget = Vec2.distance(keeper.position, targetPos);

    if (distToTarget > 0.01) {
      return {
        type: ActionType.MOVE,
        direction: Vec2.normalize(Vec2.sub(targetPos, keeper.position)),
      };
    }

    return { type: ActionType.IDLE };
  }

  private getNearestPlayer(pos: Vector2D, players: Player[]): Player | null {
    if (players.length === 0) return null;
    let nearest = players[0];
    let minDist = Vec2.distance(pos, players[0].position);

    for (let i = 1; i < players.length; i++) {
      const d = Vec2.distance(pos, players[i].position);
      if (d < minDist) {
        minDist = d;
        nearest = players[i];
      }
    }
    return nearest;
  }

  private findBestPassTarget(
    passer: Player,
    teammates: Player[],
    opponents: Player[],
    teamSide: TeamSide
  ): Player | null {
    const candidates = teammates.filter((t) => t.id !== passer.id && !t.isGoalkeeper);
    if (candidates.length === 0) return null;

    let bestPlayer: Player | null = null;
    let highestScore = -999;

    for (const mate of candidates) {
      const dist = Vec2.distance(passer.position, mate.position);
      if (dist < 0.08 || dist > 0.7) continue;

      // Forward advancement reward
      const forwardProgress = teamSide === 'left'
        ? mate.position.x - passer.position.x
        : passer.position.x - mate.position.x;

      // Check lane obstruction by opponents
      let isObstructed = false;
      for (const opp of opponents) {
        const oppDistToLine = this.distanceToSegment(opp.position, passer.position, mate.position);
        if (oppDistToLine < 0.04) {
          isObstructed = true;
          break;
        }
      }

      const score = (forwardProgress * 2.0) - (dist * 0.5) - (isObstructed ? 3.0 : 0);
      if (score > highestScore) {
        highestScore = score;
        bestPlayer = mate;
      }
    }

    return bestPlayer;
  }

  private distanceToSegment(p: Vector2D, a: Vector2D, b: Vector2D): number {
    const l2 = Vec2.distance(a, b) * Vec2.distance(a, b);
    if (l2 === 0) return Vec2.distance(p, a);
    let t = ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / l2;
    t = Math.max(0, Math.min(1, t));
    return Vec2.distance(p, { x: a.x + t * (b.x - a.x), y: a.y + t * (b.y - a.y) });
  }
}
