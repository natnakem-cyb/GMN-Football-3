import { ActionType, AgentAction, Vector2D } from '../types/football';
import { AgentDecisionContext, IAgent } from './BaseAgent';
import { Vec2 } from '../engine/Vector';

export class ScriptedScenarioAgent implements IAgent {
  id: string;
  name: string;
  type: 'scripted' = 'scripted';
  scenarioType: 'defender_contain' | 'keeper_dive' | 'static_obstacle' | 'passive_patrol';

  constructor(
    id = 'scripted_ai',
    name = 'Scenario Bot',
    scenarioType: 'defender_contain' | 'keeper_dive' | 'static_obstacle' | 'passive_patrol' = 'defender_contain'
  ) {
    this.id = id;
    this.name = name;
    this.scenarioType = scenarioType;
  }

  decide(context: AgentDecisionContext): AgentAction {
    const { player, ball, opponents } = context;

    if (this.scenarioType === 'static_obstacle') {
      return { type: ActionType.IDLE };
    }

    if (this.scenarioType === 'passive_patrol') {
      // Move up and down on Y axis
      const patrolY = Math.sin(context.matchTime * 2) * 0.2;
      const targetPos: Vector2D = { x: player.position.x, y: patrolY };
      return {
        type: ActionType.MOVE,
        direction: Vec2.normalize(Vec2.sub(targetPos, player.position)),
      };
    }

    if (this.scenarioType === 'keeper_dive') {
      const ballPos2D: Vector2D = { x: ball.position.x, y: ball.position.y };
      const dist = Vec2.distance(player.position, ballPos2D);

      if (dist < 0.08 && (ball.velocity.x !== 0 || ball.velocity.y !== 0)) {
        return { type: ActionType.TACKLE };
      }

      // Track ball Y position along goal line
      const targetPos: Vector2D = {
        x: player.position.x,
        y: Math.max(-0.06, Math.min(0.06, ball.position.y * 0.9)),
      };
      return {
        type: ActionType.MOVE,
        direction: Vec2.normalize(Vec2.sub(targetPos, player.position)),
      };
    }

    // Default: 'defender_contain'
    const ballPos2D: Vector2D = { x: ball.position.x, y: ball.position.y };
    const distToBall = Vec2.distance(player.position, ballPos2D);

    if (distToBall < 0.06) {
      return { type: ActionType.TACKLE };
    }

    // Shadow opponent between ball and goal
    const goalCenter: Vector2D = { x: player.team === 'left' ? -1.0 : 1.0, y: 0 };
    const midpoint = Vec2.lerp(ballPos2D, goalCenter, 0.4);

    return {
      type: ActionType.MOVE,
      direction: Vec2.normalize(Vec2.sub(midpoint, player.position)),
    };
  }
}
