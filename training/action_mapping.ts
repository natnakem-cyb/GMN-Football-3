import { ActionType, AgentAction, Vector2D } from '../src/types/football';
import { ACTION_SPACE_SIZE, ACTION_SCHEMA_VERSION } from '../src/engine/Contract';

export { ACTION_SPACE_SIZE, ACTION_SCHEMA_VERSION };

/**
 * Standard Canonical Discrete Action Space Mapping for GMN-Football-3 RL Environment.
 * Size: Discrete(19) matching GMN-Football-3 authoritative action specifications.
 *
 * ID | NAME                      | EFFECT (GMN-Football Semantics)
 * ---+---------------------------+---------------------------------------------------
 *  0 | action_idle               | No-op; existing sticky movement/dribble/sprint unaffected
 *  1 | action_left               | Move left (-x direction) (sticky)
 *  2 | action_top_left           | Move top-left (-x, -y direction) (sticky)
 *  3 | action_top                | Move top (-y direction) (sticky)
 *  4 | action_top_right          | Move top-right (+x, -y direction) (sticky)
 *  5 | action_right              | Move right (+x direction) (sticky)
 *  6 | action_bottom_right       | Move bottom-right (+x, +y direction) (sticky)
 *  7 | action_bottom             | Move bottom (+y direction) (sticky)
 *  8 | action_bottom_left        | Move bottom-left (-x, +y direction) (sticky)
 *  9 | action_long_pass          | Long pass with loft 0.35, power 1.0
 * 10 | action_high_pass          | High lobbed pass with loft 0.45, power 0.85
 * 11 | action_short_pass         | Ground pass with loft 0.0, power 0.75
 * 12 | action_shot               | Direct shot on opponent goal with loft 0.15, power 0.95
 * 13 | action_sprint             | Start sprinting (sticky; 1.3x speed, faster stamina drain)
 * 14 | action_release_direction  | Stop active directional movement (clears target/velocity)
 * 15 | action_release_sprint     | Stop sprinting (sticky sprint released)
 * 16 | action_sliding            | Execute slide tackle / dispossession attempt
 * 17 | action_dribble            | Start close dribbling (sticky; 0.6x speed, higher control)
 * 18 | action_release_dribble    | Stop close dribbling (sticky dribble released)
 */

const SQRT_HALF = 0.7071067811865476;

export function mapDiscreteAction(actionIdx: number): AgentAction {
  if (typeof actionIdx !== 'number' || !Number.isInteger(actionIdx) || actionIdx < 0 || actionIdx >= ACTION_SPACE_SIZE) {
    throw new Error(
      `[GMN Action Mapping Error] Invalid action index: ${actionIdx}. Expected integer in range [0, ${ACTION_SPACE_SIZE - 1}].`
    );
  }

  switch (actionIdx) {
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
    case 16: // SLIDING
      return { type: ActionType.TACKLE };
    case 17: // DRIBBLE
      return { type: ActionType.DRIBBLE };
    case 18: // RELEASE_DRIBBLE
      return { type: ActionType.RELEASE_DRIBBLE };
    default:
      throw new Error(`[GMN Action Mapping Error] Unhandled action index: ${actionIdx}`);
  }
}

