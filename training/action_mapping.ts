import { ActionType, AgentAction, Vector2D } from '../src/types/football';

/**
 * Standard Discrete Action Space Mapping for GMN-Football RL Environment.
 * Size: Discrete(19) matching Google Research Football (GRF) 19-action set.
 *
 * ID | NAME                      | EFFECT
 * ---+---------------------------+---------------------------------------------------
 *  0 | action_idle               | No-op; sticky actions unaffected
 *  1 | action_left               | Move left (sticky)
 *  2 | action_top_left           | Move top-left (sticky)
 *  3 | action_top                | Move top (sticky)
 *  4 | action_top_right          | Move top-right (sticky)
 *  5 | action_right              | Move right (sticky)
 *  6 | action_bottom_right       | Move bottom-right (sticky)
 *  7 | action_bottom             | Move bottom (sticky)
 *  8 | action_bottom_left        | Move bottom-left (sticky)
 *  9 | action_long_pass          | Long pass, target auto-determined by movement direction
 * 10 | action_high_pass          | High pass, same targeting logic
 * 11 | action_short_pass         | Short pass, same targeting logic
 * 12 | action_shot               | Shot toward opponent goal
 * 13 | action_sprint             | Start sprinting (sticky; faster, worse ball handling)
 * 14 | action_release_direction  | Clear current movement direction
 * 15 | action_release_sprint     | Stop sprinting
 * 16 | action_sliding            | Slide tackle (only effective without ball possession)
 * 17 | action_dribble            | Start dribbling (sticky; slower, harder to dispossess)
 * 18 | action_release_dribble    | Stop dribbling
 */
export const ACTION_SPACE_SIZE = 19;

const SQRT_HALF = 0.7071067811865476;

export function mapDiscreteAction(actionIdx: number, currentHeading = 0): AgentAction {
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
      return { type: ActionType.IDLE };
  }
}
