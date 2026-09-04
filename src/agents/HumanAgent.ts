import { ActionType, AgentAction, Vector2D } from '../types/football';
import { AgentDecisionContext, IAgent } from './BaseAgent';
import { Vec2 } from '../engine/Vector';
import { SeededRNG } from '../engine/SeededRNG';

export class HumanAgent implements IAgent {
  id = 'human_player';
  name = 'Human User';
  type: 'human' = 'human';

  private pressedKeys = new Set<string>();
  private virtualMovement: Vector2D = { x: 0, y: 0 };
  private pendingAction: AgentAction | null = null;
  private rng: SeededRNG = new SeededRNG(42);

  constructor(seed = 42) {
    this.rng = new SeededRNG(seed);
    if (typeof window !== 'undefined') {
      window.addEventListener('keydown', this.handleKeyDown);
      window.addEventListener('keyup', this.handleKeyUp);
    }
  }

  public setSeed(seed: number): void {
    this.rng.setSeed(seed);
  }

  destroy(): void {
    if (typeof window !== 'undefined') {
      window.removeEventListener('keydown', this.handleKeyDown);
      window.removeEventListener('keyup', this.handleKeyUp);
    }
  }

  private handleKeyDown = (e: KeyboardEvent): void => {
    // Avoid hijacking typing in text inputs or textareas
    if (
      document.activeElement &&
      (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA')
    ) {
      return;
    }

    const key = e.key.toLowerCase();
    this.pressedKeys.add(key);

    if ([' ', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'w', 'a', 's', 'd', 'j', 'k', 'l', 'e'].includes(key)) {
      e.preventDefault();
    }

    // Action triggers on key press
    if (key === 'k') {
      // Shoot
      this.pendingAction = { type: ActionType.SHOT, power: 0.85 };
    } else if (key === 'j' || key === ' ') {
      // Short Pass
      this.pendingAction = { type: ActionType.SHORT_PASS, power: 0.7 };
    } else if (key === 'l') {
      // Long / High Pass
      this.pendingAction = { type: ActionType.HIGH_PASS, power: 0.85 };
    } else if (key === 'e' || key === 'q') {
      // Tackle / Switch Player
      this.pendingAction = { type: ActionType.TACKLE };
    }
  };

  private handleKeyUp = (e: KeyboardEvent): void => {
    this.pressedKeys.delete(e.key.toLowerCase());
  };

  public setVirtualMovement(v: Vector2D): void {
    this.virtualMovement = v;
  }

  public triggerAction(action: AgentAction): void {
    this.pendingAction = action;
  }

  decide(context: AgentDecisionContext): AgentAction {
    const { player, ball } = context;

    // Check if there is an explicit action queued
    if (this.pendingAction) {
      const act = this.pendingAction;
      this.pendingAction = null;

      if (act.type === ActionType.SHOT) {
        // Target opponent goal
        const targetGoalX = player.team === 'left' ? 1.0 : -1.0;
        const rngVal = context.rng ? context.rng.next() : this.rng.next();
        const shootDir = Vec2.sub({ x: targetGoalX, y: (rngVal - 0.5) * 0.08 }, player.position);
        return {
          type: ActionType.SHOT,
          direction: Vec2.normalize(shootDir),
          power: act.power || 0.85,
        };
      }

      if (act.type === ActionType.SHORT_PASS || act.type === ActionType.HIGH_PASS) {
        // Find best open teammate ahead
        const candidates = context.teammates.filter((t) => t.id !== player.id);
        if (candidates.length > 0) {
          // Find teammate closest in direction the player is moving/facing
          const headingDir = Vec2.fromAngle(player.heading);
          let bestMate = candidates[0];
          let bestScore = -999;

          candidates.forEach((tm) => {
            const dirToMate = Vec2.normalize(Vec2.sub(tm.position, player.position));
            const align = Vec2.dot(headingDir, dirToMate);
            const dist = Vec2.distance(player.position, tm.position);
            const score = align * 2 - dist;
            if (score > bestScore) {
              bestScore = score;
              bestMate = tm;
            }
          });

          const passDir = Vec2.normalize(Vec2.sub(bestMate.position, player.position));
          return {
            type: act.type,
            direction: passDir,
            power: Math.min(1.0, Vec2.distance(player.position, bestMate.position) * 2.5 + 0.3),
            targetPlayerId: bestMate.id,
          };
        }
      }

      if (act.type === ActionType.TACKLE) {
        return { type: ActionType.TACKLE };
      }
    }

    // Directional movement from keyboard or virtual joystick
    let dx = this.virtualMovement.x;
    let dy = this.virtualMovement.y;

    if (this.pressedKeys.has('w') || this.pressedKeys.has('arrowup')) dy -= 1;
    if (this.pressedKeys.has('s') || this.pressedKeys.has('arrowdown')) dy += 1;
    if (this.pressedKeys.has('a') || this.pressedKeys.has('arrowleft')) dx -= 1;
    if (this.pressedKeys.has('d') || this.pressedKeys.has('arrowright')) dx += 1;

    const isSprinting = this.pressedKeys.has('shift');

    if (dx !== 0 || dy !== 0) {
      const norm = Vec2.normalize({ x: dx, y: dy });
      return {
        type: isSprinting ? ActionType.SPRINT : ActionType.MOVE,
        direction: norm,
      };
    }

    return { type: ActionType.IDLE };
  }
}
