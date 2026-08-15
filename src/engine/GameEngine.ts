import {
  ActionType,
  AgentAction,
  Ball,
  FormationType,
  GameMode,
  MatchEvent,
  MatchScore,
  MatchStateStatus,
  MatchStats,
  Player,
  PlayerRole,
  ReplayFrame,
  RLObservation,
  RLStepResult,
  ScenarioConfig,
  TeamConfig,
  TeamSide,
  Vector2D,
  Vector3D,
} from '../types/football';
import { PITCH, getFormationPositions } from './Rules';
import { PhysicsEngine } from './Physics';
import { Vec2, Vec3 } from './Vector';
import { ObservationEncoder } from './ObservationEncoder';

export class GameEngine {
  public ball: Ball;
  public players: Player[] = [];
  public score: MatchScore = { left: 0, right: 0 };
  public status: MatchStateStatus = 'playing';
  public gameMode: GameMode = GameMode.KickOff;
  public matchTimeSeconds = 0;
  public tickCount = 0;
  public activeScenario: ScenarioConfig | null = null;
  public controlledPlayerId: string | null = null;

  public teamLeftConfig: TeamConfig = {
    id: 'team_left',
    name: 'FC Alpha (Blue)',
    shortName: 'ALP',
    color: '#3b82f6',
    accentColor: '#60a5fa',
    textColor: '#ffffff',
    formation: '4-3-3',
    controller: 'human',
    aiDifficulty: 'medium',
    tactics: { aggression: 0.7, pressLine: 0.6, passingDirectness: 0.6, width: 0.8 },
  };

  public teamRightConfig: TeamConfig = {
    id: 'team_right',
    name: 'Red Devils (Red)',
    shortName: 'RED',
    color: '#ef4444',
    accentColor: '#f87171',
    textColor: '#ffffff',
    formation: '4-3-3',
    controller: 'rule_based',
    aiDifficulty: 'medium',
    tactics: { aggression: 0.7, pressLine: 0.6, passingDirectness: 0.6, width: 0.8 },
  };

  public stats: MatchStats = {
    possession: { left: 50, right: 50 },
    shots: { left: 0, right: 0 },
    shotsOnTarget: { left: 0, right: 0 },
    passes: { left: 0, right: 0 },
    completedPasses: { left: 0, right: 0 },
    tackles: { left: 0, right: 0 },
    interceptions: { left: 0, right: 0 },
    goals: { left: 0, right: 0 },
    possessionHistory: [],
    shotLocations: [],
    heatmapData: { left: [], right: [], ball: [] },
  };

  public replayBuffer: ReplayFrame[] = [];
  public maxReplayFrames = 3000;
  public events: MatchEvent[] = [];

  private possessionTicks = { left: 0, right: 0 };
  private goalResetTimer = 0;
  private currentPassTracking: { passerId: string; team: TeamSide; targetId?: string } | null = null;

  constructor() {
    this.ball = this.createDefaultBall();
    this.initDefaultMatch('4-3-3', '4-3-3', 11);
  }

  private createDefaultBall(): Ball {
    return {
      position: { x: 0, y: 0, z: 0 },
      velocity: { x: 0, y: 0, z: 0 },
      rotation: { x: 0, y: 0, z: 0 },
      ownerId: null,
      lastOwnerId: null,
      lastOwnerTeam: null,
      isInAir: false,
      trail: [],
    };
  }

  public initDefaultMatch(
    leftFormation: FormationType = '4-3-3',
    rightFormation: FormationType = '4-3-3',
    numPlayers = 11
  ): void {
    this.players = [];
    this.ball = this.createDefaultBall();
    this.score = { left: 0, right: 0 };
    this.matchTimeSeconds = 0;
    this.tickCount = 0;
    this.status = 'playing';
    this.events = [];
    this.replayBuffer = [];
    this.possessionTicks = { left: 0, right: 0 };
    this.activeScenario = null;

    this.teamLeftConfig.formation = leftFormation;
    this.teamRightConfig.formation = rightFormation;
    this.gameMode = GameMode.KickOff;

    const leftPositions = getFormationPositions(leftFormation, 'left', numPlayers);
    const rightPositions = getFormationPositions(rightFormation, 'right', numPlayers);

    // Create Left Team Players
    leftPositions.forEach((posData, idx) => {
      const player: Player = {
        id: `left_${idx + 1}`,
        name: idx === 0 ? 'Goalkeeper A' : `Player #${idx + 1}`,
        number: idx + 1,
        team: 'left',
        role: posData.role,
        position: { ...posData.pos },
        targetPosition: { ...posData.pos },
        velocity: { x: 0, y: 0 },
        heading: 0, // facing right
        stamina: 100,
        isSprinting: false,
        isTackling: false,
        tackleCooldown: 0,
        hasBall: false,
        isGoalkeeper: posData.role === 'GK',
        stats: {
          speed: 82 + (idx % 4) * 3,
          stamina: 85,
          kickPower: 80 + (idx % 3) * 5,
          passingAccuracy: 84,
          dribbling: 82,
          tackling: posData.role === 'CB' || posData.role === 'GK' ? 88 : 70,
        },
        yellowCards: 0,
        redCard: false,
      };
      this.players.push(player);
    });

    // Create Right Team Players
    rightPositions.forEach((posData, idx) => {
      const player: Player = {
        id: `right_${idx + 1}`,
        name: idx === 0 ? 'Goalkeeper B' : `Player #${idx + 1}`,
        number: idx + 1,
        team: 'right',
        role: posData.role,
        position: { ...posData.pos },
        targetPosition: { ...posData.pos },
        velocity: { x: 0, y: 0 },
        heading: Math.PI, // facing left
        stamina: 100,
        isSprinting: false,
        isTackling: false,
        tackleCooldown: 0,
        hasBall: false,
        isGoalkeeper: posData.role === 'GK',
        stats: {
          speed: 80 + (idx % 4) * 3,
          stamina: 85,
          kickPower: 80 + (idx % 3) * 5,
          passingAccuracy: 82,
          dribbling: 80,
          tackling: posData.role === 'CB' || posData.role === 'GK' ? 88 : 70,
        },
        yellowCards: 0,
        redCard: false,
      };
      this.players.push(player);
    });

    // Default controlled player is first striker/midfielder of left team
    const defaultControllable = this.players.find((p) => p.team === 'left' && !p.isGoalkeeper);
    this.controlledPlayerId = defaultControllable ? defaultControllable.id : this.players[0]?.id || null;

    // Ball starts at center
    this.ball.position = { x: 0, y: 0, z: 0 };
    this.ball.velocity = { x: 0, y: 0, z: 0 };

    this.recordEvent('kickoff', 'Match Kickoff', { x: 0, y: 0 });
  }

  public loadScenario(scenario: ScenarioConfig): void {
    this.activeScenario = {
      ...scenario,
      objectives: scenario.objectives.map((o) => ({ ...o, isCompleted: false, isFailed: false })),
    };
    this.players = [];
    this.ball = this.createDefaultBall();
    this.score = { left: 0, right: 0 };
    this.matchTimeSeconds = 0;
    this.tickCount = 0;
    this.status = 'playing';
    this.events = [];
    this.replayBuffer = [];
    this.possessionTicks = { left: 0, right: 0 };
    this.gameMode = scenario.id.startsWith('academy') ? GameMode.Normal : GameMode.KickOff;

    this.ball.position = { ...scenario.setup.ball };

    // Setup left players
    scenario.setup.leftPlayers.forEach((pData, idx) => {
      const player: Player = {
        id: `left_${idx + 1}`,
        name: `Trainee #${idx + 1}`,
        number: idx === 0 ? 10 : idx + 1,
        team: 'left',
        role: pData.role,
        position: { ...pData.pos },
        targetPosition: { ...pData.pos },
        velocity: { x: 0, y: 0 },
        heading: 0,
        stamina: 100,
        isSprinting: false,
        isTackling: false,
        tackleCooldown: 0,
        hasBall: false,
        isGoalkeeper: pData.role === 'GK',
        stats: {
          speed: 88,
          stamina: 95,
          kickPower: 86,
          passingAccuracy: 88,
          dribbling: 90,
          tackling: 75,
        },
        yellowCards: 0,
        redCard: false,
      };
      this.players.push(player);

      if (pData.isControlled || idx === 0) {
        this.controlledPlayerId = player.id;
      }
    });

    // Setup right players
    scenario.setup.rightPlayers.forEach((pData, idx) => {
      const player: Player = {
        id: `right_${idx + 1}`,
        name: pData.role === 'GK' ? 'Academy Keeper' : `Opponent Defender #${idx + 1}`,
        number: idx === 0 ? 1 : idx + 2,
        team: 'right',
        role: pData.role,
        position: { ...pData.pos },
        targetPosition: { ...pData.pos },
        velocity: { x: 0, y: 0 },
        heading: Math.PI,
        stamina: 100,
        isSprinting: false,
        isTackling: false,
        tackleCooldown: 0,
        hasBall: false,
        isGoalkeeper: pData.role === 'GK',
        stats: {
          speed: 80,
          stamina: 85,
          kickPower: 80,
          passingAccuracy: 75,
          dribbling: 70,
          tackling: 84,
        },
        yellowCards: 0,
        redCard: false,
      };
      this.players.push(player);
    });

    this.recordEvent('kickoff', `Started Scenario: ${scenario.name}`, { x: this.ball.position.x, y: this.ball.position.y });
  }

  public step(
    actionMap: Map<string, AgentAction>,
    dt = 1 / 60
  ): RLStepResult {
    const prevBallX = this.ball.position.x;
    let goalScoredThisTick: TeamSide | null = null;
    let eventDescription: string | undefined;

    if (this.status !== 'paused') {
      this.tickCount++;
      this.matchTimeSeconds += dt;

      // Handle goal celebration cooldown
      if (this.goalResetTimer > 0) {
        this.goalResetTimer--;
        if (this.goalResetTimer <= 0) {
          if (this.activeScenario) {
            this.loadScenario(this.activeScenario);
          } else {
            this.resetToKickoff();
          }
        }
      } else {
        // 1. Process Actions for each player
        this.players.forEach((player) => {
          const action = actionMap.get(player.id) || { type: ActionType.IDLE };
          this.applyPlayerAction(player, action);
        });

        // 2. Physics updates for players
        this.players.forEach((player) => {
          PhysicsEngine.updatePlayer(player, this.ball, dt);
        });

        // 3. Ball physics
        PhysicsEngine.updateBall(this.ball, this.players, dt);

        // 4. Ball pickup / interception checks
        this.checkBallPossession();

        // 5. Goal & boundary checks
        goalScoredThisTick = this.checkGoalAndBoundaries();

        // 6. Scenario objective evaluations
        this.evaluateScenarioConditions();

        // 7. Update statistics & heatmaps (every 10 ticks)
        if (this.tickCount % 10 === 0) {
          this.updateStatistics();
        }
      }

      // Record snapshot to replay buffer
      this.recordReplayFrame();
    }

    // Generate RL observation & reward
    const observation = ObservationEncoder.encode(
      this.players,
      this.ball,
      this.controlledPlayerId,
      this.score,
      this.tickCount,
      this.activeScenario ? this.activeScenario.timeLimitSeconds * 60 : 3600,
      this.gameMode
    );

    const { reward, checkpoint } = ObservationEncoder.computeReward(
      prevBallX,
      this.ball.position.x,
      goalScoredThisTick,
      'left'
    );

    const isGoalScored = this.score.left > 0 || this.score.right > 0 || this.status === 'goal';
    const isOpponentPossession = Boolean(
      this.activeScenario?.terminateOnOpponentPossession &&
      this.ball.ownerId &&
      this.players.find((p) => p.id === this.ball.ownerId)?.team === 'right'
    );
    const isTerminated = this.status === 'fulltime' || isGoalScored || isOpponentPossession;
    const isTruncated = this.activeScenario ? this.matchTimeSeconds >= this.activeScenario.timeLimitSeconds : false;

    return {
      observation,
      reward,
      terminated: isTerminated,
      truncated: isTruncated,
      info: {
        score: { ...this.score },
        event: eventDescription,
        checkpointReward: checkpoint,
        ballDistanceToGoal: Vec2.distance({ x: this.ball.position.x, y: this.ball.position.y }, { x: 1.0, y: 0 }),
      },
    };
  }

  private applyPlayerAction(player: Player, action: AgentAction): void {
    // 1. Update sticky flags
    if (action.type === ActionType.SPRINT) {
      player.isSprinting = true;
    } else if (action.type === ActionType.RELEASE_SPRINT) {
      player.isSprinting = false;
    }

    if (action.type === ActionType.DRIBBLE) {
      player.isDribbling = true;
    } else if (action.type === ActionType.RELEASE_DRIBBLE) {
      player.isDribbling = false;
    }

    if (action.type === ActionType.MOVE) {
      if (action.direction) {
        player.stickyDirection = { ...action.direction };
      }
    } else if (action.type === ActionType.RELEASE_DIRECTION) {
      player.stickyDirection = null;
    }

    // If active action is taken during KickOff, transition to Normal mode
    if (this.gameMode === GameMode.KickOff && action.type !== ActionType.IDLE) {
      this.gameMode = GameMode.Normal;
    }

    // 2. Handle specific action types
    switch (action.type) {
      case ActionType.MOVE:
        if (action.direction) {
          const speedMultiplier = player.isSprinting ? 1.3 : player.isDribbling ? 0.6 : 1.0;
          const stepDist = (player.stats.speed / 100) * 0.15 * speedMultiplier;
          player.targetPosition = {
            x: player.position.x + action.direction.x * stepDist,
            y: player.position.y + action.direction.y * stepDist,
          };
          player.heading = Vec2.angle(action.direction);
        }
        break;

      case ActionType.RELEASE_DIRECTION:
        player.targetPosition = { ...player.position };
        player.velocity = { x: 0, y: 0 };
        break;

      case ActionType.DRIBBLE:
        {
          const dribbleDir = action.direction || (player.stickyDirection ? player.stickyDirection : Vec2.fromAngle(player.heading));
          const stepDist = (player.stats.speed / 100) * 0.08;
          player.targetPosition = {
            x: player.position.x + dribbleDir.x * stepDist,
            y: player.position.y + dribbleDir.y * stepDist,
          };
          player.heading = Vec2.angle(dribbleDir);
        }
        break;

      case ActionType.SHORT_PASS:
        if (player.hasBall || this.ball.ownerId === player.id) {
          const dir = action.direction || (player.stickyDirection ? player.stickyDirection : Vec2.fromAngle(player.heading));
          const power = action.power || 0.75;
          PhysicsEngine.kickBall(this.ball, player, dir, power, 0);

          this.stats.passes[player.team]++;
          this.currentPassTracking = {
            passerId: player.id,
            team: player.team,
            targetId: action.targetPlayerId,
          };

          this.recordEvent('pass', `${player.name} played short pass`, player.position, player.team);
        }
        break;

      case ActionType.LONG_PASS:
      case ActionType.HIGH_PASS:
        if (player.hasBall || this.ball.ownerId === player.id) {
          const dir = action.direction || (player.stickyDirection ? player.stickyDirection : Vec2.fromAngle(player.heading));
          const power = action.power || (action.type === ActionType.LONG_PASS ? 1.0 : 0.85);
          const loft = action.type === ActionType.LONG_PASS ? 0.35 : 0.45;
          PhysicsEngine.kickBall(this.ball, player, dir, power, loft);

          this.stats.passes[player.team]++;
          this.currentPassTracking = {
            passerId: player.id,
            team: player.team,
            targetId: action.targetPlayerId,
          };

          this.recordEvent('pass', `${player.name} launched ${action.type === ActionType.LONG_PASS ? 'long' : 'high'} pass`, player.position, player.team);
        }
        break;

      case ActionType.SHOT:
        if (player.hasBall || this.ball.ownerId === player.id) {
          const opponentGoalX = player.team === 'left' ? 1.0 : -1.0;
          const defaultDir = Vec2.sub({ x: opponentGoalX, y: 0.0 }, player.position);
          const dir = action.direction || Vec2.normalize(defaultDir);
          const power = action.power || 0.95;

          PhysicsEngine.kickBall(this.ball, player, dir, power, 0.15);

          this.stats.shots[player.team]++;
          this.stats.shotsOnTarget[player.team]++;
          this.stats.shotLocations.push({
            x: player.position.x,
            y: player.position.y,
            team: player.team,
            isGoal: false,
          });

          this.recordEvent('shot', `${player.name} fired a shot at goal!`, player.position, player.team);
        }
        break;

      case ActionType.SLIDING:
      case ActionType.TACKLE:
        if (!player.hasBall && this.ball.ownerId !== player.id) {
          const success = PhysicsEngine.executeTackle(player, this.ball, this.players);
          if (success) {
            this.stats.tackles[player.team]++;
            this.recordEvent('tackle', `${player.name} executed a clean slide tackle!`, player.position, player.team);
          }
        }
        break;

      case ActionType.SWITCH_PLAYER:
        if (player.team === 'left') {
          // Switch to teammate closest to ball
          const teammates = this.players.filter((p) => p.team === 'left' && !p.isGoalkeeper);
          const ballPos2D = { x: this.ball.position.x, y: this.ball.position.y };
          let nearest = teammates[0];
          let minDist = 999;
          teammates.forEach((tm) => {
            if (tm.id !== this.controlledPlayerId) {
              const d = Vec2.distance(tm.position, ballPos2D);
              if (d < minDist) {
                minDist = d;
                nearest = tm;
              }
            }
          });
          if (nearest) {
            this.controlledPlayerId = nearest.id;
          }
        }
        break;

      case ActionType.IDLE:
      case ActionType.SPRINT:
      case ActionType.RELEASE_SPRINT:
      case ActionType.RELEASE_DRIBBLE:
      default:
        if (player.stickyDirection) {
          const speedMultiplier = player.isSprinting ? 1.3 : player.isDribbling ? 0.6 : 1.0;
          const stepDist = (player.stats.speed / 100) * 0.15 * speedMultiplier;
          player.targetPosition = {
            x: player.position.x + player.stickyDirection.x * stepDist,
            y: player.position.y + player.stickyDirection.y * stepDist,
          };
          player.heading = Vec2.angle(player.stickyDirection);
        }
        break;
    }
  }

  private checkBallPossession(): void {
    if (this.ball.isInAir && this.ball.position.z > 0.04) {
      return; // Ball too high to be controlled on ground
    }

    const ballPos2D: Vector2D = { x: this.ball.position.x, y: this.ball.position.y };

    for (const player of this.players) {
      const dist = Vec2.distance(player.position, ballPos2D);
      const catchRadius = player.isGoalkeeper ? PITCH.goalkeeperCatchRadius : PhysicsEngine.BALL_CONTROL_DIST;

      if (dist < catchRadius && !player.isTackling) {
        // If ball was unowned or changing owner
        if (this.ball.ownerId !== player.id) {
          // If a pass was in progress
          if (this.currentPassTracking) {
            if (this.currentPassTracking.team === player.team) {
              if (this.currentPassTracking.passerId !== player.id) {
                this.stats.completedPasses[player.team]++;
              }
            } else {
              this.stats.interceptions[player.team]++;
              this.recordEvent('interception', `${player.name} intercepted the ball`, player.position, player.team);
            }
            this.currentPassTracking = null;
          }

          // Assign possession
          this.players.forEach((p) => (p.hasBall = false));
          player.hasBall = true;
          this.ball.ownerId = player.id;
          this.ball.lastOwnerId = player.id;
          this.ball.lastOwnerTeam = player.team;

          // If left team gained possession and controlled player wasn't active
          if (player.team === 'left' && this.teamLeftConfig.controller === 'human') {
            this.controlledPlayerId = player.id;
          }
        }
        break;
      }
    }
  }

  private checkGoalAndBoundaries(): TeamSide | null {
    const { x, y } = this.ball.position;

    // Check Right Goal (Left team scores)
    if (x >= PITCH.maxX) {
      if (y >= PITCH.goalMinY && y <= PITCH.goalMaxY) {
        this.score.left++;
        this.stats.goals.left++;
        if (this.stats.shotLocations.length > 0) {
          this.stats.shotLocations[this.stats.shotLocations.length - 1].isGoal = true;
        }
        this.recordEvent('goal', `GOAL! Team Left scored! (${this.score.left} - ${this.score.right})`, { x, y }, 'left');
        this.status = 'goal';
        this.goalResetTimer = 100; // ~1.6s pause
        return 'left';
      } else {
        this.handleOutOfBounds('goal_kick_right');
      }
    }

    // Check Left Goal (Right team scores)
    if (x <= PITCH.minX) {
      if (y >= PITCH.goalMinY && y <= PITCH.goalMaxY) {
        this.score.right++;
        this.stats.goals.right++;
        if (this.stats.shotLocations.length > 0) {
          this.stats.shotLocations[this.stats.shotLocations.length - 1].isGoal = true;
        }
        this.recordEvent('goal', `GOAL! Team Right scored! (${this.score.left} - ${this.score.right})`, { x, y }, 'right');
        this.status = 'goal';
        this.goalResetTimer = 100;
        return 'right';
      } else {
        this.handleOutOfBounds('goal_kick_left');
      }
    }

    // Touchlines (Y bounds)
    if (y < PITCH.minY || y > PITCH.maxY) {
      this.handleOutOfBounds('throw_in');
    }

    return null;
  }

  private handleOutOfBounds(type: string): void {
    this.ball.velocity = { x: 0, y: 0, z: 0 };
    this.ball.position.x = Math.max(PITCH.minX + 0.05, Math.min(PITCH.maxX - 0.05, this.ball.position.x));
    this.ball.position.y = Math.max(PITCH.minY + 0.02, Math.min(PITCH.maxY - 0.02, this.ball.position.y));
    this.ball.position.z = 0;
    this.ball.ownerId = null;
  }

  public resetToKickoff(): void {
    this.status = 'playing';
    this.gameMode = GameMode.KickOff;
    this.ball = this.createDefaultBall();
    this.players.forEach((p) => {
      p.hasBall = false;
      p.velocity = { x: 0, y: 0 };
      p.position = { ...p.targetPosition };
    });
  }

  private evaluateScenarioConditions(): void {
    if (!this.activeScenario) return;

    // Check time limit
    if (this.matchTimeSeconds >= this.activeScenario.timeLimitSeconds) {
      const timeObj = this.activeScenario.objectives.find((o) => o.id === 'within_time');
      if (timeObj && !timeObj.isCompleted) {
        timeObj.isFailed = true;
      }
    }

    // Goal scoring objective
    if (this.score.left > 0) {
      const scoreObj = this.activeScenario.objectives.find((o) => o.id === 'score_goal' || o.id === 'win_match');
      if (scoreObj) {
        scoreObj.isCompleted = true;
      }
      const timeObj = this.activeScenario.objectives.find((o) => o.id === 'within_time');
      if (timeObj && this.matchTimeSeconds <= this.activeScenario.timeLimitSeconds) {
        timeObj.isCompleted = true;
      }
    }

    // Passing objective
    if (this.stats.completedPasses.left >= 1) {
      const passObj = this.activeScenario.objectives.find((o) => o.id === 'complete_pass');
      if (passObj) passObj.isCompleted = true;
    }
    if (this.stats.completedPasses.left >= 2) {
      const triObj = this.activeScenario.objectives.find((o) => o.id === 'create_triangle');
      if (triObj) triObj.isCompleted = true;
    }
  }

  private updateStatistics(): void {
    if (this.ball.ownerId) {
      const owner = this.players.find((p) => p.id === this.ball.ownerId);
      if (owner) {
        this.possessionTicks[owner.team]++;
      }
    }

    const totalPossTicks = this.possessionTicks.left + this.possessionTicks.right;
    if (totalPossTicks > 0) {
      this.stats.possession.left = Math.round((this.possessionTicks.left / totalPossTicks) * 100);
      this.stats.possession.right = 100 - this.stats.possession.left;
    }

    // Possession history for recharts
    if (this.tickCount % 60 === 0) {
      this.stats.possessionHistory.push({
        time: Math.round(this.matchTimeSeconds),
        left: this.stats.possession.left,
        right: this.stats.possession.right,
      });
      if (this.stats.possessionHistory.length > 30) {
        this.stats.possessionHistory.shift();
      }
    }

    // Heatmap positions sampling
    this.stats.heatmapData.ball.push({ x: this.ball.position.x, y: this.ball.position.y });
    if (this.stats.heatmapData.ball.length > 100) {
      this.stats.heatmapData.ball.shift();
    }
  }

  private recordReplayFrame(): void {
    if (this.replayBuffer.length >= this.maxReplayFrames) {
      this.replayBuffer.shift();
    }

    const frame: ReplayFrame = {
      tick: this.tickCount,
      timestamp: Date.now(),
      matchTimeSeconds: this.matchTimeSeconds,
      ball: {
        position: { ...this.ball.position },
        velocity: { ...this.ball.velocity },
        ownerId: this.ball.ownerId,
      },
      players: this.players.map((p) => ({
        id: p.id,
        team: p.team,
        position: { ...p.position },
        velocity: { ...p.velocity },
        heading: p.heading,
        stamina: p.stamina,
        isTackling: p.isTackling,
        hasBall: p.hasBall,
      })),
      score: { ...this.score },
      event: this.events[this.events.length - 1],
    };

    this.replayBuffer.push(frame);
  }

  public recordEvent(
    type: MatchEvent['type'],
    description: string,
    position?: Vector2D,
    team?: TeamSide,
    playerId?: string
  ): void {
    const event: MatchEvent = {
      id: `evt_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
      timeSeconds: this.matchTimeSeconds,
      type,
      team,
      playerId,
      description,
      position,
    };
    this.events.push(event);
    if (this.events.length > 50) {
      this.events.shift();
    }
  }
}
