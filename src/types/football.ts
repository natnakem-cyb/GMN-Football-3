export interface Vector2D {
  x: number;
  y: number;
}

export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export type TeamSide = 'left' | 'right';

export type PlayerRole = 'GK' | 'CB' | 'LB' | 'RB' | 'CDM' | 'CM' | 'LM' | 'RM' | 'CAM' | 'ST' | 'LW' | 'RW';

export interface PlayerStats {
  speed: number;
  stamina: number;
  kickPower: number;
  passingAccuracy: number;
  dribbling: number;
  tackling: number;
}

export enum GameMode {
  Normal = 0,
  KickOff = 1,
  GoalKick = 2,
  FreeKick = 3,
  Corner = 4,
  ThrowIn = 5,
  Penalty = 6,
}

export interface Player {
  id: string;
  name: string;
  number: number;
  team: TeamSide;
  role: PlayerRole;
  position: Vector2D;
  targetPosition: Vector2D;
  velocity: Vector2D;
  heading: number; // in radians
  stamina: number; // 0 to 100
  isSprinting: boolean;
  isDribbling?: boolean;
  stickyDirection?: Vector2D | null;
  isTackling: boolean;
  tackleCooldown: number;
  hasBall: boolean;
  isGoalkeeper: boolean;
  stats: PlayerStats;
  actionState?: string;
  yellowCards: number;
  redCard: boolean;
}

export interface Ball {
  position: Vector3D;
  velocity: Vector3D;
  rotation: Vector3D;
  ownerId: string | null;
  lastOwnerId: string | null;
  lastOwnerTeam: TeamSide | null;
  isInAir: boolean;
  isShotInFlight: boolean;
  trail: Vector3D[];
}

export type MatchStateStatus = 'warmup' | 'kickoff' | 'playing' | 'goal' | 'out_of_bounds' | 'corner' | 'goalkick' | 'halftime' | 'fulltime' | 'paused';

export interface MatchScore {
  left: number;
  right: number;
}

export type FormationType = '4-3-3' | '4-4-2' | '3-5-2' | '5-3-2' | '1-2-1' | '1-1-1' | '1-0';

export interface FormationNode {
  role: PlayerRole;
  xRatio: number; // 0 to 1 relative to own half
  yRatio: number; // 0 to 1 (0 = top/left wing, 0.5 = center, 1 = bottom/right wing)
}

export interface TeamConfig {
  id: string;
  name: string;
  shortName: string;
  color: string;
  accentColor: string;
  textColor: string;
  formation: FormationType;
  controller: 'human' | 'rule_based' | 'neural' | 'scripted' | 'heuristic';
  aiDifficulty: 'easy' | 'medium' | 'hard' | 'master';
  tactics: {
    aggression: number; // 0..1
    pressLine: number; // 0..1
    passingDirectness: number; // 0..1
    width: number; // 0..1
  };
}

export enum ActionType {
  IDLE = 'IDLE',
  MOVE = 'MOVE',
  LONG_PASS = 'LONG_PASS',
  HIGH_PASS = 'HIGH_PASS',
  SHORT_PASS = 'SHORT_PASS',
  SHOT = 'SHOT',
  SPRINT = 'SPRINT',
  RELEASE_DIRECTION = 'RELEASE_DIRECTION',
  RELEASE_SPRINT = 'RELEASE_SPRINT',
  SLIDING = 'SLIDING',
  TACKLE = 'TACKLE',
  DRIBBLE = 'DRIBBLE',
  RELEASE_DRIBBLE = 'RELEASE_DRIBBLE',
  SWITCH_PLAYER = 'SWITCH_PLAYER',
  CLEAR_BALL = 'CLEAR_BALL',
}

export interface AgentAction {
  type: ActionType;
  direction?: Vector2D;
  power?: number; // 0 to 1
  targetPlayerId?: string;
  switchTargetId?: string;
}

export interface MatchEvent {
  id: string;
  timeSeconds: number;
  type: 'goal' | 'shot' | 'shot_saved' | 'shot_missed' | 'pass' | 'interception' | 'tackle' | 'foul' | 'kickoff' | 'out_of_bounds' | 'scenario_complete' | 'scenario_failed';
  team?: TeamSide;
  playerId?: string;
  playerName?: string;
  description: string;
  position?: Vector2D;
}

export interface MatchStats {
  possession: { left: number; right: number }; // percentages
  shots: { left: number; right: number };
  shotsOnTarget: { left: number; right: number };
  passes: { left: number; right: number };
  completedPasses: { left: number; right: number };
  tackles: { left: number; right: number };
  interceptions: { left: number; right: number };
  fouls: { left: number; right: number };
  yellowCards: { left: number; right: number };
  redCards: { left: number; right: number };
  goals: { left: number; right: number };
  possessionHistory: { time: number; left: number; right: number }[];
  shotLocations: { x: number; y: number; team: TeamSide; isGoal: boolean }[];
  heatmapData: { left: Vector2D[]; right: Vector2D[]; ball: Vector2D[] };
}

export interface ReplayFrame {
  tick: number;
  timestamp: number;
  matchTimeSeconds: number;
  ball: {
    position: Vector3D;
    velocity: Vector3D;
    ownerId: string | null;
  };
  players: {
    id: string;
    team: TeamSide;
    position: Vector2D;
    velocity: Vector2D;
    heading: number;
    stamina: number;
    isTackling: boolean;
    hasBall: boolean;
  }[];
  score: MatchScore;
  event?: MatchEvent;
}

export interface ScenarioObjective {
  id: string;
  text: string;
  isCompleted: boolean;
  isFailed: boolean;
}

export interface ScenarioConfig {
  id: string;
  stage: number;
  name: string;
  codeName: string;
  description: string;
  instructions: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Master';
  teamLeftPlayers: number;
  teamRightPlayers: number;
  hasGoalkeeperLeft: boolean;
  hasGoalkeeperRight: boolean;
  timeLimitSeconds: number;
  setup: {
    ball: Vector3D;
    leftPlayers: { role: PlayerRole; pos: Vector2D; isControlled?: boolean }[];
    rightPlayers: { role: PlayerRole; pos: Vector2D }[];
    positionJitter?: number;
  };
  objectives: ScenarioObjective[];
  terminateOnOpponentPossession?: boolean;
  rewards: {
    scoring: number;
    completion: number;
  };
}

export interface RLObservation {
  // GRF SMM / Vector structure
  leftTeamPositions: number[][]; // [x, y] normalized -1..1, -0.42..0.42
  leftTeamVelocities: number[][];
  rightTeamPositions: number[][];
  rightTeamVelocities: number[][];
  ballPosition: [number, number, number]; // [x, y, z]
  ballVelocity: [number, number, number];
  ballOwnedTeam: -1 | 0 | 1; // -1: None, 0: Left, 1: Right
  ballOwnedPlayer: number; // index or -1
  activePlayerIndex: number;
  gameMode: number;
  score: [number, number];
  stepsLeft: number;
  rawVector: number[]; // 115-float standard SMM vector representation
}

export interface RLStepResult {
  observation: RLObservation;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  info: {
    score: MatchScore;
    event?: string;
    checkpointReward: number;
    ballDistanceToGoal: number;
  };
}
