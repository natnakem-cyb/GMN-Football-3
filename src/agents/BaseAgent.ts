import { ActionType, AgentAction, Ball, Player, TeamSide } from '../types/football';

export interface AgentDecisionContext {
  player: Player;
  teammates: Player[];
  opponents: Player[];
  ball: Ball;
  allPlayers: Player[];
  teamSide: TeamSide;
  controlledPlayerId: string | null;
  matchTime: number;
}

export interface IAgent {
  id: string;
  name: string;
  type: 'human' | 'rule_based' | 'neural' | 'scripted';
  decide(context: AgentDecisionContext): AgentAction;
  reset?(): void;
}
