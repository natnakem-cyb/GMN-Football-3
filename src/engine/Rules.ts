import { FormationNode, FormationType, Player, PlayerRole, TeamSide, Vector2D } from '../types/football';

// Google Research Football standard coordinate system:
// Pitch extends from x: -1.0 to 1.0 (length = 2.0), y: -0.42 to 0.42 (width = 0.84)
// Goals are located at x: -1.0 and x: 1.0, between y: -0.07 and 0.07
export const PITCH = {
  minX: -1.0,
  maxX: 1.0,
  minY: -0.42,
  maxY: 0.42,
  width: 2.0,
  height: 0.84,
  
  // Goal specifications
  goalWidth: 0.14, // from y = -0.07 to y = 0.07
  goalMinY: -0.07,
  goalMaxY: 0.07,
  goalHeight: 0.05, // z height
  goalDepth: 0.04,

  // Penalty box specs
  penaltyBoxLength: 0.33,
  penaltyBoxWidth: 0.45,
  goalBoxLength: 0.11,
  goalBoxWidth: 0.22,
  penaltySpotDist: 0.22,
  centerCircleRadius: 0.18,
  cornerArcRadius: 0.03,

  // Physics constants
  ballRadius: 0.015,
  playerRadius: 0.022,
  goalkeeperCatchRadius: 0.05,
};

export const FORMATIONS: Record<FormationType, FormationNode[]> = {
  '4-3-3': [
    { role: 'GK', xRatio: 0.05, yRatio: 0.5 },
    { role: 'LB', xRatio: 0.25, yRatio: 0.15 },
    { role: 'CB', xRatio: 0.22, yRatio: 0.38 },
    { role: 'CB', xRatio: 0.22, yRatio: 0.62 },
    { role: 'RB', xRatio: 0.25, yRatio: 0.85 },
    { role: 'CM', xRatio: 0.48, yRatio: 0.3 },
    { role: 'CM', xRatio: 0.44, yRatio: 0.5 },
    { role: 'CM', xRatio: 0.48, yRatio: 0.7 },
    { role: 'LW', xRatio: 0.75, yRatio: 0.15 },
    { role: 'ST', xRatio: 0.82, yRatio: 0.5 },
    { role: 'RW', xRatio: 0.75, yRatio: 0.85 },
  ],
  '4-4-2': [
    { role: 'GK', xRatio: 0.05, yRatio: 0.5 },
    { role: 'LB', xRatio: 0.25, yRatio: 0.15 },
    { role: 'CB', xRatio: 0.22, yRatio: 0.38 },
    { role: 'CB', xRatio: 0.22, yRatio: 0.62 },
    { role: 'RB', xRatio: 0.25, yRatio: 0.85 },
    { role: 'LM', xRatio: 0.5, yRatio: 0.15 },
    { role: 'CM', xRatio: 0.48, yRatio: 0.38 },
    { role: 'CM', xRatio: 0.48, yRatio: 0.62 },
    { role: 'RM', xRatio: 0.5, yRatio: 0.85 },
    { role: 'ST', xRatio: 0.8, yRatio: 0.4 },
    { role: 'ST', xRatio: 0.8, yRatio: 0.6 },
  ],
  '3-5-2': [
    { role: 'GK', xRatio: 0.05, yRatio: 0.5 },
    { role: 'CB', xRatio: 0.22, yRatio: 0.25 },
    { role: 'CB', xRatio: 0.2, yRatio: 0.5 },
    { role: 'CB', xRatio: 0.22, yRatio: 0.75 },
    { role: 'LM', xRatio: 0.48, yRatio: 0.1 },
    { role: 'CM', xRatio: 0.45, yRatio: 0.35 },
    { role: 'CAM', xRatio: 0.58, yRatio: 0.5 },
    { role: 'CM', xRatio: 0.45, yRatio: 0.65 },
    { role: 'RM', xRatio: 0.48, yRatio: 0.9 },
    { role: 'ST', xRatio: 0.8, yRatio: 0.38 },
    { role: 'ST', xRatio: 0.8, yRatio: 0.62 },
  ],
  '5-3-2': [
    { role: 'GK', xRatio: 0.05, yRatio: 0.5 },
    { role: 'LB', xRatio: 0.25, yRatio: 0.12 },
    { role: 'CB', xRatio: 0.2, yRatio: 0.3 },
    { role: 'CB', xRatio: 0.18, yRatio: 0.5 },
    { role: 'CB', xRatio: 0.2, yRatio: 0.7 },
    { role: 'RB', xRatio: 0.25, yRatio: 0.88 },
    { role: 'CM', xRatio: 0.48, yRatio: 0.3 },
    { role: 'CM', xRatio: 0.46, yRatio: 0.5 },
    { role: 'CM', xRatio: 0.48, yRatio: 0.7 },
    { role: 'ST', xRatio: 0.78, yRatio: 0.4 },
    { role: 'ST', xRatio: 0.78, yRatio: 0.6 },
  ],
  '1-2-1': [
    // 5v5 formation
    { role: 'GK', xRatio: 0.05, yRatio: 0.5 },
    { role: 'CB', xRatio: 0.25, yRatio: 0.5 },
    { role: 'LM', xRatio: 0.5, yRatio: 0.2 },
    { role: 'RM', xRatio: 0.5, yRatio: 0.8 },
    { role: 'ST', xRatio: 0.75, yRatio: 0.5 },
  ],
  '1-1-1': [
    // 3v3 / 4v4 formation
    { role: 'GK', xRatio: 0.05, yRatio: 0.5 },
    { role: 'CB', xRatio: 0.3, yRatio: 0.5 },
    { role: 'CM', xRatio: 0.55, yRatio: 0.5 },
    { role: 'ST', xRatio: 0.8, yRatio: 0.5 },
  ],
  '1-0': [
    { role: 'GK', xRatio: 0.05, yRatio: 0.5 },
  ],
};

export function getFormationPositions(
  formation: FormationType,
  team: TeamSide,
  numPlayers: number
): { role: PlayerRole; pos: Vector2D }[] {
  const nodes = FORMATIONS[formation] || FORMATIONS['4-3-3'];
  const activeNodes = nodes.slice(0, numPlayers);
  
  return activeNodes.map((node) => {
    let x: number;
    let y: number;
    
    if (team === 'left') {
      // Left team occupies [-1.0 .. 0.0] primarily, stretching to midfield
      x = PITCH.minX + node.xRatio * (PITCH.width * 0.6);
      y = PITCH.minY + node.yRatio * PITCH.height;
    } else {
      // Right team occupies [1.0 .. 0.0]
      x = PITCH.maxX - node.xRatio * (PITCH.width * 0.6);
      y = PITCH.maxY - node.yRatio * PITCH.height;
    }
    
    return {
      role: node.role,
      pos: { x, y },
    };
  });
}

/**
 * Computes the x-coordinate of the offside line for the defending team.
 * The offside line is the position of the second-deepest defending player.
 */
export function computeOffsideLineX(defendingTeamPlayers: Player[], defendingTeam: TeamSide): number {
  if (defendingTeamPlayers.length === 0) return 0;
  if (defendingTeamPlayers.length === 1) return defendingTeamPlayers[0].position.x;

  if (defendingTeam === 'right') {
    // Defends x ≈ maxX (1.0). Sort descending by x (deepest defenders first).
    const sorted = [...defendingTeamPlayers].sort((a, b) => b.position.x - a.position.x);
    return sorted[1].position.x;
  } else {
    // Defends x ≈ minX (-1.0). Sort ascending by x (deepest defenders first).
    const sorted = [...defendingTeamPlayers].sort((a, b) => a.position.x - b.position.x);
    return sorted[1].position.x;
  }
}
