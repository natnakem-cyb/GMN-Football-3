import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ActionType, AgentAction } from '../src/types/football';
import { mapDiscreteAction, ACTION_SPACE_SIZE } from './action_mapping';
import { RuleBasedAgent } from '../src/agents/RuleBasedAgent';

interface FeatureStats {
  min: number;
  max: number;
  sum: number;
  count: number;
}

export function runObservationAndActionAudit(totalSteps = 100000) {
  console.log('==================================================');
  console.log('1. OBSERVATION SPACE AUDIT (100,000 steps)');
  console.log('==================================================');

  const engine = new GameEngine();
  const scenario = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_empty_goal')!;
  engine.loadScenario(scenario);

  const featureStats: FeatureStats[] = Array.from({ length: 115 }, () => ({
    min: Infinity,
    max: -Infinity,
    sum: 0,
    count: 0,
  }));

  let globalMin = Infinity;
  let globalMax = -Infinity;
  let nanCount = 0;
  let infCount = 0;
  let totalRewards = 0;
  let rewardMin = Infinity;
  let rewardMax = -Infinity;
  let rewardSum = 0;
  let rewardSqSum = 0;

  // Track episode stats
  let episodeCount = 0;
  let currentEpisodeLength = 0;
  const episodeLengths: number[] = [];
  let goalsScored = 0;

  // Run across all 15 actions + random actions + dynamic movements
  for (let step = 0; step < totalSteps; step++) {
    currentEpisodeLength++;
    const actionIdx = Math.floor(Math.random() * ACTION_SPACE_SIZE);
    const controlledPlayer = engine.players.find((p) => p.id === engine.controlledPlayerId) || engine.players[0];
    const action = mapDiscreteAction(actionIdx, controlledPlayer?.heading || 0);

    const actionMap = new Map<string, AgentAction>();
    if (controlledPlayer) {
      actionMap.set(controlledPlayer.id, action);
    }

    const stepResult = engine.step(actionMap, 1 / 60);
    const obs = stepResult.observation.rawVector;
    const rew = stepResult.reward;

    // Reward stats
    rewardMin = Math.min(rewardMin, rew);
    rewardMax = Math.max(rewardMax, rew);
    rewardSum += rew;
    rewardSqSum += rew * rew;
    totalRewards++;

    if (stepResult.info.event === 'GOAL') {
      goalsScored++;
    }

    // Observation stats
    for (let d = 0; d < 115; d++) {
      const val = obs[d] ?? 0;
      if (Number.isNaN(val)) {
        nanCount++;
      } else if (!Number.isFinite(val)) {
        infCount++;
      } else {
        featureStats[d].min = Math.min(featureStats[d].min, val);
        featureStats[d].max = Math.max(featureStats[d].max, val);
        featureStats[d].sum += val;
        featureStats[d].count++;
        globalMin = Math.min(globalMin, val);
        globalMax = Math.max(globalMax, val);
      }
    }

    if (stepResult.terminated || stepResult.truncated) {
      episodeCount++;
      episodeLengths.push(currentEpisodeLength);
      currentEpisodeLength = 0;
      engine.loadScenario(scenario);
    }
  }

  console.log(`Steps Completed: ${totalSteps.toLocaleString()}`);
  console.log(`Global Minimum: ${globalMin.toFixed(4)}`);
  console.log(`Global Maximum: ${globalMax.toFixed(4)}`);
  console.log(`NaN Count: ${nanCount}`);
  console.log(`Infinity Count: ${infCount}`);

  console.log('\n--- Selected Feature Range Summary (0..114) ---');
  console.log('Dims  0-43 (Left Players: x, y, vx, vy):');
  console.log(`  x: [${featureStats[0].min.toFixed(3)}, ${featureStats[0].max.toFixed(3)}], y: [${featureStats[1].min.toFixed(3)}, ${featureStats[1].max.toFixed(3)}], vx: [${featureStats[22].min.toFixed(3)}, ${featureStats[22].max.toFixed(3)}], vy: [${featureStats[23].min.toFixed(3)}, ${featureStats[23].max.toFixed(3)}]`);
  console.log('Dims 44-87 (Right Players: x, y, vx, vy):');
  console.log(`  Range: [${featureStats[44].min.toFixed(3)}, ${featureStats[44].max.toFixed(3)}] (empty in academy_empty_goal = -1.0)`);
  console.log('Dims 88-93 (Ball: x, y, z, vx, vy, vz):');
  console.log(`  ball.x: [${featureStats[88].min.toFixed(3)}, ${featureStats[88].max.toFixed(3)}], ball.y: [${featureStats[89].min.toFixed(3)}, ${featureStats[89].max.toFixed(3)}], ball.z: [${featureStats[90].min.toFixed(3)}, ${featureStats[90].max.toFixed(3)}]`);
  console.log(`  ball.vx: [${featureStats[91].min.toFixed(3)}, ${featureStats[91].max.toFixed(3)}], ball.vy: [${featureStats[92].min.toFixed(3)}, ${featureStats[92].max.toFixed(3)}], ball.vz: [${featureStats[93].min.toFixed(3)}, ${featureStats[93].max.toFixed(3)}]`);
  console.log('Dims 94-96 (Ball Ownership [none, left, right]):');
  console.log(`  none: [${featureStats[94].min}, ${featureStats[94].max}], left: [${featureStats[95].min}, ${featureStats[95].max}], right: [${featureStats[96].min}, ${featureStats[96].max}]`);
  console.log('Dims 97-107 (Active Player One-Hot, 11 slots):');
  console.log(`  val: [${featureStats[97].min.toFixed(3)}, ${featureStats[97].max.toFixed(3)}]`);
  console.log('Dims 108-114 (GameMode One-Hot [Normal, KickOff, GoalKick, FreeKick, Corner, ThrowIn, Penalty]):');
  console.log(`  mode: [${featureStats[108].min}, ${featureStats[108].max}]`);

  const rewardMean = rewardSum / totalRewards;
  const rewardVariance = rewardSqSum / totalRewards - rewardMean * rewardMean;
  const rewardStd = Math.sqrt(Math.max(0, rewardVariance));

  console.log('\n==================================================');
  console.log('8. REWARD AUDIT (over 100,000 steps)');
  console.log('==================================================');
  console.log(`- Min Reward: ${rewardMin.toFixed(4)}`);
  console.log(`- Max Reward: ${rewardMax.toFixed(4)}`);
  console.log(`- Mean Reward: ${rewardMean.toFixed(6)}`);
  console.log(`- Std Reward: ${rewardStd.toFixed(6)}`);
  console.log(`- Goals in 100k random steps: ${goalsScored}`);

  console.log('\n==================================================');
  console.log('9. EPISODE STATISTICS');
  console.log('==================================================');
  const minEpLen = Math.min(...episodeLengths);
  const maxEpLen = Math.max(...episodeLengths);
  const avgEpLen = episodeLengths.reduce((a, b) => a + b, 0) / Math.max(1, episodeLengths.length);
  console.log(`- Total Episodes: ${episodeCount}`);
  console.log(`- Min Episode Length: ${minEpLen} steps`);
  console.log(`- Max Episode Length: ${maxEpLen} steps (capped by scenario maxSteps: ${scenario.maxSteps || 900})`);
  console.log(`- Avg Episode Length: ${avgEpLen.toFixed(1)} steps`);

  console.log('\n==================================================');
  console.log('2. ACTION EFFECTIVENESS AUDIT');
  console.log('==================================================');
  auditAllActions();
}

function auditAllActions() {
  const actionNames = [
    'IDLE',
    'LEFT',
    'TOP_LEFT',
    'TOP',
    'TOP_RIGHT',
    'RIGHT',
    'BOTTOM_RIGHT',
    'BOTTOM',
    'BOTTOM_LEFT',
    'LONG_PASS',
    'HIGH_PASS',
    'SHORT_PASS',
    'SHOT',
    'SPRINT',
    'RELEASE_DIRECTION',
    'RELEASE_SPRINT',
    'SLIDING',
    'DRIBBLE',
    'RELEASE_DRIBBLE',
  ];

  console.log('| IDX | ACTION | ENGINE TYPE | PLAYER DELTA (dx, dy) | BALL DELTA | REWARD | VALID? |');
  console.log('|---|---|---|---|---|---|---|');

  for (let a = 0; a < ACTION_SPACE_SIZE; a++) {
    const engine = new GameEngine();
    const scenario = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_empty_goal')!;
    engine.loadScenario(scenario);

    // Give player possession for ball-action testing
    engine.players[0].hasBall = true;
    engine.players[0].position = { x: 0.5, y: 0 };
    engine.ball.ownerId = engine.players[0].id;
    engine.ball.position = { x: 0.5, y: 0, z: 0 };

    const pInit = { ...engine.players[0].position };
    const bInit = { ...engine.ball.position };
    let totalReward = 0;

    // Run action for 30 steps
    for (let t = 0; t < 30; t++) {
      const p = engine.players[0];
      const action = mapDiscreteAction(a, p.heading);
      const actionMap = new Map<string, AgentAction>([[p.id, action]]);
      const res = engine.step(actionMap, 1 / 60);
      totalReward += res.reward;
    }

    const pFinal = engine.players[0].position;
    const bFinal = engine.ball.position;
    const dx = pFinal.x - pInit.x;
    const dy = pFinal.y - pInit.y;
    const bdx = bFinal.x - bInit.x;
    const bdy = bFinal.y - bInit.y;

    const mapped = mapDiscreteAction(a, 0);
    const valid =
      (a === 0 && (Math.abs(dx) < 0.1 || mapped.type === ActionType.IDLE)) ||
      (a > 0 &&
        (Math.abs(dx) > 0.001 ||
          Math.abs(dy) > 0.001 ||
          Math.abs(bdx) > 0.001 ||
          Math.abs(bdy) > 0.001 ||
          totalReward !== 0 ||
          mapped.type === ActionType.SLIDING ||
          mapped.type === ActionType.TACKLE ||
          mapped.type === ActionType.DRIBBLE ||
          mapped.type === ActionType.RELEASE_DIRECTION ||
          mapped.type === ActionType.RELEASE_SPRINT ||
          mapped.type === ActionType.RELEASE_DRIBBLE));

    console.log(
      `| ${a.toString().padStart(2, ' ')} | ${actionNames[a].padEnd(18, ' ')} | ${mapped.type.padEnd(18, ' ')} | (${dx >= 0 ? '+' : ''}${dx.toFixed(3)}, ${dy >= 0 ? '+' : ''}${dy.toFixed(3)}) | (${bdx >= 0 ? '+' : ''}${bdx.toFixed(3)}, ${bdy >= 0 ? '+' : ''}${bdy.toFixed(3)}) | ${totalReward.toFixed(3)} | ${valid ? 'YES' : 'NO'} |`
    );
  }
}

runObservationAndActionAudit(100000);
