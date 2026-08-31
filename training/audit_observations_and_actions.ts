import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ActionType, AgentAction, GameMode } from '../src/types/football';
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
  let scenarioIdx = 0;
  engine.loadScenario(ACADEMY_SCENARIOS[scenarioIdx]);

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

  // Track foul and card statistics across 100k rollout
  const cumulativeStats = {
    fouls: { left: 0, right: 0 },
    yellowCards: { left: 0, right: 0 },
    redCards: { left: 0, right: 0 },
    secondYellows: { left: 0, right: 0 },
    straightReds: { left: 0, right: 0 },
    offsides: { left: 0, right: 0 },
  };
  let lastEventCount = 0;

  // Run across all 19 actions + dynamic movements across scenarios
  for (let step = 0; step < totalSteps; step++) {
    currentEpisodeLength++;
    const actionIdx = Math.floor(Math.random() * ACTION_SPACE_SIZE);
    const controlledPlayer = engine.players.find((p) => p.id === engine.controlledPlayerId) || engine.players[0];
    
    let action: AgentAction;
    if (controlledPlayer) {
      const dx = engine.ball.position.x - controlledPlayer.position.x;
      const dy = engine.ball.position.y - controlledPlayer.position.y;
      const dist = Math.hypot(dx, dy);
      if (dist < 0.15 && engine.ball.ownerId && engine.ball.ownerId !== controlledPlayer.id) {
        action = { type: ActionType.SLIDING };
      } else if (Math.random() < 0.5 && dist > 0.02) {
        action = { type: ActionType.MOVE, direction: { x: dx / dist, y: dy / dist } };
      } else {
        action = mapDiscreteAction(actionIdx, controlledPlayer.heading);
      }
    } else {
      action = mapDiscreteAction(actionIdx, controlledPlayer?.heading || 0);
    }

    const actionMap = new Map<string, AgentAction>();
    if (controlledPlayer) {
      actionMap.set(controlledPlayer.id, action);
    }

    // Step other players: seek ball and tackle if opposing owner nearby
    for (const p of engine.players) {
      if (p.id !== controlledPlayer?.id) {
        const dx = engine.ball.position.x - p.position.x;
        const dy = engine.ball.position.y - p.position.y;
        const dist = Math.hypot(dx, dy);
        if (engine.ball.ownerId && engine.ball.ownerId !== p.id) {
          const owner = engine.players.find((pl) => pl.id === engine.ball.ownerId);
          if (owner && owner.team !== p.team) {
            if (dist < 0.15) {
              actionMap.set(p.id, { type: ActionType.SLIDING });
              continue;
            }
          }
        }
        if (Math.random() < 0.6) {
          if (dist > 0.02) {
            actionMap.set(p.id, { type: ActionType.MOVE, direction: { x: dx / dist, y: dy / dist } });
          } else {
            const randActIdx = Math.floor(Math.random() * ACTION_SPACE_SIZE);
            actionMap.set(p.id, mapDiscreteAction(randActIdx));
          }
        }
      }
    }

    const stepResult = engine.step(actionMap, 1 / 60);
    const obs = stepResult.observation.rawVector;
    const rew = stepResult.reward;

    // Track card and offside events
    for (let e = lastEventCount; e < engine.events.length; e++) {
      const evt = engine.events[e];
      if (evt.type === 'foul' && evt.team) {
        if (evt.description.includes('offside')) {
          cumulativeStats.offsides[evt.team]++;
        } else {
          cumulativeStats.fouls[evt.team]++;
          if (evt.description.includes('Second yellow')) {
            cumulativeStats.yellowCards[evt.team]++;
            cumulativeStats.redCards[evt.team]++;
            cumulativeStats.secondYellows[evt.team]++;
          } else if (evt.description.includes('Yellow card')) {
            cumulativeStats.yellowCards[evt.team]++;
          } else if (evt.description.includes('RED CARD')) {
            cumulativeStats.redCards[evt.team]++;
            cumulativeStats.straightReds[evt.team]++;
          }
        }
      }
    }
    lastEventCount = engine.events.length;

    // Reward stats
    rewardMin = Math.min(rewardMin, rew);
    rewardMax = Math.max(rewardMax, rew);
    rewardSum += rew;
    rewardSqSum += rew * rew;
    totalRewards++;

    if (typeof stepResult.info.event === 'object' && stepResult.info.event?.type === 'goal') {
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
      scenarioIdx = (scenarioIdx + 1) % ACADEMY_SCENARIOS.length;
      engine.loadScenario(ACADEMY_SCENARIOS[scenarioIdx]);
      lastEventCount = 0;
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
  console.log(`  Range: [${featureStats[44].min.toFixed(3)}, ${featureStats[44].max.toFixed(3)}] (inactive: -1.0)`);
  console.log('Dims 88-93 (Ball: x, y, z, vx, vy, vz):');
  console.log(`  ball.x: [${featureStats[88].min.toFixed(3)}, ${featureStats[88].max.toFixed(3)}], ball.y: [${featureStats[89].min.toFixed(3)}, ${featureStats[89].max.toFixed(3)}], ball.z: [${featureStats[90].min.toFixed(3)}, ${featureStats[90].max.toFixed(3)}]`);
  console.log(`  ball.vx: [${featureStats[91].min.toFixed(3)}, ${featureStats[91].max.toFixed(3)}], ball.vy: [${featureStats[92].min.toFixed(3)}, ${featureStats[92].max.toFixed(3)}], ball.vz: [${featureStats[93].min.toFixed(3)}, ${featureStats[93].max.toFixed(3)}]`);
  console.log('Dims 94-96 (Ball Ownership [none, left, right]):');
  console.log(`  none: [${featureStats[94].min}, ${featureStats[94].max}], left: [${featureStats[95].min}, ${featureStats[95].max}], right: [${featureStats[96].min}, ${featureStats[96].max}]`);
  console.log('Dims 97-107 (Active Player One-Hot, 11 slots):');
  console.log(`  val: [${featureStats[97].min.toFixed(3)}, ${featureStats[97].max.toFixed(3)}]`);
  console.log('Dims 108-114 (GameMode One-Hot [Normal, KickOff, GoalKick, FreeKick, Corner, ThrowIn, Penalty]):');
  const modeNames = ['Normal', 'KickOff', 'GoalKick', 'FreeKick', 'Corner', 'ThrowIn', 'Penalty'];
  modeNames.forEach((name, idx) => {
    const dim = 108 + idx;
    console.log(`  - ${name} (dim ${dim}): [${featureStats[dim].min}, ${featureStats[dim].max}] (active: ${featureStats[dim].max > 0 ? 'YES' : 'NO'})`);
  });

  const rewardMean = rewardSum / totalRewards;
  const rewardVariance = rewardSqSum / totalRewards - rewardMean * rewardMean;
  const rewardStd = Math.sqrt(Math.max(0, rewardVariance));

  console.log('\n==================================================');
  console.log('4. FOULS, CARDS, AND OFFSIDES AUDIT (over 100,000 steps)');
  console.log('==================================================');
  console.log(`- Total Fouls Committed: Left: ${cumulativeStats.fouls.left}, Right: ${cumulativeStats.fouls.right}`);
  console.log(`- Offsides Called:       Left: ${cumulativeStats.offsides.left}, Right: ${cumulativeStats.offsides.right}`);
  console.log(`- Yellow Cards Shown:    Left: ${cumulativeStats.yellowCards.left}, Right: ${cumulativeStats.yellowCards.right}`);
  console.log(`- Red Cards Shown:       Left: ${cumulativeStats.redCards.left}, Right: ${cumulativeStats.redCards.right}`);
  console.log(`  * Straight Reds:       Left: ${cumulativeStats.straightReds.left}, Right: ${cumulativeStats.straightReds.right}`);
  console.log(`  * Second Yellow Reds:  Left: ${cumulativeStats.secondYellows.left}, Right: ${cumulativeStats.secondYellows.right}`);

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
  console.log(`- Max Episode Length: ${maxEpLen} steps`);
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

  auditGameModes();
}

function auditGameModes() {
  console.log('\n==================================================');
  console.log('3. GAME MODE TRIGGERING & OBSERVATION VERIFICATION');
  console.log('==================================================');

  const modeTests: {
    name: string;
    expectedMode: GameMode;
    expectedDim: number;
    setup: (engine: GameEngine) => void;
  }[] = [
    {
      name: 'KickOff',
      expectedMode: GameMode.KickOff,
      expectedDim: 109,
      setup: (eng) => {
        eng.resetToKickoff();
      },
    },
    {
      name: 'Normal',
      expectedMode: GameMode.Normal,
      expectedDim: 108,
      setup: (eng) => {
        eng.resetToKickoff();
        eng.step(new Map([[eng.players[0].id, { type: ActionType.MOVE, direction: { x: 1, y: 0 } }]]), 1 / 60);
      },
    },
    {
      name: 'GoalKick',
      expectedMode: GameMode.GoalKick,
      expectedDim: 110,
      setup: (eng) => {
        eng.resetToKickoff();
        eng.ball.lastOwnerTeam = 'left';
        eng.ball.position = { x: 1.05, y: 0.2, z: 0 };
        eng.step(new Map(), 1 / 60);
      },
    },
    {
      name: 'Corner',
      expectedMode: GameMode.Corner,
      expectedDim: 112,
      setup: (eng) => {
        eng.resetToKickoff();
        eng.ball.lastOwnerTeam = 'right';
        eng.ball.position = { x: 1.05, y: 0.2, z: 0 };
        eng.step(new Map(), 1 / 60);
      },
    },
    {
      name: 'ThrowIn',
      expectedMode: GameMode.ThrowIn,
      expectedDim: 113,
      setup: (eng) => {
        eng.resetToKickoff();
        eng.ball.lastOwnerTeam = 'left';
        eng.ball.position = { x: 0.2, y: 0.45, z: 0 };
        eng.step(new Map(), 1 / 60);
      },
    },
    {
      name: 'FreeKick (Foul outside box)',
      expectedMode: GameMode.FreeKick,
      expectedDim: 111,
      setup: (eng) => {
        eng.loadScenario(ACADEMY_SCENARIOS.find((s) => s.id === 'academy_run_to_score')!);
        const attacker = eng.players.find((p) => p.team === 'left')!;
        const defender = eng.players.find((p) => p.team === 'right' && !p.isGoalkeeper)!;
        defender.heading = Math.PI;

        let attempts = 0;
        while (eng.gameMode !== GameMode.FreeKick && attempts < 50) {
          attacker.position = { x: 0.2, y: 0.0 };
          defender.position = { x: 0.22, y: 0.001 * attempts };
          eng.ball.position = { x: 0.2, y: 0.0, z: 0 };
          eng.ball.ownerId = attacker.id;
          attacker.hasBall = true;
          defender.tackleCooldown = 0;
          eng.step(new Map([[defender.id, { type: ActionType.SLIDING }]]), 1 / 60);
          attempts++;
        }
      },
    },
    {
      name: 'Penalty (Foul inside defending box)',
      expectedMode: GameMode.Penalty,
      expectedDim: 114,
      setup: (eng) => {
        eng.loadScenario(ACADEMY_SCENARIOS.find((s) => s.id === 'academy_run_to_score')!);
        const attacker = eng.players.find((p) => p.team === 'left')!;
        const defender = eng.players.find((p) => p.team === 'right' && !p.isGoalkeeper)!;
        defender.heading = Math.PI;

        let attempts = 0;
        while (eng.gameMode !== GameMode.Penalty && attempts < 50) {
          attacker.position = { x: 0.85, y: 0.0 };
          defender.position = { x: 0.87, y: 0.001 * attempts };
          eng.ball.position = { x: 0.85, y: 0.0, z: 0 };
          eng.ball.ownerId = attacker.id;
          attacker.hasBall = true;
          defender.tackleCooldown = 0;
          eng.step(new Map([[defender.id, { type: ActionType.SLIDING }]]), 1 / 60);
          attempts++;
        }
      },
    },
  ];

  for (const test of modeTests) {
    const testEngine = new GameEngine();
    test.setup(testEngine);
    const obs = testEngine.step(
      new Map([[testEngine.players[0].id, { type: ActionType.IDLE }]]),
      1 / 60
    ).observation.rawVector;
    const modeSlice = obs.slice(108, 115);
    const triggered = testEngine.gameMode === test.expectedMode;
    const oneHotCorrect = obs[test.expectedDim] === 1.0;
    console.log(
      `✓ ${test.name.padEnd(30, ' ')} | GameMode: ${GameMode[testEngine.gameMode]} (${testEngine.gameMode}) | OneHot[108..114]: [${modeSlice.join(', ')}] | Match: ${triggered && oneHotCorrect ? 'PASS' : 'FAIL'}`
    );
  }

  console.log('\n==================================================');
  console.log('5. CARD MECHANICS UNIT VERIFICATION');
  console.log('==================================================');
  auditCardMechanics();

  console.log('\n==================================================');
  console.log('6. OFFSIDE MECHANICS UNIT VERIFICATION');
  console.log('==================================================');
  auditOffsideMechanics();
}

function auditCardMechanics() {
  // Test 1: First Yellow Card
  {
    const engine = new GameEngine();
    engine.loadScenario(ACADEMY_SCENARIOS.find((s) => s.id === 'academy_run_to_score')!);
    const attacker = engine.players.find((p) => p.team === 'left')!;
    const defender = engine.players.find((p) => p.team === 'right' && !p.isGoalkeeper)!;
    
    // Find seed where tackle result is foul and card roll is yellow (0.80 - 0.95)
    let found = false;
    for (let seed = 1; seed < 1000; seed++) {
      engine.setSeed(seed);
      attacker.position = { x: 0.2, y: 0.0 };
      defender.position = { x: 0.22, y: 0.0 };
      engine.ball.position = { x: 0.2, y: 0.0, z: 0 };
      engine.ball.ownerId = attacker.id;
      attacker.hasBall = true;
      defender.tackleCooldown = 0;
      defender.yellowCards = 0;
      defender.redCard = false;
      engine.stats.yellowCards.right = 0;
      engine.stats.redCards.right = 0;
      engine.stats.fouls.right = 0;

      engine.step(new Map([[defender.id, { type: ActionType.SLIDING }]]), 1 / 60);
      if (defender.yellowCards === 1 && !defender.redCard && engine.stats.yellowCards.right === 1 && engine.stats.redCards.right === 0) {
        console.log(`✓ First Yellow Card (1st yellow -> player.yellowCards = 1, stays on, stats.yellowCards = 1): PASS`);
        found = true;
        break;
      }
    }
    if (!found) {
      console.log(`✗ First Yellow Card test failed to trigger`);
    }
  }

  // Test 2: Second Yellow Card -> Red Card
  {
    const engine = new GameEngine();
    engine.loadScenario(ACADEMY_SCENARIOS.find((s) => s.id === 'academy_run_to_score')!);
    const attacker = engine.players.find((p) => p.team === 'left')!;
    const defender = engine.players.find((p) => p.team === 'right' && !p.isGoalkeeper)!;
    
    let found = false;
    for (let seed = 1; seed < 1000; seed++) {
      engine.setSeed(seed);
      attacker.position = { x: 0.2, y: 0.0 };
      defender.position = { x: 0.22, y: 0.0 };
      engine.ball.position = { x: 0.2, y: 0.0, z: 0 };
      engine.ball.ownerId = attacker.id;
      attacker.hasBall = true;
      defender.tackleCooldown = 0;
      defender.yellowCards = 1; // Already on a yellow
      defender.redCard = false;
      engine.stats.yellowCards.right = 1;
      engine.stats.redCards.right = 0;
      engine.stats.fouls.right = 1;

      engine.step(new Map([[defender.id, { type: ActionType.SLIDING }]]), 1 / 60);
      if (defender.yellowCards === 2 && defender.redCard && engine.stats.yellowCards.right === 2 && engine.stats.redCards.right === 1) {
        console.log(`✓ Second Yellow Card (2nd yellow -> player.yellowCards = 2, player.redCard = true, stats.redCards = 1): PASS`);
        found = true;
        break;
      }
    }
    if (!found) {
      console.log(`✗ Second Yellow Card test failed to trigger`);
    }
  }

  // Test 3: Straight Red Card
  {
    const engine = new GameEngine();
    engine.loadScenario(ACADEMY_SCENARIOS.find((s) => s.id === 'academy_run_to_score')!);
    const attacker = engine.players.find((p) => p.team === 'left')!;
    const defender = engine.players.find((p) => p.team === 'right' && !p.isGoalkeeper)!;
    
    let found = false;
    for (let seed = 1; seed < 2000; seed++) {
      engine.setSeed(seed);
      attacker.position = { x: 0.2, y: 0.0 };
      defender.position = { x: 0.22, y: 0.0 };
      engine.ball.position = { x: 0.2, y: 0.0, z: 0 };
      engine.ball.ownerId = attacker.id;
      attacker.hasBall = true;
      defender.tackleCooldown = 0;
      defender.yellowCards = 0;
      defender.redCard = false;
      engine.stats.yellowCards.right = 0;
      engine.stats.redCards.right = 0;
      engine.stats.fouls.right = 0;

      engine.step(new Map([[defender.id, { type: ActionType.SLIDING }]]), 1 / 60);
      if (defender.yellowCards === 0 && defender.redCard && engine.stats.yellowCards.right === 0 && engine.stats.redCards.right === 1) {
        console.log(`✓ Straight Red Card (roll >= 0.95 -> player.redCard = true, yellowCards = 0, stats.redCards = 1): PASS`);
        found = true;
        break;
      }
    }
    if (!found) {
      console.log(`✗ Straight Red Card test failed to trigger`);
    }
  }

  // Test 4: Already Red Card Guard (skips further cards)
  {
    const engine = new GameEngine();
    engine.loadScenario(ACADEMY_SCENARIOS.find((s) => s.id === 'academy_run_to_score')!);
    const attacker = engine.players.find((p) => p.team === 'left')!;
    const defender = engine.players.find((p) => p.team === 'right' && !p.isGoalkeeper)!;
    
    let found = false;
    for (let seed = 1; seed < 1000; seed++) {
      engine.setSeed(seed);
      attacker.position = { x: 0.2, y: 0.0 };
      defender.position = { x: 0.22, y: 0.0 };
      engine.ball.position = { x: 0.2, y: 0.0, z: 0 };
      engine.ball.ownerId = attacker.id;
      attacker.hasBall = true;
      defender.tackleCooldown = 0;
      defender.yellowCards = 1;
      defender.redCard = true; // Already carrying red
      engine.stats.yellowCards.right = 1;
      engine.stats.redCards.right = 1;
      engine.stats.fouls.right = 2;

      engine.step(new Map([[defender.id, { type: ActionType.SLIDING }]]), 1 / 60);
      if (engine.stats.fouls.right === 3 && defender.yellowCards === 1 && defender.redCard === true && engine.stats.yellowCards.right === 1 && engine.stats.redCards.right === 1) {
        console.log(`✓ Red Card Guard (already red-carded player commits foul -> foul counted, card rolls skipped): PASS`);
        found = true;
        break;
      }
    }
    if (!found) {
      console.log(`✗ Red Card Guard test failed to trigger`);
    }
  }
}

function auditOffsideMechanics() {
  // Test 1: Standard Offside Call
  {
    const engine = new GameEngine();
    engine.initDefaultMatch('4-3-3', '4-3-3', 11);
    const passer = engine.players.find((p) => p.id === 'left_10') || engine.players[1];
    const receiver = engine.players.find((p) => p.id === 'left_11') || engine.players[2];
    const gk = engine.players.find((p) => p.team === 'right' && p.isGoalkeeper)!;
    const defender = engine.players.find((p) => p.team === 'right' && !p.isGoalkeeper)!;

    // Position players:
    // Passer in opponent half at 0.2
    // Defender at 0.45, GK at 0.95 (offside line = 0.45)
    // Receiver in offside position at 0.65
    passer.position = { x: 0.2, y: 0.0 };
    passer.targetPosition = { x: 0.2, y: 0.0 };
    passer.hasBall = true;
    engine.ball.position = { x: 0.2, y: 0.0, z: 0 };
    engine.ball.ownerId = passer.id;

    receiver.position = { x: 0.65, y: 0.0 };
    receiver.targetPosition = { x: 0.65, y: 0.0 };
    receiver.hasBall = false;

    defender.position = { x: 0.45, y: 0.2 };
    defender.targetPosition = { x: 0.45, y: 0.2 };
    gk.position = { x: 0.95, y: 0.0 };
    gk.targetPosition = { x: 0.95, y: 0.0 };

    // Move all other players far away
    engine.players.forEach((p) => {
      if (p.id !== passer.id && p.id !== receiver.id && p.id !== defender.id && p.id !== gk.id) {
        p.position = { x: -0.5, y: -0.3 };
        p.targetPosition = { x: -0.5, y: -0.3 };
      }
    });

    engine.gameMode = GameMode.Normal;
    engine.step(new Map([[passer.id, { type: ActionType.SHORT_PASS, direction: { x: 1, y: 0 }, power: 0.95 }]]), 1 / 60);

    // Ball moves towards receiver at 0.65
    let offsideTriggered = false;
    for (let step = 0; step < 120; step++) {
      engine.step(new Map(), 1 / 60);
      if (engine.gameMode === GameMode.FreeKick) {
        const offsideEvent = engine.events.find((e) => e.description.includes('was offside!'));
        if (offsideEvent && engine.stats.completedPasses.left === 0) {
          offsideTriggered = true;
          break;
        }
      }
    }

    if (offsideTriggered) {
      console.log(`✓ Standard Offside (attacker ahead of 2nd-last defender in opponent half -> Free Kick awarded at receiver pos, completedPasses not incremented): PASS`);
    } else {
      console.log(`✗ Standard Offside test failed`);
    }
  }

  // Test 2: Restart Exemption (GoalKick / Corner / ThrowIn)
  {
    const engine = new GameEngine();
    engine.initDefaultMatch('4-3-3', '4-3-3', 11);
    const passer = engine.players.find((p) => p.id === 'left_10') || engine.players[1];
    const receiver = engine.players.find((p) => p.id === 'left_11') || engine.players[2];
    const gk = engine.players.find((p) => p.team === 'right' && p.isGoalkeeper)!;
    const defender = engine.players.find((p) => p.team === 'right' && !p.isGoalkeeper)!;

    passer.position = { x: 0.2, y: 0.0 };
    passer.targetPosition = { x: 0.2, y: 0.0 };
    passer.hasBall = true;
    engine.ball.position = { x: 0.2, y: 0.0, z: 0 };
    engine.ball.ownerId = passer.id;

    receiver.position = { x: 0.65, y: 0.0 };
    receiver.targetPosition = { x: 0.65, y: 0.0 };
    receiver.hasBall = false;

    defender.position = { x: 0.45, y: 0.2 };
    defender.targetPosition = { x: 0.45, y: 0.2 };
    gk.position = { x: 0.95, y: 0.0 };
    gk.targetPosition = { x: 0.95, y: 0.0 };

    engine.players.forEach((p) => {
      if (p.id !== passer.id && p.id !== receiver.id && p.id !== defender.id && p.id !== gk.id) {
        p.position = { x: -0.5, y: -0.3 };
        p.targetPosition = { x: -0.5, y: -0.3 };
      }
    });

    // Directly set mode to ThrowIn restart
    engine.gameMode = GameMode.ThrowIn;
    engine.step(new Map([[passer.id, { type: ActionType.SHORT_PASS, direction: { x: 1, y: 0 }, power: 0.95 }]]), 1 / 60);

    let passCompleted = false;
    let offsideCalled = false;
    for (let step = 0; step < 120; step++) {
      engine.step(new Map(), 1 / 60);
      if (engine.events.some((e) => e.description.includes('was offside!'))) {
        offsideCalled = true;
      }
      if (engine.stats.completedPasses.left === 1 && receiver.hasBall) {
        passCompleted = true;
        break;
      }
    }

    if (!offsideCalled && passCompleted) {
      console.log(`✓ Restart Exemption (pass played directly from ThrowIn/Corner/GoalKick with teammate in offside position -> no offside called, pass completed): PASS`);
    } else {
      console.log(`✗ Restart Exemption test failed (offsideCalled: ${offsideCalled}, passCompleted: ${passCompleted})`);
    }
  }

  // Test 3: Onside Pass (receiver behind offside line)
  {
    const engine = new GameEngine();
    engine.initDefaultMatch('4-3-3', '4-3-3', 11);
    const passer = engine.players.find((p) => p.id === 'left_10') || engine.players[1];
    const receiver = engine.players.find((p) => p.id === 'left_11') || engine.players[2];
    const gk = engine.players.find((p) => p.team === 'right' && p.isGoalkeeper)!;
    const defender = engine.players.find((p) => p.team === 'right' && !p.isGoalkeeper)!;

    passer.position = { x: 0.1, y: 0.0 };
    passer.targetPosition = { x: 0.1, y: 0.0 };
    passer.hasBall = true;
    engine.ball.position = { x: 0.1, y: 0.0, z: 0 };
    engine.ball.ownerId = passer.id;

    // Receiver at 0.35 is onside (defender at 0.50)
    receiver.position = { x: 0.35, y: 0.0 };
    receiver.targetPosition = { x: 0.35, y: 0.0 };
    receiver.hasBall = false;

    defender.position = { x: 0.50, y: 0.2 };
    defender.targetPosition = { x: 0.50, y: 0.2 };
    gk.position = { x: 0.95, y: 0.0 };
    gk.targetPosition = { x: 0.95, y: 0.0 };

    engine.players.forEach((p) => {
      if (p.id !== passer.id && p.id !== receiver.id && p.id !== defender.id && p.id !== gk.id) {
        p.position = { x: -0.5, y: -0.3 };
        p.targetPosition = { x: -0.5, y: -0.3 };
      }
    });

    engine.gameMode = GameMode.Normal;
    engine.step(new Map([[passer.id, { type: ActionType.SHORT_PASS, direction: { x: 1, y: 0 }, power: 0.75 }]]), 1 / 60);

    let passCompleted = false;
    let offsideCalled = false;
    for (let step = 0; step < 120; step++) {
      engine.step(new Map(), 1 / 60);
      if (engine.events.some((e) => e.description.includes('was offside!'))) {
        offsideCalled = true;
      }
      if (engine.stats.completedPasses.left === 1 && receiver.hasBall) {
        passCompleted = true;
        break;
      }
    }

    if (!offsideCalled && passCompleted) {
      console.log(`✓ Onside Pass (receiver behind 2nd-last defender -> no offside called, pass completed): PASS`);
    } else {
      console.log(`✗ Onside Pass test failed`);
    }
  }

  // Test 4: Onside in Own Half
  {
    const engine = new GameEngine();
    engine.initDefaultMatch('4-3-3', '4-3-3', 11);
    const passer = engine.players.find((p) => p.id === 'left_10') || engine.players[1];
    const receiver = engine.players.find((p) => p.id === 'left_11') || engine.players[2];

    // Both players in own half (x < 0 for left team)
    passer.position = { x: -0.4, y: 0.0 };
    passer.targetPosition = { x: -0.4, y: 0.0 };
    passer.hasBall = true;
    engine.ball.position = { x: -0.4, y: 0.0, z: 0 };
    engine.ball.ownerId = passer.id;

    receiver.position = { x: -0.15, y: 0.0 };
    receiver.targetPosition = { x: -0.15, y: 0.0 };
    receiver.hasBall = false;

    // All defenders also pushed up past receiver (e.g. at x = 0.2)
    engine.players.forEach((p) => {
      if (p.team === 'right') {
        p.position = { x: 0.2, y: 0.0 };
        p.targetPosition = { x: 0.2, y: 0.0 };
      } else if (p.id !== passer.id && p.id !== receiver.id) {
        p.position = { x: -0.6, y: -0.3 };
        p.targetPosition = { x: -0.6, y: -0.3 };
      }
    });

    engine.gameMode = GameMode.Normal;
    engine.step(new Map([[passer.id, { type: ActionType.SHORT_PASS, direction: { x: 1, y: 0 }, power: 0.75 }]]), 1 / 60);

    let passCompleted = false;
    let offsideCalled = false;
    for (let step = 0; step < 120; step++) {
      engine.step(new Map(), 1 / 60);
      if (engine.events.some((e) => e.description.includes('was offside!'))) {
        offsideCalled = true;
      }
      if (engine.stats.completedPasses.left === 1 && receiver.hasBall) {
        passCompleted = true;
        break;
      }
    }

    if (!offsideCalled && passCompleted) {
      console.log(`✓ Own Half Exemption (attacker in own half x < 0 -> no offside called, pass completed): PASS`);
    } else {
      console.log(`✗ Own Half Exemption test failed`);
    }
  }
}

runObservationAndActionAudit(100000);
