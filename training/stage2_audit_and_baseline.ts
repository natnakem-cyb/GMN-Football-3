import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ActionType, AgentAction } from '../src/types/football';
import { mapDiscreteAction, ACTION_SPACE_SIZE } from './action_mapping';
import { RuleBasedAgent } from '../src/agents/RuleBasedAgent';
import { Vec2 } from '../src/engine/Vector';

export function runStage2Audit() {
  console.log('==================================================');
  console.log('1. STAGE 2 SCENARIO INSPECTION (academy_run_to_score)');
  console.log('==================================================');

  const scenario = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_run_to_score')!;
  console.log(`Scenario: ${scenario.name} (${scenario.id})`);
  console.log(`Description: ${scenario.description}`);
  console.log(`Time Limit: ${scenario.timeLimitSeconds}s (${scenario.timeLimitSeconds * 60} ticks)`);
  console.log(`Left Players (${scenario.teamLeftPlayers}):`, scenario.setup.leftPlayers);
  console.log(`Right Players (${scenario.teamRightPlayers}):`, scenario.setup.rightPlayers);
  console.log(`Ball Initial Pos:`, scenario.setup.ball);
  console.log(`Objectives:`, scenario.objectives.map((o) => o.text));

  // Initialize engine and inspect setup
  const engine = new GameEngine();
  engine.loadScenario(scenario);

  console.log(`\nEngine Initial State:`);
  console.log(`- Controlled Player: ${engine.controlledPlayerId} (pos: ${JSON.stringify(engine.players[0].position)})`);
  console.log(`- Opponents Count: ${engine.players.filter((p) => p.team === 'right').length}`);
  const defender = engine.players.find((p) => p.role === 'CB');
  const keeper = engine.players.find((p) => p.role === 'GK');
  console.log(`  - Defender CB: pos=${JSON.stringify(defender?.position)}, speed=${defender?.stats.speed}`);
  console.log(`  - Goalkeeper GK: pos=${JSON.stringify(keeper?.position)}, speed=${keeper?.stats.speed}`);
  console.log(`- Ball: pos=${JSON.stringify(engine.ball.position)}, ownerId=${engine.ball.ownerId}`);

  // Test 1: Random Policy Baseline on Stage 2 (100 episodes)
  console.log('\n==================================================');
  console.log('2. RANDOM POLICY BASELINE (100 episodes)');
  console.log('==================================================');
  {
    let goals = 0;
    let dispossessed = 0;
    let timeouts = 0;
    const rewards: number[] = [];
    const lengths: number[] = [];
    let totalPossessionTicks = 0;
    let totalTicks = 0;

    const botDefender = new RuleBasedAgent('bot_cb', 'CB Defender', 'medium');
    const botGK = new RuleBasedAgent('bot_gk', 'GK', 'medium');

    for (let ep = 0; ep < 100; ep++) {
      const eng = new GameEngine();
      eng.loadScenario(scenario);

      let epRew = 0;
      let epLen = 0;
      let done = false;

      while (!done && epLen < 1200) {
        epLen++;
        totalTicks++;
        const player = eng.players[0];
        if (player.hasBall || eng.ball.ownerId === player.id) {
          totalPossessionTicks++;
        }

        const actionIdx = Math.floor(Math.random() * ACTION_SPACE_SIZE);
        const action = mapDiscreteAction(actionIdx, player.heading);

        const actionMap = new Map<string, AgentAction>([[player.id, action]]);

        // Opponents
        eng.players.forEach((p) => {
          if (p.team === 'right') {
            const bot = p.role === 'GK' ? botGK : botDefender;
            actionMap.set(
              p.id,
              bot.decide({
                player: p,
                teammates: eng.players.filter((x) => x.team === p.team),
                opponents: eng.players.filter((x) => x.team !== p.team),
                ball: eng.ball,
                allPlayers: eng.players,
                teamSide: p.team,
                controlledPlayerId: eng.controlledPlayerId,
                matchTime: eng.matchTimeSeconds,
              })
            );
          }
        });

        const res = eng.step(actionMap, 1 / 60);
        epRew += res.reward;

        if (res.terminated || res.truncated) {
          done = true;
          if (eng.score.left > 0 || (typeof res.info.event === 'object' && res.info.event?.type === 'goal')) {
            goals++;
          } else if (eng.ball.ownerId && eng.players.find((p) => p.id === eng.ball.ownerId)?.team === 'right') {
            dispossessed++;
          } else {
            timeouts++;
          }
        }
      }
      rewards.push(epRew);
      lengths.push(epLen);
    }

    const meanRew = rewards.reduce((a, b) => a + b, 0) / 100;
    const meanLen = lengths.reduce((a, b) => a + b, 0) / 100;
    const possRate = (totalPossessionTicks / totalTicks) * 100;

    console.log(`Random Policy Results (100 episodes):`);
    console.log(`- Success / Goal Rate: ${(goals / 100) * 100}% (${goals}/100)`);
    console.log(`- Dispossessed by Defender/GK: ${(dispossessed / 100) * 100}% (${dispossessed}/100)`);
    console.log(`- Timeouts: ${(timeouts / 100) * 100}% (${timeouts}/100)`);
    console.log(`- Mean Episode Reward: ${meanRew.toFixed(4)}`);
    console.log(`- Mean Episode Length: ${meanLen.toFixed(1)} steps (~${(meanLen / 60).toFixed(2)}s)`);
    console.log(`- Ball Possession Rate: ${possRate.toFixed(1)}%`);
  }

  // Test 2: Scripted Policy Baseline on Stage 2 (100 episodes)
  console.log('\n==================================================');
  console.log('3. SCRIPTED / RULE-BASED BASELINE (100 episodes)');
  console.log('==================================================');
  {
    let goals = 0;
    let dispossessed = 0;
    const rewards: number[] = [];
    const lengths: number[] = [];
    let totalPossessionTicks = 0;
    let totalTicks = 0;

    const striker = new RuleBasedAgent('striker', 'Striker', 'hard');
    const botDefender = new RuleBasedAgent('bot_cb', 'CB Defender', 'medium');
    const botGK = new RuleBasedAgent('bot_gk', 'GK', 'medium');

    for (let ep = 0; ep < 100; ep++) {
      const eng = new GameEngine();
      eng.loadScenario(scenario);

      let epRew = 0;
      let epLen = 0;
      let done = false;

      while (!done && epLen < 1200) {
        epLen++;
        totalTicks++;
        const player = eng.players[0];
        if (player.hasBall || eng.ball.ownerId === player.id) {
          totalPossessionTicks++;
        }

        const action = striker.decide({
          player,
          teammates: [player],
          opponents: eng.players.filter((p) => p.team === 'right'),
          ball: eng.ball,
          allPlayers: eng.players,
          teamSide: 'left',
          controlledPlayerId: player.id,
          matchTime: eng.matchTimeSeconds,
        });

        const actionMap = new Map<string, AgentAction>([[player.id, action]]);

        eng.players.forEach((p) => {
          if (p.team === 'right') {
            const bot = p.role === 'GK' ? botGK : botDefender;
            actionMap.set(
              p.id,
              bot.decide({
                player: p,
                teammates: eng.players.filter((x) => x.team === p.team),
                opponents: eng.players.filter((x) => x.team !== p.team),
                ball: eng.ball,
                allPlayers: eng.players,
                teamSide: p.team,
                controlledPlayerId: eng.controlledPlayerId,
                matchTime: eng.matchTimeSeconds,
              })
            );
          }
        });

        const res = eng.step(actionMap, 1 / 60);
        epRew += res.reward;

        if (res.terminated || res.truncated) {
          done = true;
          if (eng.score.left > 0 || (typeof res.info.event === 'object' && res.info.event?.type === 'goal')) {
            goals++;
          } else {
            dispossessed++;
          }
        }
      }
      rewards.push(epRew);
      lengths.push(epLen);
    }

    const meanRew = rewards.reduce((a, b) => a + b, 0) / 100;
    const meanLen = lengths.reduce((a, b) => a + b, 0) / 100;
    const possRate = (totalPossessionTicks / totalTicks) * 100;

    console.log(`Scripted Agent Results (100 episodes):`);
    console.log(`- Success / Goal Rate: ${(goals / 100) * 100}% (${goals}/100)`);
    console.log(`- Dispossessed / Saved: ${(dispossessed / 100) * 100}% (${dispossessed}/100)`);
    console.log(`- Mean Episode Reward: ${meanRew.toFixed(4)}`);
    console.log(`- Mean Steps to Goal: ${meanLen.toFixed(1)} steps (~${(meanLen / 60).toFixed(2)}s)`);
    console.log(`- Ball Possession Rate: ${possRate.toFixed(1)}%`);
  }

  // Test 3: Action Semantics Audit for Stage 2
  console.log('\n==================================================');
  console.log('4. ACTION SEMANTICS AUDIT ON STAGE 2');
  console.log('==================================================');
  {
    const eng = new GameEngine();
    eng.loadScenario(scenario);
    const obs = eng.step(new Map(), 1 / 60).observation;

    console.log(`Observation Dimensions: ${obs.rawVector.length}`);
    console.log(`- Own Player Pos (dim 0,1): (${obs.rawVector[0].toFixed(3)}, ${obs.rawVector[1].toFixed(3)})`);
    console.log(`- Own Player Vel (dim 22,23): (${obs.rawVector[22].toFixed(3)}, ${obs.rawVector[23].toFixed(3)})`);
    console.log(`- Right Player 1 (GK) Pos (dim 44,45): (${obs.rawVector[44].toFixed(3)}, ${obs.rawVector[45].toFixed(3)})`);
    console.log(`- Right Player 2 (CB) Pos (dim 46,47): (${obs.rawVector[46].toFixed(3)}, ${obs.rawVector[47].toFixed(3)})`);
    console.log(`- Ball Pos (dim 88,89,90): (${obs.rawVector[88].toFixed(3)}, ${obs.rawVector[89].toFixed(3)}, ${obs.rawVector[90].toFixed(3)})`);
    console.log(`- Ball Ownership (dim 94,95,96): [${obs.rawVector[94]}, ${obs.rawVector[95]}, ${obs.rawVector[96]}]`);
    console.log(`- Game Mode (dim 108..114): [${obs.rawVector.slice(108, 115).join(', ')}]`);
  }
}

runStage2Audit();
