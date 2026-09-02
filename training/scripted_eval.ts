import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { ActionType, AgentAction } from '../src/types/football';
import { RuleBasedAgent } from '../src/agents/RuleBasedAgent';
import { Vec2 } from '../src/engine/Vector';

export function evaluateScriptedAgent(numEpisodes = 100) {
  console.log('==================================================');
  console.log('5. SCRIPTED POLICY BASELINE');
  console.log(`Evaluating RuleBasedAgent on academy_empty_goal (${numEpisodes} episodes)`);
  console.log('==================================================');

  const scenario = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_empty_goal')!;
  let goals = 0;
  const rewards: number[] = [];
  const stepsList: number[] = [];

  const agent = new RuleBasedAgent('scripted_striker', 'Academy Striker', 'medium');

  for (let ep = 0; ep < numEpisodes; ep++) {
    const engine = new GameEngine();
    engine.loadScenario(scenario);

    let epReward = 0;
    let epSteps = 0;
    let done = false;

    while (!done && epSteps < 900) {
      epSteps++;
      const player = engine.players[0];

      const action = agent.decide({
        player,
        teammates: [player],
        opponents: [],
        ball: engine.ball,
        allPlayers: engine.players,
        teamSide: 'left',
        controlledPlayerId: player.id,
        matchTime: engine.matchTimeSeconds,
      });

      const actionMap = new Map<string, AgentAction>([[player.id, action]]);
      const res = engine.step(actionMap, 1 / 60);
      epReward += res.reward;

      if (res.terminated || res.truncated) {
        done = true;
        if (engine.score.left > 0 || (typeof res.info.event === 'string' && res.info.event.toLowerCase().includes('goal'))) {
          goals++;
        }
      }
    }

    rewards.push(epReward);
    stepsList.push(epSteps);
  }

  const successRate = (goals / numEpisodes) * 100;
  const meanReward = rewards.reduce((a, b) => a + b, 0) / numEpisodes;
  const meanSteps = stepsList.reduce((a, b) => a + b, 0) / numEpisodes;
  const minSteps = Math.min(...stepsList);
  const maxSteps = Math.max(...stepsList);

  console.log(`Results:`);
  console.log(`- Success Rate: ${successRate.toFixed(1)}% (${goals}/${numEpisodes} goals)`);
  console.log(`- Mean Episode Reward: ${meanReward.toFixed(4)}`);
  console.log(`- Mean Steps to Goal: ${meanSteps.toFixed(1)} steps (~${(meanSteps / 60).toFixed(2)}s)`);
  console.log(`- Min Steps: ${minSteps} | Max Steps: ${maxSteps}`);
  console.log('==================================================\n');
}

evaluateScriptedAgent(100);
