import React, { useEffect, useRef, useState, useCallback } from 'react';
import { GameEngine } from './engine/GameEngine';
import { HumanAgent } from './agents/HumanAgent';
import { RuleBasedAgent } from './agents/RuleBasedAgent';
import { NeuralHeuristicAgent } from './agents/NeuralHeuristicAgent';
import { TrainedPolicyAgent } from './agents/TrainedPolicyAgent';
import { ScriptedScenarioAgent } from './agents/ScriptedScenarioAgent';
import { ActionType, AgentAction, FormationType, RLStepResult, ScenarioConfig, TeamConfig, Vector2D } from './types/football';
import { ACADEMY_SCENARIOS } from './scenarios/ScenarioRegistry';

import { PitchCanvas } from './components/PitchCanvas';
import { Scoreboard } from './components/Scoreboard';
import { MatchControls } from './components/MatchControls';
import { AgentArenaPanel } from './components/AgentArenaPanel';
import { ScenarioSelector } from './components/ScenarioSelector';
import { ReplayAnalyzer } from './components/ReplayAnalyzer';
import { TacticalAnalytics } from './components/TacticalAnalytics';
import { RLGymnasiumPanel } from './components/RLGymnasiumPanel';
import { GeminiTacticalCoach } from './components/GeminiTacticalCoach';
import { ControlsHelpModal } from './components/ControlsHelpModal';

import {
  Trophy,
  GraduationCap,
  Film,
  Cpu,
  BarChart3,
  Bot,
  Sparkles,
  Zap,
  RotateCcw,
  Target,
  RefreshCw,
  Shield,
  Layers,
} from 'lucide-react';

type TabType = 'arena' | 'academy' | 'replay' | 'gymnasium' | 'analytics' | 'coach';

export default function App() {
  const engineRef = useRef<GameEngine>(new GameEngine());
  const humanAgentRef = useRef<HumanAgent>(new HumanAgent());
  const ruleAgentLeftRef = useRef<RuleBasedAgent>(new RuleBasedAgent('rule_left', 'Rule AI Left', 'medium'));
  const ruleAgentRightRef = useRef<RuleBasedAgent>(new RuleBasedAgent('rule_right', 'Rule AI Right', 'medium'));
  const neuralAgentRef = useRef<NeuralHeuristicAgent>(new NeuralHeuristicAgent());
  const trainedAgentRef = useRef<TrainedPolicyAgent | null>(null);
  const scriptedAgentRef = useRef<ScriptedScenarioAgent>(new ScriptedScenarioAgent());
  const warnedNeuralFallbackRef = useRef<boolean>(false);

  const [activeTab, setActiveTab] = useState<TabType>('arena');
  const [isPlaying, setIsPlaying] = useState(true);
  const [speedMultiplier, setSpeedMultiplier] = useState(1.0);
  const [showRadar, setShowRadar] = useState(true);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [isReplayMode, setIsReplayMode] = useState(false);
  const [replayFrameIndex, setReplayFrameIndex] = useState(0);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);

  // Engine React State Bridge
  const [, setRenderTrigger] = useState(0);
  const [lastStepResult, setLastStepResult] = useState<RLStepResult | null>(null);

  const engine = engineRef.current;

  // Cleanup on unmount & Async Model Loading
  useEffect(() => {
    let mounted = true;
    TrainedPolicyAgent.create('/models/mappo_policy.onnx')
      .then((agent) => {
        if (mounted) {
          trainedAgentRef.current = agent;
          setIsModelLoading(false);
        }
      })
      .catch((err) => {
        console.error('[TrainedPolicyAgent] Failed to load ONNX policy model:', err);
        if (mounted) {
          setModelError(err?.message || 'Failed to load model');
          setIsModelLoading(false);
        }
      });

    return () => {
      mounted = false;
      humanAgentRef.current.destroy();
    };
  }, []);

  // Main Simulation Animation Loop
  useEffect(() => {
    let animId: number;
    let lastTime = performance.now();
    let accumulatedTime = 0;
    const baseTickInterval = 1000 / 60; // 60Hz tick

    const loop = (currentTime: number) => {
      const deltaMs = currentTime - lastTime;
      lastTime = currentTime;

      if (isPlaying && !isReplayMode) {
        accumulatedTime += deltaMs * speedMultiplier;
        const tickInterval = baseTickInterval;

        // Run steps up to accumulated time (with a max of 10 ticks per frame to prevent freeze)
        let ticksRun = 0;
        while (accumulatedTime >= tickInterval && ticksRun < 10) {
          const actionMap = new Map<string, AgentAction>();

          const leftPlayersCount = engine.players.filter((p) => p.team === 'left').length;
          const is3v1Scenario = engine.activeScenario?.id === 'academy_3_vs_1_with_keeper' || leftPlayersCount === 3;

          // Gather decisions for each player
          engine.players.forEach((player) => {
            const isLeft = player.team === 'left';
            const teamConfig = isLeft ? engine.teamLeftConfig : engine.teamRightConfig;
            const teammates = engine.players.filter((p) => p.team === player.team);
            const opponents = engine.players.filter((p) => p.team !== player.team);

            const context = {
              player,
              teammates,
              opponents,
              ball: engine.ball,
              allPlayers: engine.players,
              teamSide: player.team,
              controlledPlayerId: engine.controlledPlayerId,
              matchTime: engine.matchTimeSeconds,
              gameMode: engine.gameMode,
            };

            let action: AgentAction = { type: ActionType.IDLE };

            if (isLeft && teamConfig.controller === 'human' && player.id === engine.controlledPlayerId) {
              action = humanAgentRef.current.decide(context);
            } else if (teamConfig.controller === 'neural') {
              if (isLeft && trainedAgentRef.current && is3v1Scenario) {
                action = trainedAgentRef.current.decide(context);
              } else {
                if (!warnedNeuralFallbackRef.current) {
                  console.warn(
                    `[controller=neural] fallback: no valid trained policy for team=${player.team}, scenario=${engine.activeScenario?.id ?? 'custom'}; using rule_based`
                  );
                  warnedNeuralFallbackRef.current = true;
                }
                const ruleAgent = isLeft ? ruleAgentLeftRef.current : ruleAgentRightRef.current;
                action = ruleAgent.decide(context);
              }
            } else if (teamConfig.controller === 'heuristic') {
              action = neuralAgentRef.current.decide(context);
            } else if (teamConfig.controller === 'scripted') {
              action = scriptedAgentRef.current.decide(context);
            } else {
              const ruleAgent = isLeft ? ruleAgentLeftRef.current : ruleAgentRightRef.current;
              action = ruleAgent.decide(context);
            }

            actionMap.set(player.id, action);
          });

          // Execute Step
          const stepRes = engine.step(actionMap, 1 / 60);
          setLastStepResult(stepRes);

          accumulatedTime -= tickInterval;
          ticksRun++;
        }

        setRenderTrigger((prev) => prev + 1);
      }

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [isPlaying, isReplayMode, speedMultiplier, engine]);

  // Handle Manual Step
  const handleStep = useCallback(() => {
    const actionMap = new Map<string, AgentAction>();
    const leftPlayersCount = engine.players.filter((p) => p.team === 'left').length;
    const is3v1Scenario = engine.activeScenario?.id === 'academy_3_vs_1_with_keeper' || leftPlayersCount === 3;

    engine.players.forEach((player) => {
      const isLeft = player.team === 'left';
      const teamConfig = isLeft ? engine.teamLeftConfig : engine.teamRightConfig;
      const context = {
        player,
        teammates: engine.players.filter((p) => p.team === player.team),
        opponents: engine.players.filter((p) => p.team !== player.team),
        ball: engine.ball,
        allPlayers: engine.players,
        teamSide: player.team,
        controlledPlayerId: engine.controlledPlayerId,
        matchTime: engine.matchTimeSeconds,
        gameMode: engine.gameMode,
      };

      let action: AgentAction = { type: ActionType.IDLE };
      if (isLeft && teamConfig.controller === 'human' && player.id === engine.controlledPlayerId) {
        action = humanAgentRef.current.decide(context);
      } else if (teamConfig.controller === 'neural') {
        if (isLeft && trainedAgentRef.current && is3v1Scenario) {
          action = trainedAgentRef.current.decide(context);
        } else {
          if (!warnedNeuralFallbackRef.current) {
            console.warn(
              `[controller=neural] fallback: no valid trained policy for team=${player.team}, scenario=${engine.activeScenario?.id ?? 'custom'}; using rule_based`
            );
            warnedNeuralFallbackRef.current = true;
          }
          const ruleAgent = isLeft ? ruleAgentLeftRef.current : ruleAgentRightRef.current;
          action = ruleAgent.decide(context);
        }
      } else if (teamConfig.controller === 'heuristic') {
        action = neuralAgentRef.current.decide(context);
      } else if (teamConfig.controller === 'scripted') {
        action = scriptedAgentRef.current.decide(context);
      } else {
        const ruleAgent = isLeft ? ruleAgentLeftRef.current : ruleAgentRightRef.current;
        action = ruleAgent.decide(context);
      }
      actionMap.set(player.id, action);
    });

    const res = engine.step(actionMap, 1 / 60);
    setLastStepResult(res);
    setRenderTrigger((prev) => prev + 1);
  }, [engine]);

  // Handle Match Reset
  const handleReset = useCallback(() => {
    if (engine.activeScenario) {
      engine.loadScenario(engine.activeScenario);
    } else {
      engine.resetToKickoff();
    }
    setRenderTrigger((prev) => prev + 1);
  }, [engine]);

  // Preset match configuration
  const handleApplyPreset = (type: 'human_vs_ai' | 'ai_vs_ai' | 'neural_vs_rule') => {
    if (type === 'human_vs_ai') {
      engine.teamLeftConfig.controller = 'human';
      engine.teamRightConfig.controller = 'rule_based';
    } else if (type === 'ai_vs_ai') {
      engine.teamLeftConfig.controller = 'rule_based';
      engine.teamRightConfig.controller = 'rule_based';
    } else if (type === 'neural_vs_rule') {
      engine.teamLeftConfig.controller = 'neural';
      engine.teamRightConfig.controller = 'rule_based';
    }
    setRenderTrigger((prev) => prev + 1);
  };

  // Scenario Loader
  const handleSelectScenario = (scenario: ScenarioConfig) => {
    engine.loadScenario(scenario);
    if (scenario.id === 'academy_3_vs_1_with_keeper') {
      engine.teamLeftConfig.controller = 'neural';
    } else if (engine.teamLeftConfig.controller === 'neural') {
      engine.teamLeftConfig.controller = 'human';
    }
    setIsPlaying(true);
    setIsReplayMode(false);
    setActiveTab('arena');
    setRenderTrigger((prev) => prev + 1);
  };

  // Free play
  const handleFreePlay = () => {
    engine.initDefaultMatch('4-3-3', '4-3-3', 11);
    if (engine.teamLeftConfig.controller === 'neural') {
      engine.teamLeftConfig.controller = 'human';
    }
    setIsPlaying(true);
    setIsReplayMode(false);
    setActiveTab('arena');
    setRenderTrigger((prev) => prev + 1);
  };

  // On-screen action trigger (for mouse / touch)
  const handleVirtualAction = (type: ActionType) => {
    humanAgentRef.current.triggerAction({ type, power: 0.85 });
  };

  // Current Replay Frame if in replay mode
  const currentReplayFrame = isReplayMode
    ? engine.replayBuffer[replayFrameIndex] || engine.replayBuffer[engine.replayBuffer.length - 1]
    : null;

  const displayBall = currentReplayFrame ? currentReplayFrame.ball : engine.ball;
  const displayPlayers = currentReplayFrame
    ? engine.players.map((p, idx) => {
        const rf = currentReplayFrame.players[idx];
        return rf
          ? {
              ...p,
              position: { ...rf.position },
              velocity: { ...rf.velocity },
              heading: rf.heading,
              stamina: rf.stamina,
              hasBall: rf.hasBall,
              isTackling: rf.isTackling,
            }
          : p;
      })
    : engine.players;

  const displayScore = currentReplayFrame ? currentReplayFrame.score : engine.score;
  const displayTime = currentReplayFrame ? currentReplayFrame.matchTimeSeconds : engine.matchTimeSeconds;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-emerald-500 selection:text-slate-950">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-40 bg-slate-900/90 backdrop-blur-md border-b border-slate-800 px-4 py-3">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center text-slate-950 font-black shadow-lg shadow-emerald-950/40 text-lg">
              ⚽
            </div>
            <div>
              <h1 className="text-base font-extrabold text-white tracking-tight flex items-center gap-2">
                GMN-Football <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-400 font-semibold border border-emerald-800/40">v1.0 Arena</span>
              </h1>
              <p className="text-[11px] text-slate-400">
                Google Research Football Web Simulation, RL Gym & AI Agent Platform
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center gap-1 bg-slate-950/80 p-1 rounded-xl border border-slate-800 overflow-x-auto max-w-full">
            <button
              id="tab-arena"
              onClick={() => setActiveTab('arena')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeTab === 'arena'
                  ? 'bg-emerald-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Target className="w-3.5 h-3.5" /> Pitch Arena
            </button>

            <button
              id="tab-academy"
              onClick={() => setActiveTab('academy')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeTab === 'academy'
                  ? 'bg-emerald-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <GraduationCap className="w-3.5 h-3.5" /> Academy Scenarios
            </button>

            <button
              id="tab-replay"
              onClick={() => {
                setActiveTab('replay');
                setIsReplayMode(true);
                setReplayFrameIndex(Math.max(0, engine.replayBuffer.length - 1));
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeTab === 'replay'
                  ? 'bg-amber-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Film className="w-3.5 h-3.5" /> Replay Studio
            </button>

            <button
              id="tab-gymnasium"
              onClick={() => setActiveTab('gymnasium')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeTab === 'gymnasium'
                  ? 'bg-purple-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Cpu className="w-3.5 h-3.5" /> RL Gym
            </button>

            <button
              id="tab-analytics"
              onClick={() => setActiveTab('analytics')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeTab === 'analytics'
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" /> Analytics
            </button>

            <button
              id="tab-coach"
              onClick={() => setActiveTab('coach')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeTab === 'coach'
                  ? 'bg-gradient-to-r from-amber-500 to-emerald-500 text-slate-950 shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" /> AI Coach
            </button>
          </nav>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 space-y-5">
        {/* Live Scoreboard */}
        <Scoreboard
          score={displayScore}
          matchTimeSeconds={displayTime}
          status={engine.status}
          teamLeft={engine.teamLeftConfig}
          teamRight={engine.teamRightConfig}
          possession={engine.stats.possession}
          scenarioName={engine.activeScenario?.name}
        />

        {/* Pitch Canvas View (Rendered in Arena, Academy, and Replay tabs) */}
        <div className="space-y-3">
          <PitchCanvas
            ball={displayBall as any}
            players={displayPlayers}
            teamLeftConfig={engine.teamLeftConfig}
            teamRightConfig={engine.teamRightConfig}
            controlledPlayerId={engine.controlledPlayerId}
            cameraMode="full"
            showRadar={showRadar}
            onPlayerClick={(id) => {
              const p = engine.players.find((pl) => pl.id === id);
              if (p && p.team === 'left') {
                engine.controlledPlayerId = p.id;
                setRenderTrigger((prev) => prev + 1);
              }
            }}
          />

          {/* On-Screen Touch / Action Bar for Quick Play & Testing */}
          {engine.teamLeftConfig.controller === 'human' && (
            <div className="flex flex-wrap items-center justify-between gap-2 p-3 bg-slate-900/90 border border-slate-800 rounded-xl">
              <div className="text-xs text-slate-400 flex items-center gap-2">
                <span className="font-semibold text-slate-200">Active Player:</span>
                <span className="text-emerald-400 font-bold">
                  {engine.players.find((p) => p.id === engine.controlledPlayerId)?.name || 'Player #10'}
                </span>
                <span className="text-[11px] text-slate-500 hidden sm:inline">(Click teammate on pitch to switch)</span>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleVirtualAction(ActionType.SHORT_PASS)}
                  className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs shadow-md transition-all"
                >
                  Pass [J / Space]
                </button>
                <button
                  onClick={() => handleVirtualAction(ActionType.HIGH_PASS)}
                  className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs shadow-md transition-all"
                >
                  Lob [L]
                </button>
                <button
                  onClick={() => handleVirtualAction(ActionType.SHOT)}
                  className="px-3 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-xs shadow-md transition-all"
                >
                  Shoot [K]
                </button>
                <button
                  onClick={() => handleVirtualAction(ActionType.TACKLE)}
                  className="px-3 py-1.5 rounded-lg bg-red-600 hover:bg-red-500 text-white font-bold text-xs shadow-md transition-all"
                >
                  Tackle [E]
                </button>
              </div>
            </div>
          )}

          {/* Match Play/Pause & Speed Controls */}
          <MatchControls
            isPlaying={isPlaying}
            onTogglePlay={() => setIsPlaying(!isPlaying)}
            onStep={handleStep}
            onReset={handleReset}
            speed={speedMultiplier}
            onSpeedChange={setSpeedMultiplier}
            showRadar={showRadar}
            onToggleRadar={() => setShowRadar(!showRadar)}
            onOpenHelp={() => setIsHelpOpen(true)}
            isHumanControlled={engine.teamLeftConfig.controller === 'human'}
          />
        </div>

        {/* Tab Specific Views */}
        {activeTab === 'arena' && (
          <AgentArenaPanel
            teamLeft={engine.teamLeftConfig}
            teamRight={engine.teamRightConfig}
            is3v1Scenario={engine.activeScenario?.id === 'academy_3_vs_1_with_keeper' || engine.players.filter((p) => p.team === 'left').length === 3}
            isModelLoading={isModelLoading}
            onUpdateTeamLeft={(cfg) => {
              Object.assign(engine.teamLeftConfig, cfg);
              if (cfg.formation) {
                engine.initDefaultMatch(cfg.formation, engine.teamRightConfig.formation, 11);
              }
              setRenderTrigger((prev) => prev + 1);
            }}
            onUpdateTeamRight={(cfg) => {
              Object.assign(engine.teamRightConfig, cfg);
              if (cfg.formation) {
                engine.initDefaultMatch(engine.teamLeftConfig.formation, cfg.formation, 11);
              }
              setRenderTrigger((prev) => prev + 1);
            }}
            onApplyPresetMatchup={handleApplyPreset}
          />
        )}

        {activeTab === 'academy' && (
          <ScenarioSelector
            activeScenario={engine.activeScenario}
            onSelectScenario={handleSelectScenario}
            onFreePlay={handleFreePlay}
            matchTimeSeconds={engine.matchTimeSeconds}
          />
        )}

        {activeTab === 'replay' && (
          <ReplayAnalyzer
            replayFrames={engine.replayBuffer}
            events={engine.events}
            currentFrameIndex={replayFrameIndex}
            onSeekFrame={setReplayFrameIndex}
            isReplayMode={isReplayMode}
            scenarioName={engine.activeScenario?.id || 'academy_3_vs_1_with_keeper'}
            onImportTrace={(importedFrames, importedEvents) => {
              engine.replayBuffer = importedFrames;
              engine.events = importedEvents;
              setReplayFrameIndex(0);
              setIsReplayMode(true);
              setRenderTrigger((prev) => prev + 1);
            }}
            onToggleReplayMode={(active) => {
              setIsReplayMode(active);
              if (!active) {
                setActiveTab('arena');
              }
            }}
          />
        )}

        {activeTab === 'gymnasium' && (
          <RLGymnasiumPanel
            lastStepResult={lastStepResult}
            onEnvReset={handleReset}
            onEnvStepAction={(actionType) => {
              const actionMap = new Map<string, AgentAction>();
              if (engine.controlledPlayerId) {
                actionMap.set(engine.controlledPlayerId, { type: actionType });
              }
              const res = engine.step(actionMap, 1 / 60);
              setLastStepResult(res);
              setRenderTrigger((prev) => prev + 1);
            }}
            stepCount={engine.tickCount}
          />
        )}

        {activeTab === 'analytics' && (
          <TacticalAnalytics
            stats={engine.stats}
            teamLeft={engine.teamLeftConfig}
            teamRight={engine.teamRightConfig}
          />
        )}

        {activeTab === 'coach' && (
          <GeminiTacticalCoach
            stats={engine.stats}
            teamLeft={engine.teamLeftConfig}
            teamRight={engine.teamRightConfig}
            score={engine.score}
            eventsSummary={engine.events.map((e) => `${e.timeSeconds.toFixed(0)}s: ${e.description}`).join('; ')}
          />
        )}
      </main>

      {/* Controls Help Modal */}
      <ControlsHelpModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />
    </div>
  );
}
