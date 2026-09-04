import React from 'react';
import { FormationType, TeamConfig } from '../types/football';
import { Bot, User, Brain, Sliders, Shield, Award, AlertTriangle } from 'lucide-react';

interface AgentArenaPanelProps {
  teamLeft: TeamConfig;
  teamRight: TeamConfig;
  onUpdateTeamLeft: (config: Partial<TeamConfig>) => void;
  onUpdateTeamRight: (config: Partial<TeamConfig>) => void;
  onApplyPresetMatchup: (type: 'human_vs_ai' | 'ai_vs_ai' | 'neural_vs_rule') => void;
  is3v1Scenario?: boolean;
  isModelLoading?: boolean;
  modelError?: string | null;
}

export const AgentArenaPanel: React.FC<AgentArenaPanelProps> = ({
  teamLeft,
  teamRight,
  onUpdateTeamLeft,
  onUpdateTeamRight,
  onApplyPresetMatchup,
  is3v1Scenario = false,
  isModelLoading = false,
  modelError = null,
}) => {
  const formations: FormationType[] = ['4-3-3', '4-4-2', '3-5-2', '5-3-2', '1-2-1'];

  return (
    <div id="agent-arena-panel" className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-5 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <Bot className="w-5 h-5 text-emerald-400" /> AI Agent Arena & Team Configuration
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Configure agent controllers, tactical formations, and neural policy difficulty.
          </p>
        </div>

        {/* Quick Presets */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onApplyPresetMatchup('human_vs_ai')}
            className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-blue-400 border border-blue-500/20"
          >
            Human vs AI
          </button>
          <button
            onClick={() => onApplyPresetMatchup('ai_vs_ai')}
            className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-emerald-400 border border-emerald-500/20"
          >
            AI vs AI
          </button>
          <button
            onClick={() => onApplyPresetMatchup('neural_vs_rule')}
            className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-purple-400 border border-purple-500/20"
          >
            Neural vs Rule
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left Team (Home) Config */}
        <div className="p-4 rounded-xl bg-slate-950/60 border border-blue-500/30">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="w-3.5 h-3.5 rounded-full" style={{ backgroundColor: teamLeft.color }} />
              <span className="font-bold text-slate-100">{teamLeft.name} (Left)</span>
            </div>
            <span className="text-xs text-blue-400 font-semibold px-2 py-0.5 bg-blue-950 rounded-full border border-blue-800">
              Home
            </span>
          </div>

          <div className="space-y-3">
            <div>
              <label className="text-xs font-semibold text-slate-300 mb-1 block">Controller Type</label>
              <select
                value={teamLeft.controller}
                onChange={(e) => onUpdateTeamLeft({ controller: e.target.value as any })}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100 font-medium focus:ring-1 focus:ring-blue-500"
              >
                <option value="human">👤 Human Player (Keyboard / Touch)</option>
                <option value="rule_based">🤖 Tactical Rule AI</option>
                {!modelError ? (
                  <option value="neural">
                    🧠 Neural Policy (Trained MAPPO — ONNX / 127-dim)
                  </option>
                ) : (
                  <option value="neural" disabled>
                    🧠 Neural Policy (Unavailable: {modelError})
                  </option>
                )}
                <option value="heuristic">📊 Heuristic Bot (Untrained Baseline)</option>
                <option value="scripted">📜 Scripted Scenario Bot</option>
              </select>
              {modelError && (
                <div className="mt-2 p-2 rounded-lg bg-amber-950/80 border border-amber-600/80 text-[11px] text-amber-300 flex items-center gap-1.5 font-semibold animate-fadeIn">
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                  <span>Neural Model Unavailable – Reverting to Rule-Based ({modelError})</span>
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-semibold text-slate-300 mb-1 block">Formation</label>
                <select
                  value={teamLeft.formation}
                  onChange={(e) => onUpdateTeamLeft({ formation: e.target.value as any })}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100 font-medium"
                >
                  {formations.map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-300 mb-1 block">AI Difficulty</label>
                <select
                  value={teamLeft.aiDifficulty}
                  onChange={(e) => onUpdateTeamLeft({ aiDifficulty: e.target.value as any })}
                  disabled={teamLeft.controller === 'human'}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100 font-medium disabled:opacity-40"
                >
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                  <option value="master">Master</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Right Team (Away) Config */}
        <div className="p-4 rounded-xl bg-slate-950/60 border border-red-500/30">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="w-3.5 h-3.5 rounded-full" style={{ backgroundColor: teamRight.color }} />
              <span className="font-bold text-slate-100">{teamRight.name} (Right)</span>
            </div>
            <span className="text-xs text-red-400 font-semibold px-2 py-0.5 bg-red-950 rounded-full border border-red-800">
              Away
            </span>
          </div>

          <div className="space-y-3">
            <div>
              <label className="text-xs font-semibold text-slate-300 mb-1 block">Controller Type</label>
              <select
                value={teamRight.controller}
                onChange={(e) => onUpdateTeamRight({ controller: e.target.value as any })}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100 font-medium focus:ring-1 focus:ring-red-500"
              >
                <option value="rule_based">🤖 Tactical Rule AI</option>
                <option value="heuristic">📊 Heuristic Bot (Untrained Baseline)</option>
                <option value="human">👤 Human Player</option>
                <option value="scripted">📜 Scripted Scenario Bot</option>
                <option value="neural" disabled title="Trained policy is left-side only (trained to attack the right goal)">
                  🧠 Neural Policy (Left-side only, no mirroring)
                </option>
              </select>
              <p className="text-[11px] text-slate-400 mt-1">
                Neural policy is left-side only (trained to attack the right goal).
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-semibold text-slate-300 mb-1 block">Formation</label>
                <select
                  value={teamRight.formation}
                  onChange={(e) => onUpdateTeamRight({ formation: e.target.value as any })}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100 font-medium"
                >
                  {formations.map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-300 mb-1 block">AI Difficulty</label>
                <select
                  value={teamRight.aiDifficulty}
                  onChange={(e) => onUpdateTeamRight({ aiDifficulty: e.target.value as any })}
                  disabled={teamRight.controller === 'human'}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100 font-medium disabled:opacity-40"
                >
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                  <option value="master">Master</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
