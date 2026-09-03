import React, { useState } from 'react';
import { ActionType, RLObservation, RLStepResult } from '../types/football';
import { Cpu, Terminal, Play, RotateCcw, Activity, Award, CheckSquare, Layers } from 'lucide-react';

interface RLGymnasiumPanelProps {
  lastStepResult: RLStepResult | null;
  onEnvReset: () => void;
  onEnvStepAction: (actionType: ActionType) => void;
  stepCount: number;
}

export const RLGymnasiumPanel: React.FC<RLGymnasiumPanelProps> = ({
  lastStepResult,
  onEnvReset,
  onEnvStepAction,
  stepCount,
}) => {
  const [vectorViewFilter, setVectorViewFilter] = useState<'all' | 'players' | 'ball' | 'match'>('all');

  const obs = lastStepResult?.observation;
  const rawVector = obs?.rawVector || [];

  return (
    <div id="rl-gymnasium-panel" className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-5 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-purple-400" /> Gymnasium RL Environment & SMM Vector Inspector
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Standard 115-float Google Research Football observation tensor, reward signals, and step interface.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onEnvReset}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700"
          >
            <RotateCcw className="w-3.5 h-3.5" /> env.reset()
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Step Reward & Telemetry */}
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
            <h4 className="text-xs font-bold text-slate-300 flex items-center gap-1.5 mb-3">
              <Award className="w-4 h-4 text-amber-400" /> Reward & Episode State
            </h4>

            <div className="space-y-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Step Count:</span>
                <span className="font-mono font-bold text-slate-100">{stepCount}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Immediate Reward:</span>
                <span
                  className={`font-mono font-bold ${
                    (lastStepResult?.reward || 0) > 0
                      ? 'text-emerald-400'
                      : (lastStepResult?.reward || 0) < 0
                      ? 'text-red-400'
                      : 'text-slate-300'
                  }`}
                >
                  {(lastStepResult?.reward || 0).toFixed(4)}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Checkpoint Reward:</span>
                <span
                  className={`font-mono font-semibold ${
                    (lastStepResult?.info.checkpointReward || 0) > 0
                      ? 'text-emerald-400'
                      : (lastStepResult?.info.checkpointReward || 0) < 0
                      ? 'text-red-400'
                      : 'text-slate-300'
                  }`}
                >
                  {(lastStepResult?.info.checkpointReward || 0).toFixed(4)}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Terminated / Truncated:</span>
                <span className="font-mono text-slate-300">
                  {lastStepResult?.terminated ? 'True (Terminated)' : lastStepResult?.truncated ? 'True (Truncated)' : 'False'}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Ball Dist to Goal:</span>
                <span className="font-mono text-blue-400">
                  {(lastStepResult?.info.ballDistanceToGoal || 0).toFixed(3)}
                </span>
              </div>
            </div>
          </div>

          {/* Action Step Trigger Panel */}
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
            <h4 className="text-xs font-bold text-slate-300 flex items-center gap-1.5 mb-2.5">
              <Terminal className="w-4 h-4 text-purple-400" /> Manual Discrete Action Trigger
            </h4>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => onEnvStepAction(ActionType.MOVE)}
                className="px-2 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700"
              >
                Action: MOVE
              </button>
              <button
                onClick={() => onEnvStepAction(ActionType.SPRINT)}
                className="px-2 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700"
              >
                Action: SPRINT
              </button>
              <button
                onClick={() => onEnvStepAction(ActionType.SHORT_PASS)}
                className="px-2 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-blue-400 border border-blue-500/20"
              >
                Action: PASS
              </button>
              <button
                onClick={() => onEnvStepAction(ActionType.SHOT)}
                className="px-2 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-amber-400 border border-amber-500/20"
              >
                Action: SHOT
              </button>
              <button
                onClick={() => onEnvStepAction(ActionType.TACKLE)}
                className="px-2 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-red-400 border border-red-500/20"
              >
                Action: TACKLE
              </button>
              <button
                onClick={() => onEnvStepAction(ActionType.IDLE)}
                className="px-2 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-400 border border-slate-700"
              >
                Action: IDLE
              </button>
            </div>
          </div>
        </div>

        {/* 115-Float SMM Observation Vector Tensor Matrix */}
        <div className="lg:col-span-2 p-4 rounded-xl bg-slate-950/60 border border-slate-800 flex flex-col">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
              <Layers className="w-4 h-4 text-emerald-400" /> SMM Vector Inspector (115 Dimensions)
            </h4>
            <div className="text-[10px] text-slate-400 font-mono">
              Shape: (115,) Float32
            </div>
          </div>

          <p className="text-[11px] text-slate-400 mb-2">
            Normalized coordinates: [-1..1, -0.42..0.42], velocities, ball z-axis, ownership flags.
          </p>

          <div className="flex-1 bg-slate-900 border border-slate-800 rounded-lg p-3 overflow-y-auto max-h-56 font-mono text-[11px] leading-relaxed">
            <div className="grid grid-cols-5 sm:grid-cols-8 gap-1.5 text-center">
              {rawVector.map((val, idx) => (
                <div
                  key={idx}
                  title={`Dim [${idx}] = ${val.toFixed(4)}`}
                  className={`px-1 py-0.5 rounded text-[10px] truncate border ${
                    Math.abs(val) > 0.001
                      ? val > 0
                        ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800/60'
                        : 'bg-blue-950/80 text-blue-300 border-blue-800/60'
                      : 'bg-slate-950 text-slate-600 border-slate-800'
                  }`}
                >
                  <span className="text-[8px] opacity-40 block">{idx}</span>
                  {val.toFixed(2)}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
