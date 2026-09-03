import React from 'react';
import { AgentCreditMetrics } from '../types/telemetry';
import { Users, Award, Zap, Shield, Compass, BarChart2 } from 'lucide-react';

interface MultiAgentCreditMatrixProps {
  metrics: AgentCreditMetrics[];
  onSelectAgent?: (playerId: string) => void;
  selectedPlayerId?: string | null;
}

export const MultiAgentCreditMatrix: React.FC<MultiAgentCreditMatrixProps> = ({
  metrics,
  onSelectAgent,
  selectedPlayerId,
}) => {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-400 text-center">
        No active multi-agent team deployed on pitch.
      </div>
    );
  }

  // Calculate total reward contribution for relative percentage
  const totalReward = metrics.reduce((acc, m) => acc + Math.max(0, m.rewardContribution), 0) || 1;

  return (
    <div className="rounded-xl bg-slate-900/90 border border-slate-800 overflow-hidden shadow-lg">
      {/* Header */}
      <div className="px-4 py-3 bg-slate-800/60 border-b border-slate-700/60 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center">
            <Users className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-slate-100">Cooperative Multi-Agent Credit Assignment</h3>
            <p className="text-[11px] text-slate-400">
              Counterfactual advantage decomposition & joint policy contribution
            </p>
          </div>
        </div>

        <div className="text-[11px] text-slate-400 flex items-center gap-1.5 font-mono">
          <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-emerald-400 font-semibold">
            {metrics.length} Controllable Agents
          </span>
        </div>
      </div>

      {/* Credit Share Bar */}
      <div className="p-3 bg-slate-950/40 border-b border-slate-800">
        <div className="flex justify-between text-[10px] text-slate-400 mb-1 font-mono">
          <span>Joint Reward Contribution Share</span>
          <span>100% Team Credit</span>
        </div>
        <div className="w-full h-2.5 rounded-full bg-slate-800 overflow-hidden flex">
          {metrics.map((m, idx) => {
            const sharePct = ((Math.max(0, m.rewardContribution) / totalReward) * 100).toFixed(1);
            const colors = ['bg-emerald-500', 'bg-blue-500', 'bg-purple-500', 'bg-amber-500', 'bg-cyan-500'];
            const color = colors[idx % colors.length];
            return (
              <div
                key={m.playerId}
                className={`${color} h-full transition-all duration-300 relative group`}
                style={{ width: `${sharePct}%` }}
                title={`${m.playerName} (${m.role}): ${sharePct}%`}
              />
            );
          })}
        </div>
      </div>

      {/* Agents Grid Cards */}
      <div className="p-3.5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {metrics.map((m, idx) => {
          const isSelected = selectedPlayerId === m.playerId;
          const sharePct = ((Math.max(0, m.rewardContribution) / totalReward) * 100).toFixed(1);

          return (
            <div
              key={m.playerId}
              onClick={() => onSelectAgent?.(m.playerId)}
              className={`p-3 rounded-xl border transition-all cursor-pointer ${
                isSelected
                  ? 'bg-slate-800/90 border-emerald-500 shadow-md shadow-emerald-950/30'
                  : 'bg-slate-950/60 border-slate-800/90 hover:border-slate-700'
              }`}
            >
              {/* Agent Title & Role */}
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-xs font-bold text-slate-200">
                    {idx + 1}
                  </div>
                  <div>
                    <div className="text-xs font-bold text-slate-200">{m.playerName}</div>
                    <div className="text-[10px] text-slate-400 font-mono">{m.playerId}</div>
                  </div>
                </div>

                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-800 font-mono">
                  {m.role}
                </span>
              </div>

              {/* Counterfactual Advantage Value */}
              <div className="p-2 rounded-lg bg-slate-900/90 border border-slate-800/80 mb-2.5">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-slate-400 flex items-center gap-1">
                    <Zap className="w-3 h-3 text-amber-400" />
                    Advantage $A_i$:
                  </span>
                  <span className="font-mono font-bold text-amber-400">
                    {m.counterfactualAdvantage >= 0 ? `+${m.counterfactualAdvantage}` : m.counterfactualAdvantage}
                  </span>
                </div>
                <div className="flex items-center justify-between text-[11px] mt-1">
                  <span className="text-slate-400 flex items-center gap-1">
                    <Award className="w-3 h-3 text-emerald-400" />
                    Credit Share:
                  </span>
                  <span className="font-mono font-bold text-emerald-400">{sharePct}%</span>
                </div>
              </div>

              {/* Specific Performance Metrics */}
              <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                <div className="p-1.5 rounded bg-slate-900/60 border border-slate-800/50">
                  <div className="text-slate-500">Space Creation</div>
                  <div className="text-slate-200 font-bold">{m.spaceCreationScore} pts</div>
                </div>
                <div className="p-1.5 rounded bg-slate-900/60 border border-slate-800/50">
                  <div className="text-slate-500">Pass Accuracy</div>
                  <div className="text-blue-400 font-bold">{m.passCompletionRate}%</div>
                </div>
                <div className="p-1.5 rounded bg-slate-900/60 border border-slate-800/50">
                  <div className="text-slate-500">Key Passes</div>
                  <div className="text-purple-400 font-bold">{m.keyPasses}</div>
                </div>
                <div className="p-1.5 rounded bg-slate-900/60 border border-slate-800/50">
                  <div className="text-slate-500">Discipline</div>
                  <div className="text-emerald-400 font-bold">{m.positionalDiscipline}%</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
