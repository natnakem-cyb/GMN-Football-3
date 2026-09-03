import React, { useState } from 'react';
import { PolicyActionDistribution } from '../types/telemetry';
import {
  BrainCircuit,
  Eye,
  Crosshair,
  Gauge,
  ArrowRight,
  TrendingUp,
  Shield,
  Layers,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

interface PolicyActionOverlayProps {
  distribution: PolicyActionDistribution | null;
  onSelectAction?: (actionIndex: number) => void;
  showAttentionVectors: boolean;
  onToggleAttentionVectors: () => void;
  isNeuralActive: boolean;
}

export const PolicyActionOverlay: React.FC<PolicyActionOverlayProps> = ({
  distribution,
  onSelectAction,
  showAttentionVectors,
  onToggleAttentionVectors,
  isNeuralActive,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!distribution) {
    return (
      <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800 text-xs text-slate-400 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BrainCircuit className="w-4 h-4 text-slate-500 animate-pulse" />
          <span>Select an agent on the pitch to inspect live neural policy logits and value estimates.</span>
        </div>
      </div>
    );
  }

  // Sort actions by probability descending
  const sortedActions = [...distribution.actions].sort((a, b) => b.probability - a.probability);
  const topActions = sortedActions.slice(0, 5);

  const getCategoryBadge = (category: string) => {
    switch (category) {
      case 'pass':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/40';
      case 'shot':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      case 'defense':
        return 'bg-rose-500/20 text-rose-300 border-rose-500/40';
      case 'move':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
      default:
        return 'bg-purple-500/20 text-purple-300 border-purple-500/40';
    }
  };

  const getCategoryBarColor = (category: string) => {
    switch (category) {
      case 'pass':
        return 'bg-blue-500';
      case 'shot':
        return 'bg-amber-500';
      case 'defense':
        return 'bg-rose-500';
      case 'move':
        return 'bg-emerald-500';
      default:
        return 'bg-purple-500';
    }
  };

  // Critic value gauge color
  const valueColor =
    distribution.valueEstimate > 0.3
      ? 'text-emerald-400'
      : distribution.valueEstimate > -0.1
      ? 'text-amber-400'
      : 'text-rose-400';

  const valueBarWidth = Math.min(100, Math.max(0, (distribution.valueEstimate + 1.0) * 50));

  return (
    <div className="rounded-xl bg-slate-900/95 border border-slate-800 shadow-xl overflow-hidden backdrop-blur-md">
      {/* Header Strip */}
      <div className="flex items-center justify-between px-3.5 py-2.5 bg-slate-800/60 border-b border-slate-700/60">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-lg bg-purple-500/20 border border-purple-500/40 flex items-center justify-center">
            <BrainCircuit className="w-3.5 h-3.5 text-purple-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-slate-100">Neural Policy Inference</span>
              <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-slate-300 border border-slate-700">
                Agent: {distribution.playerId}
              </span>
              <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-purple-950 text-purple-300 border border-purple-800">
                Role: {distribution.role}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onToggleAttentionVectors}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all ${
              showAttentionVectors
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50'
                : 'bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-700'
            }`}
            title="Toggle spatial attention cones and passing lanes on pitch"
          >
            <Eye className="w-3 h-3" />
            <span>Attention Cones</span>
          </button>

          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-800"
          >
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Main Body */}
      {isExpanded && (
        <div className="p-3.5 grid grid-cols-1 md:grid-cols-12 gap-4">
          {/* Left Column: Critic Value V(s) & Spatial Attention */}
          <div className="md:col-span-4 space-y-3">
            {/* Value Function Gauge */}
            <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80">
              <div className="flex items-center justify-between text-[11px] mb-1">
                <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                  <Gauge className="w-3 h-3 text-slate-400" />
                  Critic State Value V(s)
                </span>
                <span className={`font-mono font-bold ${valueColor}`}>
                  {distribution.valueEstimate >= 0 ? `+${distribution.valueEstimate}` : distribution.valueEstimate}
                </span>
              </div>

              {/* Progress bar */}
              <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden relative">
                <div
                  className="h-full bg-gradient-to-r from-rose-500 via-amber-500 to-emerald-500 transition-all duration-300"
                  style={{ width: `${valueBarWidth}%` }}
                />
                <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-slate-400/50" />
              </div>

              <div className="flex justify-between text-[9px] text-slate-500 mt-1 font-mono">
                <span>Defensive (-1.0)</span>
                <span>Neutral (0.0)</span>
                <span>Goal Adv (+1.0)</span>
              </div>
            </div>

            {/* Spatial Attention & Passing Lane Clearance */}
            <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80 space-y-1.5 text-xs">
              <div className="text-[11px] font-semibold text-slate-300 flex items-center gap-1.5">
                <Crosshair className="w-3 h-3 text-cyan-400" />
                Spatial Attention
              </div>

              {distribution.attention ? (
                <div className="space-y-1 text-[11px]">
                  <div className="flex justify-between text-slate-400">
                    <span>Target Receiver:</span>
                    <span className="font-semibold text-slate-200">
                      {distribution.attention.targetPlayerId || 'In Space'}
                    </span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Lane Clearance:</span>
                    <span className="font-mono font-bold text-cyan-400">
                      {distribution.attention.passClearanceProb}%
                    </span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Shot Goal Vector:</span>
                    <span className="font-mono font-bold text-amber-400">
                      {distribution.attention.shotAngleClearance}%
                    </span>
                  </div>
                </div>
              ) : (
                <div className="text-[11px] text-slate-500">No active passing vector computed.</div>
              )}
            </div>

            {/* Top Decision Summary */}
            <div className="p-2.5 rounded-lg bg-purple-950/30 border border-purple-900/40 text-xs">
              <div className="text-[11px] text-purple-300 font-semibold mb-1">Argmax Action</div>
              <div className="flex items-center justify-between">
                <span className="font-bold text-white text-sm">{distribution.bestActionName}</span>
                <span className="font-mono font-extrabold text-purple-400">{distribution.confidence}%</span>
              </div>
            </div>
          </div>

          {/* Right Column: Categorical Softmax Action Distribution */}
          <div className="md:col-span-8 space-y-2">
            <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
              <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                <Layers className="w-3 h-3 text-slate-400" />
                Categorical Action Distribution (Top 5 of 19 Discrete Actions)
              </span>
              <span className="text-[10px] text-slate-500">Softmax $\pi_\theta(a|s)$</span>
            </div>

            <div className="space-y-1.5">
              {topActions.map((action, rank) => {
                const pct = (action.probability * 100).toFixed(1);
                return (
                  <div
                    key={action.index}
                    onClick={() => onSelectAction?.(action.index)}
                    className="p-1.5 rounded-lg bg-slate-950/60 border border-slate-800/80 hover:border-slate-700 transition-all cursor-pointer group"
                  >
                    <div className="flex items-center justify-between text-xs mb-1">
                      <div className="flex items-center gap-2">
                        <span className="w-4 text-[10px] font-mono text-slate-500">#{rank + 1}</span>
                        <span className="font-semibold text-slate-200 group-hover:text-emerald-400 transition-colors">
                          {action.name}
                        </span>
                        <span
                          className={`px-1.5 py-0.2 text-[9px] font-bold rounded border uppercase ${getCategoryBadge(
                            action.category
                          )}`}
                        >
                          {action.category}
                        </span>
                      </div>

                      <div className="flex items-center gap-2 font-mono">
                        <span className="text-[10px] text-slate-500">logit: {action.logit.toFixed(2)}</span>
                        <span className="font-bold text-slate-100 text-xs w-11 text-right">{pct}%</span>
                      </div>
                    </div>

                    <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                      <div
                        className={`h-full ${getCategoryBarColor(action.category)} transition-all duration-300`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
