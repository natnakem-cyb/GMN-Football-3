import React from 'react';
import { Play, Pause, StepForward, RotateCcw, Zap, Compass, Keyboard, ShieldAlert } from 'lucide-react';

interface MatchControlsProps {
  isPlaying: boolean;
  onTogglePlay: () => void;
  onStep: () => void;
  onReset: () => void;
  speed: number;
  onSpeedChange: (speed: number) => void;
  showRadar: boolean;
  onToggleRadar: () => void;
  onOpenHelp: () => void;
  isHumanControlled: boolean;
}

export const MatchControls: React.FC<MatchControlsProps> = ({
  isPlaying,
  onTogglePlay,
  onStep,
  onReset,
  speed,
  onSpeedChange,
  showRadar,
  onToggleRadar,
  onOpenHelp,
  isHumanControlled,
}) => {
  return (
    <div
      id="match-controls"
      className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/90 border border-slate-800 p-3 rounded-2xl shadow-lg mt-3"
    >
      {/* Play / Step / Reset Group */}
      <div className="flex items-center gap-2">
        <button
          id="btn-play-pause"
          onClick={onTogglePlay}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl font-semibold text-sm transition-all shadow-md ${
            isPlaying
              ? 'bg-amber-600 hover:bg-amber-500 text-white'
              : 'bg-emerald-600 hover:bg-emerald-500 text-white'
          }`}
        >
          {isPlaying ? (
            <>
              <Pause className="w-4 h-4" /> Pause
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" /> Play
            </>
          )}
        </button>

        <button
          id="btn-step-frame"
          onClick={onStep}
          disabled={isPlaying}
          title="Step 1 Tick Forward"
          className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 text-sm font-medium transition-all"
        >
          <StepForward className="w-4 h-4" />
          <span className="hidden sm:inline">Step</span>
        </button>

        <button
          id="btn-reset-match"
          onClick={onReset}
          title="Reset Match to Kickoff"
          className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium transition-all"
        >
          <RotateCcw className="w-4 h-4" />
          <span className="hidden sm:inline">Reset</span>
        </button>
      </div>

      {/* Speed Multipliers */}
      <div className="flex items-center gap-1 bg-slate-950/80 p-1 rounded-xl border border-slate-800">
        {[0.5, 1.0, 2.0, 5.0].map((s) => (
          <button
            key={s}
            id={`btn-speed-${s}`}
            onClick={() => onSpeedChange(s)}
            className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
              speed === s
                ? 'bg-emerald-500 text-slate-950 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {s}x
          </button>
        ))}
      </div>

      {/* Auxiliary Controls: Radar & Controls Modal */}
      <div className="flex items-center gap-2">
        <button
          id="btn-toggle-radar"
          onClick={onToggleRadar}
          className={`flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium border transition-all ${
            showRadar
              ? 'bg-slate-800 border-emerald-500/50 text-emerald-400'
              : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
          }`}
        >
          <Compass className="w-4 h-4" />
          <span className="hidden sm:inline">Radar Minimap</span>
        </button>

        <button
          id="btn-controls-help"
          onClick={onOpenHelp}
          className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-amber-400 border border-amber-500/30 text-xs font-semibold transition-all"
        >
          <Keyboard className="w-4 h-4" />
          <span>Controls Guide</span>
        </button>
      </div>
    </div>
  );
};
