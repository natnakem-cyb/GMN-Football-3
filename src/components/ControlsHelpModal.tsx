import React from 'react';
import { X, Keyboard, Gamepad2, Move, Target, Zap, Shield, RefreshCw } from 'lucide-react';

interface ControlsHelpModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ControlsHelpModal: React.FC<ControlsHelpModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="w-full max-w-lg bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-800 bg-slate-950/60">
          <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
            <Keyboard className="w-4 h-4 text-amber-400" /> Controls & Keyboard Shortcuts Guide
          </h3>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-5 space-y-4 max-h-[80vh] overflow-y-auto">
          {/* Movement */}
          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <Move className="w-3.5 h-3.5 text-blue-400" /> Player Movement
              </span>
              <div className="flex gap-1">
                <kbd className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[11px] font-mono text-slate-200">W</kbd>
                <kbd className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[11px] font-mono text-slate-200">A</kbd>
                <kbd className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[11px] font-mono text-slate-200">S</kbd>
                <kbd className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[11px] font-mono text-slate-200">D</kbd>
                <span className="text-slate-500 text-xs self-center">or</span>
                <kbd className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[11px] font-mono text-slate-200">Arrows</kbd>
              </div>
            </div>
            <p className="text-[11px] text-slate-400">
              Steer the active player (indicated by yellow marker) across the pitch in 360 degrees.
            </p>
          </div>

          {/* Sprint */}
          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-emerald-400" /> Sprint Acceleration
              </span>
              <kbd className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[11px] font-mono text-slate-200">
                Shift (Hold)
              </kbd>
            </div>
            <p className="text-[11px] text-slate-400">
              Accelerate to maximum speed (+35% velocity). Drains stamina ring.
            </p>
          </div>

          {/* Pass & Lob */}
          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <RefreshCw className="w-3.5 h-3.5 text-blue-400" /> Short Pass / High Lob Pass
              </span>
              <div className="flex gap-1.5">
                <kbd className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[11px] font-mono text-slate-200">J</kbd>
                <span className="text-slate-500 text-xs">/</span>
                <kbd className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[11px] font-mono text-slate-200">Space</kbd>
                <span className="text-slate-500 text-xs">|</span>
                <kbd className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[11px] font-mono text-slate-200">L (High)</kbd>
              </div>
            </div>
            <p className="text-[11px] text-slate-400">
              Direct the ball along the ground [J/Space] or loft over defenders [L] to an open teammate.
            </p>
          </div>

          {/* Shoot */}
          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <Target className="w-3.5 h-3.5 text-amber-400" /> Strike at Goal (Shot)
              </span>
              <kbd className="px-2.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-[11px] font-mono text-amber-400 font-bold">
                K
              </kbd>
            </div>
            <p className="text-[11px] text-slate-400">
              Power a shot toward the goal corners with aerodynamic trajectory.
            </p>
          </div>

          {/* Slide Tackle / Switch */}
          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <Shield className="w-3.5 h-3.5 text-red-400" /> Slide Tackle / Switch Player
              </span>
              <kbd className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[11px] font-mono text-slate-200">
                E / Q
              </kbd>
            </div>
            <p className="text-[11px] text-slate-400">
              When defending: perform a slide tackle to dispossess opponents or switch to the nearest teammate.
            </p>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/60 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-bold text-xs shadow-md transition-all"
          >
            Got It
          </button>
        </div>
      </div>
    </div>
  );
};
