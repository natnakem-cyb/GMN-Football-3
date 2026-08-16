import React, { useState } from 'react';
import { MatchEvent, ReplayFrame } from '../types/football';
import { Film, Play, Pause, SkipBack, SkipForward, Bookmark, FastForward } from 'lucide-react';

interface ReplayAnalyzerProps {
  replayFrames: ReplayFrame[];
  events: MatchEvent[];
  currentFrameIndex: number;
  onSeekFrame: (index: number) => void;
  isReplayMode: boolean;
  onToggleReplayMode: (active: boolean) => void;
}

export const ReplayAnalyzer: React.FC<ReplayAnalyzerProps> = ({
  replayFrames,
  events,
  currentFrameIndex,
  onSeekFrame,
  isReplayMode,
  onToggleReplayMode,
}) => {
  const [isPlayingReplay, setIsPlayingReplay] = useState(false);
  const totalFrames = replayFrames.length;
  const currentFrame = replayFrames[currentFrameIndex] || replayFrames[totalFrames - 1];

  const handleBookmarkClick = (event: MatchEvent) => {
    // Find closest frame to event time
    const targetIdx = replayFrames.findIndex(
      (f) => Math.abs(f.matchTimeSeconds - event.timeSeconds) < 0.1
    );
    if (targetIdx !== -1) {
      onSeekFrame(targetIdx);
    }
  };

  return (
    <div id="replay-analyzer-panel" className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-4 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <Film className="w-5 h-5 text-emerald-400" /> Match Replay & Frame-by-Frame Analyzer
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Deterministic state timeline scrubber with key event bookmarks and player telemetry.
          </p>
        </div>

        <button
          onClick={() => onToggleReplayMode(!isReplayMode)}
          className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all ${
            isReplayMode
              ? 'bg-amber-600 text-white border-amber-500 shadow-md'
              : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
          }`}
        >
          {isReplayMode ? '🔴 Exit Replay Studio' : '🎬 Enter Replay Studio'}
        </button>
      </div>

      {totalFrames === 0 ? (
        <div className="text-center py-8 text-slate-500 text-xs">
          No replay frames recorded yet. Start simulation to record frames.
        </div>
      ) : (
        <div className="space-y-4">
          {/* Timeline Scrubber */}
          <div>
            <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5">
              <span>
                Frame: <strong className="text-white">{currentFrameIndex + 1}</strong> / {totalFrames}
              </span>
              <span>
                Time:{' '}
                <strong className="text-amber-400 font-mono">
                  {currentFrame ? `${currentFrame.matchTimeSeconds.toFixed(1)}s` : '0.0s'}
                </strong>
              </span>
            </div>

            <input
              type="range"
              min={0}
              max={Math.max(0, totalFrames - 1)}
              value={currentFrameIndex}
              onChange={(e) => onSeekFrame(Number(e.target.value))}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
            />
          </div>

          {/* Stepper Buttons */}
          <div className="flex items-center justify-center gap-2">
            <button
              onClick={() => onSeekFrame(0)}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs"
              title="First Frame"
            >
              <SkipBack className="w-4 h-4" />
            </button>
            <button
              onClick={() => onSeekFrame(Math.max(0, currentFrameIndex - 10))}
              className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold"
            >
              -10 Ticks
            </button>
            <button
              onClick={() => onSeekFrame(Math.max(0, currentFrameIndex - 1))}
              className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold"
            >
              -1 Tick
            </button>
            <button
              onClick={() => onSeekFrame(Math.min(totalFrames - 1, currentFrameIndex + 1))}
              className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold"
            >
              +1 Tick
            </button>
            <button
              onClick={() => onSeekFrame(Math.min(totalFrames - 1, currentFrameIndex + 10))}
              className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold"
            >
              +10 Ticks
            </button>
            <button
              onClick={() => onSeekFrame(totalFrames - 1)}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs"
              title="Latest Frame"
            >
              <SkipForward className="w-4 h-4" />
            </button>
          </div>

          {/* Key Event Bookmarks */}
          <div>
            <h4 className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 mb-2">
              <Bookmark className="w-3.5 h-3.5 text-amber-400" /> Match Event Bookmarks
            </h4>

            {events.length === 0 ? (
              <p className="text-xs text-slate-500">No match events recorded yet.</p>
            ) : (
              <div className="flex flex-wrap gap-2 max-h-36 overflow-y-auto pr-1">
                {events.map((evt, idx) => (
                  <button
                    key={`${evt.id}_${idx}`}
                    onClick={() => handleBookmarkClick(evt)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-medium border flex items-center gap-1.5 transition-all ${
                      evt.type === 'goal'
                        ? 'bg-amber-950/80 text-amber-300 border-amber-500/50 hover:bg-amber-900'
                        : evt.type === 'shot'
                        ? 'bg-blue-950/80 text-blue-300 border-blue-500/40 hover:bg-blue-900'
                        : evt.type === 'tackle'
                        ? 'bg-red-950/80 text-red-300 border-red-500/40 hover:bg-red-900'
                        : 'bg-slate-950 text-slate-300 border-slate-800 hover:bg-slate-800'
                    }`}
                  >
                    <span>{evt.type === 'goal' ? '⚽' : evt.type === 'shot' ? '🎯' : '🛡️'}</span>
                    <span>{evt.timeSeconds.toFixed(1)}s</span>
                    <span className="text-[11px] opacity-80">{evt.description}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
