import React, { useState, useRef } from 'react';
import { MatchEvent, ReplayFrame } from '../types/football';
import { Film, SkipBack, SkipForward, Bookmark, Download, Upload, CheckCircle, AlertCircle, FileText } from 'lucide-react';
import { exportReplayToJsonl, parseJsonlTrace, TraceMetadata } from '../utils/replaySerializer';

interface ReplayAnalyzerProps {
  replayFrames: ReplayFrame[];
  events: MatchEvent[];
  currentFrameIndex: number;
  onSeekFrame: (index: number) => void;
  isReplayMode: boolean;
  onToggleReplayMode: (active: boolean) => void;
  scenarioName?: string;
  onImportTrace?: (frames: ReplayFrame[], events: MatchEvent[], metadata: TraceMetadata) => void;
}

export const ReplayAnalyzer: React.FC<ReplayAnalyzerProps> = ({
  replayFrames,
  events,
  currentFrameIndex,
  onSeekFrame,
  isReplayMode,
  onToggleReplayMode,
  scenarioName = 'academy_3_vs_1_with_keeper',
  onImportTrace,
}) => {
  const [importStatus, setImportStatus] = useState<{ msg: string; type: 'success' | 'error' } | null>(null);
  const [importedMeta, setImportedMeta] = useState<TraceMetadata | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const totalFrames = replayFrames.length;
  const currentFrame = replayFrames[currentFrameIndex] || replayFrames[totalFrames - 1];

  const handleBookmarkClick = (event: MatchEvent) => {
    const targetIdx = replayFrames.findIndex(
      (f) => Math.abs(f.matchTimeSeconds - event.timeSeconds) < 0.1
    );
    if (targetIdx !== -1) {
      onSeekFrame(targetIdx);
    }
  };

  const handleExportJsonl = () => {
    if (replayFrames.length === 0) return;
    const jsonlContent = exportReplayToJsonl(replayFrames, events, scenarioName);
    const blob = new Blob([jsonlContent], { type: 'application/x-ndjson;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    link.href = url;
    link.download = `replay_${scenarioName}_${timestamp}.jsonl`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const text = evt.target?.result as string;
        const { metadata, frames, events: parsedEvents } = parseJsonlTrace(text);
        setImportedMeta(metadata);
        setImportStatus({
          msg: `Successfully imported ${frames.length} frames from ${file.name} (Scenario: ${metadata.scenario})`,
          type: 'success',
        });
        if (onImportTrace) {
          onImportTrace(frames, parsedEvents, metadata);
        }
        if (!isReplayMode) {
          onToggleReplayMode(true);
        }
      } catch (err: any) {
        setImportStatus({
          msg: `Failed to parse trace file: ${err.message || 'Invalid JSONL format'}`,
          type: 'error',
        });
      }
    };
    reader.readAsText(file);
  };

  return (
    <div id="replay-analyzer-panel" className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-4 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <Film className="w-5 h-5 text-emerald-400" /> Match Replay & Frame-by-Frame Analyzer
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Deterministic state timeline scrubber with JSONL trace import/export, key event bookmarks, and telemetry.
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* Export JSONL Button */}
          <button
            onClick={handleExportJsonl}
            disabled={totalFrames === 0}
            className="px-3 py-1.5 rounded-xl text-xs font-semibold border border-slate-700 bg-slate-800 text-slate-200 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5 transition-all shadow-sm"
            title="Export full replay trace to .jsonl file"
          >
            <Download className="w-3.5 h-3.5 text-emerald-400" />
            Export Trace (.jsonl)
          </button>

          {/* Import JSONL Button */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept=".jsonl,.json"
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-3 py-1.5 rounded-xl text-xs font-semibold border border-slate-700 bg-slate-800 text-slate-200 hover:bg-slate-700 flex items-center gap-1.5 transition-all shadow-sm"
            title="Import an RL episode trace or match replay"
          >
            <Upload className="w-3.5 h-3.5 text-cyan-400" />
            Import Trace (.jsonl)
          </button>

          {/* Toggle Replay Mode */}
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
      </div>

      {importStatus && (
        <div
          className={`mb-4 p-3 rounded-xl border text-xs flex items-center gap-2 ${
            importStatus.type === 'success'
              ? 'bg-emerald-950/60 border-emerald-500/40 text-emerald-300'
              : 'bg-rose-950/60 border-rose-500/40 text-rose-300'
          }`}
        >
          {importStatus.type === 'success' ? (
            <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
          ) : (
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          )}
          <span>{importStatus.msg}</span>
        </div>
      )}

      {importedMeta && (
        <div className="mb-4 p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 text-[11px] text-slate-400 flex flex-wrap items-center gap-4">
          <span className="flex items-center gap-1 text-slate-300">
            <FileText className="w-3.5 h-3.5 text-cyan-400" />
            <strong>Trace Meta:</strong> {importedMeta.scenario}
          </span>
          {importedMeta.seed !== null && <span>Seed: {importedMeta.seed}</span>}
          {importedMeta.checkpoint && <span>Checkpoint: {importedMeta.checkpoint}</span>}
          <span>Recorded: {new Date(importedMeta.recorded_at).toLocaleString()}</span>
        </div>
      )}

      {totalFrames === 0 ? (
        <div className="text-center py-8 text-slate-500 text-xs">
          No replay frames recorded yet. Start simulation or import a .jsonl trace to analyze frames.
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
