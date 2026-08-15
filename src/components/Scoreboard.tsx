import React from 'react';
import { MatchScore, MatchStateStatus, TeamConfig } from '../types/football';
import { Shield, Trophy, Activity, Radio } from 'lucide-react';

interface ScoreboardProps {
  score: MatchScore;
  matchTimeSeconds: number;
  status: MatchStateStatus;
  teamLeft: TeamConfig;
  teamRight: TeamConfig;
  possession: { left: number; right: number };
  scenarioName?: string;
}

export const Scoreboard: React.FC<ScoreboardProps> = ({
  score,
  matchTimeSeconds,
  status,
  teamLeft,
  teamRight,
  possession,
  scenarioName,
}) => {
  const minutes = Math.floor(matchTimeSeconds / 60);
  const seconds = Math.floor(matchTimeSeconds % 60);
  const timeFormatted = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

  return (
    <div
      id="match-scoreboard"
      className="w-full bg-slate-900/90 backdrop-blur-md border border-slate-800 rounded-2xl p-4 shadow-xl mb-4"
    >
      <div className="flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Left Team Badge & Name */}
        <div className="flex items-center gap-3 w-full md:w-1/3 justify-start">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center font-bold text-white shadow-md transition-transform"
            style={{ backgroundColor: teamLeft.color }}
          >
            {teamLeft.shortName}
          </div>
          <div>
            <div className="font-bold text-slate-100 flex items-center gap-2 text-base">
              {teamLeft.name}
              <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 font-medium">
                {teamLeft.controller.toUpperCase()}
              </span>
            </div>
            <div className="text-xs text-slate-400 flex items-center gap-2">
              <span>Formation: {teamLeft.formation}</span>
              <span>•</span>
              <span className="text-blue-400 font-semibold">{possession.left}% Poss.</span>
            </div>
          </div>
        </div>

        {/* Center: Score & Match Clock */}
        <div className="flex flex-col items-center justify-center">
          {scenarioName && (
            <div className="text-xs font-semibold text-emerald-400 bg-emerald-950/60 border border-emerald-800/60 px-3 py-0.5 rounded-full mb-1 flex items-center gap-1.5">
              <Trophy className="w-3 h-3" />
              {scenarioName}
            </div>
          )}

          <div className="flex items-center gap-4 bg-slate-950/80 border border-slate-800 px-6 py-2 rounded-2xl shadow-inner">
            <span className="text-3xl font-extrabold text-white font-mono tracking-tight">{score.left}</span>
            <div className="flex flex-col items-center">
              <span className="text-lg font-bold text-amber-400 font-mono tracking-widest">{timeFormatted}</span>
              <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                {status === 'playing' ? (
                  <span className="flex items-center gap-1 text-emerald-400">
                    <Radio className="w-2.5 h-2.5 animate-pulse" /> LIVE
                  </span>
                ) : status === 'goal' ? (
                  <span className="text-amber-400 font-bold animate-bounce">⚽ GOAL!</span>
                ) : (
                  status.toUpperCase()
                )}
              </span>
            </div>
            <span className="text-3xl font-extrabold text-white font-mono tracking-tight">{score.right}</span>
          </div>
        </div>

        {/* Right Team Badge & Name */}
        <div className="flex items-center gap-3 w-full md:w-1/3 justify-end text-right">
          <div>
            <div className="font-bold text-slate-100 flex items-center justify-end gap-2 text-base">
              <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 font-medium">
                {teamRight.controller.toUpperCase()}
              </span>
              {teamRight.name}
            </div>
            <div className="text-xs text-slate-400 flex items-center justify-end gap-2">
              <span className="text-red-400 font-semibold">{possession.right}% Poss.</span>
              <span>•</span>
              <span>Formation: {teamRight.formation}</span>
            </div>
          </div>
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center font-bold text-white shadow-md transition-transform"
            style={{ backgroundColor: teamRight.color }}
          >
            {teamRight.shortName}
          </div>
        </div>
      </div>
    </div>
  );
};
