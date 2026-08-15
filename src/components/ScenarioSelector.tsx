import React from 'react';
import { ScenarioConfig } from '../types/football';
import { ACADEMY_SCENARIOS } from '../scenarios/ScenarioRegistry';
import { GraduationCap, CheckCircle2, XCircle, Clock, Trophy, Play, Star } from 'lucide-react';

interface ScenarioSelectorProps {
  activeScenario: ScenarioConfig | null;
  onSelectScenario: (scenario: ScenarioConfig) => void;
  onFreePlay: () => void;
  matchTimeSeconds: number;
}

export const ScenarioSelector: React.FC<ScenarioSelectorProps> = ({
  activeScenario,
  onSelectScenario,
  onFreePlay,
  matchTimeSeconds,
}) => {
  return (
    <div id="scenario-selector-panel" className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-5 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <GraduationCap className="w-5 h-5 text-emerald-400" /> Google Research Football Academy Curriculum
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Progressive scenario-based training curriculum inspired by GRF Academy benchmarks.
          </p>
        </div>

        <button
          onClick={onFreePlay}
          className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all ${
            !activeScenario
              ? 'bg-emerald-600 text-white border-emerald-500 shadow-md'
              : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
          }`}
        >
          ⚽ Free Play Match
        </button>
      </div>

      {/* Scenarios Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {ACADEMY_SCENARIOS.map((scenario) => {
          const isSelected = activeScenario?.id === scenario.id;
          const allCompleted = isSelected && scenario.objectives.every((o) => o.isCompleted);
          const hasFailed = isSelected && scenario.objectives.some((o) => o.isFailed);

          return (
            <div
              key={scenario.id}
              onClick={() => onSelectScenario(scenario)}
              className={`cursor-pointer rounded-xl p-4 border transition-all relative overflow-hidden ${
                isSelected
                  ? 'bg-slate-800/90 border-emerald-500 shadow-lg shadow-emerald-950/40 ring-1 ring-emerald-500'
                  : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900/50'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-400 bg-emerald-950/80 px-2 py-0.5 rounded-md border border-emerald-800/40">
                  Stage {scenario.stage}
                </span>

                <span
                  className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                    scenario.difficulty === 'Beginner'
                      ? 'bg-blue-950 text-blue-400 border border-blue-800'
                      : scenario.difficulty === 'Intermediate'
                      ? 'bg-amber-950 text-amber-400 border border-amber-800'
                      : 'bg-purple-950 text-purple-400 border border-purple-800'
                  }`}
                >
                  {scenario.difficulty}
                </span>
              </div>

              <h4 className="text-sm font-bold text-slate-100 mb-1">{scenario.name}</h4>
              <p className="text-xs text-slate-400 line-clamp-2 mb-3">{scenario.description}</p>

              {/* Objectives List */}
              <div className="space-y-1.5 border-t border-slate-800/80 pt-2.5 mb-3">
                {scenario.objectives.map((obj) => (
                  <div key={obj.id} className="flex items-center gap-2 text-xs">
                    {isSelected && obj.isCompleted ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    ) : isSelected && obj.isFailed ? (
                      <XCircle className="w-3.5 h-3.5 text-red-400 shrink-0" />
                    ) : (
                      <div className="w-2 h-2 rounded-full bg-slate-600 shrink-0" />
                    )}
                    <span className={isSelected && obj.isCompleted ? 'text-emerald-300 font-medium' : 'text-slate-400'}>
                      {obj.text}
                    </span>
                  </div>
                ))}
              </div>

              {/* Footer status / Action */}
              <div className="flex items-center justify-between text-xs text-slate-400 pt-1">
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" /> {scenario.timeLimitSeconds}s Limit
                </span>
                <span className="flex items-center gap-1 font-semibold text-amber-400">
                  <Trophy className="w-3 h-3" /> +{scenario.rewards.completion} pts
                </span>
              </div>

              {isSelected && (
                <div className="mt-3 w-full py-1 text-center bg-emerald-500 text-slate-950 font-bold rounded-lg text-xs flex items-center justify-center gap-1">
                  <Play className="w-3 h-3 fill-current" /> Active Drill
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
