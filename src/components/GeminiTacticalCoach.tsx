import React, { useState } from 'react';
import { MatchStats, TeamConfig } from '../types/football';
import { GeminiCoachService, TacticalAnalysisResult } from '../services/geminiService';
import { Sparkles, Brain, CheckCircle, AlertTriangle, Lightbulb, Compass, RefreshCw } from 'lucide-react';

interface GeminiTacticalCoachProps {
  stats: MatchStats;
  teamLeft: TeamConfig;
  teamRight: TeamConfig;
  score: { left: number; right: number };
  eventsSummary: string;
}

export const GeminiTacticalCoach: React.FC<GeminiTacticalCoachProps> = ({
  stats,
  teamLeft,
  teamRight,
  score,
  eventsSummary,
}) => {
  const [analysis, setAnalysis] = useState<TacticalAnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleGenerateAnalysis = async () => {
    setIsLoading(true);
    try {
      const result = await GeminiCoachService.analyzeMatch(
        stats,
        teamLeft,
        teamRight,
        score,
        eventsSummary
      );
      setAnalysis(result);
    } catch (err) {
      console.error('Failed to generate tactical analysis:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div id="gemini-tactical-coach" className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-5 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-amber-400" /> AI Tactical Coach & Match Analyzer
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Generative tactical evaluation, positional structure auditing, and coaching adjustments.
          </p>
        </div>

        <button
          onClick={handleGenerateAnalysis}
          disabled={isLoading}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-amber-500 to-emerald-600 hover:from-amber-400 hover:to-emerald-500 text-slate-950 font-bold text-xs shadow-lg transition-all disabled:opacity-50"
        >
          {isLoading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" /> Analyzing Match Telemetry...
            </>
          ) : (
            <>
              <Brain className="w-4 h-4" /> Generate Tactical Audit
            </>
          )}
        </button>
      </div>

      {!analysis && !isLoading && (
        <div className="text-center py-10 border border-dashed border-slate-800 rounded-xl bg-slate-950/40">
          <Brain className="w-10 h-10 text-slate-600 mx-auto mb-2" />
          <p className="text-sm font-semibold text-slate-300">No Tactical Audit Generated Yet</p>
          <p className="text-xs text-slate-500 max-w-md mx-auto mt-1">
            Click "Generate Tactical Audit" to inspect live telemetry, spatial patterns, passing lanes, and agent performance.
          </p>
        </div>
      )}

      {analysis && (
        <div className="space-y-4">
          {/* Headline & Summary */}
          <div className="p-4 rounded-xl bg-slate-950/80 border border-amber-500/30">
            <h4 className="text-sm font-bold text-amber-300 flex items-center gap-2 mb-1.5">
              <Compass className="w-4 h-4" /> {analysis.headline}
            </h4>
            <p className="text-xs text-slate-300 leading-relaxed">{analysis.summary}</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Strengths */}
            <div className="p-4 rounded-xl bg-slate-950/60 border border-emerald-500/30">
              <h5 className="text-xs font-bold text-emerald-400 flex items-center gap-1.5 mb-2">
                <CheckCircle className="w-4 h-4" /> Tactical Strengths
              </h5>
              <ul className="space-y-1.5">
                {analysis.strengths.map((item, idx) => (
                  <li key={idx} className="text-xs text-slate-300 flex items-start gap-1.5">
                    <span className="text-emerald-400 font-bold">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Weaknesses */}
            <div className="p-4 rounded-xl bg-slate-950/60 border border-red-500/30">
              <h5 className="text-xs font-bold text-red-400 flex items-center gap-1.5 mb-2">
                <AlertTriangle className="w-4 h-4" /> Vulnerabilities & Gaps
              </h5>
              <ul className="space-y-1.5">
                {analysis.weaknesses.map((item, idx) => (
                  <li key={idx} className="text-xs text-slate-300 flex items-start gap-1.5">
                    <span className="text-red-400 font-bold">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Recommendations */}
            <div className="p-4 rounded-xl bg-slate-950/60 border border-blue-500/30">
              <h5 className="text-xs font-bold text-blue-400 flex items-center gap-1.5 mb-2">
                <Lightbulb className="w-4 h-4" /> Coaching Adjustments
              </h5>
              <ul className="space-y-1.5">
                {analysis.recommendations.map((item, idx) => (
                  <li key={idx} className="text-xs text-slate-300 flex items-start gap-1.5">
                    <span className="text-blue-400 font-bold">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Predicted Outcome */}
          <div className="text-xs text-slate-400 bg-slate-950/50 p-3 rounded-lg border border-slate-800 flex items-center justify-between">
            <span className="font-semibold text-slate-300">Tactical Forecast:</span>
            <span className="text-amber-300 font-medium">{analysis.predictedOutcome}</span>
          </div>
        </div>
      )}
    </div>
  );
};
