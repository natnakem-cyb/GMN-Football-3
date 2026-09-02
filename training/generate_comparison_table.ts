/**
 * GMN-Football-3 — Comparison Table Generator
 * Formats evaluation results into Markdown, CSV, and HTML comparison tables.
 */

import fs from 'fs';
import path from 'path';
import { AggregatedMetrics } from '../src/engine/FootballMetrics';

export class ComparisonTableGenerator {
  public static generateMarkdownTable(metricsMap: Record<string, AggregatedMetrics>): string {
    const lines: string[] = [];
    lines.push('| Policy / Baseline | Win Rate (%) | Goals Scored (Mean±Std) | Goal Diff | Possession (%) | Pass Acc (%) | Shot Acc (%) | Reward (Mean±Std) |');
    lines.push('|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|');

    for (const [name, m] of Object.entries(metricsMap)) {
      const win = `${m.winRatePct.mean.toFixed(1)}%`;
      const goals = `${m.goalsScoredPerEpisode.mean.toFixed(2)} ± ${m.goalsScoredPerEpisode.std.toFixed(2)}`;
      const diff = `${m.goalDifferencePerEpisode.mean.toFixed(2)}`;
      const poss = `${m.possessionRatePct.mean.toFixed(1)}%`;
      const pass = `${m.passCompletionRatePct.mean.toFixed(1)}%`;
      const shot = `${m.shotAccuracyPct.mean.toFixed(1)}%`;
      const rew = `${m.cumulativeReward.mean.toFixed(3)} ± ${m.cumulativeReward.std.toFixed(3)}`;

      lines.push(
        `| **${name}** | ${win} | ${goals} | ${diff} | ${poss} | ${pass} | ${shot} | ${rew} |`
      );
    }

    return lines.join('\n');
  }

  public static generateCSV(metricsMap: Record<string, AggregatedMetrics>): string {
    const headers = [
      'policy',
      'episodes',
      'win_rate_mean',
      'win_rate_std',
      'goals_scored_mean',
      'goals_scored_std',
      'goal_diff_mean',
      'possession_rate_mean',
      'pass_accuracy_mean',
      'shot_accuracy_mean',
      'reward_mean',
      'reward_std',
    ];

    const rows: string[] = [headers.join(',')];

    for (const [name, m] of Object.entries(metricsMap)) {
      const row = [
        name,
        m.totalEpisodes,
        m.winRatePct.mean.toFixed(2),
        m.winRatePct.std.toFixed(2),
        m.goalsScoredPerEpisode.mean.toFixed(3),
        m.goalsScoredPerEpisode.std.toFixed(3),
        m.goalDifferencePerEpisode.mean.toFixed(3),
        m.possessionRatePct.mean.toFixed(2),
        m.passCompletionRatePct.mean.toFixed(2),
        m.shotAccuracyPct.mean.toFixed(2),
        m.cumulativeReward.mean.toFixed(4),
        m.cumulativeReward.std.toFixed(4),
      ];
      rows.push(row.join(','));
    }

    return rows.join('\n');
  }
}
