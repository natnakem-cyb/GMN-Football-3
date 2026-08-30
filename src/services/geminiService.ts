import { MatchStats, TeamConfig } from '../types/football';

export interface TacticalAnalysisResult {
  headline: string;
  summary: string;
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
  predictedOutcome: string;
}

export class GeminiCoachService {
  private static aiClient: any = null;

  private static async getClient(): Promise<any> {
    if (!this.aiClient) {
      try {
        const apiKey =
          (typeof process !== 'undefined' && process.env?.GEMINI_API_KEY) ||
          (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_GEMINI_API_KEY);

        if (apiKey) {
          const { GoogleGenAI } = await import('@google/genai');
          this.aiClient = new GoogleGenAI({ apiKey });
        }
      } catch (err) {
        console.warn('Gemini client lazy initialization skipped:', err);
      }
    }
    return this.aiClient;
  }

  public static async analyzeMatch(
    stats: MatchStats,
    teamLeft: TeamConfig,
    teamRight: TeamConfig,
    score: { left: number; right: number },
    eventsSummary: string
  ): Promise<TacticalAnalysisResult> {
    const client = await this.getClient();

    const prompt = `You are a world-class AI Football Tactical Analyst and Coach for Google Research Football.
Analyze the following live match telemetry and provide structured tactical insights:

Match Details:
- ${teamLeft.name} (${teamLeft.formation}) vs ${teamRight.name} (${teamRight.formation})
- Score: ${teamLeft.shortName} ${score.left} - ${score.right} ${teamRight.shortName}
- Possession: ${teamLeft.shortName} ${stats.possession.left}% | ${teamRight.shortName} ${stats.possession.right}%
- Total Shots: ${stats.shots.left} vs ${stats.shots.right}
- Shots on Target: ${stats.shotsOnTarget.left} vs ${stats.shotsOnTarget.right}
- Passes Completed: ${stats.completedPasses.left}/${stats.passes.left} vs ${stats.completedPasses.right}/${stats.passes.right}
- Tackles Won: ${stats.tackles.left} vs ${stats.tackles.right}
- Interceptions: ${stats.interceptions.left} vs ${stats.interceptions.right}
- Recent Events: ${eventsSummary || 'End-to-end tactical progression'}

Provide an in-depth tactical breakdown formatted as valid JSON:
{
  "headline": "Short punchy tactical assessment (e.g., High-Press Dominance with Midfield Overloads)",
  "summary": "2-3 sentences explaining the tactical dynamic and flow of the match",
  "strengths": ["3 key tactical advantages observed"],
  "weaknesses": ["3 vulnerable areas or positional errors"],
  "recommendations": ["3 actionable adjustments for formation, passing tempo, or pressing triggers"],
  "predictedOutcome": "Brief tactical forecast for the remainder of the fixture"
}`;

    if (client) {
      try {
        const response = await client.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: prompt,
          config: {
            responseMimeType: 'application/json',
          },
        });

        if (response.text) {
          const parsed = JSON.parse(response.text);
          return parsed as TacticalAnalysisResult;
        }
      } catch (err) {
        console.warn('Gemini API call failed, generating deterministic tactical heuristic analysis:', err);
      }
    }

    // Heuristic analysis fallback if API key not present or offline
    const leftDominant = stats.possession.left > 55 || score.left > score.right;
    const shotEfficiency = stats.shots.left > 0 ? Math.round((stats.goals.left / stats.shots.left) * 100) : 0;
    const passAcc = stats.passes.left > 0 ? Math.round((stats.completedPasses.left / stats.passes.left) * 100) : 75;

    return {
      headline: leftDominant
        ? `${teamLeft.name} Controlling Half-Spaces in ${teamLeft.formation}`
        : `${teamRight.name} Exploiting Transition Lanes`,
      summary: `${teamLeft.name} registered ${stats.possession.left}% possession with ${stats.shots.left} attempts. ${
        leftDominant
          ? 'Positional play and triangular passing created numerical superiority in central channels.'
          : 'Defensive low block under pressure against swift lateral counters from the opposition.'
      }`,
      strengths: [
        `Passing circulation accuracy operating at ${passAcc}%`,
        `Defensive line compactness with ${stats.interceptions.left} recorded interceptions`,
        `Directness in attacking transition generating ${stats.shotsOnTarget.left} shots on target`,
      ],
      weaknesses: [
        stats.tackles.left < stats.tackles.right ? 'Vulnerability to counter-press in defensive third' : 'Wing spacing occasionally congested along touchline',
        passAcc < 70 ? 'Turnovers during central vertical progression' : 'Shot conversion rate requires greater composure inside the box',
        'Stamina depletion during sustained high-pressing sequences',
      ],
      recommendations: [
        'Utilize early diagonal switches [L] to exploit isolated fullbacks',
        'Trigger overlapping runs from wingbacks to overload penalty area',
        'Commit to tactical slide tackles [E] when opponents turn their back to goal',
      ],
      predictedOutcome: score.left >= score.right
        ? `Sustained possession advantage favors ${teamLeft.name} to seal victory.`
        : `Immediate tactical adjustment required to invert counter-attack momentum.`,
    };
  }
}
