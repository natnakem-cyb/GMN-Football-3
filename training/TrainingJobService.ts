import { ChildProcess, spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { WebSocket } from 'ws';

export interface TrainingJobConfig {
  algorithm: 'ppo' | 'mappo' | 'ippo';
  scenario: string;
  timesteps: number;
  resumeFrom?: string;
}

export interface TrainingMetrics {
  step: number;
  totalSteps: number;
  update?: number;
  totalUpdates?: number;
  episodes?: number;
  rollingReward?: number;
  goalRate?: number;
  valueLoss?: number;
  policyLoss?: number;
  entropy?: number;
  approxKl?: number;
  timestamp: number;
}

export interface TrainingJobInfo {
  id: string;
  config: TrainingJobConfig;
  status: 'idle' | 'running' | 'completed' | 'failed' | 'stopped';
  startTime: number;
  endTime?: number;
  pid?: number;
  currentStep: number;
  totalSteps: number;
  latestMetrics: TrainingMetrics | null;
  recentLogs: string[];
  exitCode?: number | null;
}

export class TrainingJobService {
  private static activeProcess: ChildProcess | null = null;
  private static currentJob: TrainingJobInfo | null = null;
  private static wsClients: Set<WebSocket> = new Set();
  private static sb3Buffer: Partial<TrainingMetrics> = {};

  public static registerWebSocket(ws: WebSocket) {
    this.wsClients.add(ws);
    ws.on('close', () => {
      this.wsClients.delete(ws);
    });

    // Send current status immediately upon connection
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: 'TRAINING_STATUS',
          data: this.getStatus(),
        })
      );
    }
  }

  public static broadcast(message: any) {
    const payload = JSON.stringify(message);
    for (const client of this.wsClients) {
      if (client.readyState === WebSocket.OPEN) {
        try {
          client.send(payload);
        } catch (e) {
          // ignore
        }
      }
    }
  }

  public static getStatus(): {
    isRunning: boolean;
    currentJob: TrainingJobInfo | null;
    latestMetrics: TrainingMetrics | null;
    recentLogs: string[];
  } {
    const isRunning = this.currentJob !== null && this.currentJob.status === 'running';
    return {
      isRunning,
      currentJob: this.currentJob,
      latestMetrics: this.currentJob?.latestMetrics || null,
      recentLogs: this.currentJob?.recentLogs || [],
    };
  }

  /**
   * Starts a new training job with normalized payload.
   */
  public static startJob(config: TrainingJobConfig): TrainingJobInfo {
    if (this.activeProcess || (this.currentJob && this.currentJob.status === 'running')) {
      throw new Error('A training job is already actively running.');
    }

    const { algorithm, scenario, timesteps, resumeFrom } = config;
    if (!['ppo', 'mappo', 'ippo'].includes(algorithm)) {
      throw new Error(`Invalid algorithm: ${algorithm}. Must be ppo, mappo, or ippo.`);
    }
    if (!timesteps || timesteps <= 0) {
      throw new Error('Timesteps must be a positive integer.');
    }

    const jobId = `job_${algorithm}_${Date.now()}`;
    const pyArgs: string[] = [];

    // Map to actual script flags per Task 2 requirements:
    // - PPO (train_ppo.py): --timesteps, --scenario, --resume, --checkpoint
    // - MAPPO (train_mappo.py): --timesteps, --scenario, --checkpoint-name, --resume
    // - IPPO (train_ippo.py): timesteps (positional or optional), --resume, --checkpoint
    if (algorithm === 'ppo') {
      pyArgs.push('training/train_ppo.py', '--timesteps', timesteps.toString(), '--scenario', scenario);
      if (resumeFrom) {
        pyArgs.push('--resume', resumeFrom);
      }
      pyArgs.push('--checkpoint', `ppo_${scenario}_${timesteps}.zip`);
    } else if (algorithm === 'mappo') {
      pyArgs.push('training/train_mappo.py', '--timesteps', timesteps.toString(), '--scenario', scenario);
      if (resumeFrom) {
        pyArgs.push('--resume', resumeFrom);
      }
      pyArgs.push('--checkpoint-name', `mappo_${scenario}_${timesteps}.pt`);
    } else if (algorithm === 'ippo') {
      pyArgs.push('training/train_ippo.py', timesteps.toString());
      if (resumeFrom) {
        pyArgs.push('--resume', resumeFrom);
      }
      pyArgs.push('--checkpoint', `ippo_${scenario}_${timesteps}.zip`);
    }

    console.log(`[TrainingJobService] Spawning python3 ${pyArgs.join(' ')}`);

    const child = spawn('python3', pyArgs, {
      cwd: process.cwd(),
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
      },
    });

    const jobInfo: TrainingJobInfo = {
      id: jobId,
      config,
      status: 'running',
      startTime: Date.now(),
      pid: child.pid,
      currentStep: 0,
      totalSteps: timesteps,
      latestMetrics: null,
      recentLogs: [],
    };

    this.currentJob = jobInfo;
    this.activeProcess = child;
    this.sb3Buffer = {};

    this.broadcast({
      type: 'TRAINING_STARTED',
      data: jobInfo,
    });

    let lineBuffer = '';

    const handleData = (chunk: Buffer) => {
      lineBuffer += chunk.toString('utf8');
      const lines = lineBuffer.split('\n');
      lineBuffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;

        // Keep last 250 log lines
        jobInfo.recentLogs.push(line);
        if (jobInfo.recentLogs.length > 250) {
          jobInfo.recentLogs.shift();
        }

        // Stream raw stdout line
        this.broadcast({
          type: 'TRAINING_STDOUT',
          data: {
            line,
            timestamp: Date.now(),
            jobId,
          },
        });

        // Parse line for metrics
        const parsedMetrics = this.parseStdoutLine(line, config);
        if (parsedMetrics) {
          jobInfo.currentStep = parsedMetrics.step;
          jobInfo.latestMetrics = parsedMetrics;
          this.broadcast({
            type: 'TRAINING_METRICS',
            data: parsedMetrics,
            jobId,
          });
        }
      }
    };

    child.stdout.on('data', handleData);
    child.stderr.on('data', handleData);

    child.on('close', (code) => {
      console.log(`[TrainingJobService] Job ${jobId} exited with code ${code}`);
      this.activeProcess = null;
      jobInfo.endTime = Date.now();
      jobInfo.exitCode = code;

      if (code === 0) {
        jobInfo.status = 'completed';
        jobInfo.currentStep = jobInfo.totalSteps;

        // Auto trigger export_onnx.py on successful completion
        this.handleAutomaticExport(jobInfo)
          .then((exportResult) => {
            this.broadcast({
              type: 'TRAINING_COMPLETED',
              data: {
                jobId,
                success: true,
                exportResult,
              },
            });
          })
          .catch((err) => {
            console.error('[TrainingJobService] ONNX export failed on completion:', err);
            this.broadcast({
              type: 'TRAINING_COMPLETED',
              data: {
                jobId,
                success: true,
                exportError: err.message || String(err),
              },
            });
          });
      } else {
        jobInfo.status = jobInfo.status === 'stopped' ? 'stopped' : 'failed';
        this.broadcast({
          type: 'TRAINING_FAILED',
          data: {
            jobId,
            exitCode: code,
            status: jobInfo.status,
          },
        });
      }
    });

    child.on('error', (err) => {
      console.error('[TrainingJobService] Process error:', err);
      jobInfo.status = 'failed';
      this.activeProcess = null;
      this.broadcast({
        type: 'TRAINING_FAILED',
        data: {
          jobId,
          error: err.message,
        },
      });
    });

    return jobInfo;
  }

  /**
   * Stops/kills the currently running training job.
   */
  public static stopJob(): boolean {
    if (!this.activeProcess || !this.currentJob) {
      return false;
    }

    this.currentJob.status = 'stopped';
    try {
      this.activeProcess.kill('SIGTERM');
      setTimeout(() => {
        if (this.activeProcess) {
          try {
            this.activeProcess.kill('SIGKILL');
          } catch (e) {
            // ignore
          }
        }
      }, 2000);
    } catch (e) {
      console.warn('[TrainingJobService] Error stopping process:', e);
    }

    this.broadcast({
      type: 'TRAINING_STOPPED',
      data: { jobId: this.currentJob.id },
    });

    return true;
  }

  /**
   * Parses standard periodic progress lines from train_mappo, train_ippo, or train_ppo.
   */
  private static parseStdoutLine(line: string, config: TrainingJobConfig): TrainingMetrics | null {
    // 1. MAPPO: [Step 10000 / 200000] Update 39/781 | Completed Episodes: 142 | Rolling Reward (last 50): +0.3210 | Goal Rate: 42.0% | Val Loss: 0.04120 | Entropy: 2.4512
    const mappoMatch = line.match(
      /\[Step\s+(\d+)\s*\/\s*(\d+)\]\s*Update\s+(\d+)\/(\d+)\s*\|\s*Completed Episodes:\s*(\d+)\s*\|\s*Rolling Reward[^:]*:\s*([+-]?[\d\.]+)\s*\|\s*Goal Rate:\s*([\d\.]+)%\s*\|\s*Val Loss:\s*([\d\.]+)\s*\|\s*Entropy:\s*([\d\.]+)/
    );
    if (mappoMatch) {
      return {
        step: parseInt(mappoMatch[1], 10),
        totalSteps: parseInt(mappoMatch[2], 10),
        update: parseInt(mappoMatch[3], 10),
        totalUpdates: parseInt(mappoMatch[4], 10),
        episodes: parseInt(mappoMatch[5], 10),
        rollingReward: parseFloat(mappoMatch[6]),
        goalRate: parseFloat(mappoMatch[7]),
        valueLoss: parseFloat(mappoMatch[8]),
        entropy: parseFloat(mappoMatch[9]),
        timestamp: Date.now(),
      };
    }

    // 2. IPPO: [Step 10000] Completed Episodes:   24 | Rolling Mean Reward (last 50): +0.1234 | Rolling Goal Rate:  25.0%
    const ippoMatch = line.match(
      /\[Step\s+(\d+)\]\s*Completed Episodes:\s*(\d+)\s*\|\s*Rolling Mean Reward[^:]*:\s*([+-]?[\d\.]+)\s*\|\s*Rolling Goal Rate:\s*([\d\.]+)%/
    );
    if (ippoMatch) {
      return {
        step: parseInt(ippoMatch[1], 10),
        totalSteps: config.timesteps,
        episodes: parseInt(ippoMatch[2], 10),
        rollingReward: parseFloat(ippoMatch[3]),
        goalRate: parseFloat(ippoMatch[4]),
        timestamp: Date.now(),
      };
    }

    // 3. Stable-Baselines3 (train_ppo) table lines:
    // | ep_rew_mean          | 0.354    |
    // | total_timesteps      | 1000     |
    // | value_loss           | 0.052    |
    // | entropy_loss         | -2.45    |
    // | loss                 | 0.012    |
    // | approx_kl            | 0.0084   |
    const sb3Match = line.match(/\|\s*([a-zA-Z0-9_]+)\s*\|\s*([+-]?[\d\.eE-]+)\s*\|/);
    if (sb3Match) {
      const key = sb3Match[1].trim();
      const val = parseFloat(sb3Match[2].trim());
      if (key === 'total_timesteps') {
        this.sb3Buffer.step = Math.floor(val);
      } else if (key === 'ep_rew_mean') {
        this.sb3Buffer.rollingReward = val;
      } else if (key === 'value_loss') {
        this.sb3Buffer.valueLoss = val;
      } else if (key === 'entropy_loss') {
        this.sb3Buffer.entropy = Math.abs(val);
      } else if (key === 'loss') {
        this.sb3Buffer.policyLoss = val;
      } else if (key === 'approx_kl') {
        this.sb3Buffer.approxKl = val;
      }

      if (this.sb3Buffer.step !== undefined) {
        const metrics: TrainingMetrics = {
          step: this.sb3Buffer.step,
          totalSteps: config.timesteps,
          rollingReward: this.sb3Buffer.rollingReward,
          valueLoss: this.sb3Buffer.valueLoss,
          entropy: this.sb3Buffer.entropy,
          policyLoss: this.sb3Buffer.policyLoss,
          approxKl: this.sb3Buffer.approxKl,
          timestamp: Date.now(),
        };
        return metrics;
      }
    }

    return null;
  }

  /**
   * Automatically executes training/export_onnx.py when a training run completes.
   */
  public static async handleAutomaticExport(jobInfo: TrainingJobInfo): Promise<{
    onnxPath: string;
    sidecarPath: string;
  }> {
    const { algorithm, scenario, timesteps } = jobInfo.config;
    console.log(`[TrainingJobService] Running automatic ONNX export for ${algorithm} (${scenario})...`);

    let sourcePt = `training/models/mappo_${scenario}_${timesteps}.pt`;
    if (!fs.existsSync(sourcePt)) {
      // Fallback candidates
      if (fs.existsSync(`training/models/mappo_${scenario}_trained.pt`)) {
        sourcePt = `training/models/mappo_${scenario}_trained.pt`;
      } else if (fs.existsSync('training/models/mappo_academy_3_vs_1_with_keeper_trained.pt')) {
        sourcePt = 'training/models/mappo_academy_3_vs_1_with_keeper_trained.pt';
      }
    }

    const outputOnnx = `public/models/mappo_${scenario}_${timesteps}.onnx`;

    const exportArgs = [
      'training/export_onnx.py',
      '--checkpoint',
      sourcePt,
      '--output',
      outputOnnx,
      '--scenario',
      scenario,
      '--algorithm',
      algorithm.toUpperCase(),
    ];

    return new Promise((resolve, reject) => {
      const p = spawn('python3', exportArgs, { stdio: 'inherit' });
      p.on('close', (code) => {
        if (code === 0) {
          // If this was the academy_3_vs_1_with_keeper scenario, also copy to default mappo_policy.onnx
          if (scenario === 'academy_3_vs_1_with_keeper' && fs.existsSync(outputOnnx)) {
            try {
              fs.copyFileSync(outputOnnx, 'public/models/mappo_policy.onnx');
              if (fs.existsSync(outputOnnx + '.json')) {
                fs.copyFileSync(outputOnnx + '.json', 'public/models/mappo_policy.onnx.json');
              }
              console.log('[TrainingJobService] Synced active mappo_policy.onnx with completed run.');
            } catch (e) {
              console.warn('[TrainingJobService] Failed to copy to mappo_policy.onnx:', e);
            }
          }

          resolve({
            onnxPath: outputOnnx,
            sidecarPath: outputOnnx + '.json',
          });
        } else {
          reject(new Error(`export_onnx.py exited with code ${code}`));
        }
      });
      p.on('error', reject);
    });
  }
}
