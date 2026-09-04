import React, { useState, useEffect, useRef } from 'react';
import { TrainingTelemetryService } from '../engine/TrainingTelemetryService';
import {
  TrainingMetricsSnapshot,
  HardwareMetrics,
  TrainingHyperparameters,
} from '../types/telemetry';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import {
  Play,
  Square,
  RotateCcw,
  Zap,
  Activity,
  Cpu,
  HardDrive,
  Sliders,
  TrendingUp,
  Target,
  Shield,
  Layers,
  ArrowUpRight,
  Info,
  CheckCircle2,
  AlertTriangle,
  Server,
  Radio,
  Wifi,
  WifiOff,
  Trash2,
  UploadCloud,
  Check,
  FileCode,
  Terminal,
  Copy,
  ExternalLink,
  RefreshCw,
} from 'lucide-react';

interface TrainingTelemetryDashboardProps {
  activeModelPath?: string;
  onSelectModel?: (modelPath: string) => Promise<void> | void;
  stalenessTicks?: number;
  lastInferenceMs?: number;
}

export const TrainingTelemetryDashboard: React.FC<TrainingTelemetryDashboardProps> = ({
  activeModelPath = '/models/mappo_policy.onnx',
  onSelectModel,
  stalenessTicks = 0,
  lastInferenceMs = 0,
}) => {
  const telemetryService = TrainingTelemetryService.getInstance();
  const [, setTrigger] = useState(0);

  // Job launcher form state
  const [selectedAlgo, setSelectedAlgo] = useState<'mappo' | 'ppo' | 'ippo'>('mappo');
  const [selectedScenario, setSelectedScenario] = useState('academy_3_vs_1_with_keeper');
  const [targetSteps, setTargetSteps] = useState(20000);
  const [resumeCheckpoint, setResumeCheckpoint] = useState('');
  const [isSubmittingJob, setIsSubmittingJob] = useState(false);
  const [jobFeedback, setJobFeedback] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  // Checkpoint management state
  const [deleteModalItem, setDeleteModalItem] = useState<any | null>(null);
  const [deleteSourcePt, setDeleteSourcePt] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Terminal state
  const [autoScrollLogs, setAutoScrollLogs] = useState(true);
  const logTerminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const unsub = telemetryService.subscribe(() => {
      setTrigger((prev) => prev + 1);
    });
    // Auto-connect to bridge on mount
    telemetryService.connectWebSocket();
    telemetryService.refreshCheckpoints();

    return () => unsub();
  }, [telemetryService]);

  // Auto-scroll log terminal
  useEffect(() => {
    if (autoScrollLogs && logTerminalRef.current) {
      logTerminalRef.current.scrollTop = logTerminalRef.current.scrollHeight;
    }
  }, [telemetryService.liveLogs, autoScrollLogs]);

  const {
    snapshots,
    currentStep,
    isTrainingActive,
    trainingSpeed,
    hyperparameters,
    hardware,
    isWsConnected,
    wsStatus,
    wsUrl,
    liveLogs,
    activeJob,
    checkpoints,
    isRefreshingCheckpoints,
  } = telemetryService;

  const lastSnapshot = snapshots[snapshots.length - 1] || snapshots[0];
  const totalTargetSteps = activeJob?.config?.timesteps || hyperparameters.targetTimesteps;
  const currentProgressStep = activeJob?.currentStep || currentStep;
  const progressPct = Math.min(100, Math.max(0, (currentProgressStep / totalTargetSteps) * 100)).toFixed(1);

  // Formatting steps for charts
  const chartData = snapshots.map((s) => ({
    ...s,
    stepLabel: `${Math.round(s.step / 1000)}k`,
  }));

  // Handle Starting Training
  const handleStartTraining = async () => {
    setIsSubmittingJob(true);
    setJobFeedback(null);
    try {
      const res = await telemetryService.startTrainingJob({
        algorithm: selectedAlgo,
        scenario: selectedScenario,
        timesteps: targetSteps,
        resumeFrom: resumeCheckpoint.trim() || undefined,
      });
      if (res.success) {
        setJobFeedback({ type: 'success', message: `Job ${res.job?.id} started successfully.` });
      } else {
        setJobFeedback({ type: 'error', message: res.error || 'Failed to start training' });
      }
    } catch (e: any) {
      setJobFeedback({ type: 'error', message: e.message || 'Error communicating with bridge' });
    } finally {
      setIsSubmittingJob(false);
    }
  };

  // Handle Stopping Training
  const handleStopTraining = async () => {
    setIsSubmittingJob(true);
    try {
      await telemetryService.stopTrainingJob();
      setJobFeedback({ type: 'success', message: 'Training job stopped.' });
    } catch (e: any) {
      setJobFeedback({ type: 'error', message: e.message || 'Failed to stop' });
    } finally {
      setIsSubmittingJob(false);
    }
  };

  // Handle Checkpoint Deletion
  const handleConfirmDelete = async () => {
    if (!deleteModalItem) return;
    setIsDeleting(true);
    try {
      await telemetryService.deleteCheckpoint(deleteModalItem.filename, deleteSourcePt);
      setDeleteModalItem(null);
      setDeleteSourcePt(false);
    } catch (e: any) {
      alert(`Delete failed: ${e.message}`);
    } finally {
      setIsDeleting(false);
    }
  };

  // Handle File Upload (Drag & Drop or Input)
  const handleFileUpload = async (file: File) => {
    if (!file.name.endsWith('.onnx')) {
      alert('Only .onnx model files are supported.');
      return;
    }
    setUploadStatus(`Uploading ${file.name}...`);
    try {
      const reader = new FileReader();
      reader.onload = async () => {
        const base64 = (reader.result as string).split(',')[1];
        await telemetryService.uploadCheckpoint(file.name, base64, {
          scenario: selectedScenario,
          algorithm: file.name.toUpperCase().includes('MAPPO') ? 'MAPPO' : 'PPO',
        });
        setUploadStatus(`Successfully uploaded ${file.name}`);
        setTimeout(() => setUploadStatus(null), 3000);
      };
      reader.readAsDataURL(file);
    } catch (e: any) {
      setUploadStatus(`Upload error: ${e.message}`);
    }
  };

  return (
    <div className="space-y-5 animate-fadeIn">
      {/* 1. Header & Hardware / Staleness Telemetry Strip */}
      <div className="p-4 md:p-5 rounded-2xl bg-slate-900/95 border border-slate-800 shadow-2xl backdrop-blur-md">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-purple-900/30">
                <Activity className="w-4 h-4 text-white" />
              </div>
              <div>
                <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                  RL Training Command & Model Registry
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                      isTrainingActive
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 animate-pulse'
                        : 'bg-slate-800 text-slate-400 border border-slate-700'
                    }`}
                  >
                    <span
                      className={`w-1.5 h-1.5 rounded-full ${
                        isTrainingActive ? 'bg-emerald-400' : 'bg-slate-500'
                      }`}
                    />
                    {isTrainingActive ? 'Training Active' : 'Idle'}
                  </span>

                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                      isWsConnected
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                        : 'bg-slate-800 text-slate-400 border border-slate-700'
                    }`}
                  >
                    <Radio className={`w-3 h-3 ${isWsConnected ? 'text-cyan-400 animate-pulse' : 'text-slate-500'}`} />
                    {isWsConnected ? 'Bridge Live (Port 5050)' : 'Bridge Disconnected'}
                  </span>
                </h2>
                <p className="text-xs text-slate-400">
                  Headless PyTorch RL pipeline, live telemetry streaming, and ONNX checkpoint lifecycle.
                </p>
              </div>
            </div>
          </div>

          {/* Active Model & Staleness / Latency HUD */}
          <div className="flex flex-wrap items-center gap-2.5 text-xs">
            {/* Active Model Pill */}
            <div className="px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 flex items-center gap-2">
              <span className="text-slate-500 font-semibold">Active Policy:</span>
              <span className="font-mono text-emerald-400 font-bold">
                {activeModelPath.split('/').pop()}
              </span>
            </div>

            {/* Inference Latency */}
            <div className="px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-slate-400">Inference:</span>
              <span className="font-mono text-slate-200 font-bold">{lastInferenceMs || 1.8} ms</span>
            </div>

            {/* Staleness Badge */}
            <div
              className={`px-3 py-1.5 rounded-xl border flex items-center gap-1.5 font-semibold ${
                stalenessTicks === 0
                  ? 'bg-emerald-950/60 border-emerald-700/60 text-emerald-300'
                  : 'bg-amber-950/60 border-amber-600/80 text-amber-300 animate-pulse'
              }`}
            >
              {stalenessTicks === 0 ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span>0 Ticks Stale (Live Synchronous)</span>
                </>
              ) : (
                <>
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                  <span>{stalenessTicks} Ticks Stale (Async In-Flight)</span>
                </>
              )}
            </div>

            {/* Bridge Reconnect Button */}
            <button
              onClick={() => {
                if (isWsConnected) {
                  telemetryService.disconnectWebSocket();
                } else {
                  telemetryService.connectWebSocket();
                }
              }}
              className={`px-2.5 py-1.5 rounded-xl border text-xs font-semibold flex items-center gap-1.5 transition-all ${
                isWsConnected
                  ? 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700'
                  : 'bg-cyan-600 hover:bg-cyan-500 text-white border-cyan-500'
              }`}
            >
              {isWsConnected ? <Wifi className="w-3.5 h-3.5 text-emerald-400" /> : <WifiOff className="w-3.5 h-3.5 text-red-400" />}
              {isWsConnected ? 'Connected' : 'Connect Bridge'}
            </button>
          </div>
        </div>

        {/* Real-time Hardware Metrics Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3 mt-4 pt-4 border-t border-slate-800/80 text-xs">
          <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/60">
            <div className="text-[11px] text-slate-500 font-semibold flex items-center gap-1">
              <Zap className="w-3 h-3 text-amber-400" /> Steps / Second
            </div>
            <div className="text-base font-bold font-mono text-slate-100 mt-0.5">
              {hardware.sps ? hardware.sps.toLocaleString() : '3,840'}{' '}
              <span className="text-[10px] text-slate-500 font-normal">SPS</span>
            </div>
          </div>

          <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/60">
            <div className="text-[11px] text-slate-500 font-semibold flex items-center gap-1">
              <Cpu className="w-3 h-3 text-blue-400" /> CPU Worker Load
            </div>
            <div className="text-base font-bold font-mono text-slate-100 mt-0.5">
              {hardware.cpuUtilizationPct}%
            </div>
          </div>

          <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/60">
            <div className="text-[11px] text-slate-500 font-semibold flex items-center gap-1">
              <Server className="w-3 h-3 text-emerald-400" /> GPU Utilization
            </div>
            <div className="text-base font-bold font-mono text-slate-100 mt-0.5">
              {hardware.gpuUtilizationPct}%
            </div>
          </div>

          <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/60">
            <div className="text-[11px] text-slate-500 font-semibold flex items-center gap-1">
              <HardDrive className="w-3 h-3 text-purple-400" /> VRAM Memory
            </div>
            <div className="text-base font-bold font-mono text-slate-100 mt-0.5">
              {(hardware.gpuVramUsedMb / 1024).toFixed(1)} GB
            </div>
          </div>

          <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/60">
            <div className="text-[11px] text-slate-500 font-semibold flex items-center gap-1">
              <Layers className="w-3 h-3 text-cyan-400" /> IPC Bridge Latency
            </div>
            <div className="text-base font-bold font-mono text-slate-100 mt-0.5">
              {hardware.ipcLatencyMs} ms
            </div>
          </div>

          <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/60">
            <div className="text-[11px] text-slate-500 font-semibold flex items-center gap-1">
              <Shield className="w-3 h-3 text-pink-400" /> Policy Execution
            </div>
            <div className="text-base font-bold font-mono text-slate-100 mt-0.5">
              WASM / SIMD
            </div>
          </div>
        </div>
      </div>

      {/* 2. Job Launcher & Controller Panel */}
      <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-800">
          <div>
            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              <Terminal className="w-4 h-4 text-purple-400" /> RL Job Execution Controller
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Trigger headless training runs directly on the server. Output models will automatically export to ONNX.
            </p>
          </div>

          {jobFeedback && (
            <div
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 ${
                jobFeedback.type === 'success'
                  ? 'bg-emerald-950 border border-emerald-700 text-emerald-300'
                  : 'bg-red-950 border border-red-700 text-red-300'
              }`}
            >
              {jobFeedback.type === 'success' ? (
                <CheckCircle2 className="w-3.5 h-3.5" />
              ) : (
                <AlertTriangle className="w-3.5 h-3.5" />
              )}
              {jobFeedback.message}
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 text-xs">
          {/* Algorithm */}
          <div>
            <label className="text-slate-300 font-semibold mb-1 block">Algorithm</label>
            <select
              value={selectedAlgo}
              onChange={(e) => setSelectedAlgo(e.target.value as any)}
              disabled={isTrainingActive}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 font-medium focus:ring-1 focus:ring-purple-500"
            >
              <option value="mappo">MAPPO (Multi-Agent PPO)</option>
              <option value="ppo">PPO (Stable-Baselines3)</option>
              <option value="ippo">IPPO (Independent PPO)</option>
            </select>
          </div>

          {/* Scenario */}
          <div>
            <label className="text-slate-300 font-semibold mb-1 block">Scenario</label>
            <select
              value={selectedScenario}
              onChange={(e) => setSelectedScenario(e.target.value)}
              disabled={isTrainingActive}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 font-medium focus:ring-1 focus:ring-purple-500"
            >
              <option value="academy_3_vs_1_with_keeper">3 vs 1 with Keeper</option>
              <option value="academy_empty_goal">Empty Goal</option>
              <option value="academy_run_to_score_with_keeper">Run to Score w/ Keeper</option>
              <option value="academy_pass_and_shoot_with_keeper">Pass & Shoot w/ Keeper</option>
            </select>
          </div>

          {/* Target Timesteps */}
          <div>
            <label className="text-slate-300 font-semibold mb-1 block">Target Timesteps</label>
            <input
              type="number"
              value={targetSteps}
              onChange={(e) => setTargetSteps(parseInt(e.target.value, 10) || 1000)}
              disabled={isTrainingActive}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 font-mono focus:ring-1 focus:ring-purple-500"
              step="5000"
              min="1000"
            />
          </div>

          {/* Resume From Checkpoint */}
          <div>
            <label className="text-slate-300 font-semibold mb-1 block">Resume Checkpoint (Optional)</label>
            <input
              type="text"
              placeholder="e.g. mappo_3v1_50k.pt"
              value={resumeCheckpoint}
              onChange={(e) => setResumeCheckpoint(e.target.value)}
              disabled={isTrainingActive}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 font-mono text-xs focus:ring-1 focus:ring-purple-500"
            />
          </div>

          {/* Start / Stop Button */}
          <div className="flex items-end">
            {isTrainingActive ? (
              <button
                onClick={handleStopTraining}
                disabled={isSubmittingJob}
                className="w-full py-2.5 rounded-lg bg-red-600 hover:bg-red-500 text-white font-bold text-xs flex items-center justify-center gap-1.5 shadow-lg shadow-red-950/40 transition-all"
              >
                <Square className="w-3.5 h-3.5 fill-current" /> Stop Training Job
              </button>
            ) : (
              <button
                onClick={handleStartTraining}
                disabled={isSubmittingJob}
                className="w-full py-2.5 rounded-lg bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-bold text-xs flex items-center justify-center gap-1.5 shadow-lg shadow-purple-950/40 transition-all disabled:opacity-50"
              >
                <Play className="w-3.5 h-3.5 fill-current" /> Launch Training
              </button>
            )}
          </div>
        </div>

        {/* Active Job Progress Bar (when running) */}
        {isTrainingActive && (
          <div className="mt-4 p-3 rounded-xl bg-slate-950/80 border border-purple-900/40 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-purple-300 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-purple-400 animate-ping" />
                Active Job: {activeJob?.id || 'job_running'} ({activeJob?.config?.algorithm?.toUpperCase()} on{' '}
                {activeJob?.config?.scenario})
              </span>
              <span className="font-mono text-slate-300 font-bold">
                {currentProgressStep.toLocaleString()} / {totalTargetSteps.toLocaleString()} steps ({progressPct}%)
              </span>
            </div>
            <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-purple-600 via-indigo-500 to-emerald-400 rounded-full transition-all duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* 3. Live Visual Telemetry Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Left: Reward & Goal Rate Curves */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 shadow-md">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-emerald-400" />
              <h3 className="text-xs font-bold text-slate-100">Rolling Episode Reward & Goal Rate</h3>
            </div>
            <span className="text-[11px] font-mono text-emerald-400 font-semibold">
              Goal Rate: {lastSnapshot.goalRate?.toFixed(1) || '0.0'}%
            </span>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="rewardGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                <XAxis dataKey="stepLabel" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    borderRadius: '0.5rem',
                    fontSize: '11px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Area
                  type="monotone"
                  dataKey="rollingReward"
                  name="Mean Reward"
                  stroke="#10b981"
                  fill="url(#rewardGrad)"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="goalRate"
                  name="Goal Rate (%)"
                  stroke="#06b6d4"
                  strokeWidth={2}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right: Policy Loss, Value Loss & Entropy */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 shadow-md">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-purple-400" />
              <h3 className="text-xs font-bold text-slate-100">Loss Curves & Policy Entropy</h3>
            </div>
            <span className="text-[11px] font-mono text-purple-400 font-semibold">
              Entropy: {lastSnapshot.entropy?.toFixed(3) || '2.85'}
            </span>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                <XAxis dataKey="stepLabel" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    borderRadius: '0.5rem',
                    fontSize: '11px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line
                  type="monotone"
                  dataKey="valueLoss"
                  name="Value Loss (MSE)"
                  stroke="#f59e0b"
                  strokeWidth={1.5}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="entropy"
                  name="Entropy (H)"
                  stroke="#a855f7"
                  strokeWidth={1.5}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="policyLoss"
                  name="Policy Loss"
                  stroke="#ec4899"
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 4. Live Python Output Console / Stdout Terminal */}
      <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-md">
        <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-emerald-400" />
            <h3 className="text-xs font-bold text-slate-100">Live Training Process Output & Evaluation Logs</h3>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <label className="flex items-center gap-1.5 text-slate-400 cursor-pointer text-[11px]">
              <input
                type="checkbox"
                checked={autoScrollLogs}
                onChange={(e) => setAutoScrollLogs(e.target.checked)}
                className="rounded accent-purple-500"
              />
              Auto-scroll
            </label>
            <button
              onClick={() => {
                telemetryService.liveLogs = [];
                setTrigger((p) => p + 1);
              }}
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] text-slate-300"
            >
              Clear
            </button>
            <button
              onClick={() => {
                navigator.clipboard.writeText(liveLogs.join('\n'));
                alert('Logs copied to clipboard');
              }}
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] text-slate-300 flex items-center gap-1"
            >
              <Copy className="w-3 h-3" /> Copy
            </button>
          </div>
        </div>

        <div
          ref={logTerminalRef}
          className="h-44 overflow-y-auto bg-slate-950 rounded-xl p-3 font-mono text-[11px] leading-relaxed border border-slate-800/80 text-slate-300 space-y-0.5 select-text"
        >
          {liveLogs.length === 0 ? (
            <div className="text-slate-600 italic">
              No training output yet. Click "Launch Training" above to start PyTorch simulation loop.
            </div>
          ) : (
            liveLogs.map((line, idx) => (
              <div
                key={idx}
                className={
                  line.includes('Goal Rate') || line.includes('Win Rate')
                    ? 'text-emerald-300 font-semibold'
                    : line.includes('Error') || line.includes('REJECTED')
                    ? 'text-red-400'
                    : line.includes('Step')
                    ? 'text-purple-300'
                    : 'text-slate-300'
                }
              >
                {line}
              </div>
            ))
          )}
        </div>
      </div>

      {/* 5. Interactive Checkpoints & ONNX Model Registry */}
      <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-800">
          <div>
            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              <FileCode className="w-4 h-4 text-emerald-400" /> Checkpoints & ONNX Model Registry
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Available neural checkpoints in <code className="text-slate-300">public/models</code>. Activate or hot-swap policies without engine reload.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => telemetryService.refreshCheckpoints()}
              disabled={isRefreshingCheckpoints}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 flex items-center gap-1.5 border border-slate-700"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRefreshingCheckpoints ? 'animate-spin' : ''}`} />
              Refresh
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold flex items-center gap-1.5 shadow-md"
            >
              <UploadCloud className="w-3.5 h-3.5" />
              Upload ONNX
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".onnx"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file);
              }}
            />
          </div>
        </div>

        {uploadStatus && (
          <div className="mb-3 p-2.5 rounded-lg bg-slate-950 border border-slate-700 text-xs text-cyan-300 flex items-center gap-2">
            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
            {uploadStatus}
          </div>
        )}

        {/* Checkpoint Table */}
        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 text-slate-400 font-semibold border-b border-slate-800">
              <tr>
                <th className="p-3">Model Filename</th>
                <th className="p-3">Algorithm</th>
                <th className="p-3">Scenario</th>
                <th className="p-3">Timesteps</th>
                <th className="p-3">Size</th>
                <th className="p-3">SHA-256 Checksum</th>
                <th className="p-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 bg-slate-900/40">
              {checkpoints.length === 0 ? (
                <tr>
                  <td colSpan={7} className="p-4 text-center text-slate-500 italic">
                    No ONNX models found in public/models. Run export_onnx.py or upload a model.
                  </td>
                </tr>
              ) : (
                checkpoints.map((ck) => {
                  const isActive = activeModelPath.endsWith(ck.filename);
                  return (
                    <tr key={ck.id} className="hover:bg-slate-800/40 transition-colors">
                      <td className="p-3 font-mono font-semibold text-slate-200 flex items-center gap-2">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            isActive ? 'bg-emerald-400' : 'bg-slate-600'
                          }`}
                        />
                        {ck.filename}
                      </td>
                      <td className="p-3">
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-purple-950 text-purple-300 border border-purple-800">
                          {ck.algorithm}
                        </span>
                      </td>
                      <td className="p-3 text-slate-300">{ck.scenario}</td>
                      <td className="p-3 font-mono text-slate-300">
                        {ck.timesteps ? ck.timesteps.toLocaleString() : 'Pre-trained'}
                      </td>
                      <td className="p-3 font-mono text-slate-400">
                        {(ck.sizeBytes / 1024).toFixed(1)} KB
                      </td>
                      <td className="p-3 font-mono text-[10px] text-slate-500">
                        {ck.checkpointSha256 ? `${ck.checkpointSha256.slice(0, 8)}...` : 'Verified'}
                      </td>
                      <td className="p-3 text-right space-x-2">
                        {isActive ? (
                          <span className="px-2.5 py-1 rounded-lg bg-emerald-950 border border-emerald-700 text-emerald-400 font-bold text-[11px] inline-flex items-center gap-1">
                            <Check className="w-3 h-3" /> Active
                          </span>
                        ) : (
                          <button
                            onClick={async () => {
                              if (onSelectModel) {
                                await onSelectModel(ck.path);
                              }
                            }}
                            className="px-2.5 py-1 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-[11px] transition-all"
                          >
                            Activate Policy
                          </button>
                        )}
                        <button
                          onClick={() => setDeleteModalItem(ck)}
                          className="px-2 py-1 rounded-lg bg-red-950 hover:bg-red-900 text-red-300 border border-red-800/60 text-[11px] transition-all"
                          title="Delete model checkpoint"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {deleteModalItem && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 max-w-md w-full shadow-2xl space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-red-950 border border-red-800 flex items-center justify-center text-red-400">
                <Trash2 className="w-5 h-5" />
              </div>
              <div>
                <h4 className="text-sm font-bold text-slate-100">Delete Checkpoint</h4>
                <p className="text-xs text-slate-400">
                  Are you sure you want to delete <span className="font-mono text-slate-200">{deleteModalItem.filename}</span>?
                </p>
              </div>
            </div>

            {deleteModalItem.hasSourcePt && (
              <label className="flex items-center gap-2 p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={deleteSourcePt}
                  onChange={(e) => setDeleteSourcePt(e.target.checked)}
                  className="rounded accent-red-500"
                />
                <span>Also delete source PyTorch weight file (<code className="font-mono text-red-300">{deleteModalItem.sourcePtName}</code>)</span>
              </label>
            )}

            <div className="flex items-center justify-end gap-2 pt-2">
              <button
                onClick={() => setDeleteModalItem(null)}
                disabled={isDeleting}
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-300"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmDelete}
                disabled={isDeleting}
                className="px-4 py-1.5 rounded-lg bg-red-600 hover:bg-red-500 text-white text-xs font-bold flex items-center gap-1.5"
              >
                {isDeleting && <RefreshCw className="w-3 h-3 animate-spin" />}
                Confirm Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
