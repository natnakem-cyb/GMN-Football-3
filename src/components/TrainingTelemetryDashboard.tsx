import React, { useState, useEffect } from 'react';
import {
  TrainingTelemetryService,
} from '../engine/TrainingTelemetryService';
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
  Pause,
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
} from 'lucide-react';

export const TrainingTelemetryDashboard: React.FC = () => {
  const telemetryService = TrainingTelemetryService.getInstance();
  const [, setTrigger] = useState(0);

  useEffect(() => {
    const unsub = telemetryService.subscribe(() => {
      setTrigger((prev) => prev + 1);
    });
    return () => unsub();
  }, [telemetryService]);

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
  } = telemetryService;

  const lastSnapshot = snapshots[snapshots.length - 1] || snapshots[0];
  const progressPct = ((currentStep / hyperparameters.targetTimesteps) * 100).toFixed(1);

  // Formatting steps for charts (e.g., "120k")
  const chartData = snapshots.map((s) => ({
    ...s,
    stepLabel: `${Math.round(s.step / 1000)}k`,
  }));

  return (
    <div className="space-y-5 animate-fadeIn">
      {/* 1. Header & Live Training Command Strip */}
      <div className="p-4 md:p-5 rounded-2xl bg-slate-900/95 border border-slate-800 shadow-2xl backdrop-blur-md">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-purple-900/30">
                <Activity className="w-4 h-4 text-white" />
              </div>
              <div>
                <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                  Multi-Agent PPO Training Cockpit
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
                    {isTrainingActive ? 'Training Active' : 'Loop Paused'}
                  </span>

                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                      isWsConnected
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                        : 'bg-slate-800 text-slate-400 border border-slate-700'
                    }`}
                  >
                    <Radio className={`w-3 h-3 ${isWsConnected ? 'text-cyan-400 animate-pulse' : 'text-slate-500'}`} />
                    {isWsConnected ? 'PyTorch Bridge Live' : 'Browser Simulator'}
                  </span>
                </h2>
                <p className="text-xs text-slate-400">
                  Real-time neural optimization telemetry, surrogate objectives, and hardware utilization
                </p>
              </div>
            </div>
          </div>

          {/* Action & Speed Controls */}
          <div className="flex flex-wrap items-center gap-2.5">
            {/* Speed Selector */}
            <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
              <span className="px-2 text-[11px] font-semibold text-slate-400">Speed:</span>
              {[
                { label: '1x (500 SPS)', val: 1 },
                { label: '5x (2.5k SPS)', val: 5 },
                { label: '10x Turbo', val: 10 },
              ].map((sp) => (
                <button
                  key={sp.val}
                  onClick={() => telemetryService.setTrainingSpeed(sp.val)}
                  className={`px-2 py-1 rounded-lg text-xs font-semibold transition-all ${
                    trainingSpeed === sp.val
                      ? 'bg-purple-600 text-white shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {sp.label}
                </button>
              ))}
            </div>

            {/* Run / Pause Toggle */}
            <button
              id="btn-toggle-training"
              onClick={() => telemetryService.toggleTraining()}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold shadow-lg transition-all ${
                isTrainingActive
                  ? 'bg-amber-500 hover:bg-amber-400 text-slate-950'
                  : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-950/40'
              }`}
            >
              {isTrainingActive ? (
                <>
                  <Pause className="w-4 h-4 fill-current" /> Pause Training
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" /> Start Training Loop
                </>
              )}
            </button>

            {/* Single Step Batch */}
            <button
              onClick={() => telemetryService.stepTrainingBatch()}
              disabled={isTrainingActive}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 text-xs font-semibold border border-slate-700 transition-all"
              title="Step forward by 1 PPO rollout buffer (256 steps)"
            >
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              Step Batch (+256)
            </button>

            {/* Reset */}
            <button
              onClick={() => telemetryService.resetMetrics()}
              className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 border border-slate-700 transition-all"
              title="Reset training telemetry curve"
            >
              <RotateCcw className="w-4 h-4" />
            </button>

            {/* WebSocket Bridge Toggle */}
            <button
              onClick={() => {
                if (isWsConnected) {
                  telemetryService.disconnectWebSocket();
                } else {
                  telemetryService.connectWebSocket();
                }
              }}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold border transition-all ${
                isWsConnected
                  ? 'bg-cyan-950/80 border-cyan-600 text-cyan-200 hover:bg-cyan-900/80 shadow-sm'
                  : wsStatus === 'connecting'
                  ? 'bg-amber-950/80 border-amber-700 text-amber-200'
                  : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
              }`}
              title={isWsConnected ? `Connected to ${wsUrl}` : `Connect to PyTorch Training Bridge (${wsUrl})`}
            >
              {isWsConnected ? (
                <>
                  <Wifi className="w-3.5 h-3.5 text-cyan-400" />
                  <span className="hidden sm:inline">Bridge Connected</span>
                </>
              ) : wsStatus === 'connecting' ? (
                <>
                  <Radio className="w-3.5 h-3.5 text-amber-400 animate-spin" />
                  <span className="hidden sm:inline">Connecting...</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-3.5 h-3.5 text-slate-400" />
                  <span className="hidden sm:inline">Connect Bridge</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Hardware & Throughput Status Bar */}
        <div className="mt-4 pt-4 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 text-xs">
          <div className="flex items-center gap-2 p-2 rounded-lg bg-slate-950/60 border border-slate-800/60">
            <Cpu className="w-4 h-4 text-emerald-400 shrink-0" />
            <div>
              <div className="text-[10px] text-slate-400 uppercase font-bold">Throughput</div>
              <div className="font-mono font-bold text-slate-100">{hardware.sps.toLocaleString()} SPS</div>
            </div>
          </div>

          <div className="flex items-center gap-2 p-2 rounded-lg bg-slate-950/60 border border-slate-800/60">
            <Server className="w-4 h-4 text-purple-400 shrink-0" />
            <div>
              <div className="text-[10px] text-slate-400 uppercase font-bold">GPU VRAM</div>
              <div className="font-mono font-bold text-slate-100">
                {(hardware.gpuVramUsedMb / 1024).toFixed(2)} / {(hardware.gpuVramTotalMb / 1024).toFixed(0)} GB
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 p-2 rounded-lg bg-slate-950/60 border border-slate-800/60">
            <Activity className="w-4 h-4 text-indigo-400 shrink-0" />
            <div>
              <div className="text-[10px] text-slate-400 uppercase font-bold">GPU Compute</div>
              <div className="font-mono font-bold text-slate-100">{hardware.gpuUtilizationPct}% Core</div>
            </div>
          </div>

          <div className="flex items-center gap-2 p-2 rounded-lg bg-slate-950/60 border border-slate-800/60">
            <Layers className="w-4 h-4 text-blue-400 shrink-0" />
            <div>
              <div className="text-[10px] text-slate-400 uppercase font-bold">Workers</div>
              <div className="font-mono font-bold text-slate-100">{hardware.workerCount} PettingZoo</div>
            </div>
          </div>

          <div className="flex items-center gap-2 p-2 rounded-lg bg-slate-950/60 border border-slate-800/60">
            <HardDrive className="w-4 h-4 text-amber-400 shrink-0" />
            <div>
              <div className="text-[10px] text-slate-400 uppercase font-bold">Rollout Buffer</div>
              <div className="font-mono font-bold text-slate-100">
                {hardware.bufferSize} / {hardware.bufferCapacity} steps
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 p-2 rounded-lg bg-slate-950/60 border border-slate-800/60">
            <Zap className="w-4 h-4 text-cyan-400 shrink-0" />
            <div>
              <div className="text-[10px] text-slate-400 uppercase font-bold">IPC Bridge</div>
              <div className="font-mono font-bold text-slate-100">{hardware.ipcLatencyMs} ms ping</div>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Primary KPI Metric Cards Strip */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {/* Progress */}
        <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-semibold mb-1">Total Steps</div>
          <div className="text-lg font-bold font-mono text-slate-100">
            {currentStep.toLocaleString()}
          </div>
          <div className="text-[10px] text-slate-500 font-mono mt-0.5">
            {progressPct}% of {hyperparameters.targetTimesteps.toLocaleString()}
          </div>
          <div className="w-full h-1.5 rounded-full bg-slate-800 mt-2 overflow-hidden">
            <div className="h-full bg-purple-500" style={{ width: `${progressPct}%` }} />
          </div>
        </div>

        {/* Rolling Reward */}
        <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-semibold mb-1">Rolling Reward (R̄₅₀)</div>
          <div className="text-lg font-bold font-mono text-emerald-400">
            {lastSnapshot.rollingReward >= 0 ? `+${lastSnapshot.rollingReward}` : lastSnapshot.rollingReward}
          </div>
          <div className="text-[10px] text-emerald-500/80 flex items-center gap-0.5 mt-0.5">
            <ArrowUpRight className="w-3 h-3" /> Converging to goal
          </div>
        </div>

        {/* Goal Rate */}
        <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-semibold mb-1">Goal Rate</div>
          <div className="text-lg font-bold font-mono text-blue-400">
            {lastSnapshot.goalRate}%
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">Win/conversion metric</div>
        </div>

        {/* Value Loss */}
        <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-semibold mb-1">Value Loss (L_VF)</div>
          <div className="text-lg font-bold font-mono text-amber-400">
            {lastSnapshot.valueLoss}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">Critic MSE baseline</div>
        </div>

        {/* Entropy */}
        <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-semibold mb-1">Policy Entropy S[π]</div>
          <div className="text-lg font-bold font-mono text-purple-400">
            {lastSnapshot.entropy}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">Exploration schedule</div>
        </div>

        {/* Approx KL */}
        <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-semibold mb-1">Approx KL Divergence</div>
          <div className="text-lg font-bold font-mono text-cyan-400">
            {lastSnapshot.approxKl}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">Trust region stability</div>
        </div>
      </div>

      {/* 3. Real-Time Recharts Telemetry Graphs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Graph 1: Actor vs Critic Loss */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 shadow-md">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <TrendingUp className="w-3.5 h-3.5 text-purple-400" />
                PPO Loss Surfaces: Policy Loss (L_CLIP) & Value Loss (L_VF)
              </h3>
              <p className="text-[11px] text-slate-400">
                Surrogate clipped objective optimization alongside centralized critic MSE
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300 border border-slate-700">
              Update #{lastSnapshot.update}
            </span>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="valLossGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="polLossGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#a855f7" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#a855f7" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="stepLabel" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={10} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                <Area
                  type="monotone"
                  dataKey="valueLoss"
                  name="Value Loss (L_VF)"
                  stroke="#f59e0b"
                  fill="url(#valLossGrad)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="policyLoss"
                  name="Policy Loss (L_CLIP)"
                  stroke="#a855f7"
                  fill="url(#polLossGrad)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Graph 2: Policy Entropy & Approx KL */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 shadow-md">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <Shield className="w-3.5 h-3.5 text-cyan-400" />
                Exploration Entropy S[π] & Approx KL Divergence
              </h3>
              <p className="text-[11px] text-slate-400">
                Monitors exploration decay and guards against destructive policy collapse
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-cyan-400 border border-slate-700">
              KL ≤ 0.02 Safe
            </span>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="stepLabel" stroke="#64748b" fontSize={10} />
                <YAxis yAxisId="entropy" stroke="#a855f7" fontSize={10} domain={[1.0, 3.2]} />
                <YAxis yAxisId="kl" orientation="right" stroke="#06b6d4" fontSize={10} domain={[0, 0.03]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                <Line
                  yAxisId="entropy"
                  type="monotone"
                  dataKey="entropy"
                  name="Entropy S[π]"
                  stroke="#c084fc"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  yAxisId="kl"
                  type="monotone"
                  dataKey="approxKl"
                  name="Approx KL Divergence"
                  stroke="#06b6d4"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Graph 3: Checkpoint Reward & Goal Rate */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 shadow-md">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <Target className="w-3.5 h-3.5 text-emerald-400" />
                Rolling Episodic Reward & Goal Rate Convergence
              </h3>
              <p className="text-[11px] text-slate-400">
                Performance across 50-episode moving window on academy_3_vs_1_with_keeper
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-950 text-emerald-300 border border-emerald-800">
              Peak: {lastSnapshot.goalRate}% Goals
            </span>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="stepLabel" stroke="#64748b" fontSize={10} />
                <YAxis yAxisId="rew" stroke="#10b981" fontSize={10} domain={[-0.5, 1.2]} />
                <YAxis yAxisId="goal" orientation="right" stroke="#3b82f6" fontSize={10} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                <Line
                  yAxisId="rew"
                  type="monotone"
                  dataKey="rollingReward"
                  name="Mean Reward (R̄₅₀)"
                  stroke="#10b981"
                  strokeWidth={2.5}
                  dot={false}
                />
                <Line
                  yAxisId="goal"
                  type="monotone"
                  dataKey="goalRate"
                  name="Goal Rate %"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Graph 4: Gradient Norm & Learning Rate Schedule */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 shadow-md">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-amber-400" />
                Gradient Norms ||∇θ|| & Adaptive LR Schedule
              </h3>
              <p className="text-[11px] text-slate-400">
                Max gradient clipping (0.5) and cosine decay scheduling
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-amber-300 border border-slate-700">
              LR: {lastSnapshot.learningRate}
            </span>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="stepLabel" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#f59e0b" fontSize={10} domain={[0, 0.6]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                <Line
                  type="monotone"
                  dataKey="gradNorm"
                  name="Gradient Norm (||∇θ||)"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="clipFraction"
                  name="Clip Fraction"
                  stroke="#ec4899"
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 4. Interactive Hyperparameter Tuning & Architecture Inspection */}
      <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 shadow-md">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-purple-400" />
            <h3 className="text-xs font-bold text-slate-100">
              Multi-Agent PPO Algorithm Hyperparameters & Contract Configuration
            </h3>
          </div>
          <span className="text-[11px] text-slate-400 font-mono">
            Architecture: SharedActor (MLP 64×64) + CentralizedCritic (Deep Sets)
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          {/* Learning Rate */}
          <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80 space-y-1.5">
            <div className="flex justify-between font-semibold">
              <span className="text-slate-300">Learning Rate (α)</span>
              <span className="font-mono text-purple-400">{hyperparameters.learningRate}</span>
            </div>
            <input
              type="range"
              min="0.00005"
              max="0.001"
              step="0.00005"
              value={hyperparameters.learningRate}
              onChange={(e) => telemetryService.setHyperparameter('learningRate', parseFloat(e.target.value))}
              className="w-full accent-purple-500 cursor-pointer"
            />
            <div className="text-[10px] text-slate-500">Adam optimizer initial base step size</div>
          </div>

          {/* Clip Range */}
          <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80 space-y-1.5">
            <div className="flex justify-between font-semibold">
              <span className="text-slate-300">PPO Clip Range (ε)</span>
              <span className="font-mono text-cyan-400">{hyperparameters.clipRange}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.3"
              step="0.02"
              value={hyperparameters.clipRange}
              onChange={(e) => telemetryService.setHyperparameter('clipRange', parseFloat(e.target.value))}
              className="w-full accent-cyan-500 cursor-pointer"
            />
            <div className="text-[10px] text-slate-500">Surrogate objective clip bounds [1-ε, 1+ε]</div>
          </div>

          {/* Entropy Coef */}
          <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80 space-y-1.5">
            <div className="flex justify-between font-semibold">
              <span className="text-slate-300">Entropy Coef (c₂)</span>
              <span className="font-mono text-emerald-400">{hyperparameters.entropyCoef}</span>
            </div>
            <input
              type="range"
              min="0.001"
              max="0.05"
              step="0.002"
              value={hyperparameters.entropyCoef}
              onChange={(e) => telemetryService.setHyperparameter('entropyCoef', parseFloat(e.target.value))}
              className="w-full accent-emerald-500 cursor-pointer"
            />
            <div className="text-[10px] text-slate-500">Exploration incentive regularization weight</div>
          </div>

          {/* Value Coef */}
          <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80 space-y-1.5">
            <div className="flex justify-between font-semibold">
              <span className="text-slate-300">Value Coef (c₁)</span>
              <span className="font-mono text-amber-400">{hyperparameters.valueCoef}</span>
            </div>
            <input
              type="range"
              min="0.2"
              max="1.0"
              step="0.1"
              value={hyperparameters.valueCoef}
              onChange={(e) => telemetryService.setHyperparameter('valueCoef', parseFloat(e.target.value))}
              className="w-full accent-amber-500 cursor-pointer"
            />
            <div className="text-[10px] text-slate-500">Centralized critic MSE loss weighting</div>
          </div>
        </div>
      </div>
    </div>
  );
};
