# ⚽ GMN-Football-3

**A Browser-Native Football Simulation & Reinforcement-Learning Research Platform.**

One authoritative TypeScript game engine drives both an interactive browser match and a headless Python RL training pipeline (Gymnasium / PettingZoo → Stable-Baselines3 / custom PPO, IPPO, MAPPO), so an agent is always trained against the exact same physics and rules a human plays against.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Language: TypeScript](https://img.shields.io/badge/Language-TypeScript-3178C6.svg)](https://www.typescriptlang.org/)
[![Frontend: React 18](https://img.shields.io/badge/Frontend-React%2018-61DAFB.svg)](https://react.dev/)
[![Build: Vite](https://img.shields.io/badge/Build-Vite-646CFF.svg)](https://vitejs.dev/)
[![RL: Gymnasium + PettingZoo](https://img.shields.io/badge/RL-Gymnasium%20%2B%20PettingZoo-5B8DEF.svg)](https://gymnasium.farama.org/)
[![RL: SB3 PPO / IPPO / MAPPO](https://img.shields.io/badge/RL-PPO%20%2F%20IPPO%20%2F%20MAPPO-EA7C2B.svg)](https://stable-baselines3.readthedocs.io/)

> **Status:** RL-ready simulation and research platform. There is not yet a policy trained to play a full match — see [Current Status](#current-status) for exactly what has and hasn't been trained so far.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Environment Contract](#environment-contract)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Quick Start (Browser App)](#quick-start-browser-app)
- [Running the RL Training Pipeline](#running-the-rl-training-pipeline)
- [Scenarios](#scenarios)
- [Testing & Validation](#testing--validation)
- [Current Status](#current-status)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

GMN-Football-3 is built around a single design decision: **the browser game and the RL environment do not maintain separate simulators.** The TypeScript `GameEngine` in `src/engine/` is authoritative. The browser renders and controls it interactively; a headless Node.js bridge exposes the same engine to Python training code over HTTP or a binary WebSocket protocol.

```text
                    GMN FOOTBALL WORLD
                           │
            ┌──────────────┴──────────────┐
            │                              │
         HUMAN                            AI
            │                              │
       Browser UI                    RL Environment
            │                              │
       React / Canvas          Gymnasium / PettingZoo
                                            │
                              Stable-Baselines3 PPO,
                              custom IPPO / MAPPO
                                            │
                                        PyTorch
```

This means: no separate "training physics" that quietly diverges from what a human sees, and no re-implementation risk between the game and the research environment.

## Architecture

```text
                      TypeScript GameEngine (authoritative)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
  Browser / React             Headless Node bridge         Scripts (tests,
  (src/App.tsx)               (training/bridge_server.ts)  benchmarks, audits)
                                    │
                        ┌───────────┴───────────┐
                        │                       │
                    HTTP bridge            Binary WebSocket
                        │                       │
                        └───────────┬───────────┘
                                    ▼
                     Python: Gymnasium env (gmn_gym.py)
                     Python: PettingZoo env (gmn_pettingzoo.py)
                                    │
                     Stable-Baselines3 PPO  /  custom IPPO  /  custom MAPPO
                                    │
                                 PyTorch
```

Inside the engine itself:

```text
GameEngine
├── Players & Ball state
├── Physics.ts        — movement, kicking, tackling, ball flight
├── Rules.ts           — pitch geometry, formations, offside line
├── SeededRNG.ts        — Mulberry32 deterministic PRNG
├── ObservationEncoder.ts — RL observation vector + reward shaping
└── Contract.ts        — versioned observation/action schema (single source of truth on the TS side)
```

## Environment Contract

Defined in `src/engine/Contract.ts` — treat this file as authoritative if anything below drifts out of date:

| Constant | Value |
|---|---|
| `GMN_ENV_VERSION` | `3.1.0` |
| `OBSERVATION_SCHEMA_VERSION` | `simple115_v3_role` |
| `ACTION_SCHEMA_VERSION` | `discrete19_v1` |
| `BASE_OBSERVATION_DIM` | `115` (Google Research Football–style SMM/feature vector) |
| `ROLE_DIM` | `12` (one-hot over `ROLE_VOCABULARY`) |
| `OBSERVATION_DIM` | **`127`** (`BASE_OBSERVATION_DIM + ROLE_DIM`) |
| `ACTION_SPACE_SIZE` | `19` (discrete) |

The 127-float observation layout (see `ObservationEncoder.ts` for the exact offsets): left-team positions (22) → left-team velocities (22) → right-team positions (22) → right-team velocities (22) → ball position (3) → ball velocity (3) → ball ownership one-hot (3) → active-player one-hot (11) → game-mode one-hot (7) → agent role one-hot (12).

The 19 discrete actions cover 8-directional movement, idle, short/long/high pass, shot, sprint (+release), dribble (+release), release-direction, and slide tackle — see `training/action_mapping.ts` for the canonical mapping (mirrored in `TrainedPolicyAgent.ts` for in-browser inference).

Python and TypeScript each declare their own copies of these constants (`gmn_gym.py`, `gmn_pettingzoo.py`, `Contract.ts`); the bridge's `/health` endpoint cross-checks `observation_dim`/`action_space_size` at connection time and raises if they disagree.

## Technology Stack

| Layer | Technology |
|---|---|
| Simulation & game logic | TypeScript |
| UI | React 18 + Vite |
| Styling | Tailwind CSS / PostCSS |
| Charts | Recharts |
| Node bridge runtime | Node.js via `tsx` |
| Transport | HTTP (REST) and binary WebSocket (`ws`) |
| RL API (single-agent) | Gymnasium |
| RL API (multi-agent) | PettingZoo (+ SuperSuit for vectorization) |
| RL algorithms | Stable-Baselines3 PPO; custom IPPO and MAPPO implementations |
| ML backend | PyTorch |
| Browser inference | Hand-rolled MLP forward pass over exported weights (see [Known Limitations](#known-limitations) re: the shipped but unused `onnxruntime-web`/ONNX export path) |
| Deterministic RNG | Mulberry32 (`SeededRNG.ts`) |
| Optional AI match commentary | `@google/genai` (Gemini API, via `src/services/geminiService.ts`) |
| License | Apache-2.0 |

## Repository Structure

```text
GMN-Football-3/
├── src/
│   ├── engine/            # Authoritative simulation
│   │   ├── GameEngine.ts
│   │   ├── Physics.ts
│   │   ├── Rules.ts
│   │   ├── ObservationEncoder.ts
│   │   ├── SeededRNG.ts
│   │   ├── Contract.ts
│   │   ├── EventEncoder.ts
│   │   └── Vector.ts
│   ├── agents/            # Decision-making policies
│   │   ├── BaseAgent.ts
│   │   ├── HumanAgent.ts
│   │   ├── RuleBasedAgent.ts
│   │   ├── NeuralHeuristicAgent.ts
│   │   ├── ScriptedScenarioAgent.ts
│   │   ├── TrainedPolicyAgent.ts
│   │   └── mappo_weights.ts   # generated — exported trained-policy weights
│   ├── scenarios/          # Scenario/curriculum registry
│   ├── components/          # React UI
│   ├── services/           # Gemini-based match commentary (optional)
│   ├── types/
│   ├── App.tsx / main.tsx / index.css
│
├── training/               # Python + TS training/eval/bridge code
│   ├── bridge_server.ts     # Node bridge: HTTP + binary WebSocket
│   ├── action_mapping.ts
│   ├── gmn_gym.py           # Gymnasium single-agent env
│   ├── gmn_pettingzoo.py    # PettingZoo multi-agent env
│   ├── train_ppo.py / train_stage2_ppo.py
│   ├── train_ippo.py / eval_ippo_baseline.py
│   ├── train_mappo.py / mappo_networks.py / mappo_rollout.py / mappo_update.py / eval_mappo.py
│   ├── export_onnx.py / onnx_proto_builder.py
│   ├── episode_recorder.py / trace_to_frames.py / binary_event_decoder.py
│   ├── eval_checkpoint.py / eval_progress.py / eval_generalization.py / generate_comparison_table.py
│   ├── rl_validation_suite.py
│   ├── benchmark.ts / benchmark_bridge.py / benchmark_bridge_ws.py
│   ├── test_*.ts / test_*.py   # determinism, transport parity, scenario, multi-agent tests
│   ├── audit_observations_and_actions.ts / stage2_audit_and_baseline.ts
│   ├── verify_scenario_playability.ts / scripted_eval.ts
│   ├── models/              # Checkpoints (currently: smoke tests + one drill-scenario run — see Current Status)
│   └── results/             # win_rate_progress.csv, comparison_table.md/.html
│
├── public/models/           # mappo_policy.onnx (exported, not currently loaded by the browser app)
├── requirements.txt          # Python deps (root)
├── training/requirements.txt # Python deps (training-pinned — has stricter/older version bounds than the root file; prefer this one for training)
├── package.json
├── tsconfig.json             # NOTE: only includes `src/` — training/*.ts is not currently type-checked
├── vite.config.ts / tailwind.config.js / postcss.config.js
├── .env.example              # GEMINI_API_KEY (optional, for AI match commentary)
├── LICENSE (Apache-2.0)
└── CONTRIBUTING.md
```

## Quick Start (Browser App)

Requires Node.js 18+.

```bash
npm install
npm run dev        # http://localhost:3000
```

Other useful scripts:

```bash
npm run build       # tsc (src/ only) + vite build
npm run lint         # tsc --noEmit (src/ only)
npm run preview      # serve the production build
```

## Running the RL Training Pipeline

Requires Python 3.10+ and Node.js (the bridge server runs via `npx tsx`).

```bash
pip install -r training/requirements.txt
```

**1. Start the bridge** (optional — training scripts will auto-launch it if it isn't already running):

```bash
npm run bridge       # tsx training/bridge_server.ts, default port 5050
```

**2. Train.** Scripts wired into `package.json`:

```bash
npm run test:ppo             # Stable-Baselines3 PPO, short smoke run (1,000 steps)
npm run test:ippo            # Custom IPPO, short smoke run (3,072 steps)
npm run test:ippo:train      # Custom IPPO, longer run (200,000 steps)
npm run test:ippo:eval       # Evaluate/compare an IPPO checkpoint
```

MAPPO has no `package.json` shortcut yet — invoke it directly:

```bash
python3 training/train_mappo.py
python3 training/eval_mappo.py
```

Both `train_ppo.py` and the custom trainers accept a `--scenario` (or positional step-count) argument — see each script's `argparse` setup for the current options. Training currently runs against a single environment instance per process (no parallel rollout collection yet).

**3. Evaluate / inspect:**

```bash
python3 training/eval_checkpoint.py
python3 training/eval_progress.py
python3 training/generate_comparison_table.py   # regenerates training/results/comparison_table.md
```

## Scenarios

Defined in `src/scenarios/ScenarioRegistry.ts`. Currently registered:

| ID | Description |
|---|---|
| `academy_empty_goal` | 1 attacker, empty net — basic ball-approach/shooting drill |
| `academy_run_to_score` | 1 attacker vs. 1 defender + keeper |
| `academy_pass_and_shoot_with_keeper` | 2v2 passing + finishing drill |
| `academy_3_vs_1_with_keeper` | 3 attackers vs. 1 defender + keeper |
| `academy_3_vs_1_defender_2` / `_defender_3` | Harder variations, more defenders |
| `academy_3_vs_1_keeper_aggressive` | Variation with a more aggressive keeper |
| `academy_3_vs_1_shifted` / `_randomized` | Positional variations for generalization testing |
| `5_vs_5` | Small-sided full match |
| `11_vs_11` | Full-pitch full match |

Only the `academy_*` drills currently have any completed training checkpoints — see [Current Status](#current-status).

## Testing & Validation

```bash
npm test                  # test_scenarios.ts + test_determinism.ts
npm run test:scenarios
npm run test:determinism
npm run test:parity       # HTTP vs. WebSocket transport parity (Python)
npm run test:e2e          # end-to-end determinism (Python)
npm run test:multiagent   # multi-agent determinism (Python)
npm run test:pettingzoo   # PettingZoo wrapper contract test (Python)
npm run test:audit        # observation/action audit
npm run test:playability  # scenario playability verification
npm run test:validation   # rl_validation_suite.py
```

**Note:** `npm run lint` / `npm run build` only type-check `src/`. `training/*.ts` (including `bridge_server.ts`, `action_mapping.ts`) is executed directly via `tsx` and is not currently covered by `tsc`.

## Current Status

**Neural Policy Checkpoint Status:**
- The only MAPPO checkpoint currently in the repository (`mappo_academy_3_vs_1_with_keeper_trained.pt`) is a **49,920-step smoke test**, not a trained policy.
- This checkpoint was originally trained under the 115-dim observation schema and was **zero-padded** to 127 dims when the schema changed. The 12 role-feature weights are all zeros — the network has never learned role-aware behavior.
- The browser "Neural Policy" controller currently falls back to `RuleBasedAgent` because no valid trained policy exists for the 127-dim contract.
- A real training run (≥200,000 steps) under the 127-dim schema has not yet been completed.

**Implemented:**
- Deterministic, seeded (Mulberry32) TypeScript simulation shared by browser and headless paths
- 127-dim observation encoder with role information; 19-action discrete action space
- Offside, fouls/cards, goalkeeper saves, penalty/free-kick/corner/throw-in flow
- HTTP + binary WebSocket bridge with transport-parity tests
- Gymnasium (single-agent) and PettingZoo (multi-agent, left-team-only) environments
- Stable-Baselines3 PPO integration, plus custom IPPO and MAPPO implementations
- Scenario registry from 1v0 drills through 5v5 and 11v11
- Determinism, transport-parity, and observation/action audit test suites

**Not yet done — read before assuming a "trained agent" exists:**
- No full training run has completed end-to-end. `training/results/comparison_table.md` has no filled-in rows yet, and `training/models/` contains only smoke-test checkpoints plus one completed run on the simplest drill scenario (`academy_3_vs_1_with_keeper`). Nothing has been trained on `5_vs_5` or `11_vs_11`.
- Environment stepping is not parallelized (one environment instance per training process, each step a blocking round-trip to a single Node bridge process) — current throughput is well below what's typically needed for full-match RL training.
- Training is single-sided: only the left team is ever the learning agent; the opponent is always a fixed-difficulty `RuleBasedAgent`. There is no self-play or opponent-checkpoint pool yet.
- No curriculum scheduler — each training run targets one fixed scenario rather than progressing through the registry automatically.

GMN-Football-3 should currently be described as **an RL-ready football simulation and research platform**, not as a system that already plays professional-level football.

## Known Limitations

A few things worth knowing if you're extending this codebase:

- **The "Neural" controller is currently a rule-based fallback.** `App.tsx` routes `controller === 'neural'` to `RuleBasedAgent` when `TrainedPolicyAgent` has no valid checkpoint. The UI still labels the team "Neural." This will remain the case until a real 127-dim checkpoint is trained and exported.
- **ONNX export path is currently unused.** `training/export_onnx.py` and `public/models/mappo_policy.onnx` exist, and `onnxruntime-web` is a declared dependency, but `src/agents/TrainedPolicyAgent.ts` does not load the `.onnx` file — it runs a hand-written forward pass against weights baked into `src/agents/mappo_weights.ts`. Pick one path before extending the deployment pipeline further.
- **Determinism is not guaranteed for every agent.** `RuleBasedAgent` and tackle resolution (`PhysicsEngine.executeTackle`) correctly use the seeded RNG; `NeuralHeuristicAgent` and `HumanAgent` currently use `Math.random()` directly for some decisions, so browser-only opponent behavior isn't reproducible (this doesn't affect training determinism, since neither is wired into the bridge).
- **Two Python requirements files** (`requirements.txt` and `training/requirements.txt`) exist with different version bounds — `training/requirements.txt` is the one training scripts are actually validated against.
- **Contract constants are hand-duplicated** across `Contract.ts`, `gmn_gym.py`, and `gmn_pettingzoo.py`, reconciled only by a runtime health check rather than a single generated source.

## Contributing

See `CONTRIBUTING.md` for pull request and code review process. Note that file currently carries generic boilerplate (references to an unrelated CLA/project) and could use a project-specific pass.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
