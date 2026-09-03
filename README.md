⚽ GMN-Football-3
=================

A Browser-Native Football Simulation & Reinforcement-Learning Research Platform.

One authoritative TypeScript game engine drives both an interactive browser match and a headless Python RL training pipeline (Gymnasium / PettingZoo → Stable-Baselines3 / custom PPO, IPPO, MAPPO), so an agent is always trained against the exact same physics and rules a human plays against.

**License:** Apache-2.0 · **Language:** TypeScript · **Frontend:** React 18 · **Build:** Vite · **RL:** Gymnasium + PettingZoo · **RL algorithms:** SB3 PPO / custom IPPO / custom MAPPO

**Status:** RL-ready simulation and research platform, with a real deployed policy (MAPPO, ONNX) driving in-browser gameplay for one scenario. There is not yet a policy trained to play a full match — see [Current Status](#current-status) for exactly what has and hasn't been trained so far.


Table of Contents
-----------------
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
- [Corrections vs. Prior Documentation](#corrections-vs-prior-documentation)
- [Contributing](#contributing)
- [License](#license)

Overview
--------
GMN-Football-3 is built around a single design decision: the browser game and the RL environment do not maintain separate simulators. The TypeScript `GameEngine` in `src/engine/` is authoritative. The browser renders and controls it interactively; a headless Node.js bridge exposes the same engine to Python training code over HTTP or a binary WebSocket protocol.

```
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

This means: no separate "training physics" that quietly diverges from what a human sees, and no re-implementation risk between the game and the research environment. The same design also means a trained policy can be exported and loaded straight into the browser — see [Technology Stack](#technology-stack).

Architecture
------------
```
                      TypeScript GameEngine (authoritative)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
  Browser / React             Headless Node bridge         Scripts (tests,
  (src/App.tsx)               (training/bridge_server.ts)  benchmarks, audits)
        │                           │
        │                ┌───────────┴───────────┐
        │                │                       │
        │            HTTP bridge            Binary WebSocket
        │                │                       │
        │                └───────────┬───────────┘
        │                            ▼
        │             Python: Gymnasium env (gmn_gym.py)
        │             Python: PettingZoo env (gmn_pettingzoo.py)
        │                            │
        │             Stable-Baselines3 PPO / custom IPPO / custom MAPPO
        │                            │
        │                         PyTorch
        │                            │
        │                    export_onnx.py
        ▼                            ▼
  TrainedPolicyAgent.ts ◄─── public/models/mappo_policy.onnx
  (onnxruntime-web)
```

Inside the engine itself:

```
GameEngine
├── Players & Ball state
├── Physics.ts            — movement, kicking, tackling, ball flight
├── Rules.ts               — pitch geometry, formations, offside line
├── SeededRNG.ts            — Mulberry32 deterministic PRNG
├── ObservationEncoder.ts  — RL observation vector + reward shaping
└── Contract.ts            — versioned observation/action schema (single source of truth on the TS side)
```

Environment Contract
---------------------
Defined in `src/engine/Contract.ts` — treat this file as authoritative if anything below drifts out of date:

| Constant | Value |
|---|---|
| `GMN_ENV_VERSION` | `3.1.0` |
| `OBSERVATION_SCHEMA_VERSION` | `simple115_v3_role` |
| `ACTION_SCHEMA_VERSION` | `discrete19_v1` |
| `BASE_OBSERVATION_DIM` | 115 (Google Research Football–style SMM/feature vector) |
| `ROLE_DIM` | 12 (one-hot over `ROLE_VOCABULARY`) |
| `OBSERVATION_DIM` | 127 (`BASE_OBSERVATION_DIM` + `ROLE_DIM`) |
| `ACTION_SPACE_SIZE` | 19 (discrete) |

The 127-float observation layout (see `ObservationEncoder.ts` for the exact offsets): left-team positions (22) → left-team velocities (22) → right-team positions (22) → right-team velocities (22) → ball position (3) → ball velocity (3) → ball ownership one-hot (3) → active-player one-hot (11) → game-mode one-hot (7) → agent role one-hot (12).

The 19 discrete actions cover 8-directional movement, idle, short/long/high pass, shot, sprint (+release), dribble (+release), release-direction, and slide tackle — see `training/action_mapping.ts` for the canonical mapping.

Python and TypeScript each declare their own copies of these constants (`gmn_gym.py`, `gmn_pettingzoo.py`, `Contract.ts`); the bridge's `/health` endpoint cross-checks `observation_dim`/`action_space_size` at connection time and raises if they disagree. `scripts/sync_contracts.ts` generates the Python copies from `Contract.ts` as the source of truth.

Technology Stack
-----------------
| Layer | Technology |
|---|---|
| Simulation & game logic | TypeScript |
| UI | React 18 + Vite |
| Styling | Tailwind CSS / PostCSS |
| Charts | Recharts |
| Node bridge runtime | Node.js via `tsx` |
| Transport | HTTP (REST) and binary WebSocket (`ws`) |
| RL API (single-agent) | Gymnasium |
| RL API (multi-agent) | PettingZoo, with **SuperSuit vectorization actually wired up for IPPO training** (`train_ippo.py` builds a real multi-sub-environment `SuperSuit` vec-env; PPO and MAPPO training still run a single environment instance per process) |
| RL algorithms | Stable-Baselines3 PPO; custom IPPO and MAPPO implementations |
| ML backend | PyTorch |
| Browser inference | **`onnxruntime-web`, loading `public/models/mappo_policy.onnx`.** `src/agents/TrainedPolicyAgent.ts` runs real ONNX inference for the in-browser "Neural" controller. The older hand-rolled MLP path (`src/agents/mappo_weights.ts`) is explicitly `@deprecated` in the file itself and retained only for offline reference / test parity, not used in the live decision path. |
| Deterministic RNG | Mulberry32 (`SeededRNG.ts`) |
| Optional AI match commentary | `@google/genai` (Gemini API, via `src/services/geminiService.ts`) |
| License | Apache-2.0 |

Repository Structure
---------------------
```
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
│   │   ├── TrainedPolicyAgent.ts   # loads the ONNX checkpoint; this is what the "Neural" controller actually runs
│   │   └── mappo_weights.ts        # @deprecated — offline reference only, not used at runtime
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
│   ├── train_ippo.py        # uses SuperSuit for real vectorized rollout collection
│   ├── eval_ippo_baseline.py
│   ├── train_mappo.py / mappo_networks.py / mappo_rollout.py / mappo_update.py / eval_mappo.py
│   ├── export_onnx.py / onnx_proto_builder.py
│   ├── episode_recorder.py / trace_to_frames.py / binary_event_decoder.py
│   ├── eval_checkpoint.py / eval_progress.py / eval_generalization.py / generate_comparison_table.py
│   ├── rl_validation_suite.py
│   ├── benchmark.ts / benchmark_bridge.py / benchmark_bridge_ws.py
│   ├── test_*.ts / test_*.py   # determinism, transport parity, scenario, multi-agent, critic-scaling, role-differentiation tests
│   ├── audit_observations_and_actions.ts / stage2_audit_and_baseline.ts / stage2_full_validation.py
│   ├── verify_scenario_playability.ts / scripted_eval.ts
│   ├── validate_learned_policy.ts/.py
│   ├── modular_encoder.ts / modular_networks.py
│   ├── models/              # Checkpoints (currently: smoke tests for PPO/IPPO/MAPPO, plus one completed drill-scenario training run each for IPPO and MAPPO — see Current Status)
│   └── results/             # win_rate_progress.csv, generalization.csv, comparison_table.md/.html
│   # Note: this directory has grown beyond what's listed above — check `training/` directly for the current full set of scripts before assuming this list is exhaustive.
│
├── public/models/           # mappo_policy.onnx — actively loaded by the browser app (see Technology Stack)
├── requirements.txt          # Python deps (root)
├── training/requirements.txt # Python deps (training-pinned, includes SuperSuit — prefer this one for training)
├── package.json
├── tsconfig.json             # TypeScript config includes both `src/` and `training/` for strict typechecking
├── vite.config.ts / tailwind.config.js / postcss.config.js
├── .env.example              # GEMINI_API_KEY (optional, for AI match commentary)
├── LICENSE (Apache-2.0)
└── CONTRIBUTING.md            # currently generic boilerplate referencing an unrelated project — see Contributing
```

Quick Start (Browser App)
--------------------------
Requires Node.js 18+.

```
npm install
npm run dev        # http://localhost:3000
```

Other useful scripts:

```
npm run build       # tsc (src/ and training/) + vite build
npm run lint         # tsc --noEmit
npm run preview      # serve the production build
```

Running the RL Training Pipeline
----------------------------------
Requires Python 3.10+ and Node.js (the bridge server runs via `npx tsx`).

```
pip install -r training/requirements.txt
```

**1. Start the bridge** (optional — training scripts will auto-launch it if it isn't already running):

```
npm run bridge       # tsx training/bridge_server.ts
```

**2. Train.** Scripts wired into `package.json`:

```
npm run test:ppo             # Stable-Baselines3 PPO, short smoke run (1,000 steps)
npm run test:ippo            # Custom IPPO, short smoke run (3,072 steps), vectorized via SuperSuit
npm run test:ippo:train      # Custom IPPO, longer run (200,000 steps)
npm run test:ippo:eval       # Evaluate/compare an IPPO checkpoint
```

MAPPO has no `package.json` shortcut yet — invoke it directly:

```
python3 training/train_mappo.py
python3 training/eval_mappo.py
```

Both `train_ppo.py` and the custom trainers accept a `--scenario` (or positional step-count) argument — see each script's `argparse` setup for the current options. **PPO and MAPPO still run against a single environment instance per process** (no vectorized rollout collection); **IPPO is the exception** — `train_ippo.py` builds a real `SuperSuit` vector environment with multiple sub-environments sharing one policy.

**3. Evaluate / inspect:**

```
python3 training/eval_checkpoint.py
python3 training/eval_progress.py
python3 training/generate_comparison_table.py   # regenerates training/results/comparison_table.md
```

Additional evaluation scripts also exist in `package.json` beyond the ones above — `eval:baselines`, `eval:generalization`, `eval:opponents`, `eval:ablations`, `test:browser-parity`, `validate:policy` — check `package.json`'s `scripts` block directly for the current full set.

Scenarios
---------
Defined in `src/scenarios/ScenarioRegistry.ts`. Currently registered (11 total):

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

Only the `academy_3_vs_1_with_keeper` drill currently has completed (non-smoke) training checkpoints — see [Current Status](#current-status).

Testing & Validation
----------------------
```
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

`npm run lint` (`tsc --noEmit`) and `npm run build` type-check both `src/` and `training/`, per `tsconfig.json`.

Current Status
--------------
**Neural Policy Checkpoint Status:**
- The MAPPO checkpoint the browser actually loads (via `public/models/mappo_policy.onnx`, exported from `training/models/mappo_academy_3_vs_1_with_keeper_trained.pt`) is a 200,000-step run on `academy_3_vs_1_with_keeper` under the 127-dim role-aware contract.
- This is the checkpoint actively used by the in-browser "Neural" controller (`TrainedPolicyAgent`) — it is **not** a fallback to `RuleBasedAgent`. It has not been trained on any scenario beyond `academy_3_vs_1_with_keeper`.
- Smoke-test-only checkpoints also exist for PPO (`academy_empty_goal`) and IPPO (`academy_3_vs_1_with_keeper`), alongside a completed (non-smoke) IPPO run on the same scenario.
- Nothing has been trained on `5_vs_5` or `11_vs_11`.

**Implemented:**
- Deterministic, seeded (Mulberry32) TypeScript simulation shared by browser and headless paths
- 127-dim observation encoder with role information; 19-action discrete action space
- Offside, fouls/cards, goalkeeper saves, penalty/free-kick/corner/throw-in flow
- HTTP + binary WebSocket bridge with transport-parity tests
- Gymnasium (single-agent) and PettingZoo (multi-agent, left-team-only) environments
- Stable-Baselines3 PPO integration, plus custom IPPO and MAPPO implementations
- **Real vectorized rollout collection for IPPO via SuperSuit** (PPO and MAPPO are not yet vectorized)
- ONNX export and browser-side ONNX inference for the trained MAPPO policy, actively used by the live match UI
- Scenario registry from 1v0 drills through 5v5 and 11v11
- Determinism, transport-parity, and observation/action audit test suites

**Not yet done — read before assuming a fully "trained agent" exists:**
- No policy has been trained end-to-end on anything beyond the `academy_3_vs_1_with_keeper` drill. `training/results/comparison_table.md` has no filled-in rows yet.
- PPO and MAPPO training still run a single environment instance per process, each step a blocking round-trip to a single Node bridge process — throughput for those two algorithms is well below what's typically needed for full-match RL training. (IPPO's SuperSuit vectorization is a partial exception — verify whether its vectorized sub-environments still each open their own bridge connection before assuming this fully removes the bottleneck.)
- Training is single-sided: only the left team is ever the learning agent; the opponent is always a fixed-difficulty `RuleBasedAgent`. There is no self-play or opponent-checkpoint pool wired into training yet, though references to self-play exist in `training/modular_networks.py` — its current functional status should be verified rather than assumed.
- No confirmed automatic curriculum scheduler across the scenario registry, though `training/train_stage2_ppo.py` and `src/components/ScenarioSelector.tsx` reference curriculum-related concepts — verify their actual behavior before relying on them.
- No spatial/SMM/CNN observation path — the contract is a flat 127-float vector, despite "SMM"/"CNN" appearing as comparative references in a few files.

GMN-Football-3 should currently be described as an RL-ready football simulation and research platform with one real deployed policy for one drill scenario — not as a system that already plays professional-level football.

Known Limitations
-------------------
- Determinism is not guaranteed for every agent. `RuleBasedAgent` and tackle resolution (`PhysicsEngine.executeTackle`) correctly use the seeded RNG; `NeuralHeuristicAgent` and `HumanAgent` currently use `Math.random()` directly for some decisions, so browser-only opponent behavior isn't reproducible (this doesn't affect training determinism, since neither is wired into the bridge).
- Two Python requirements files (`requirements.txt` and `training/requirements.txt`) exist with different version bounds — `training/requirements.txt` is the one training scripts are actually validated against.
- Contract constants are hand-duplicated across `Contract.ts`, `gmn_gym.py`, and `gmn_pettingzoo.py`, reconciled by a runtime health check; `scripts/sync_contracts.ts` generates the Python side from `Contract.ts`.
- `training/` contains substantially more scripts (duplicate `.ts`/`.py` pairs for several eval and validation tasks, a `modular_encoder`/`modular_networks` pair, stage-2 audit/validation scripts) than are documented in this README's Repository Structure section — treat that section as a guide to the most important files, not an exhaustive list.

Corrections vs. Prior Documentation
--------------------------------------
Earlier project documentation (including a previous version of this README and two independent technical-analysis documents) stated the following, which this version corrects after direct verification against the current code:

| Prior claim | Verified current state |
|---|---|
| "The browser 'Neural Policy' controller currently falls back to `RuleBasedAgent`" | False. `App.tsx` routes the `neural` controller directly to `TrainedPolicyAgent`, which runs real ONNX inference. |
| "ONNX export path is currently unused... runs a hand-written forward pass against `mappo_weights.ts`" | False. `TrainedPolicyAgent.create()` loads `public/models/mappo_policy.onnx` via `onnxruntime-web`; `mappo_weights.ts` is explicitly `@deprecated` and used only for offline/test-parity reference. |
| "Environment stepping is not parallelized... one environment instance per process" (stated as a blanket fact) | Partially false. `train_ippo.py` uses `SuperSuit` to build a real multi-sub-environment vectorized environment. PPO and MAPPO remain single-instance. |

If you're extending this project's documentation further, verify claims like these against the actual code rather than carrying them forward — this codebase has previously accumulated stale claims about its own capabilities across multiple documents.

Contributing
------------
See `CONTRIBUTING.md` for pull request and code review process. That file currently carries generic boilerplate (referencing an unrelated TensorFlow/Tensor2Tensor project's CLA and process) and needs a project-specific rewrite.

License
-------
Apache License 2.0 — see `LICENSE`.
