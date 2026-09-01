# ⚽ GMN-Football-3

### Modern Browser-Native Football Simulation & Reinforcement Learning Platform

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Language: TypeScript](https://img.shields.io/badge/Language-TypeScript-3178C6.svg)](https://www.typescriptlang.org/)
[![Frontend: React](https://img.shields.io/badge/Frontend-React-61DAFB.svg)](https://react.dev/)
[![Build: Vite](https://img.shields.io/badge/Build-Vite-646CFF.svg)](https://vitejs.dev/)
[![RL: Gymnasium](https://img.shields.io/badge/RL-Gymnasium-5B8DEF.svg)](https://gymnasium.farama.org/)
[![RL: SB3 PPO](https://img.shields.io/badge/RL-Stable--Baselines3%20PPO-EA7C2B.svg)](https://stable-baselines3.readthedocs.io/)

> **GMN-Football-3 is a browser-native football simulation and reinforcement-learning research platform built around one authoritative TypeScript game engine.**
>
> The same simulation is designed to run interactively in the browser and headlessly for reinforcement-learning workloads, with deterministic state transitions, versioned environment contracts, scenario-driven training, and a modern Gymnasium → Stable-Baselines3 → PyTorch pipeline.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Core Design Principles](#core-design-principles)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Authoritative Simulation Engine](#authoritative-simulation-engine)
- [Agent Architecture](#agent-architecture)
- [Football Rules and Match State](#football-rules-and-match-state)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Environment Contract](#environment-contract)
- [RL Training Architecture](#rl-training-architecture)
- [Transport Layer](#transport-layer)
- [Determinism and Reproducibility](#determinism-and-reproducibility)
- [Scenarios and Curriculum](#scenarios-and-curriculum)
- [Events and Metrics](#events-and-metrics)
- [Validation and Testing](#validation-and-testing)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [Running the RL Environment](#running-the-rl-environment)
- [Project Development Workflow](#project-development-workflow)
- [Current Status](#current-status)
- [Roadmap](#roadmap)
- [Research Direction](#research-direction)
- [Design Constraints](#design-constraints)
- [Contributing](#contributing)
- [License](#license)

---

# Project Overview

GMN-Football-3 is intended to be more than a conventional football game.

It is a **shared simulation platform** where:

```text
                    GMN FOOTBALL WORLD
                           │
            ┌──────────────┴──────────────┐
            │                             │
         HUMAN                           AI
            │                             │
       Browser UI                    RL Environment
            │                             │
       React/Canvas                  Gymnasium
                                          │
                                  Stable-Baselines3
                                          │
                                      PyTorch
```

The central architectural decision is that the browser game and RL environment do **not** maintain separate football simulators.

The TypeScript `GameEngine` is the authoritative simulation. The browser presents it interactively, while a headless Node.js process exposes it to Python training code through the bridge layer.

---

# Objectives

## Primary Objective

Build a deterministic 2D football world that can function simultaneously as:

1. an interactive football game;
2. a headless simulation environment;
3. a reproducible reinforcement-learning benchmark;
4. a foundation for multi-agent football research.

## AI Objective

The long-term objective is to progress from low-level control to increasingly intelligent football behavior:

```text
Movement
   ↓
Ball Approach
   ↓
Possession
   ↓
Dribbling
   ↓
Passing
   ↓
Shooting
   ↓
Positioning
   ↓
Defending
   ↓
Coordination
   ↓
Self-Play
   ↓
11-vs-11 Football Intelligence
```

## Engineering Objective

Maintain a strict separation between:

```text
Simulation
Training Interface
Transport
Learning Algorithm
Presentation
```

so changes to one layer do not require rewriting the entire system.

---

# Core Design Principles

## 1. One Authoritative Simulation

The same `GameEngine.ts` implementation serves:

- browser gameplay;
- headless execution;
- scripted evaluation;
- RL training.

This prevents the RL system from learning in a different football world than the interactive application.

## 2. Deterministic Simulation

All stochastic simulation mechanics use seeded randomness.

The intended invariant is:

```text
same seed
+
same initial state
+
same action trajectory
=
same state / observation trajectory
```

The project uses a Mulberry32-based seeded RNG.

## 3. Versioned Environment Contracts

Current authoritative values:

```text
GMN_ENV_VERSION            = 3.0.0
OBSERVATION_SCHEMA_VERSION = simple115_v2
ACTION_SCHEMA_VERSION      = discrete19_v1

OBSERVATION_DIM            = 115
ACTION_SPACE_SIZE          = 19
```

These are centralized in `src/engine/Contract.ts`.

## 4. Clean Reset Semantics

A reset returns the exact initial observation at `t = 0`; reset must not advance the physics clock as an implicit first step.

## 5. Transport Independence

The authoritative engine is independent of the transport mechanism:

```text
TypeScript GameEngine
        │
        ├── HTTP bridge
        │
        └── Binary WebSocket bridge
```

The Gymnasium layer can select WebSocket or HTTP transport.

---

# Architecture

## High-Level Architecture

```text
                                 GMN-FOOTBALL-3
                                        │
                                        ▼
                         ┌──────────────────────────┐
                         │  TypeScript Game Engine   │
                         │      AUTHORITATIVE        │
                         └────────────┬─────────────┘
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       │                              │                              │
       ▼                              ▼                              ▼
 Browser / React                Headless Node                  Test / Benchmark
       │                              │                              │
   Canvas / UI                 Bridge Server                     Scripts
       │                              │
       │                   ┌──────────┴──────────┐
       │                   │                     │
       │                 HTTP                Binary WS
       │                   │                     │
       │                   └──────────┬──────────┘
       │                              ▼
       │                     Python Gymnasium
       │                              │
       │                              ▼
       │                 Stable-Baselines3 PPO
       │                              │
       │                           PyTorch
       │
       └────────────────────── same simulation ──────────────────────
```

## Internal Engine Architecture

```text
GameEngine
│
├── Players
├── Ball
├── Physics
├── Possession
├── Rules
├── Match State
├── Scenarios
├── Agents
├── Seeded RNG
└── Observation Generation
```

## RL Architecture

```text
PPO
 │
 ▼
GMNFootballEnv
 │
 ├── reset()
 ├── step(action)
 ├── observation_space
 └── action_space
 │
 ▼
Transport Adapter
 │
 ├── Binary WebSocket
 └── HTTP
 │
 ▼
Node Bridge
 │
 ▼
GameEngine
 │
 ▼
ObservationEncoder
 │
 ▼
115-float observation
```

---

# Technology Stack

| Layer | Technology |
|---|---|
| Primary language | TypeScript |
| UI | React 18 |
| Build/dev tooling | Vite |
| Styling | Tailwind / PostCSS / CSS |
| UI utilities | Lucide React, `clsx`, `tailwind-merge` |
| Data visualization | Recharts |
| Node bridge runtime | Node.js + `tsx` |
| HTTP transport | HTTP/REST |
| Binary transport | WebSocket |
| WebSocket library | `ws` |
| RL API | Gymnasium |
| RL algorithm | Stable-Baselines3 PPO |
| ML backend | PyTorch |
| Simulation RNG | Mulberry32 seeded PRNG |
| Contract layer | TypeScript `Contract.ts` |
| License | Apache-2.0 |

---

# Repository Structure

```text
GMN-Football-3/
│
├── src/
│   ├── agents/
│   │   ├── BaseAgent.ts
│   │   ├── HumanAgent.ts
│   │   ├── NeuralHeuristicAgent.ts
│   │   ├── RuleBasedAgent.ts
│   │   └── ScriptedScenarioAgent.ts
│   │
│   ├── components/
│   ├── engine/
│   │   ├── Contract.ts
│   │   ├── GameEngine.ts
│   │   ├── ObservationEncoder.ts
│   │   ├── Physics.ts
│   │   ├── Rules.ts
│   │   ├── SeededRNG.ts
│   │   └── Vector.ts
│   │
│   ├── scenarios/
│   ├── services/
│   ├── types/
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
│
├── training/
│   ├── action_mapping.ts
│   ├── audit_observations_and_actions.ts
│   ├── benchmark.ts
│   ├── benchmark_bridge.py
│   ├── benchmark_bridge_ws.py
│   ├── bridge_server.ts
│   ├── eval_checkpoint.py
│   ├── gmn_gym.py
│   ├── rl_validation_suite.py
│   ├── scripted_eval.ts
│   ├── stage2_audit_and_baseline.ts
│   ├── stage2_full_validation.py
│   ├── test_determinism.ts
│   ├── test_e2e_determinism.py
│   ├── test_env.py
│   ├── test_scenarios.ts
│   ├── test_transport_parity.py
│   ├── train_ppo.py
│   ├── train_stage2_ppo.py
│   └── verify_scenario_playability.ts
│
├── .env.example
├── CONTRIBUTING.md
├── LICENSE
├── metadata.json
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
└── postcss.config.js
```

---

# Authoritative Simulation Engine

The `src/engine/` directory is the core of GMN-Football-3.

## `GameEngine.ts`

Central simulation orchestrator.

Responsibilities include:

- player state;
- ball state;
- physics updates;
- possession;
- match state;
- actions;
- scoring;
- scenarios;
- rule transitions;
- observations;
- deterministic state progression.

## `Physics.ts`

Handles movement and physical interactions.

## `Rules.ts`

Encodes football and match-level rules.

## `ObservationEncoder.ts`

Converts authoritative simulation state into the canonical RL observation vector.

## `SeededRNG.ts`

Provides deterministic pseudo-randomness for simulation events.

## `Vector.ts`

Shared vector mathematics.

## `Contract.ts`

The authoritative interface specification for:

- environment version;
- observation schema;
- action schema;
- observation dimension;
- action-space size;
- event-code mapping;
- environment metadata.

---

# Agent Architecture

GMN separates **the football simulation** from **the controller controlling an actor**.

Current controller types include:

```text
BaseAgent
├── HumanAgent
├── RuleBasedAgent
├── NeuralHeuristicAgent
└── ScriptedScenarioAgent
```

Conceptually:

```text
                    GameEngine
                        │
                ┌───────┴───────┐
                │               │
              Player          Player
                │               │
             Controller      Controller
                │               │
        ┌───────┼────────┐      │
        │       │        │      │
      Human   Rule     Neural   PPO
```

The current bridge-driven single-agent configuration gives the RL policy the controlled player while automated agents drive the remaining players.

This design provides a clean path toward future shared-policy and multi-agent control.

---

# Football Rules and Match State

GMN-Football-3 increasingly models football as a **stateful match simulation**, rather than only a movement sandbox.

## Match modes

```text
Normal
KickOff
GoalKick
FreeKick
Corner
ThrowIn
Penalty
```

## Event categories

```text
goal
shot
shot_saved
shot_missed
pass
interception
tackle
foul
kickoff
out_of_bounds
scenario_complete
scenario_failed
```

The engine has also been expanded with football-specific logic including:

- offside;
- tackle fouls;
- disciplinary state;
- yellow/red card handling;
- goalkeeper saves;
- shot placement;
- shot tracking;
- shooting metrics;
- match-state transitions.

This moves the simulation toward a rule-bearing football state machine.

---

# Observation Space

## Canonical Shape

```text
(127,)
dtype = float32
```

> **Breaking Change (115 → 127 Migration):**  
> In GMN v3.1.0 (`simple115_v3_role`), the observation vector was extended by 12 floats (`115..126`) representing the agent's self-role one-hot vector across `ROLE_VOCABULARY` (GK, CB, LB, RB, CDM, CM, LM, RM, LW, RW, CAM, ST).  
> Checkpoints trained on the legacy 115-dim observation space are incompatible and will raise a descriptive runtime error if loaded without re-training.  
> **Commands to re-train and re-export:**
> ```bash
> # Start bridge server
> npx tsx training/bridge_server.ts
> # Run MAPPO training (Python)
> python3 training/train_mappo.py --scenario academy_3_vs_1_with_keeper --timesteps 50000
> # Export ONNX & browser weights
> python3 training/export_onnx.py --checkpoint training/models/mappo_academy_3_vs_1_with_keeper_trained.pt
> ```

The Gymnasium environment exposes:

```python
spaces.Box(
    low=-5.0,
    high=5.0,
    shape=(127,),
    dtype=np.float32,
)
```

## Layout

| Offset | Length | Description |
|---|---:|---|
| `0..21` | 22 | Left-team player positions `(x,y)` |
| `22..43` | 22 | Left-team player velocities `(dx,dy)` |
| `44..65` | 22 | Right-team player positions `(x,y)` |
| `66..87` | 22 | Right-team player velocities `(dx,dy)` |
| `88..90` | 3 | Ball position `(x,y,z)` |
| `91..93` | 3 | Ball velocity `(dx,dy,dz)` |
| `94..96` | 3 | Possession one-hot |
| `97..107` | 11 | Viewpoint / Active controlled player one-hot |
| `108..114` | 7 | Match-mode one-hot |
| `115..126` | 12 | Agent self-role one-hot (`ROLE_VOCABULARY`) |

Unused player slots are represented according to the current observation encoder.

---

# Action Space

## Canonical Shape

```text
Discrete(19)
```

| ID | Action | Type |
|---:|---|---|
| 0 | `action_idle` | No-op |
| 1 | `action_left` | Directional |
| 2 | `action_top_left` | Directional |
| 3 | `action_top` | Directional |
| 4 | `action_top_right` | Directional |
| 5 | `action_right` | Directional |
| 6 | `action_bottom_right` | Directional |
| 7 | `action_bottom` | Directional |
| 8 | `action_bottom_left` | Directional |
| 9 | `action_long_pass` | Pass |
| 10 | `action_high_pass` | Pass |
| 11 | `action_short_pass` | Pass |
| 12 | `action_shot` | Shot |
| 13 | `action_sprint` | Movement |
| 14 | `action_release_direction` | Movement |
| 15 | `action_release_sprint` | Movement |
| 16 | `action_sliding` | Tackle |
| 17 | `action_dribble` | Ball control |
| 18 | `action_release_dribble` | Ball control |

The interface is intentionally semantic and follows a GRF-style football action vocabulary while being executed by GMN's own engine.

---

# Environment Contract

The current authoritative contract is:

```text
GMN_ENV_VERSION            = 3.1.0
OBSERVATION_SCHEMA_VERSION = simple115_v3_role
ACTION_SCHEMA_VERSION      = discrete19_v1

OBSERVATION_DIM            = 127
ACTION_SPACE_SIZE          = 19
```

`Contract.ts` is the single source of truth for the environment and neural interface.

## Contract consumers

```text
Contract.ts
    │
    ├── GameEngine
    ├── ObservationEncoder
    ├── bridge_server.ts
    ├── action_mapping.ts
    ├── gmn_gym.py
    └── training/evaluation
```

## Versioning policy

Any observation or action semantic change should:

1. update the schema version;
2. update TypeScript validation;
3. update Python validation;
4. rerun deterministic tests;
5. rerun audits;
6. evaluate checkpoint compatibility.

---

# RL Training Architecture

The current learning path is:

```text
                PPO
                 │
                 ▼
        Stable-Baselines3
                 │
              PyTorch
                 │
                 ▼
             Gymnasium
                 │
                 ▼
         GMNFootballEnv
                 │
       ┌─────────┴─────────┐
       │                   │
   Binary WS              HTTP
       │                   │
       └─────────┬─────────┘
                 ▼
            Node Bridge
                 │
                 ▼
            GameEngine
                 │
                 ▼
         ObservationEncoder
```

`training/gmn_gym.py` provides the standard Gymnasium environment lifecycle and validates the 127-dimensional observation and 19-action contract.

## RL Modes

- **Single-Agent PPO**: `training/train_ppo.py` for single controlled agent curriculum scenarios.
- **Multi-Agent MAPPO**: `training/train_mappo.py` using a Deep Sets centralized critic with permutation-invariant dual aggregation (`[mean_pool, max_pool]`).
- **IPPO (Deprecated)**: IPPO is deprecated and superseded by MAPPO; root cause analysis is documented in `training/ippo_credit_assignment_report.md`.

### Regenerating Browser Neural Policy Weights (Requires Local Run)

The environment's observation contract migrated from 115 dimensions to 127 dimensions to support explicit per-agent role one-hot channels. Existing checkpoints in `training/models/` (`mappo_academy_3_vs_1_with_keeper_trained.pt`) were trained under the legacy 115-dim contract.

To regenerate weights compatible with the current browser runtime, execute the following commands in your local training environment:

```bash
# 1. Retrain to produce a 127-dim checkpoint (obs_dim now 127 after the role-channel migration)
python3 training/train_mappo.py --scenario academy_3_vs_1_with_keeper --timesteps 200000

# 2. Re-export both ONNX and the browser weights from the NEW checkpoint
python3 training/export_onnx.py --checkpoint training/models/mappo_academy_3_vs_1_with_keeper_trained.pt
```

> **Note**: Until Step 1 and 2 are executed in a local training run, `src/agents/mappo_weights.ts` contains legacy 115-dim weights. Runtime shape assertions in `TrainedPolicyAgent.ts` and `export_onnx.py` will actively block execution and fall back safely to Tactical Rule AI to prevent silent NaN logits.
>
> **Team B / Right-Side Neural Mirroring**: Neural policies are currently trained strictly for Left-side attackers targeting the Right goal. Right-team neural play (mirroring coordinates/actions) is planned future work.


---

# Transport Layer

## HTTP

```text
Python
  ↓
HTTP
  ↓
Node bridge
  ↓
GameEngine
```

The repository currently documents approximately **350–500 steps/sec** for the Python/Node HTTP roundtrip.

## Binary WebSocket

The current system also includes binary WebSocket transport:

```text
training/bridge_server.ts
training/benchmark_bridge_ws.py
training/test_transport_parity.py
```

The Gymnasium wrapper can use:

```text
ws://127.0.0.1:5050
```

instead of HTTP.

## Transport parity

The project verifies that transport changes do not change environment semantics:

```text
same seed
+
same action trajectory
+
HTTP
        ↕
Binary WebSocket

Expected:
same observations
same rewards
same termination
same match result
```

This enables transport optimization without changing the authoritative simulation.

---

# Determinism and Reproducibility

## Target invariant

```text
Seed = S
Initial State = I
Actions = A

          ↓

Trajectory T
```

Repeated under the same engine version must produce the same trajectory.

## Deterministic components

- seeded RNG;
- scenario initialization;
- reset semantics;
- action mapping;
- engine state updates;
- transport parity.

## Benefits

Deterministic execution enables:

- RL experiment reproduction;
- regression testing;
- checkpoint comparison;
- bug reproduction;
- replay reconstruction;
- transport verification.

---

# Scenarios and Curriculum

The scenario system allows task-specific training and validation.

The repository currently validates **six academy/match scenarios**.

A representative curriculum:

```text
Basic Control
      ↓
Run to Goal
      ↓
Keeper / Finish
      ↓
Pass + Shot
      ↓
Small-Sided Decision Making
      ↓
Multi-Agent Coordination
      ↓
Full Match
```

Each scenario can define:

- initial player placement;
- ball state;
- objective;
- completion condition;
- failure condition;
- reward context;
- evaluation context.

Scenario testing covers setup, spawning, observation correctness, and playability.

---

# Events and Metrics

The environment exposes a shared event-code vocabulary:

```text
goal
shot
shot_saved
shot_missed
pass
interception
tackle
foul
kickoff
out_of_bounds
scenario_complete
scenario_failed
```

Events create a common foundation for:

```text
              Game Events
                   │
          ┌────────┼────────┐
          │        │        │
       Metrics   Replay     RL
          │        │        │
       Analytics Debugging Reward Design
```

This allows football analytics and reward design to evolve without coupling them directly to raw vector indices.

---

# Validation and Testing

## TypeScript

```bash
npm run lint
npm run test
npm run test:scenarios
npm run test:determinism
npm run test:audit
npm run test:scripted
npm run test:playability
```

## Python

```bash
npm run test:env
npm run test:e2e
npm run test:parity
npm run test:validation
```

## PPO smoke test

```bash
npm run test:ppo
```

or:

```bash
python3 training/train_ppo.py 1000
```

## Recommended validation order

```text
Engine correctness
      ↓
Contract correctness
      ↓
Determinism
      ↓
Transport parity
      ↓
Gymnasium compliance
      ↓
Baseline evaluation
      ↓
PPO
```

---

# Performance

The current repository reports approximately:

| Component | Measured throughput |
|---|---:|
| Raw TypeScript engine | ~33,000 steps/sec |
| Python ↔ Node HTTP bridge | ~350–500 steps/sec |

The raw simulation is substantially faster than the current HTTP transport path.

The WebSocket path is the current transport optimization direction. A definitive current WebSocket speedup should be measured with:

```bash
npm run benchmark:ws
```

Rather than hard-coding an unverified throughput claim.

## Scaling priority

```text
Correctness
   ↓
Learning validity
   ↓
Transport optimization
   ↓
Batching / vectorization
   ↓
Multi-agent scale
```

---

# Quick Start

## Install

```bash
npm install
```

## Start the browser application

```bash
npm run dev
```

Open:

```text
http://localhost:3000
```

## Type check

```bash
npm run lint
```

## Production build

```bash
npm run build
```

---

# Running the RL Environment

## Start bridge

```bash
npm run bridge
```

Default bridge address:

```text
127.0.0.1:5050
```

It can be overridden using:

```text
GMN_BRIDGE_PORT
```

## Validate Gymnasium

```bash
python3 training/test_env.py
```

## Validate end-to-end determinism

```bash
python3 training/test_e2e_determinism.py
```

## Validate transport parity

```bash
npm run test:parity
```

## Run PPO smoke test

```bash
python3 training/train_ppo.py 1000
```

---

# Project Development Workflow

Recommended loop:

```text
1. Change simulation
        ↓
2. npm run lint
        ↓
3. Scenario validation
        ↓
4. Determinism tests
        ↓
5. Observation/action audit
        ↓
6. Transport parity
        ↓
7. Gymnasium validation
        ↓
8. Scripted baseline
        ↓
9. PPO smoke test
        ↓
10. Benchmark
```

## Rules for environment changes

A change to:

- physics;
- ball handling;
- possession;
- fouls;
- shooting;
- goalkeeping;
- offside;
- restart logic;
- player movement;

is potentially an **RL environment change**.

Those changes should therefore be accompanied by deterministic and environment-level regression tests.

---

# Current Status

## Implemented

```text
✅ Browser-native football application
✅ Custom TypeScript simulation engine
✅ Headless Node execution
✅ React presentation layer
✅ Agent abstraction
✅ Deterministic seeded RNG
✅ Strict environment contracts
✅ 115-dimensional observation space
✅ 19-action discrete action space
✅ Scenario registry
✅ Six scenario validation
✅ Event-code system
✅ Football rule/state expansion
✅ Offside
✅ Fouls / disciplinary mechanics
✅ Goalkeeper save logic
✅ Improved shooting / targeting
✅ Gymnasium environment
✅ Stable-Baselines3 PPO path
✅ HTTP bridge
✅ Binary WebSocket bridge
✅ Transport parity testing
✅ Determinism testing
✅ Observation/action audits
✅ Scripted evaluation
✅ Benchmark tooling
✅ Stage-2 RL tooling
```

## Active research areas

```text
⏳ Stronger PPO learning results
⏳ Shared-policy MARL
⏳ Centralized multi-agent critic
⏳ Self-play
⏳ Historical-opponent league
⏳ Full 11-vs-11 learned teams
⏳ Emergent tactical behavior
```

GMN-Football-3 should currently be described as an **RL-ready football simulation and research platform**, not as a finished autonomous 11-vs-11 football intelligence system.

---

# Roadmap

## Phase 1 — Simulation Foundation

```text
Simulation
Determinism
Contracts
Rules
Scenarios
Validation
```

**Status: Substantially implemented**

## Phase 2 — Single-Agent RL

```text
Gymnasium
PPO
Scenario curriculum
Checkpoint evaluation
```

**Status: Active**

## Phase 3 — Multi-Agent Learning

```text
Shared policy
Multiple learned players
Centralized critic
```

**Status: Research target**

## Phase 4 — Self-Play

```text
Current policy
       ↕
Opponent policy
       ↕
Historical policies
```

**Status: Research target**

## Phase 5 — Team Intelligence

Evaluate:

```text
spacing
support
positioning
pressing
passing lanes
defensive shape
attacking coordination
```

## Phase 6 — 11-vs-11

Target:

```text
11 learned agents
        vs
11 learned agents
```

with:

- self-play;
- historical opponents;
- reproducible evaluation;
- tactical metrics;
- long-horizon behavioral analysis.

---

# Research Direction

GMN-Football-3 sits at the intersection of:

```text
Football Simulation
        +
Reinforcement Learning
        +
Multi-Agent Systems
        +
Behavioral Cloning
        +
Self-Play
```

A key research direction is to combine proven multi-agent football techniques—shared policies, centralized critics, attention, opponent stabilization, and self-play—with GMN's deterministic simulation and modern Gymnasium/SB3 architecture.

The system should evolve from:

```text
single player
     ↓
small-sided learning
     ↓
shared multi-agent policy
     ↓
team coordination
     ↓
self-play
     ↓
11-vs-11
```

The ultimate objective is not merely scoring goals. It is learning behaviors such as:

```text
movement
possession
passing
shooting
positioning
defending
support
spacing
pressing
team coordination
tactical adaptation
```

without hard-coding those behaviors into the agent policy.

---

# Design Constraints

## Simulation Is Authoritative

Do not implement duplicate football physics in Python.

## Contracts Are Explicit

Do not silently change observation or action semantics.

## Determinism Is Required

Randomness must be seeded and reproducible.

## Transport Is Replaceable

HTTP/WebSocket are interfaces to the simulation, not alternate simulation implementations.

## Training Must Be Measurable

A policy is not successful merely because:

```text
loss decreases
reward is non-zero
PPO finishes training
```

Success must be demonstrated against explicit baselines and reproducible evaluation.

## Research Claims Must Match Evidence

The project has a strong simulation and RL infrastructure foundation. Full multi-agent football intelligence remains an active research objective.

---

# Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

Before submitting simulation changes, run:

```bash
npm run lint
npm run test
npm run test:determinism
npm run test:audit
npm run test:parity
```

For RL environment changes, also run:

```bash
python3 training/test_env.py
python3 training/test_e2e_determinism.py
```

---

# License

GMN-Football-3 is released under the **Apache License 2.0**.

See [`LICENSE`](LICENSE).

---

## Project Links

- **Repository:** https://github.com/natnakem-cyb/GMN-Football-3
- **Wiki:** https://github.com/natnakem-cyb/GMN-Football-3/wiki
- **License:** [`LICENSE`](LICENSE)
- **Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## Architecture at a Glance

```text
                                  GMN FOOTBALL
                                       │
                         ┌─────────────┴─────────────┐
                         │                           │
                    Interactive                  Training
                         │                           │
                    React/Canvas                Gymnasium
                         │                           │
                         ▼                    ┌──────┴──────┐
                 TypeScript Engine            │             │
                    AUTHORITATIVE           HTTP          WS
                         │                    │             │
          ┌──────────────┼──────────────┐     └──────┬──────┘
          │              │              │            │
       Physics         Rules         Agents           ▼
          │              │              │       Node Bridge
       Ball/Players     Match State   Human           │
       Possession       Offside       RuleBased       ▼
       Shooting        Fouls/Cards    Neural      GameEngine
       Goalkeeping     Restarts       PPO             │
          │              │              │             ▼
          └──────────────┼──────────────┘       Observation
                         │                           │
                    Event / State              115 floats
                         │                           │
                         └──────────────┬────────────┘
                                        ▼
                                 SB3 PPO / PyTorch
                                        │
                                        ▼
                            Future Multi-Agent RL
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                           ▼
                       Self-Play                  11v11
```

> **GMN-Football-3 is being built as a deterministic football world first, an RL environment second, and eventually a multi-agent football intelligence platform.**
