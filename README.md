# ⚽ GMN-Football

### A Modern Web-Based Football Simulation, AI Agent Arena, Replay Analyzer, And Reinforcement-Learning Platform.

GMN-Football is an independent web-based reimplementation and research platform.

The project aims to bring the core ideas of GRF into a modern browser-native environment built around **TypeScript, React, Vite, deterministic simulation, AI agents, scenario-based training, replay analysis, and modern reinforcement learning**.

The long-term goal is to create a football simulation that can be used both interactively in the browser and as a training environment for intelligent football agents.

> **Project status:** Early development / architecture phase.

---

## 🎯 Vision

GMN-Football is designed around one central idea:

> **Build a modern football simulation that can serve both humans and AI agents from the same underlying game environment.**

The project is intended to evolve into a platform where:

```text
                 GMN-FOOTBALL
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   Simulation       AI Agents       Replay
        │              │              │
        ▼              ▼              ▼
   Scenarios       RL Training     Analysis
        │              │
        └───────┬──────┘
                ▼
        Intelligent Football
```

A human should be able to play and observe the game, while an AI agent should be able to interact with the exact same simulation through a well-defined environment interface.

---

# 🚀 Project Goals

GMN-Football is being developed toward the following capabilities:

* 🏟️ Browser-based football simulation
* 🤖 AI-controlled football players
* 🧠 Reinforcement-learning environments
* 🎯 Scenario-based training
* 📈 Curriculum learning
* 🎬 Match replay and analysis
* 🧪 Deterministic simulation for reproducible experiments
* 🧩 Extensible player/action systems
* 🌐 Browser-native inference
* 📦 Neural-network model export
* ⚡ Headless simulation for high-speed training
* 🔬 Research and experimentation

---

# 🧠 Relationship to Google Research Football

GMN-Football is **inspired by Google Research Football**, but it is not intended to simply reproduce the original Python/C++ repository.

The original GRF project provides an important reference for:

* football environment design
* agent actions
* observations
* scenarios
* rewards
* multi-agent interaction
* reinforcement-learning workflows

GMN-Football takes those concepts as architectural and behavioral inspiration while pursuing a modern web-native implementation.

The original GRF project uses a legacy Python/C++ architecture and historically relied on TensorFlow 1.x and OpenAI Baselines for PPO training. GMN-Football is intended to provide a cleaner foundation for modern browser-based simulation and modern RL workflows.

---

# 🏗️ Architecture

The target architecture is:

```text
                    GMN-Football
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Football Engine          RL Environment
              │                     │
              │                Gymnasium API
              │                     │
              │                     ▼
              │                PPO / SB3
              │                     │
              │                     ▼
              │                 PyTorch
              │                     │
              │                     ▼
              │                    ONNX
              │                     │
              └──────────────┬──────┘
                             ▼
                         Web Agent
```

The simulation should remain independent of the AI system.

This allows the same football engine to support:

```text
Human Player
     │
     ├── Keyboard
     ├── Gamepad
     └── Browser controls

AI Player
     │
     ├── Rule-based agent
     ├── Scripted agent
     ├── PPO agent
     └── Neural-network agent
```

---

# 🎮 Simulation Layer

The simulation layer is responsible for the actual football world.

It is expected to contain systems for:

```text
Game World
 ├── Pitch
 ├── Players
 ├── Teams
 ├── Ball
 ├── Goals
 ├── Physics
 ├── Collision
 ├── Possession
 ├── Movement
 ├── Passing
 ├── Shooting
 ├── Tackling
 ├── Goalkeeping
 ├── Match Clock
 └── Game Rules
```

The simulation should not depend on React components.

Instead:

```text
React UI
   │
   ▼
Game Controller
   │
   ▼
Game Engine
   │
   ├── World
   ├── Physics
   ├── Players
   ├── Ball
   └── Match
```

React is responsible for presentation.

The simulation engine is responsible for the game.

---

# 🤖 AI Agent System

GMN-Football is designed to support multiple types of agents.

### Rule-Based Agents

Simple deterministic or heuristic behavior.

```text
Ball position
      ↓
Distance
      ↓
Decision rules
      ↓
Action
```

### Scripted Agents

Scenario-specific behavior used for testing.

### Neural Agents

Machine-learning policies trained externally or inside a compatible RL environment.

```text
Observation
     ↓
Neural Network
     ↓
Action
```

### Future Agents

The architecture is intended to eventually support:

* PPO
* imitation learning
* self-play
* multi-agent reinforcement learning
* tactical agents
* learned goalkeeper agents
* learned passing/decision systems

---

# 🎯 Scenario System

Scenarios are a core part of GMN-Football.

Rather than beginning with a full 11-vs-11 match, the environment can progressively teach agents individual football skills.

A planned curriculum can follow:

```text
Stage 1
academy_empty_goal
        │
        ▼
Stage 2
academy_run_to_score
        │
        ▼
Stage 3
academy_pass_and_shoot_with_keeper
        │
        ▼
Stage 4
academy_3_vs_1_with_keeper
        │
        ▼
Stage 5
5_vs_5
        │
        ▼
Stage 6
11_vs_11
```

Each scenario should define:

* initial world state
* player positions
* ball position
* team configuration
* objectives
* available actions
* termination conditions
* rewards
* optional randomness

---

# 🧠 Reinforcement Learning

One of the major goals of GMN-Football is to create a modern RL-compatible football environment.

The target interface follows the familiar environment model:

```python
observation, info = env.reset()

observation, reward, terminated, truncated, info = env.step(action)
```

The conceptual environment interface is:

```text
reset()
   │
   ▼
Observation
   │
   ▼
Agent
   │
   ▼
Action
   │
   ▼
Environment
   │
   ├── New Observation
   ├── Reward
   ├── Termination
   └── Info
```

The project is intended to support a modern:

```text
GMN-Football
      ↓
Gymnasium
      ↓
Stable-Baselines3
      ↓
PPO
      ↓
PyTorch
```

workflow.

---

# 🏋️ Curriculum Learning

Training directly on a complex 11-vs-11 environment can be difficult.

GMN-Football therefore aims to support curriculum learning.

An agent can first learn:

```text
movement
```

then:

```text
ball control
```

then:

```text
shooting
```

then:

```text
dribbling
```

then:

```text
passing
```

then:

```text
attacking
```

then:

```text
team coordination
```

and eventually:

```text
full-match football
```

The objective is to transfer useful learned representations between increasingly difficult scenarios where the observation and action spaces remain compatible.

---

# 👁️ Observation System

The environment should provide a standardized observation representation for AI agents.

A future observation pipeline can look like:

```text
Game State
    │
    ▼
Observation Encoder
    │
    ├── Player position
    ├── Ball position
    ├── Player velocity
    ├── Ball velocity
    ├── Possession
    ├── Team information
    ├── Opponent information
    └── Match state
    │
    ▼
Agent Observation
```

Possible future representations include:

### Structured observations

Fast vector/tensor representations suitable for MLP policies.

### Spatial observations

Grid or feature-map representations suitable for CNN policies.

### Visual observations

Rendered or synthetic image observations.

The observation format should remain explicitly versioned so trained models can be reproduced reliably.

---

# 🎮 Action System

The action system separates the agent's decision from the underlying simulation.

Conceptually:

```text
Agent
  │
  ▼
Action
  │
  ▼
Action Executor
  │
  ▼
Football Engine
```

Possible football actions include:

* movement
* sprint
* short pass
* long/high pass
* shooting
* dribbling
* defensive pressure
* tackling
* goalkeeper pressure
* player switching

The exact action mapping should remain centralized so that the same mapping is used during:

```text
training
evaluation
replay
browser inference
```

---

# 🎬 Replay System

GMN-Football is intended to support complete match replay.

A replay should contain enough information to reconstruct a match:

```text
Match Metadata
      │
      ▼
Initial State
      │
      ▼
Game Frames
      │
      ▼
Actions
      │
      ▼
Events
      │
      ▼
Final Result
```

This will enable:

* replay playback
* tactical analysis
* agent comparison
* training evaluation
* debugging
* performance analysis
* AI-generated commentary

---

# 📊 Match Analysis

The platform is intended to expose statistics and analytical information such as:

* possession
* shots
* shots on target
* goals
* passes
* successful passes
* interceptions
* tackles
* player movement
* ball movement
* territory
* expected future metrics
* agent rewards
* scenario success rate

These metrics can be used both by humans and RL training pipelines.

---

# 🤖 AI-Assisted Analysis

GMN-Football includes an AI integration layer intended for higher-level analysis and tooling.

The current web project includes Google's GenAI SDK as a dependency.

Potential uses include:

* tactical analysis
* replay explanations
* scenario generation
* coaching feedback
* match summaries
* agent behavior analysis
* training recommendations

AI services should remain outside the real-time football physics loop.

The simulation must remain deterministic and performant without requiring an external AI API.

---

# 🌐 Web Technology

The current application is built with:

* **React**
* **TypeScript**
* **Vite**
* **Tailwind CSS**
* **Recharts**
* **Lucide React**
* **Google GenAI**

These dependencies are reflected in the current project configuration.

The web application currently uses Vite for development and production builds.

---

# 💻 Development

## Requirements

Recommended:

* Node.js
* npm
* Git
* modern Chromium/Firefox/Safari browser

Clone the repository:

```bash
git clone https://github.com/natnakem-cyb/GMN-Football.git
cd GMN-Football
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The project currently uses Vite with the development server configured on port `3000`.

---

# 🔨 Build

Create a production build:

```bash
npm run build
```

The build performs TypeScript checking followed by the Vite production build.

Preview the production build:

```bash
npm run preview
```

Type-check the project:

```bash
npm run lint
```

---

# 📁 Target Project Structure

As development progresses, the project is intended to evolve toward a structure similar to:

```text
GMN-Football/
│
├── src/
│   │
│   ├── engine/
│   │   ├── GameEngine.ts
│   │   ├── World.ts
│   │   ├── Physics.ts
│   │   └── Clock.ts
│   │
│   ├── football/
│   │   ├── Player.ts
│   │   ├── Ball.ts
│   │   ├── Team.ts
│   │   ├── Pitch.ts
│   │   └── Match.ts
│   │
│   ├── actions/
│   │   ├── Action.ts
│   │   ├── ActionSet.ts
│   │   └── ActionExecutor.ts
│   │
│   ├── observations/
│   │   ├── Observation.ts
│   │   └── ObservationEncoder.ts
│   │
│   ├── rewards/
│   │   └── RewardSystem.ts
│   │
│   ├── scenarios/
│   │   ├── Scenario.ts
│   │   └── academy/
│   │
│   ├── agents/
│   │   ├── Agent.ts
│   │   ├── RuleBasedAgent.ts
│   │   └── NeuralAgent.ts
│   │
│   ├── replay/
│   │   └── Replay.ts
│   │
│   ├── rl/
│   │   └── Environment.ts
│   │
│   └── ui/
│
├── training/
│   ├── configs/
│   └── colab/
│
├── models/
│
└── README.md
```

This structure is a development target rather than a claim about every directory currently present.

---

# ☁️ Google Colab Training

A planned training workflow is:

```text
GitHub
   │
   ▼
Google Colab
   │
   ▼
GMN-Football Environment
   │
   ▼
Gymnasium
   │
   ▼
Stable-Baselines3 PPO
   │
   ▼
PyTorch
   │
   ▼
Checkpoint
   │
   ▼
Model Export
```

Training can use GPU-enabled Colab runtimes.

Scenario-by-scenario training can then follow the curriculum:

```text
Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6
```

Checkpoints should be stored externally, for example in Google Drive or another persistent storage system.

---

# 📦 Model Export

The long-term objective is to export trained policies for browser inference.

The intended pipeline is:

```text
SB3 PPO
   │
   ▼
PyTorch Actor
   │
   ▼
ONNX
   │
   ▼
JavaScript / Web Runtime
   │
   ▼
GMN-Football
```

A model export should be accompanied by metadata describing:

```text
model version
observation version
observation shape
action version
action mapping
normalization
scenario
training configuration
```

This prevents a model from being accidentally executed against an incompatible observation or action schema.

---

# 🧪 Deterministic Simulation

Reproducibility is a major design goal.

The simulation should support seeded environments:

```text
seed
 │
 ▼
Initial State
 │
 ▼
Deterministic Simulation
 │
 ▼
Reproducible Match
```

This is particularly important for:

* reinforcement learning
* debugging
* regression tests
* replay generation
* agent evaluation
* scientific experiments

---

# ⚡ Headless Simulation

The same simulation engine should eventually support two modes.

### Visual mode

```text
Game Engine
     ↓
Renderer
     ↓
Browser
```

### Headless mode

```text
Game Engine
     ↓
No Renderer
     ↓
Maximum Simulation Speed
     ↓
RL Training
```

Headless simulation is essential for scaling reinforcement-learning experiments.

The agent should never require the UI to be rendered during training unless visual observations are explicitly being used.

---

# 🧪 Testing Philosophy

GMN-Football should test the simulation independently from the UI.

Important tests include:

### Physics

```text
ball movement
player movement
collision
friction
velocity
```

### Football rules

```text
goals
possession
out-of-bounds
match clock
```

### Actions

```text
pass
shoot
move
tackle
sprint
```

### Scenarios

```text
initial state
objective
termination
reward
```

### RL

```text
reset
step
observation
action
reward
termination
```

### Determinism

```text
same seed
+
same actions
=
same result
```

---

# 🗺️ Roadmap

## Phase 1 — Foundation

* [x] React/Vite/TypeScript project
* [ ] Define simulation architecture
* [ ] Separate UI from simulation
* [ ] Create game-state model
* [ ] Create deterministic clock
* [ ] Create player model
* [ ] Create ball model

## Phase 2 — Football Engine

* [ ] Pitch
* [ ] Teams
* [ ] Player movement
* [ ] Ball physics
* [ ] Possession
* [ ] Passing
* [ ] Shooting
* [ ] Tackling
* [ ] Goalkeeper
* [ ] Match rules

## Phase 3 — Scenarios

* [ ] Academy empty goal
* [ ] Run to score
* [ ] Pass and shoot
* [ ] 3v1
* [ ] 5v5
* [ ] 11v11

## Phase 4 — AI Agents

* [ ] Rule-based agent
* [ ] Scripted scenario agents
* [ ] Agent interface
* [ ] Neural agent interface
* [ ] Multi-agent architecture

## Phase 5 — Reinforcement Learning

* [ ] Observation encoder
* [ ] Action space
* [ ] Reward system
* [ ] Gymnasium-compatible environment
* [ ] SB3 PPO integration
* [ ] Headless training
* [ ] Curriculum learning
* [ ] Evaluation framework

## Phase 6 — Model Deployment

* [ ] PyTorch policy export
* [ ] ONNX export
* [ ] Browser inference
* [ ] Model metadata
* [ ] Versioned observation/action schemas

## Phase 7 — Research Platform

* [ ] Self-play
* [ ] Multi-agent RL
* [ ] Agent tournaments
* [ ] Replay analysis
* [ ] Tactical analytics
* [ ] AI coaching
* [ ] Scenario builder
* [ ] Training dashboard

---

# 🔬 Research Applications

GMN-Football can eventually be used for experiments involving:

* reinforcement learning
* curriculum learning
* multi-agent reinforcement learning
* imitation learning
* self-play
* tactical planning
* hierarchical agents
* neural football controllers
* agent evaluation
* behavior analysis
* simulation-based research

---

# 🤝 Contributing

Contributions are welcome.

Before submitting significant changes:

1. Understand the simulation architecture.
2. Keep simulation logic independent from React UI.
3. Avoid introducing nondeterministic behavior without a clear reason.
4. Add tests for new game mechanics.
5. Keep action and observation schemas versionable.
6. Do not couple the simulation directly to a specific RL framework.
7. Run the TypeScript checks before submitting changes.

See `CONTRIBUTING.md` for project contribution guidelines.

---

# 📜 License

This project is released under the license included in this repository.

GMN-Football is an independent project inspired by the ideas and research surrounding Google Research Football.

It is not an official Google Research project.

---

# 🙏 Acknowledgements

GMN-Football is inspired by the research and open-source work behind **Google Research Football**, including the original football simulation and reinforcement-learning environment.

The project also draws inspiration from the broader open-source reinforcement-learning ecosystem.

---

# ⚠️ Project Status

GMN-Football is currently under active development.

The repository is being built toward a complete football simulation and AI research environment. Features described in the roadmap are **planned capabilities and should not be interpreted as already implemented**.

The current web application is based on React, TypeScript, Vite and related frontend tooling.

---

# ⭐ Project Philosophy

GMN-Football follows five core principles:

### 1. Simulation First

The football world must remain independent of the UI and AI systems.

### 2. Deterministic When Needed

The same state, seed and actions should produce reproducible results.

### 3. AI-Ready

Every important game mechanic should be accessible through a clean environment interface.

### 4. Framework Independent

The football simulator should not be permanently tied to PPO, TensorFlow, PyTorch, or any particular RL framework.

### 5. Browser Native

The ultimate goal is to make trained football intelligence capable of running directly inside a modern web application.

---

## ⚽ GMN-Football

**A football simulation built for humans, agents, and reinforcement learning.**

```text
SIMULATE
    ↓
OBSERVE
    ↓
ACT
    ↓
LEARN
    ↓
TRAIN
    ↓
DEPLOY
    ↓
PLAY
```
