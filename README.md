# ⚽ GMN-Football

A browser-native football simulation with an AI agent arena, replay analysis, and a working reinforcement-learning training pipeline.

GMN-Football is inspired by Google Research Football but is an independent reimplementation built around TypeScript, React, and Vite, with a headless bridge that exposes the same simulation engine to a Python/Gymnasium RL stack.

> **Project status:** Active development. Core simulation engine, five agent types, the RL bridge, and PPO training are implemented and working. Match physics, agent behavior, and scenario coverage are still expanding — see Roadmap below.

---

## 🎯 What this is

A human plays the game through the browser UI. An AI agent — rule-based, scripted, or a trained neural policy — plays through the exact same `GameEngine`, either in-browser or headlessly via an HTTP bridge to Python. That shared-engine design is the core idea the project is built around, and it's already functioning end-to-end for the `academy_empty_goal` scenario, including a trained PPO checkpoint.

---

## 🏗️ Architecture (as built)

GMN-Football
│
├── src/engine/ Deterministic game core
│ GameEngine.ts Match state, step loop, scenario loading
│ Physics.ts Ball/player movement, collision
│ Rules.ts Possession, goals, out-of-bounds
│ ObservationEncoder.ts 115-float observation vector
│ Vector.ts 2D vector math
│
├── src/agents/ Pluggable agent implementations
│ BaseAgent.ts
│ HumanAgent.ts
│ RuleBasedAgent.ts
│ ScriptedScenarioAgent.ts
│ NeuralHeuristicAgent.ts
│
├── src/scenarios/ScenarioRegistry.ts Scenario definitions
├── src/services/geminiService.ts Gemini-backed tactical coach
├── src/types/football.ts Shared domain types
├── src/components/ React UI (PitchCanvas, Scoreboard,
│ AgentArenaPanel, ReplayAnalyzer,
│ RLGymnasiumPanel, TacticalAnalytics,
│ GeminiTacticalCoach, MatchControls,
│ ScenarioSelector, ControlsHelpModal)
└── src/App.tsx, main.tsx, index.css

training/ RL pipeline (TypeScript + Python)
├── bridge_server.ts Headless HTTP wrapper around GameEngine
├── gmn_gym.py Gymnasium Env — talks to the bridge
├── action_mapping.ts Discrete(15) action ↔ engine action
├── train_ppo.py PPO training via Stable-Baselines3
├── train_stage2_ppo.py
├── eval_checkpoint.py
├── rl_validation_suite.py / stage2_full_validation.py
├── audit_observations_and_actions.ts / stage2_audit_and_baseline.ts
├── benchmark.ts / benchmark_bridge.py
├── test_determinism.ts / test_env.py
└── models/ppo_academy_empty_goal_smoke.zip Trained checkpoint

The simulation engine has no dependency on React. The RL bridge runs the same `GameEngine` used by the browser, headless, over HTTP — so training and interactive play are guaranteed to see identical game logic.

---

## 🧠 RL Environment

`gmn_gym.py` implements a standard Gymnasium `Env`:

- **Observation space:** `Box(115,)`, float32
- **Action space:** `Discrete(15)` — move, pass, shoot, sprint, tackle, dribble, idle
- **Protocol:** the Python env starts (or connects to) `training/bridge_server.ts` over HTTP, which runs the real TypeScript engine and returns state after each step.

- Python (gmn_gym.py)
│ HTTP
▼
bridge_server.ts
│
▼
GameEngine.ts (same engine the browser uses)

A smoke-trained PPO checkpoint for `academy_empty_goal` is checked into `training/models/`.

---

## 💻 Development

**Requirements:** Node.js, [Bun](https://bun.sh) (this repo is built and locked with Bun — `bun.lock` is committed), Python 3.10+ for the training scripts, Git.

```bash
git clone https://github.com/natnakem-cyb/GMN-Football-2.git
cd GMN-Football-2
bun install
bun run dev
```

Dev server runs on port `3000`.

Set `GEMINI_API_KEY` in `.env` (see `.env.example`) to enable the in-app Gemini tactical coach — the simulation and RL pipeline run fully without it.

---

## 📜 Scripts

| Command | Purpose |
|---|---|
| `dev` | Start Vite dev server |
| `build` | Type-check (`tsc`) then production build |
| `preview` | Preview production build |
| `lint` | Type-check only |
| `bridge` | Run the headless bridge server standalone |
| `headless` | Run the headless benchmark |
| `test:env` | Sanity-check the Python Gymnasium env |
| `test:ppo` | Short PPO training smoke test |
| `test:validation` | Full RL validation suite |
| `test:determinism` | Verify seeded runs reproduce identical results |
| `test:audit` | Audit observation/action ranges |
| `test:scripted` | Run scripted-agent evaluation |

Python training scripts (`train_ppo.py`, `eval_checkpoint.py`, etc.) require `pip install -r training/requirements.txt` (gymnasium, stable-baselines3, torch, numpy, requests).

---

## 🎯 Scenarios

Currently registered in `ScenarioRegistry.ts`:

- `academy_empty_goal` — Stage 1 of the planned curriculum; has a trained PPO checkpoint

Planned curriculum (not yet implemented): `academy_run_to_score` → `academy_pass_and_shoot_with_keeper` → `academy_3_vs_1_with_keeper` → `5_vs_5` → `11_vs_11`.

---

## 🗺️ Roadmap

**Done**
- [x] React/Vite/TypeScript project
- [x] Deterministic game engine (physics, rules, clock)
- [x] Player/ball/match state model
- [x] UI separated from simulation logic
- [x] Rule-based, scripted, human, and neural-heuristic agents
- [x] Observation encoder (115-float vector)
- [x] Discrete(15) action space + mapping
- [x] Gymnasium-compatible environment via HTTP bridge
- [x] PPO training (Stable-Baselines3) for `academy_empty_goal`
- [x] Determinism, audit, and validation test scripts

**In progress / planned**
- [ ] Remaining academy scenarios (run-to-score → 11v11)
- [ ] Full match rules (offside, fouls, cards)
- [ ] Reward-system module beyond scenario-level scoring
- [ ] Dedicated replay data format (currently UI-level analysis only)
- [ ] ONNX export for browser-native inference
- [ ] Multi-agent / self-play training
- [ ] Curriculum learning across scenario stages

---

## 🤝 Contributing

1. Keep simulation logic (`src/engine/`) independent of React.
2. Don't introduce nondeterministic behavior without a clear reason — `test:determinism` must pass.
3. Add or update tests in `training/` when changing observation/action schemas.
4. Run `bun run lint` before submitting changes.

See `CONTRIBUTING.md`.

---

## 📜 License

Apache-2.0. GMN-Football is an independent project inspired by Google Research Football; it is not an official Google project.

