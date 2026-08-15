# ⚽ GMN-Football-3

### Modern Browser-Native Football Simulation & Reinforcement Learning Platform

GMN-Football-3 is a browser-native football simulation and research platform where the core physics and match simulation engine is shared, unmodified, between interactive human play in the browser and headless reinforcement learning training in Python.

```text
       TypeScript Football Simulation Engine (Autoritative)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       Interactive UI                   HTTP Bridge
   (React, Canvas, Replay)       (bridge_server.ts / REST)
                                              │
                                              ▼
                                     Gymnasium Environment
                                      (training/gmn_gym.py)
                                              │
                                              ▼
                                      Stable-Baselines3
                                      (PPO / PyTorch)
```

---

## 🎯 Key Architectural Pillars

1. **Shared Engine Contract**: The same `GameEngine.ts` simulation executes in the web browser (React UI) and in headless Node.js processes for RL training.
2. **Deterministic Simulation & Seed Propagation**: Pure Mulberry32 PRNG (`SeededRNG.ts`) drives all stochastic mechanics (tackles, deflections, ball possession changes). The same seed + action trajectory produces bitwise identical states and observations.
3. **Strict Versioned Interface Contracts**:
   - `OBSERVATION_DIM`: Exactly **115 floats** (`simple115_v2` representation)
   - `ACTION_SPACE_SIZE`: Exactly **19 discrete actions** (matching canonical GRF action set)
   - Contract versions: `GMN_ENV_VERSION = "3.0.0"`, `OBSERVATION_SCHEMA_VERSION = "simple115_v2"`, `ACTION_SCHEMA_VERSION = "grf19_v1"`
4. **Clean Reset Semantics**: `/reset` returns the exact initial observation at $t = 0$ without stepping the physics engine.
5. **High-Throughput Raw Simulation**:
   - **TypeScript Engine**: ~33,000 steps/second (~30 µs latency per step)
   - **HTTP Bridge**: ~350–500 steps/second end-to-end Python/Node roundtrip.

---

## 🎮 Action Space (19 Discrete Actions)

| ID | Name | Action Type | Details |
|:--:|:-----|:------------|:--------|
| `0` | `action_idle` | No-op | Maintain momentum |
| `1` | `action_left` | Directional | Move Left (-x) |
| `2` | `action_top_left` | Directional | Move Top-Left (-x, -y) |
| `3` | `action_top` | Directional | Move Top (-y) |
| `4` | `action_top_right` | Directional | Move Top-Right (+x, -y) |
| `5` | `action_right` | Directional | Move Right (+x) |
| `6` | `action_bottom_right` | Directional | Move Bottom-Right (+x, +y) |
| `7` | `action_bottom` | Directional | Move Bottom (+y) |
| `8` | `action_bottom_left` | Directional | Move Bottom-Left (-x, +y) |
| `9` | `action_long_pass` | Pass | High-power pass in movement direction |
| `10` | `action_high_pass` | Pass | Lofted chip pass |
| `11` | `action_short_pass` | Pass | Low grounded pass to teammate |
| `12` | `action_shot` | Shot | Powerful drive toward opponent goal |
| `13` | `action_sprint` | Movement | Engage sticky sprint mode |
| `14` | `action_release_direction` | Movement | Stop directional input |
| `15` | `action_release_sprint` | Movement | Disengage sticky sprint |
| `16` | `action_sliding` | Tackle | Slide tackle (active only when defending) |
| `17` | `action_dribble` | Ball Control | Engage close-control dribble mode |
| `18` | `action_release_dribble` | Ball Control | Disengage close-control dribble |

---

## 📊 Observation Space (115 Floats)

| Offset Range | Length | Description |
|:------------:|:------:|:------------|
| `0 .. 21` | 22 | Left team player positions $(x, y)$ (up to 11 players; unused slots padded with -1) |
| `22 .. 43` | 22 | Left team player velocities $(\Delta x, \Delta y)$ |
| `44 .. 65` | 22 | Right team player positions $(x, y)$ |
| `66 .. 87` | 22 | Right team player velocities $(\Delta x, \Delta y)$ |
| `88 .. 90` | 3 | Ball position $(x, y, z)$ |
| `91 .. 93` | 3 | Ball velocity $(\Delta x, \Delta y, \Delta z)$ |
| `94 .. 96` | 3 | Ball possession one-hot: `[no_one, left_team, right_team]` |
| `97 .. 107` | 11 | Active controlled player one-hot across team roster |
| `108 .. 114` | 7 | Match mode one-hot: `[Normal, KickOff, GoalKick, FreeKick, Corner, ThrowIn, Penalty]` |

---

## 🏃 Quickstart & Commands

### Development Server (Web UI)
```bash
npm run dev
```
Launches the interactive React application on `http://localhost:3000`.

### Type-checking & Build
```bash
npm run lint      # Runs tsc --noEmit
npm run build     # Builds production bundle to dist/
```

### Automated Testing Suite
```bash
npm run test             # Runs scenario validation + engine determinism tests
npm run test:scenarios   # Validates all 6 academy and match scenarios
npm run test:determinism # Verifies bitwise determinism and seed reproducibility
npm run test:audit       # Runs 100,000-step observation and action audit
```

### Benchmarks
```bash
npm run benchmark        # Benchmarks raw TypeScript GameEngine throughput (~33k steps/sec)
npm run benchmark:bridge # Benchmarks Python-to-Node HTTP bridge roundtrip
```

### Python RL Environment & PPO Training
```bash
# 1. Start bridge server (or let gmn_gym auto-start it)
npm run bridge

# 2. Run Gymnasium environment validation
python3 training/test_env.py

# 3. Run End-to-End determinism check
python3 training/test_e2e_determinism.py

# 4. Run PPO smoke test (1000 timesteps)
python3 training/train_ppo.py 1000
```

---

## 📋 Hardening & Alignment Fixes Implemented

1. **Fix 1: Seed Propagation** — Integrated `SeededRNG` (Mulberry32) across `GameEngine`, `PhysicsEngine`, and `/reset` endpoint.
2. **Fix 2: Clean Reset Semantics** — Refactored `GameEngine.getObservation()` and `/reset` to return pure initial state without advancing physics clock.
3. **Fix 3: Strict Observation Contract** — Enforced exact `(115,)` observation validation in both TypeScript (`ObservationEncoder.ts`) and Python (`gmn_gym.py`), eliminating silent padding.
4. **Fix 4: Versioned Contract Constants** — Centralized `Contract.ts` defining canonical versions, observation shapes, and action spaces.
5. **Fix 5: Centralized Action Mapping** — Audited discrete action index range (`0..18`) in `action_mapping.ts` and `bridge_server.ts`.
6. **Fix 6: Automated Scenario Validation** — Created `training/test_scenarios.ts` covering setup, player spawning, initial observations, and execution across all 6 scenarios.
7. **Fix 7: End-to-End Determinism Tests** — Built bitwise trajectory verification in TypeScript (`test_determinism.ts`) and Python (`test_e2e_determinism.py`).
8. **Fix 8: Gymnasium Environment Compliance** — Updated `gmn_gym.py` with standard `Box(-5.0, 5.0, shape=(115,))` and `Discrete(19)` spaces and proper lifecycle methods.
9. **Fix 9: Real PPO Smoke Test** — Standardized `train_ppo.py` for Stable-Baselines3 PPO policy training, checkpoint saving, and evaluation.
10. **Fix 10: Project Identity & Documentation** — Cleaned up project naming, metadata, scripts, and documentation.
11. **Fix 11: Measured Bottlenecks** — Quantified raw engine throughput (33,000 steps/sec) and HTTP bridge roundtrip performance.
