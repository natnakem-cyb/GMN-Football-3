import os
import time
import struct
import json
import subprocess
import requests
import numpy as np
import websockets.sync.client
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

OBSERVATION_DIM = 115
ACTION_SPACE_SIZE = 19
GMN_ENV_VERSION = "3.0.0"

EVENT_CODE_MAP = [
    None,
    'goal',
    'shot',
    'shot_saved',
    'shot_missed',
    'pass',
    'interception',
    'tackle',
    'foul',
    'kickoff',
    'out_of_bounds',
    'scenario_complete',
    'scenario_failed',
]


class GMNFootballEnv(gym.Env):
    """
    Gymnasium Environment wrapper for GMN-Football-3 Simulation.
    Communicates with the headless TypeScript GameEngine via binary WebSocket or HTTP Bridge.

    Authoritative Contract:
    - Observation Space: Box(-5.0, 5.0, shape=(115,), dtype=np.float32)
    - Action Space: Discrete(19)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        scenario: str = "academy_empty_goal",
        host: str = "127.0.0.1",
        port: int = 5050,
        auto_start_bridge: bool = True,
        render_mode: Optional[str] = None,
        use_ws: bool = True,
    ):
        super().__init__()

        self.scenario = scenario
        self.host = host
        self.port = int(os.environ.get("GMN_BRIDGE_PORT", port))
        self.base_url = f"http://{self.host}:{self.port}"
        self.ws_url = f"ws://{self.host}:{self.port}"
        self.auto_start_bridge = auto_start_bridge
        self.render_mode = render_mode
        self.use_ws = use_ws
        self.bridge_process: Optional[subprocess.Popen] = None
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=3)
        self.session.mount("http://", adapter)
        self.ws_client = None

        # 1. Observation Space: Exactly 115-float SMM Vector
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

        # 2. Action Space: Discrete(19) matching GMN-Football-3 canonical action space
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        self._step_count = 0

        # Ensure Bridge Server is running
        self._ensure_bridge_running()
        if self.use_ws:
            self._connect_ws()

    def _ensure_bridge_running(self):
        """Verifies connection to bridge server or starts it via npx tsx."""
        for attempt in range(6):
            try:
                res = requests.get(f"{self.base_url}/health", timeout=1.0)
                if res.status_code == 200:
                    info = res.json()
                    bridge_obs_dim = info.get("observation_dim", 115)
                    bridge_act_dim = info.get("action_space_size", info.get("action_dim", 19))
                    if bridge_obs_dim != OBSERVATION_DIM or bridge_act_dim != ACTION_SPACE_SIZE:
                        raise RuntimeError(
                            f"[GMN-Gym Contract Mismatch] Bridge reports obs_dim={bridge_obs_dim}, "
                            f"act_dim={bridge_act_dim}; Expected obs_dim={OBSERVATION_DIM}, act_dim={ACTION_SPACE_SIZE}."
                        )
                    return
            except requests.RequestException:
                pass

            if self.auto_start_bridge and self.bridge_process is None:
                print(f"[GMN-Gym] Launching Headless Bridge Server on {self.base_url}...")
                bridge_script = os.path.join(os.path.dirname(__file__), "bridge_server.ts")
                self.bridge_process = subprocess.Popen(
                    ["npx", "tsx", bridge_script],
                    env=dict(os.environ, GMN_BRIDGE_PORT=str(self.port)),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(2.0)
            else:
                time.sleep(1.0)

    def _connect_ws(self):
        if self.ws_client is not None:
            try:
                self.ws_client.close()
            except Exception:
                pass
        self.ws_client = websockets.sync.client.connect(self.ws_url, max_size=None)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0

        target_scenario = self.scenario
        if options and "scenario" in options:
            target_scenario = options["scenario"]

        payload: Dict[str, Any] = {
            "type": "reset",
            "scenario": target_scenario,
        }
        if seed is not None:
            payload["seed"] = int(seed)

        if self.use_ws:
            if self.ws_client is None:
                self._connect_ws()
            self.ws_client.send(json.dumps(payload))
            res_text = self.ws_client.recv()
            data = json.loads(res_text)
        else:
            http_payload = {"scenario": target_scenario}
            if seed is not None:
                http_payload["seed"] = int(seed)
            res = self.session.post(f"{self.base_url}/reset", json=http_payload, timeout=5.0)
            if res.status_code != 200:
                raise RuntimeError(f"[GMN-Gym Bridge Error] /reset failed with code {res.status_code}: {res.text}")
            data = res.json()

        raw_obs = np.array(data["observation"], dtype=np.float32)
        if raw_obs.shape != (OBSERVATION_DIM,):
            raise ValueError(
                f"[GMN-Gym Contract Violation] Invalid observation shape {raw_obs.shape}. "
                f"Expected ({OBSERVATION_DIM},) floats without silent padding."
            )

        info = data.get("info", {})
        return raw_obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action} for Discrete({ACTION_SPACE_SIZE}) space.")

        self._step_count += 1

        if self.use_ws:
            if self.ws_client is None:
                self._connect_ws()

            self.ws_client.send(int(action).to_bytes(1, "little"))
            data = self.ws_client.recv()
            if not isinstance(data, (bytes, bytearray)) or len(data) != 477:
                raise RuntimeError(
                    f"[GMN-Gym WS Error] Expected 477 binary bytes, got "
                    f"{len(data) if isinstance(data, (bytes, bytearray)) else type(data)}"
                )

            reward, term, trunc, score_l, score_r, cp_reward, dist_goal, event_code = struct.unpack_from("<f??BBffB", data, 0)
            raw_obs = np.frombuffer(data, dtype="<f4", count=115, offset=17).copy()

            info: Dict[str, Any] = {
                "score": {"left": int(score_l), "right": int(score_r)},
                "checkpointReward": float(cp_reward),
                "ballDistanceToGoal": float(dist_goal),
            }
            if event_code > 0 and event_code < len(EVENT_CODE_MAP):
                ev_type = EVENT_CODE_MAP[event_code]
                if ev_type:
                    info["event"] = {"type": ev_type}

            return raw_obs, float(reward), bool(term), bool(trunc), info
        else:
            payload = {"action": int(action)}
            res = self.session.post(f"{self.base_url}/step", json=payload, timeout=5.0)
            if res.status_code != 200:
                raise RuntimeError(f"[GMN-Gym Bridge Error] /step failed with code {res.status_code}: {res.text}")
            data = res.json()

            raw_obs = np.array(data["observation"], dtype=np.float32)
            if raw_obs.shape != (OBSERVATION_DIM,):
                raise ValueError(
                    f"[GMN-Gym Contract Violation] Invalid observation shape {raw_obs.shape}. "
                    f"Expected ({OBSERVATION_DIM},) floats."
                )

            reward = float(data.get("reward", 0.0))
            terminated = bool(data.get("terminated", False))
            truncated = bool(data.get("truncated", False))
            info = data.get("info", {})

            return raw_obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            # Return dummy pitch frame representation for rgb_array mode
            return np.zeros((100, 100, 3), dtype=np.uint8)
        elif self.render_mode == "human":
            print(f"[GMNFootballEnv Step {self._step_count}] Scenario: {self.scenario}")

    def close(self):
        if self.ws_client is not None:
            try:
                self.ws_client.send(json.dumps({"type": "close"}))
                self.ws_client.close()
            except Exception:
                pass
            self.ws_client = None
        try:
            self.session.close()
        except Exception:
            pass
        if self.bridge_process is not None:
            try:
                requests.post(f"{self.base_url}/close", timeout=1.0)
            except Exception:
                pass
            self.bridge_process.terminate()
            self.bridge_process = None


# Registration for standard gym.make("GMNFootball-v0")
gym.register(
    id="GMNFootball-v0",
    entry_point="training.gmn_gym:GMNFootballEnv",
    max_episode_steps=900,
)
