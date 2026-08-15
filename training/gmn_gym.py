import os
import time
import subprocess
import requests
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class GMNFootballEnv(gym.Env):
    """
    Gymnasium Environment wrapper for Google Research Football (GMN-Football) Simulation.
    Communicates with the headless TypeScript GameEngine via persistent HTTP Bridge.

    Observation Space: Box(115,) Float32
    Action Space: Discrete(19)
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        scenario: str = "academy_empty_goal",
        host: str = "127.0.0.1",
        port: int = 5050,
        auto_start_bridge: bool = True,
    ):
        super().__init__()

        self.scenario = scenario
        self.host = host
        self.port = int(os.environ.get("GMN_BRIDGE_PORT", port))
        self.base_url = f"http://{self.host}:{self.port}"
        self.auto_start_bridge = auto_start_bridge
        self.bridge_process: Optional[subprocess.Popen] = None
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=3)
        self.session.mount("http://", adapter)

        # 1. Observation Space: 115-float SMM Tensor
        # Audited range across 100k steps is [-2.37, +2.56] (velocities x 50, pitch coords, normalized flags)
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(115,),
            dtype=np.float32,
        )

        # 2. Action Space: 19 Discrete Actions matching Google Research Football (GRF)
        self.action_space = spaces.Discrete(19)

        # Ensure Bridge Server is running
        self._ensure_bridge_running()

    def _ensure_bridge_running(self):
        """Verifies connection to bridge server or starts it via npx tsx."""
        for attempt in range(5):
            try:
                res = requests.get(f"{self.base_url}/health", timeout=1.0)
                if res.status_code == 200:
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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        payload = {
            "scenario": self.scenario,
            "seed": seed if seed is not None else int(time.time()),
        }

        if options and "scenario" in options:
            payload["scenario"] = options["scenario"]

        res = self.session.post(f"{self.base_url}/reset", json=payload, timeout=5.0)
        data = res.json()

        raw_obs = np.array(data["observation"], dtype=np.float32)
        # Ensure exact shape (115,)
        if len(raw_obs) != 115:
            padded = np.zeros(115, dtype=np.float32)
            padded[: min(115, len(raw_obs))] = raw_obs[: min(115, len(raw_obs))]
            raw_obs = padded

        info = data.get("info", {})
        return raw_obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action} for Discrete(19) space.")

        payload = {"action": int(action)}
        res = self.session.post(f"{self.base_url}/step", json=payload, timeout=5.0)
        data = res.json()

        raw_obs = np.array(data["observation"], dtype=np.float32)
        if len(raw_obs) != 115:
            padded = np.zeros(115, dtype=np.float32)
            padded[: min(115, len(raw_obs))] = raw_obs[: min(115, len(raw_obs))]
            raw_obs = padded

        reward = float(data.get("reward", 0.0))
        terminated = bool(data.get("terminated", False))
        truncated = bool(data.get("truncated", False))
        info = data.get("info", {})

        return raw_obs, reward, terminated, truncated, info

    def close(self):
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
