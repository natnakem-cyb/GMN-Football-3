"""
Episode Trace Recorder and Loader for GMN-Football-3.
Provides an on-disk JSON Lines format (.jsonl) for recording per-step RL trajectories.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


class EpisodeRecorder:
    def __init__(
        self,
        scenario: str,
        seed: Optional[int],
        agent_ids: List[str],
        checkpoint: Optional[str] = None,
        output_dir: str = "training/replays",
    ):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        seed_str = str(seed) if seed is not None else "noseed"
        
        # Ensure filename uniqueness even if multiple episodes start within the same second
        base_filename = f"{scenario}_{seed_str}_{timestamp}"
        candidate_path = os.path.join(output_dir, f"{base_filename}.jsonl")
        counter = 1
        while os.path.exists(candidate_path):
            candidate_path = os.path.join(output_dir, f"{base_filename}_{counter}.jsonl")
            counter += 1
            
        self.filepath = candidate_path
        self._file = open(self.filepath, "w", encoding="utf-8")
        metadata = {
            "type": "metadata",
            "scenario": scenario,
            "seed": seed,
            "agent_ids": list(agent_ids),
            "checkpoint": checkpoint,
            "recorded_at": timestamp,
        }
        self._file.write(json.dumps(metadata) + "\n")
        self._file.flush()

    def record_step(
        self,
        tick: int,
        actions: Dict[str, Any],
        observations: Dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        score: Dict[str, int],
        event: Optional[str] = None,
    ):
        record = {
            "tick": int(tick),
            "actions": {k: int(v) for k, v in actions.items()},
            "observations": {k: [float(x) for x in v] for k, v in observations.items()},
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "score": score,
            "event": event,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> str:
        if self._file and not self._file.closed:
            self._file.close()
        return self.filepath


def load_episode_trace(filepath: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Empty replay file at: {filepath}")
    metadata = lines[0]
    steps = lines[1:]
    return metadata, steps
