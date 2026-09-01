"""
GMN-Football-3 — Trace to Replay Converter
Converts raw .jsonl episode traces recorded by EpisodeRecorder into structured frame format.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List


def convert_trace(input_path: str, output_path: str = None) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Trace file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Trace file {input_path} is empty")

    metadata = lines[0]
    steps = lines[1:]

    frames: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    for step in steps:
        tick = step.get("tick", 0)
        time_sec = tick / 60.0
        score = step.get("score", {"left": 0, "right": 0})
        event_type = step.get("event")

        # Use first observation to extract ball/player positions
        observations = step.get("observations", {})
        obs = next(iter(observations.values())) if observations else []

        players = []
        if len(obs) >= 88:
            for i in range(11):
                px, py = obs[i * 2], obs[i * 2 + 1]
                if px != -1.0 or py != -1.0:
                    vx, vy = obs[22 + i * 2] / 50.0, obs[22 + i * 2 + 1] / 50.0
                    players.append({
                        "id": f"left_{i+1}",
                        "team": "left",
                        "position": {"x": px, "y": py},
                        "velocity": {"x": vx, "y": vy},
                    })
            for i in range(11):
                px, py = obs[44 + i * 2], obs[44 + i * 2 + 1]
                if px != -1.0 or py != -1.0:
                    vx, vy = obs[66 + i * 2] / 50.0, obs[66 + i * 2 + 1] / 50.0
                    players.append({
                        "id": f"right_{i+1}",
                        "team": "right",
                        "position": {"x": px, "y": py},
                        "velocity": {"x": vx, "y": vy},
                    })

        ball_pos = {"x": obs[88] if len(obs) > 88 else 0.0, "y": obs[89] if len(obs) > 89 else 0.0, "z": obs[90] if len(obs) > 90 else 0.0}

        frame = {
            "tick": tick,
            "matchTimeSeconds": time_sec,
            "score": score,
            "ball": {"position": ball_pos},
            "players": players,
            "reward": step.get("reward", 0.0),
            "event": event_type,
        }
        frames.append(frame)

        if event_type:
            events.append({
                "id": f"evt_{tick}",
                "timeSeconds": time_sec,
                "type": event_type,
                "description": f"Match event: {event_type}",
            })

    output_data = {
        "metadata": metadata,
        "totalFrames": len(frames),
        "events": events,
        "frames": frames,
    }

    if output_path is None:
        output_path = input_path.replace(".jsonl", "_frames.json")

    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(output_data, out_f, indent=2)

    print(f"✓ Converted {len(frames)} frames to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .jsonl trace to replay frames JSON")
    parser.add_argument("input_trace", type=str, help="Path to input .jsonl file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    convert_trace(args.input_trace, args.output)
