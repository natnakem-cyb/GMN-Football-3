"""
Verification and round-trip unit/integration tests for EpisodeRecorder in GMN-Football-3.
"""

import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.episode_recorder import EpisodeRecorder, load_episode_trace


class TestEpisodeRecorder(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="gmn_test_replays_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_multi_agent_recorder_format_and_roundtrip(self):
        scenario = "academy_3_vs_1_with_keeper"
        seed = 500000
        agent_ids = ["left_1", "left_2", "left_3"]
        checkpoint = "mappo_real_verified_run.pt"

        recorder = EpisodeRecorder(
            scenario=scenario,
            seed=seed,
            agent_ids=agent_ids,
            checkpoint=checkpoint,
            output_dir=self.test_dir,
        )

        num_steps = 20
        dummy_obs = [float(i) * 0.01 for i in range(115)]

        for tick in range(num_steps):
            actions = {"left_1": 5, "left_2": 3, "left_3": 12}
            observations = {a: dummy_obs for a in agent_ids}
            reward = 0.05 if tick < num_steps - 1 else 1.0
            terminated = tick == num_steps - 1
            truncated = False
            score = {"left": 1 if terminated else 0, "right": 0}
            event = "goal" if terminated else None

            recorder.record_step(
                tick=tick,
                actions=actions,
                observations=observations,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                score=score,
                event=event,
            )

        filepath = recorder.close()
        self.assertTrue(os.path.exists(filepath))

        # Inspect raw file lines
        with open(filepath, "r", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(raw_lines), num_steps + 1)

        # Load with load_episode_trace
        metadata, steps = load_episode_trace(filepath)

        # Validate metadata
        self.assertEqual(metadata["type"], "metadata")
        self.assertEqual(metadata["scenario"], scenario)
        self.assertEqual(metadata["seed"], seed)
        self.assertEqual(metadata["agent_ids"], agent_ids)
        self.assertEqual(metadata["checkpoint"], checkpoint)
        self.assertIn("recorded_at", metadata)

        # Validate steps
        self.assertEqual(len(steps), num_steps)
        for idx, step in enumerate(steps):
            self.assertEqual(step["tick"], idx)
            self.assertEqual(set(step["actions"].keys()), set(agent_ids))
            for a, act in step["actions"].items():
                self.assertIsInstance(act, int)
                self.assertTrue(0 <= act <= 18)
            
            self.assertEqual(set(step["observations"].keys()), set(agent_ids))
            for a, obs in step["observations"].items():
                self.assertEqual(len(obs), 115)
                for val in obs:
                    self.assertIsInstance(val, (float, int))

            self.assertIsInstance(step["reward"], float)
            self.assertIsInstance(step["terminated"], bool)
            self.assertIsInstance(step["truncated"], bool)
            self.assertIn("left", step["score"])
            self.assertIn("right", step["score"])

        self.assertTrue(steps[-1]["terminated"])
        self.assertEqual(steps[-1]["score"]["left"], 1)
        self.assertEqual(steps[-1]["event"], "goal")

    def test_single_agent_recorder_and_truncation(self):
        scenario = "academy_empty_goal"
        seed = 42
        agent_ids = ["left_1"]
        checkpoint = "ppo_empty_goal.zip"

        recorder = EpisodeRecorder(
            scenario=scenario,
            seed=seed,
            agent_ids=agent_ids,
            checkpoint=checkpoint,
            output_dir=self.test_dir,
        )

        num_steps = 10
        dummy_obs = [0.0] * 115

        for tick in range(num_steps):
            actions = {"left_1": 5}
            observations = {"left_1": dummy_obs}
            reward = 0.0
            terminated = False
            truncated = tick == num_steps - 1
            score = {"left": 0, "right": 0}
            event = None

            recorder.record_step(
                tick=tick,
                actions=actions,
                observations=observations,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                score=score,
                event=event,
            )

        filepath = recorder.close()
        metadata, steps = load_episode_trace(filepath)

        self.assertEqual(len(steps), num_steps)
        self.assertEqual(metadata["agent_ids"], ["left_1"])
        self.assertFalse(steps[-1]["terminated"])
        self.assertTrue(steps[-1]["truncated"])
        self.assertIsNone(steps[-1]["event"])

    def test_collision_avoidance(self):
        # Two recorders started with exact same parameters in the same directory
        rec1 = EpisodeRecorder("academy_empty_goal", 1234, ["left_1"], output_dir=self.test_dir)
        rec2 = EpisodeRecorder("academy_empty_goal", 1234, ["left_1"], output_dir=self.test_dir)
        p1 = rec1.close()
        p2 = rec2.close()
        self.assertNotEqual(p1, p2)
        self.assertTrue(os.path.exists(p1))
        self.assertTrue(os.path.exists(p2))


if __name__ == "__main__":
    unittest.main()
