"""
GMN-Football-3 — PettingZoo Multi-Agent Wrapper Test & Validation Suite
Validates GMNMultiAgentEnv against the PettingZoo ParallelEnv specification,
verifying agent discovery, observation/action shapes, stable agent ordering,
and multi-agent stepping.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv, OBSERVATION_DIM, ACTION_SPACE_SIZE


def test_pettingzoo_wrapper():
    print("========================================================================")
    print("GMN-FOOTBALL-3 — PETTINGZOO MULTI-AGENT WRAPPER VALIDATION")
    print("Scenario: academy_3_vs_1_with_keeper (3 controllable left agents)")
    print("========================================================================")

    env = GMNMultiAgentEnv(scenario="academy_3_vs_1_with_keeper", auto_start_bridge=True)

    try:
        # 1. Check initial agents & spaces
        print(f"Possible Agents: {env.possible_agents}")
        print(f"Active Agents: {env.agents}")
        assert len(env.possible_agents) == 3, f"Expected 3 possible agents, got {len(env.possible_agents)}"
        assert env.agents == env.possible_agents, "Active agents should match possible agents initially"

        for agent in env.possible_agents:
            obs_space = env.observation_space(agent)
            act_space = env.action_space(agent)
            assert obs_space.shape == (OBSERVATION_DIM,), f"Agent {agent} obs space shape {obs_space.shape} != (115,)"
            assert act_space.n == ACTION_SPACE_SIZE, f"Agent {agent} act space size {act_space.n} != 19"
        print("✓ Observation and Action spaces verified for all agents.")

        # 2. Reset stability and observation shapes across multiple resets
        initial_order = list(env.possible_agents)
        for reset_i in range(5):
            obs, info = env.reset(seed=1000 + reset_i)
            assert env.possible_agents == initial_order, f"Agent order unstable: {env.possible_agents} != {initial_order}"
            assert list(obs.keys()) == initial_order, f"Observation keys {list(obs.keys())} != {initial_order}"
            assert list(info.keys()) == initial_order, f"Info keys {list(info.keys())} != {initial_order}"

            for agent in initial_order:
                agent_obs = obs[agent]
                assert isinstance(agent_obs, np.ndarray), f"Observation for {agent} is not numpy array"
                assert agent_obs.shape == (OBSERVATION_DIM,), f"Observation shape {agent_obs.shape} != (115,)"
                assert agent_obs.dtype == np.float32, f"Observation dtype {agent_obs.dtype} != float32"
                assert not np.isnan(agent_obs).any(), f"Observation for {agent} contains NaNs"
                assert not np.isinf(agent_obs).any(), f"Observation for {agent} contains Infs"

        print(f"✓ Agent ordering stable across 5 resets: {initial_order}")
        print(f"✓ Reset observations verified (shape: ({OBSERVATION_DIM},), dtype: float32, no NaNs/Infs).")

        # 3. Step execution for 100 ticks with random per-agent actions
        print("\nRunning 100 multi-agent steps with random actions...")
        obs, info = env.reset(seed=42)
        total_reward = 0.0
        step_count = 0

        for step_idx in range(100):
            if not env.agents:
                print(f"Episode terminated at step {step_count}. Resetting...")
                obs, info = env.reset(seed=42 + step_idx)

            active_agents = list(env.agents)
            actions = {agent: int(np.random.randint(0, ACTION_SPACE_SIZE)) for agent in active_agents}

            obs, rews, terms, truncs, infos = env.step(actions)
            step_count += 1

            # Validate all returned dictionaries
            assert set(obs.keys()) == set(active_agents), f"Step {step_idx}: obs keys {set(obs.keys())} != {set(active_agents)}"
            assert set(rews.keys()) == set(active_agents), f"Step {step_idx}: rews keys {set(rews.keys())} != {set(active_agents)}"
            assert set(terms.keys()) == set(active_agents), f"Step {step_idx}: terms keys {set(terms.keys())} != {set(active_agents)}"
            assert set(truncs.keys()) == set(active_agents), f"Step {step_idx}: truncs keys {set(truncs.keys())} != {set(active_agents)}"
            assert set(infos.keys()) == set(active_agents), f"Step {step_idx}: infos keys {set(infos.keys())} != {set(active_agents)}"

            for agent in active_agents:
                assert obs[agent].shape == (OBSERVATION_DIM,)
                assert isinstance(rews[agent], float)
                assert isinstance(terms[agent], bool)
                assert isinstance(truncs[agent], bool)
                assert "score" in infos[agent]
                assert "checkpointReward" in infos[agent]
                assert "ballDistanceToGoal" in infos[agent]
                assert "eventCode" in infos[agent]

            total_reward += rews[active_agents[0]]

        print(f"✓ 100 multi-agent steps completed cleanly. Cumulative team reward: {total_reward:.4f}")
        print("\n========================================================================")
        print("✓ PETTINGZOO MULTI-AGENT ENVIRONMENT VALIDATION PASSED COMPLETELY!")
        print("========================================================================")

    finally:
        env.close()


if __name__ == "__main__":
    test_pettingzoo_wrapper()
