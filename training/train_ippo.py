import sys
import os
import time

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import supersuit as ss
from stable_baselines3 import PPO
from training.gmn_pettingzoo import GMNMultiAgentEnv


def run_ippo_smoke_test(timesteps: int = 3000):
    print("==================================================")
    print("GMN FOOTBALL — INDEPENDENT PPO (IPPO) SMOKE TEST")
    print(f"Target Scenario: academy_3_vs_1_with_keeper | Timesteps: {timesteps}")
    print("Architecture: Parameter-Sharing IPPO (SuperSuit + SB3)")
    print("==================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("\n1. Initializing Multi-Agent PettingZoo Environment & SuperSuit Vectorization...")
    pz_env = GMNMultiAgentEnv(scenario="academy_3_vs_1_with_keeper", auto_start_bridge=True)
    print(f"   Controllable Agents: {pz_env.possible_agents}")

    # Vectorize the 3-agent ParallelEnv into an SB3-compatible VecEnv where each agent is 1 sub-env
    vec_env = ss.pettingzoo_env_to_vec_env_v1(pz_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    print(f"   SuperSuit VecEnv created: {vec_env.num_envs} vectorized sub-environments (sharing 1 policy)")

    # SB3 v2.x / SuperSuit 3.9.x compatibility adapter:
    # ConcatVecEnv adheres to Gymnasium (reset(seed=...)) and omits deprecated .seed() method;
    # attach adapter so SB3's set_random_seed() completes without AttributeError.
    if hasattr(vec_env, "venv") and not hasattr(vec_env.venv, "seed"):
        vec_env.venv.seed = lambda seed=None: [None] * vec_env.num_envs

    try:
        print("\n2. Configuring IPPO Model (MlpPolicy, gamma=0.99, n_steps=256, batch_size=64)...")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=None,
            seed=42,
        )

        print(f"\n3. Starting Multi-Agent IPPO Training for {timesteps} steps...")
        start_time = time.time()
        model.learn(total_timesteps=timesteps)
        duration = time.time() - start_time
        fps = timesteps / max(0.001, duration)

        print(f"\n   ✓ IPPO Training completed in {duration:.2f}s ({fps:.1f} steps/sec)")

        # Save model checkpoint
        checkpoint_path = os.path.join(models_dir, "ippo_academy_3_vs_1_with_keeper_smoke.zip")
        print(f"\n4. Saving Multi-Agent Model Checkpoint to: {checkpoint_path}...")
        model.save(checkpoint_path)
        print("   ✓ Checkpoint saved successfully.")

        # Verify loading model
        print("\n5. Testing Model Loading from Checkpoint...")
        loaded_model = PPO.load(checkpoint_path)
        print("   ✓ Checkpoint loaded successfully into memory.")

        # Run direct PettingZoo evaluation without SuperSuit wrapper
        print("\n6. Running Direct Multi-Agent PettingZoo Evaluation (200 steps)...")
        eval_pz_env = GMNMultiAgentEnv(scenario="academy_3_vs_1_with_keeper", auto_start_bridge=True)
        try:
            obs_dict, info_dict = eval_pz_env.reset(seed=100)
            total_team_reward = 0.0
            step_count = 0
            episodes_completed = 0

            for _ in range(200):
                if not eval_pz_env.agents:
                    episodes_completed += 1
                    obs_dict, info_dict = eval_pz_env.reset(seed=100 + episodes_completed)

                # Predict independently for each agent using the shared policy network
                actions = {
                    agent: int(loaded_model.predict(obs_dict[agent], deterministic=True)[0])
                    for agent in eval_pz_env.agents
                }

                obs_dict, rewards, terminations, truncations, infos = eval_pz_env.step(actions)
                step_count += 1

                if eval_pz_env.possible_agents and eval_pz_env.possible_agents[0] in rewards:
                    total_team_reward += rewards[eval_pz_env.possible_agents[0]]

            print(
                f"   ✓ Multi-Agent Evaluation Rollout completed: {step_count} steps across {episodes_completed + 1} episode(s)"
            )
            print(f"   ✓ Cumulative Team Reward: {total_team_reward:+.4f}")
            print("\n==================================================")
            print("RESULT: IPPO MULTI-AGENT SMOKE TEST SUCCESSFUL")
            print("==================================================")
            return True
        finally:
            eval_pz_env.close()

    finally:
        vec_env.close()
        pz_env.close()


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    success = run_ippo_smoke_test(steps)
    sys.exit(0 if success else 1)
