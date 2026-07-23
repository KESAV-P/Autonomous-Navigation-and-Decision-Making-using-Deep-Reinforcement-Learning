"""
Learning Progression GIF Generator.

Renders and exports 3 trajectory animations comparing:
1. Untrained / Random Agent (Stage 1)
2. Partially Trained Agent (Stage 2)
3. Mastered / Current Checkpoint Agent (Stage 3)
on MiniGrid-DoorKey-8x8-v0.
"""

import os
import imageio
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from src.envs.make_env import make_env


def render_progression_gifs(
    checkpoint_path: str = "checkpoints/complex_doorkey_agent.zip",
    env_id: str = "MiniGrid-DoorKey-8x8-v0",
    seed: int = 42,
    output_dir: str = "media/gifs"
):
    os.makedirs(output_dir, exist_ok=True)

    # Stage 1: Untrained / Random Agent
    print("Generating Stage 1 (Untrained Random Agent) GIF...")
    env = make_env(env_id, seed=seed, obs_mode="symbolic", render_mode="rgb_array", use_shaping=False)
    obs, info = env.reset(seed=seed)
    frames_s1 = [env.unwrapped.render()]
    done = False
    step = 0
    while not done and step < 60:
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        frames_s1.append(env.unwrapped.render())
        done = term or trunc
        step += 1
    env.close()
    s1_path = os.path.join(output_dir, "learning_stage1_untrained.gif")
    imageio.mimsave(s1_path, frames_s1, fps=5)
    print(f"Saved {s1_path} ({len(frames_s1)} frames)")

    # Stage 2: Mid-Training (Baseline PPO on DoorKey)
    print("Generating Stage 2 (Mid-Training Agent) GIF...")
    env = make_env(env_id, seed=seed, obs_mode="symbolic", render_mode="rgb_array", use_shaping=False)
    obs, info = env.reset(seed=seed)
    frames_s2 = [env.unwrapped.render()]
    done = False
    step = 0
    if os.path.exists("checkpoints/baseline_ppo.zip"):
        model_s2 = PPO.load("checkpoints/baseline_ppo.zip")
    else:
        model_s2 = None

    while not done and step < 60:
        if model_s2 is not None:
            action, _ = model_s2.predict(obs, deterministic=False)
        else:
            action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        frames_s2.append(env.unwrapped.render())
        done = term or trunc
        step += 1
    env.close()
    s2_path = os.path.join(output_dir, "learning_stage2_midtraining.gif")
    imageio.mimsave(s2_path, frames_s2, fps=5)
    print(f"Saved {s2_path} ({len(frames_s2)} frames)")

    # Stage 3: Mastered Recurrent Agent
    print("Generating Stage 3 (Mastered Agent) GIF...")
    if os.path.exists(checkpoint_path):
        model_s3 = RecurrentPPO.load(checkpoint_path)
        env = make_env(env_id, seed=seed, obs_mode="symbolic", render_mode="rgb_array", use_shaping=False)
        obs, info = env.reset(seed=seed)
        frames_s3 = [env.unwrapped.render()]
        done = False
        step = 0
        lstm_states = None
        ep_starts = np.ones((1,), dtype=bool)

        while not done and step < 60:
            action, lstm_states = model_s3.predict(
                obs,
                state=lstm_states,
                episode_start=ep_starts,
                deterministic=False
            )
            ep_starts[0] = False
            obs, reward, term, trunc, info = env.step(action)
            frames_s3.append(env.unwrapped.render())
            done = term or trunc
            step += 1
        env.close()
        s3_path = os.path.join(output_dir, "learning_stage3_mastered.gif")
        imageio.mimsave(s3_path, frames_s3, fps=5)
        print(f"Saved {s3_path} ({len(frames_s3)} frames)")
    else:
        print(f"[WARN] Checkpoint {checkpoint_path} not found yet.")


if __name__ == "__main__":
    render_progression_gifs()
