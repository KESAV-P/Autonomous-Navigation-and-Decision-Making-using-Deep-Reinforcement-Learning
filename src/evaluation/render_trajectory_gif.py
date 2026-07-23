"""
Trajectory GIF Rendering Script.

Evaluates policies in RGB rendering mode and exports episode animations as GIFs
in media/gifs/ using imageio.
"""

import os
import argparse
import imageio
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from src.envs.make_env import make_env


def render_policy_gif(
    model,
    policy_name: str,
    env_id: str = "MiniGrid-MultiRoom-N4-S5-v0",
    seed: int = 1000,
    is_recurrent: bool = False,
    output_dir: str = "media/gifs",
    max_steps: int = 250
):
    os.makedirs(output_dir, exist_ok=True)
    # Use symbolic obs_mode for model predictions and render_mode='rgb_array' for visual frames
    env = make_env(env_id, seed=seed, obs_mode="symbolic", render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    frames = []
    # Capture initial frame
    frame = env.unwrapped.render()
    if frame is not None:
        frames.append(frame)

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    done = False
    step_count = 0
    ep_reward = 0.0

    while not done and step_count < max_steps:
        if model is None:
            # Random policy
            action = env.action_space.sample()
        elif is_recurrent:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=False
            )
            episode_starts[0] = False
        else:
            action, _ = model.predict(obs, deterministic=False)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += float(reward)
        step_count += 1
        done = terminated or truncated

        frame = env.unwrapped.render()
        if frame is not None:
            frames.append(frame)

    env.close()

    gif_filename = os.path.join(output_dir, f"{policy_name}_{env_id}.gif")
    if frames:
        imageio.mimsave(gif_filename, frames, fps=6)
        print(f"Saved trajectory GIF ({len(frames)} frames) to {gif_filename}")
    else:
        print(f"[WARN] No frames captured for {policy_name}")


def render_all_trajectories(
    mlp_checkpoint: str = "checkpoints/baseline_ppo.zip",
    recurrent_checkpoint: str = "checkpoints/curriculum_stage_3_MiniGrid-MultiRoom-N4-S5-v0.zip",
    env_id: str = "MiniGrid-MultiRoom-N4-S5-v0",
    seed: int = 1000
):
    print(f"\n--- Generating Trajectory GIFs for {env_id} (Seed {seed}) ---")

    # 1. Random Policy
    print("Rendering Random Policy trajectory...")
    render_policy_gif(None, "random_policy", env_id=env_id, seed=seed)

    # 2. MLP-PPO Policy
    if os.path.exists(mlp_checkpoint):
        print(f"Rendering MLP-PPO Policy from {mlp_checkpoint}...")
        mlp_model = PPO.load(mlp_checkpoint)
        render_policy_gif(mlp_model, "mlp_ppo_policy", env_id=env_id, seed=seed, is_recurrent=False)

    # 3. Recurrent CNN-LSTM PPO Policy
    if os.path.exists(recurrent_checkpoint):
        print(f"Rendering CNN-LSTM-PPO Policy from {recurrent_checkpoint}...")
        recurrent_model = RecurrentPPO.load(recurrent_checkpoint)
        render_policy_gif(recurrent_model, "recurrent_cnn_lstm_policy", env_id=env_id, seed=seed, is_recurrent=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render trajectory GIFs")
    parser.add_argument("--mlp_checkpoint", type=str, default="checkpoints/baseline_ppo.zip")
    parser.add_argument("--recurrent_checkpoint", type=str, default="checkpoints/curriculum_stage_3_MiniGrid-MultiRoom-N4-S5-v0.zip")
    parser.add_argument("--env_id", type=str, default="MiniGrid-MultiRoom-N4-S5-v0")
    parser.add_argument("--seed", type=int, default=1000)

    args = parser.parse_args()
    render_all_trajectories(
        mlp_checkpoint=args.mlp_checkpoint,
        recurrent_checkpoint=args.recurrent_checkpoint,
        env_id=args.env_id,
        seed=args.seed
    )
