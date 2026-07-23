"""
Live Visual Demonstration Script.

Renders a smooth Pygame live window showing the trained RL agent navigating
step-by-step in real time without window freezing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
import argparse
import numpy as np
import pygame
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from src.envs.make_env import make_env


def run_visual_demo(
    checkpoint_path: str,
    env_id: str = "MiniGrid-Empty-8x8-v0",
    is_recurrent: bool = False,
    obs_mode: str = "symbolic",
    use_shaping: bool = False,
    n_episodes: int = 3,
    delay: float = 0.1
):
    print(f"\n=======================================================")
    print(f"  Live Pygame Visual Demo: {checkpoint_path} on {env_id}")
    print(f"=======================================================")

    if is_recurrent:
        model = RecurrentPPO.load(checkpoint_path)
    else:
        model = PPO.load(checkpoint_path)

    # Initialize Pygame Display
    pygame.init()
    pygame.font.init()
    window_size = 640
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Autonomous Navigation Live Visual Demo (16x16 & 8x8 Grids)")
    font = pygame.font.SysFont("Helvetica", 22, bold=True)

    for ep in range(n_episodes):
        seed = 1000 + ep
        env = make_env(env_id, seed=seed, obs_mode=obs_mode, render_mode="rgb_array", use_shaping=use_shaping)
        obs, info = env.reset(seed=seed)
        
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        ep_reward = 0.0
        step_count = 0

        print(f"\nEpisode {ep + 1}/{n_episodes} (Seed {seed})...")

        while not done:
            # Handle Pygame window events to prevent window freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Exiting visual demo...")
                    env.close()
                    pygame.quit()
                    return

            # Capture RGB frame
            frame = env.unwrapped.render()
            if frame is not None:
                # Transpose for Pygame surface (Width, Height, Channel)
                surf_frame = np.transpose(frame, (1, 0, 2))
                surface = pygame.surfarray.make_surface(surf_frame)
                scaled_surface = pygame.transform.scale(surface, (window_size, window_size))
                screen.blit(scaled_surface, (0, 0))

                # Render HUD text overlays
                txt_env = font.render(f"Env: {env_id}", True, (255, 255, 255))
                txt_step = font.render(f"Step: {step_count}", True, (255, 255, 0))
                txt_rew = font.render(f"Reward: {ep_reward:.2f}", True, (0, 255, 0))
                
                screen.blit(txt_env, (15, 15))
                screen.blit(txt_step, (15, 45))
                screen.blit(txt_rew, (15, 75))

                pygame.display.flip()
                pygame.time.delay(int(delay * 1000))

            if is_recurrent:
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

        print(f"Episode {ep + 1} finished in {step_count} steps | Final Reward: {ep_reward:.4f}")
        pygame.time.delay(500)
        env.close()

    pygame.quit()
    print("\nVisual demo complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Live Visual Navigation Demo")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cnn_ppo.zip")
    parser.add_argument("--env_id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--is_recurrent", action="store_true")
    parser.add_argument("--use_shaping", action="store_true")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.08, help="Delay between steps in seconds")

    args = parser.parse_args()
    run_visual_demo(
        checkpoint_path=args.checkpoint,
        env_id=args.env_id,
        is_recurrent=args.is_recurrent,
        use_shaping=args.use_shaping,
        n_episodes=args.n_episodes,
        delay=args.delay
    )
