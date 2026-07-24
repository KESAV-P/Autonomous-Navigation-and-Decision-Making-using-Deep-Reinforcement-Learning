"""
Live Visual Demonstration Script with Enhanced Telemetry HUD.

Renders a smooth Pygame live window showing the trained RL agent navigating
step-by-step with real-time telemetry: Mission State, Collisions, Sensor Noise (σ),
Step Count, and Accumulated Reward.
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
import src.envs.warehouse_env  # Register Warehouse-Navigate-v0
from src.envs.make_env import make_env


def run_visual_demo(
    checkpoint_path: str,
    env_id: str = "MiniGrid-DoorKey-8x8-v0",
    is_recurrent: bool = False,
    obs_mode: str = "symbolic",
    use_shaping: bool = False,
    noise_sigma: float = 0.0,
    n_episodes: int = 3,
    delay: float = 0.08
):
    print(f"\n=======================================================")
    print(f"  Live Telemetry Visual Demo: {checkpoint_path}")
    print(f"  Env: {env_id} | Recurrent: {is_recurrent} | Noise σ: {noise_sigma}")
    print(f"=======================================================")

    if is_recurrent:
        model = RecurrentPPO.load(checkpoint_path)
    else:
        model = PPO.load(checkpoint_path)

    # Initialize Pygame Display (Canvas + Telemetry Side Panel)
    pygame.init()
    pygame.font.init()
    
    grid_render_size = 540
    hud_width = 280
    total_width = grid_render_size + hud_width
    total_height = grid_render_size

    screen = pygame.display.set_mode((total_width, total_height))
    pygame.display.set_caption("Autonomous Navigation Live Telemetry & Mission Control")

    font_title = pygame.font.SysFont("Helvetica", 20, bold=True)
    font_body = pygame.font.SysFont("Helvetica", 16, bold=False)
    font_bold = pygame.font.SysFont("Helvetica", 16, bold=True)

    for ep in range(n_episodes):
        seed = 1000 + ep
        env = make_env(
            env_id,
            seed=seed,
            obs_mode=obs_mode,
            render_mode="rgb_array",
            use_shaping=use_shaping,
            noise_sigma=noise_sigma
        )
        obs, info = env.reset(seed=seed)

        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        ep_reward = 0.0
        step_count = 0

        print(f"\nEpisode {ep + 1}/{n_episodes} (Seed {seed})...")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Exiting visual demo...")
                    env.close()
                    pygame.quit()
                    return

            frame = env.unwrapped.render()
            if frame is not None:
                screen.fill((20, 24, 30))  # Dark sleek dashboard background

                # 1. Render Grid Frame
                surf_frame = np.transpose(frame, (1, 0, 2))
                surface = pygame.surfarray.make_surface(surf_frame)
                scaled_surface = pygame.transform.scale(surface, (grid_render_size, grid_render_size))
                screen.blit(scaled_surface, (0, 0))

                # 2. Render Telemetry HUD Side Panel
                panel_x = grid_render_size + 15
                
                # Header
                txt = font_title.render("TELEMETRY PANEL", True, (0, 220, 255))
                screen.blit(txt, (panel_x, 15))

                # Divider line
                pygame.draw.line(screen, (50, 60, 80), (panel_x, 45), (total_width - 15, 45), 2)

                # Info items
                y_offset = 60

                # Model Type
                model_str = "RecurrentPPO (CNN+LSTM)" if is_recurrent else "Standard PPO (CNN)"
                t_mod_lbl = font_bold.render("Agent Architecture:", True, (180, 190, 200))
                t_mod_val = font_body.render(model_str, True, (255, 255, 255))
                screen.blit(t_mod_lbl, (panel_x, y_offset))
                screen.blit(t_mod_val, (panel_x, y_offset + 22))
                y_offset += 55

                # Environment ID
                t_env_lbl = font_bold.render("Target Environment:", True, (180, 190, 200))
                t_env_val = font_body.render(env_id, True, (255, 255, 255))
                screen.blit(t_env_lbl, (panel_x, y_offset))
                screen.blit(t_env_val, (panel_x, y_offset + 22))
                y_offset += 55

                # Mission Status
                has_cargo = info.get("has_cargo", False)
                cargo_deliv = info.get("cargo_delivered", False)
                if cargo_deliv:
                    status_str = "DELIVERED TO DOCK"
                    status_color = (0, 255, 120)
                elif has_cargo:
                    status_str = "CARGO PICKED UP"
                    status_color = (255, 220, 0)
                else:
                    status_str = "SEARCHING / NAVIGATING"
                    status_color = (100, 200, 255)

                t_stat_lbl = font_bold.render("Mission Status:", True, (180, 190, 200))
                t_stat_val = font_bold.render(status_str, True, status_color)
                screen.blit(t_stat_lbl, (panel_x, y_offset))
                screen.blit(t_stat_val, (panel_x, y_offset + 22))
                y_offset += 55

                # Metrics: Step, Reward, Collisions
                t_step = font_body.render(f"Elapsed Steps: {step_count}", True, (240, 240, 240))
                t_rew = font_body.render(f"Total Reward: {ep_reward:+.3f}", True, (0, 240, 100))
                t_col = font_body.render(f"Collisions: {info.get('collisions', 0)}", True, (255, 100, 100))
                screen.blit(t_step, (panel_x, y_offset))
                screen.blit(t_rew, (panel_x, y_offset + 25))
                screen.blit(t_col, (panel_x, y_offset + 50))
                y_offset += 85

                # Sensor Noise Indicator Bar
                t_noise_lbl = font_bold.render(f"Sensor Noise (σ = {noise_sigma:.2f}):", True, (180, 190, 200))
                screen.blit(t_noise_lbl, (panel_x, y_offset))
                
                # Draw bar
                bar_x = panel_x
                bar_y = y_offset + 25
                bar_w = 230
                bar_h = 16
                fill_w = int(bar_w * min(noise_sigma / 1.0, 1.0))
                
                pygame.draw.rect(screen, (40, 45, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
                if fill_w > 0:
                    bar_color = (0, 200, 255) if noise_sigma <= 0.2 else (255, 160, 0) if noise_sigma <= 0.5 else (255, 50, 50)
                    pygame.draw.rect(screen, bar_color, (bar_x, bar_y, fill_w, bar_h), border_radius=4)

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
    parser = argparse.ArgumentParser(description="Run Live Telemetry Navigation Demo")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/complex_doorkey_agent.zip")
    parser.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--is_recurrent", action="store_true")
    parser.add_argument("--use_shaping", action="store_true")
    parser.add_argument("--noise_sigma", type=float, default=0.0)
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.08)

    args = parser.parse_args()
    run_visual_demo(
        checkpoint_path=args.checkpoint,
        env_id=args.env_id,
        is_recurrent=args.is_recurrent,
        use_shaping=args.use_shaping,
        noise_sigma=args.noise_sigma,
        n_episodes=args.n_episodes,
        delay=args.delay
    )
