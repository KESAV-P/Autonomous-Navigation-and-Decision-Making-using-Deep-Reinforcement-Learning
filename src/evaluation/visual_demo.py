"""
Live Visual Demonstration Script.

Renders a live window (Pygame/OpenCV) showing the trained RL agent navigating
step-by-step in real time.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
import argparse
import numpy as np
import cv2
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
    print(f"  Live Visual Demo: {checkpoint_path} on {env_id}")
    print(f"=======================================================")

    if is_recurrent:
        model = RecurrentPPO.load(checkpoint_path)
    else:
        model = PPO.load(checkpoint_path)

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
            # Capture RGB visual frame
            frame = env.unwrapped.render()
            if frame is not None:
                # Display in OpenCV window
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Resize for high resolution display window
                display_frame = cv2.resize(bgr_frame, (512, 512), interpolation=cv2.INTER_NEAREST)
                
                # Add overlay text
                cv2.putText(display_frame, f"Env: {env_id}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Step: {step_count}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Reward: {ep_reward:.2f}", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Autonomous Navigation Live Visual Demo", display_frame)
                key = cv2.waitKey(int(delay * 1000))
                if key == 27 or key == ord('q'): # Press ESC or q to exit
                    print("Exiting visual demo...")
                    env.close()
                    cv2.destroyAllWindows()
                    return

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
        time.sleep(0.5)
        env.close()

    cv2.destroyAllWindows()
    print("\nVisual demo complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Live Visual Navigation Demo")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cnn_ppo.zip")
    parser.add_argument("--env_id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--is_recurrent", action="store_true")
    parser.add_argument("--use_shaping", action="store_true")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps in seconds")

    args = parser.parse_args()
    run_visual_demo(
        checkpoint_path=args.checkpoint,
        env_id=args.env_id,
        is_recurrent=args.is_recurrent,
        use_shaping=args.use_shaping,
        n_episodes=args.n_episodes,
        delay=args.delay
    )
