"""
Quick evaluation script for trained RL policy checkpoints.

Loads a given PPO or RecurrentPPO checkpoint, runs evaluation episodes across a fixed seed range,
and computes mean reward, success rate, mean steps to goal, and mean collisions.
"""

import sys
from pathlib import Path
# Add repo root directory to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import numpy as np
import yaml
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from src.envs.make_env import make_env


def evaluate_checkpoint(
    checkpoint_path: str,
    env_id: str = "MiniGrid-Empty-8x8-v0",
    n_episodes: int = 20,
    start_seed: int = 1000,
    is_recurrent: bool = False,
    obs_mode: str = "symbolic",
    deterministic: bool = False,
    verbose: bool = True
) -> dict:
    """
    Evaluates a saved checkpoint on a held-out set of environment seeds.

    Args:
        checkpoint_path: Path to the .zip model checkpoint.
        env_id: MiniGrid environment ID.
        n_episodes: Number of evaluation episodes.
        start_seed: Starting random seed for held-out evaluation episodes.
        is_recurrent: Whether the model uses RecurrentPPO (LSTM).
        obs_mode: 'symbolic' or 'rgb'.
        deterministic: Whether to use deterministic action selection.
        verbose: If True, prints evaluation progress and summary.

    Returns:
        dict containing mean_reward, success_rate, mean_steps, mean_collisions.
    """
    if is_recurrent:
        model = RecurrentPPO.load(checkpoint_path)
    else:
        model = PPO.load(checkpoint_path)

    rewards = []
    successes = []
    steps_list = []
    collisions_list = []

    for ep in range(n_episodes):
        seed = start_seed + ep
        env = make_env(env_id, seed=seed, obs_mode=obs_mode)
        obs, info = env.reset(seed=seed)
        
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        done = False
        ep_reward = 0.0
        step_count = 0

        while not done:
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=deterministic
                )
                episode_starts[0] = False
            else:
                action, _ = model.predict(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            step_count += 1
            done = terminated or truncated

        collisions = info.get('collisions', 0)
        env.close()

        rewards.append(ep_reward)
        successes.append(1.0 if ep_reward > 0 else 0.0)
        steps_list.append(step_count)
        collisions_list.append(collisions)

    metrics = {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'success_rate': float(np.mean(successes)),
        'mean_steps': float(np.mean(steps_list)),
        'mean_collisions': float(np.mean(collisions_list))
    }

    if verbose:
        print(f"\n--- Evaluation Results for {checkpoint_path} on {env_id} ---")
        print(f"Episodes Evaluated : {n_episodes} (Seeds {start_seed}..{start_seed + n_episodes - 1})")
        print(f"Deterministic      : {deterministic}")
        print(f"Success Rate       : {metrics['success_rate'] * 100:.1f}%")
        print(f"Mean Reward        : {metrics['mean_reward']:.4f} +/- {metrics['std_reward']:.4f}")
        print(f"Mean Steps to Goal : {metrics['mean_steps']:.1f}")
        print(f"Mean Collisions    : {metrics['mean_collisions']:.1f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .zip file")
    parser.add_argument("--env_id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--is_recurrent", action="store_true")
    parser.add_argument("--obs_mode", type=str, default="symbolic")
    parser.add_argument("--deterministic", action="store_true")

    args = parser.parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        env_id=args.env_id,
        n_episodes=args.n_episodes,
        start_seed=args.start_seed,
        is_recurrent=args.is_recurrent,
        obs_mode=args.obs_mode,
        deterministic=args.deterministic
    )
