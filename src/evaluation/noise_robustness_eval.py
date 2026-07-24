"""
Noise Robustness Benchmark Script.

Evaluates RecurrentPPO (CNN + LSTM) vs standard PPO (CNN-only) across a range of
observation noise levels (sigma = 0.0 to 1.0) to measure temporal memory resilience
against sensor degradation.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from src.envs.make_env import make_env


def evaluate_under_noise(
    checkpoint_path: str,
    env_id: str,
    is_recurrent: bool,
    sigmas: list,
    n_episodes: int = 20,
    start_seed: int = 5000
):
    """Evaluates policy across multiple noise levels."""
    if is_recurrent:
        model = RecurrentPPO.load(checkpoint_path)
    else:
        model = PPO.load(checkpoint_path)

    results = []

    for sigma in sigmas:
        successes = []
        steps = []
        collisions = []

        for ep in range(n_episodes):
            seed = start_seed + ep
            env = make_env(env_id, seed=seed, obs_mode="symbolic", noise_sigma=sigma, use_shaping=False)
            obs, info = env.reset(seed=seed)

            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            done = False
            step_count = 0

            while not done:
                if is_recurrent:
                    action, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True
                    )
                    episode_starts[0] = False
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                done = terminated or truncated

            is_success = bool(terminated and reward > 0)
            successes.append(is_success)
            steps.append(step_count)
            collisions.append(info.get("collisions", 0))
            env.close()

        success_rate = float(np.mean(successes) * 100.0)
        mean_steps = float(np.mean(steps))
        mean_collisions = float(np.mean(collisions))

        results.append({
            "sigma": sigma,
            "success_rate": success_rate,
            "mean_steps": mean_steps,
            "mean_collisions": mean_collisions
        })

        print(f"  Sigma = {sigma:.2f} | Success Rate: {success_rate:5.1f}% | Avg Steps: {mean_steps:5.1f} | Collisions: {mean_collisions:.1f}")

    return results


def run_noise_benchmark(
    lstm_checkpoint: str = "checkpoints/complex_doorkey_agent.zip",
    cnn_checkpoint: str = "checkpoints/cnn_ppo.zip",
    env_id: str = "MiniGrid-DoorKey-8x8-v0",
    n_episodes: int = 15
):
    print("\n=======================================================")
    print(f"  Running Noise Robustness Benchmark on {env_id}")
    print("=======================================================")

    sigmas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    print("\n[1/2] Evaluating RecurrentPPO (CNN + LSTM)...")
    lstm_res = evaluate_under_noise(lstm_checkpoint, env_id, is_recurrent=True, sigmas=sigmas, n_episodes=n_episodes)

    print("\n[2/2] Evaluating Standard PPO (CNN Only)...")
    cnn_res = evaluate_under_noise(cnn_checkpoint, env_id, is_recurrent=False, sigmas=sigmas, n_episodes=n_episodes)

    # Plotting
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    lstm_success = [r["success_rate"] for r in lstm_res]
    cnn_success = [r["success_rate"] for r in cnn_res]

    plt.plot(sigmas, lstm_success, 'o-', color='#1f77b4', linewidth=2.5, label='RecurrentPPO (CNN + LSTM Memory)')
    plt.plot(sigmas, cnn_success, 's--', color='#ff7f0e', linewidth=2.5, label='Standard PPO (CNN Only)')

    plt.title(f'Sensor Noise Degradation Robustness Benchmark ({env_id})', fontsize=14, fontweight='bold')
    plt.xlabel('Sensor Noise Level (Gaussian σ)', fontsize=12)
    plt.ylabel('Navigation Success Rate (%)', fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    chart_path = "reports/noise_robustness.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\nSaved benchmark plot to {chart_path}")

    # Generate Markdown Report
    report_path = "reports/noise_robustness_eval.md"
    with open(report_path, "w") as f:
        f.write(f"# Noise Degradation Robustness Benchmark Report\n\n")
        f.write(f"**Environment:** `{env_id}` | **Episodes per noise level:** {n_episodes}\n\n")
        f.write(f"| Noise Level (σ) | RecurrentPPO (LSTM) Success | Standard PPO (CNN) Success | LSTM Advantage |\n")
        f.write(f"|---|---|---|---|\n")
        for r_lstm, r_cnn in zip(lstm_res, cnn_res):
            diff = r_lstm["success_rate"] - r_cnn["success_rate"]
            f.write(f"| {r_lstm['sigma']:.2f} | {r_lstm['success_rate']:5.1f}% | {r_cnn['success_rate']:5.1f}% | **+{diff:+.1f}%** |\n")
    print(f"Saved benchmark report to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise Robustness Benchmark")
    parser.add_argument("--lstm_ckpt", type=str, default="checkpoints/complex_doorkey_agent.zip")
    parser.add_argument("--cnn_ckpt", type=str, default="checkpoints/cnn_ppo.zip")
    parser.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--episodes", type=int, default=15)

    args = parser.parse_args()
    run_noise_benchmark(
        lstm_checkpoint=args.lstm_ckpt,
        cnn_checkpoint=args.cnn_ckpt,
        env_id=args.env_id,
        n_episodes=args.episodes
    )
