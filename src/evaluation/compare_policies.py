"""
Comparative Policy Evaluation Script — Review 2 (Fixed).

Evaluates 4 policies across seen and held-out unseen map seeds:
1. Random Action Baseline
2. Non-Recurrent MLP-PPO Baseline
3. CNN Feature Extractor PPO (non-recurrent)
4. Recurrent CNN-LSTM PPO (per-environment best checkpoint)

Each environment uses the most appropriate trained checkpoint for CNN-LSTM.

Outputs results into reports/comparison_table.csv and reports/comparison_table.md.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import csv
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from src.envs.make_env import make_env


# Per-environment best checkpoint mapping for CNN-LSTM
RECURRENT_CHECKPOINT_MAP = {
    "MiniGrid-Empty-8x8-v0":           "checkpoints/recurrent_ppo_empty8x8.zip",
    "MiniGrid-DoorKey-8x8-v0":         "checkpoints/complex_doorkey_agent.zip",
    "MiniGrid-MultiRoom-N2-S4-v0":     "checkpoints/curriculum_stage_2_MiniGrid-MultiRoom-N2-S4-v0.zip",
    "MiniGrid-MultiRoom-N4-S5-v0":     "checkpoints/curriculum_stage_3_MiniGrid-MultiRoom-N4-S5-v0.zip",
}


def evaluate_random_policy(env_id: str, seeds: list, obs_mode: str = "symbolic") -> dict:
    rewards, successes, steps_list, collisions_list = [], [], [], []

    for seed in seeds:
        env = make_env(env_id, seed=seed, obs_mode=obs_mode)
        obs, info = env.reset(seed=seed)
        done = False
        ep_reward = 0.0
        step_count = 0

        while not done:
            action = env.action_space.sample()
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

    return {
        'mean_reward': float(np.mean(rewards)),
        'success_rate': float(np.mean(successes)),
        'mean_steps': float(np.mean(steps_list)),
        'mean_collisions': float(np.mean(collisions_list))
    }


def evaluate_trained_policy(model, env_id: str, seeds: list, is_recurrent: bool = False,
                             obs_mode: str = "symbolic") -> dict:
    rewards, successes, steps_list, collisions_list = [], [], [], []

    for seed in seeds:
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
                    deterministic=False
                )
                episode_starts[0] = False
            else:
                action, _ = model.predict(obs, deterministic=False)

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

    return {
        'mean_reward': float(np.mean(rewards)),
        'success_rate': float(np.mean(successes)),
        'mean_steps': float(np.mean(steps_list)),
        'mean_collisions': float(np.mean(collisions_list))
    }


def run_comparative_evaluation(
    mlp_checkpoint: str = "checkpoints/baseline_ppo.zip",
    cnn_checkpoint: str = "checkpoints/cnn_ppo.zip",
    environments: list = None,
    seen_seeds: list = None,
    unseen_seeds: list = None,
    output_csv: str = "reports/comparison_table.csv",
    output_md: str = "reports/comparison_table.md"
):
    if environments is None:
        environments = [
            "MiniGrid-Empty-8x8-v0",
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-MultiRoom-N2-S4-v0",
            "MiniGrid-MultiRoom-N4-S5-v0"
        ]

    if seen_seeds is None:
        seen_seeds = list(range(0, 20))
    if unseen_seeds is None:
        unseen_seeds = list(range(1000, 1020))

    # Load fixed policy models
    print(f"Loading MLP Baseline model from {mlp_checkpoint}...")
    mlp_model = PPO.load(mlp_checkpoint)

    print(f"Loading CNN-PPO model from {cnn_checkpoint}...")
    cnn_model = PPO.load(cnn_checkpoint)

    results = []

    for env_id in environments:
        print(f"\n--- Evaluating on Environment: {env_id} ---")

        # Load the per-environment best CNN-LSTM checkpoint
        rec_ckpt = RECURRENT_CHECKPOINT_MAP.get(env_id)
        print(f"Loading CNN-LSTM model from {rec_ckpt}...")
        recurrent_model = RecurrentPPO.load(rec_ckpt)

        seed_sets = [("Seen (0..19)", seen_seeds), ("Unseen (1000..1019)", unseen_seeds)]

        for seed_label, seeds in seed_sets:
            # 1. Random Baseline
            print(f"  Evaluating Random Policy on {seed_label}...")
            rand_metrics = evaluate_random_policy(env_id, seeds)
            results.append({
                "Environment": env_id,
                "Policy": "Random Baseline",
                "Evaluation Split": seed_label,
                "Success Rate (%)": f"{rand_metrics['success_rate'] * 100:.1f}",
                "Mean Reward": f"{rand_metrics['mean_reward']:.4f}",
                "Mean Steps": f"{rand_metrics['mean_steps']:.1f}",
                "Mean Collisions": f"{rand_metrics['mean_collisions']:.1f}"
            })

            # 2. MLP-PPO Baseline
            print(f"  Evaluating MLP-PPO Policy on {seed_label}...")
            mlp_metrics = evaluate_trained_policy(mlp_model, env_id, seeds, is_recurrent=False)
            results.append({
                "Environment": env_id,
                "Policy": "MLP-PPO Baseline",
                "Evaluation Split": seed_label,
                "Success Rate (%)": f"{mlp_metrics['success_rate'] * 100:.1f}",
                "Mean Reward": f"{mlp_metrics['mean_reward']:.4f}",
                "Mean Steps": f"{mlp_metrics['mean_steps']:.1f}",
                "Mean Collisions": f"{mlp_metrics['mean_collisions']:.1f}"
            })

            # 3. CNN-PPO (non-recurrent)
            print(f"  Evaluating CNN-PPO (non-recurrent) on {seed_label}...")
            cnn_metrics = evaluate_trained_policy(cnn_model, env_id, seeds, is_recurrent=False)
            results.append({
                "Environment": env_id,
                "Policy": "CNN-PPO (Extractor)",
                "Evaluation Split": seed_label,
                "Success Rate (%)": f"{cnn_metrics['success_rate'] * 100:.1f}",
                "Mean Reward": f"{cnn_metrics['mean_reward']:.4f}",
                "Mean Steps": f"{cnn_metrics['mean_steps']:.1f}",
                "Mean Collisions": f"{cnn_metrics['mean_collisions']:.1f}"
            })

            # 4. CNN-LSTM-PPO (per-environment checkpoint)
            print(f"  Evaluating CNN-LSTM-PPO (Ours) on {seed_label}...")
            rec_metrics = evaluate_trained_policy(recurrent_model, env_id, seeds, is_recurrent=True)
            results.append({
                "Environment": env_id,
                "Policy": "CNN-LSTM-PPO (Ours)",
                "Evaluation Split": seed_label,
                "Success Rate (%)": f"{rec_metrics['success_rate'] * 100:.1f}",
                "Mean Reward": f"{rec_metrics['mean_reward']:.4f}",
                "Mean Steps": f"{rec_metrics['mean_steps']:.1f}",
                "Mean Collisions": f"{rec_metrics['mean_collisions']:.1f}"
            })

    # Save to CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved CSV results to {output_csv}")

    # Generate Markdown Table
    md_content = "# Autonomous Navigation Policy Performance Comparison (Review 2)\n\n"
    md_content += "> **Note**: CNN-LSTM uses per-environment best checkpoints. "
    md_content += "`recurrent_ppo_empty8x8.zip` for Empty-8x8, `complex_doorkey_agent.zip` for DoorKey.\n\n"
    md_content += df.to_markdown(index=False)
    md_content += "\n"

    with open(output_md, "w") as f:
        f.write(md_content)
    print(f"Saved Markdown table to {output_md}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run comparative policy evaluation (Review 2)")
    parser.add_argument("--mlp_checkpoint", type=str, default="checkpoints/baseline_ppo.zip")
    parser.add_argument("--cnn_checkpoint", type=str, default="checkpoints/cnn_ppo.zip")
    parser.add_argument("--output_csv", type=str, default="reports/comparison_table.csv")
    parser.add_argument("--output_md", type=str, default="reports/comparison_table.md")

    args = parser.parse_args()
    run_comparative_evaluation(
        mlp_checkpoint=args.mlp_checkpoint,
        cnn_checkpoint=args.cnn_checkpoint,
        output_csv=args.output_csv,
        output_md=args.output_md
    )
