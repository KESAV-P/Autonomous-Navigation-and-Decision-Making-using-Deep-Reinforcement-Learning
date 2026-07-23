"""
Plotting and Visualization Generator Script.

Reads reports/comparison_table.csv and reports/curriculum_results.csv,
generating static publication-ready comparison bar charts and reward curves
in media/ and media/reward_curves/.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_comparison_charts(comparison_csv: str = "reports/comparison_table.csv", output_dir: str = "media"):
    if not os.path.exists(comparison_csv):
        print(f"[WARN] {comparison_csv} does not exist. Skipping comparison plots.")
        return

    df = pd.read_csv(comparison_csv)
    # Ensure numerical types
    df['Success Rate (%)'] = pd.to_numeric(df['Success Rate (%)'])
    df['Mean Steps'] = pd.to_numeric(df['Mean Steps'])
    df['Mean Reward'] = pd.to_numeric(df['Mean Reward'])

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reward_curves"), exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Success Rate Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    g = sns.barplot(
        data=df,
        x="Environment",
        y="Success Rate (%)",
        hue="Policy",
        errorbar=None
    )
    plt.title("Navigation Policy Success Rate Comparison Across Environments", fontsize=14, fontweight="bold")
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.xlabel("Environment", fontsize=12)
    plt.ylim(0, 105)
    plt.xticks(rotation=15)
    plt.tight_layout()
    success_chart_path = os.path.join(output_dir, "success_rate_comparison.png")
    plt.savefig(success_chart_path, dpi=300)
    plt.close()
    print(f"Saved {success_chart_path}")

    # 2. Steps to Goal Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="Environment",
        y="Mean Steps",
        hue="Policy",
        errorbar=None
    )
    plt.title("Mean Steps to Goal Comparison Across Environments", fontsize=14, fontweight="bold")
    plt.ylabel("Mean Steps", fontsize=12)
    plt.xlabel("Environment", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    steps_chart_path = os.path.join(output_dir, "steps_to_goal_comparison.png")
    plt.savefig(steps_chart_path, dpi=300)
    plt.close()
    print(f"Saved {steps_chart_path}")


def plot_curriculum_curves(curriculum_csv: str = "reports/curriculum_results.csv", output_dir: str = "media/reward_curves"):
    if not os.path.exists(curriculum_csv):
        print(f"[WARN] {curriculum_csv} does not exist. Skipping curriculum plot.")
        return

    df = pd.read_csv(curriculum_csv)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(df['stage'], df['mean_reward'], marker='o', linewidth=2.5, color='#1f77b4', label='Mean Reward')
    plt.plot(df['stage'], df['success_rate'], marker='s', linewidth=2.5, color='#2ca02c', label='Success Rate')
    
    plt.title("Recurrent CNN-LSTM PPO Performance Across Curriculum Stages", fontsize=13, fontweight='bold')
    plt.xlabel("Curriculum Stage Index", fontsize=11)
    plt.ylabel("Metric Score", fontsize=11)
    plt.xticks(df['stage'], [f"Stage {s}\n{env}" for s, env in zip(df['stage'], df['env_id'])], fontsize=9)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    curve_path = os.path.join(output_dir, "curriculum_progress.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Saved {curve_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate performance plots")
    parser.add_argument("--comparison_csv", type=str, default="reports/comparison_table.csv")
    parser.add_argument("--curriculum_csv", type=str, default="reports/curriculum_results.csv")
    parser.add_argument("--output_dir", type=str, default="media")

    args = parser.parse_args()
    plot_comparison_charts(args.comparison_csv, args.output_dir)
    plot_curriculum_curves(args.curriculum_csv, os.path.join(args.output_dir, "reward_curves"))
