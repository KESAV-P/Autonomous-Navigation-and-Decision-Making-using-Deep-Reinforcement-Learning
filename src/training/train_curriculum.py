"""
Curriculum Training Script.

Progressively trains a RecurrentPPO agent across a sequence of increasingly complex
MiniGrid environments (Empty-8x8 -> DoorKey-8x8 -> MultiRoom-N2-S4 -> MultiRoom-N4-S5),
transferring model checkpoints across stages and saving evaluation metrics.
"""

import os
import csv
import argparse
import yaml
import torch
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from src.envs.make_env import make_env
from src.models.cnn_extractor import NavCNNExtractor
from src.evaluation.quick_eval import evaluate_checkpoint


def train_curriculum(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_mode = config.get("obs_mode", "symbolic")
    features_dim = config.get("features_dim", 128)
    n_lstm_layers = config.get("n_lstm_layers", 1)
    lstm_hidden_size = config.get("lstm_hidden_size", 128)
    results_csv = config.get("results_csv", "reports/curriculum_results.csv")
    curriculum_stages = config["curriculum"]

    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    
    # Initialize CSV header if file doesn't exist
    if not os.path.exists(results_csv):
        with open(results_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["stage", "env_id", "total_timesteps", "mean_reward", "std_reward", "success_rate", "mean_steps", "mean_collisions"])

    prev_checkpoint = None

    for stage_info in curriculum_stages:
        stage = stage_info["stage"]
        env_id = stage_info["env_id"]
        timesteps = stage_info["total_timesteps"]

        print(f"\n=======================================================")
        print(f"  Curriculum Stage {stage}: {env_id} ({timesteps} timesteps)")
        print(f"=======================================================")

        raw_env = make_env(env_id=env_id, seed=seed, obs_mode=obs_mode)
        env = Monitor(raw_env)

        wandb_project = config.get("wandb_project", "drl-nav-curriculum")
        run_name = f"stage_{stage}_{env_id}"
        try:
            run = wandb.init(
                project=wandb_project,
                name=run_name,
                config={**config, **stage_info},
                sync_tensorboard=True,
                reinit=True,
                mode=os.environ.get("WANDB_MODE", "offline")
            )
            callback = WandbCallback(verbose=2)
        except Exception as e:
            print(f"[WARN] WandB init failed: {e}. Proceeding without callback.")
            run = None
            callback = None

        policy_kwargs = dict(
            features_extractor_class=NavCNNExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            n_lstm_layers=n_lstm_layers,
            lstm_hidden_size=lstm_hidden_size
        )

        if stage == 0 or prev_checkpoint is None:
            print("Initializing fresh RecurrentPPO model...")
            model = RecurrentPPO(
                policy=config.get("policy", "CnnLstmPolicy"),
                env=env,
                policy_kwargs=policy_kwargs,
                learning_rate=config.get("learning_rate", 3e-4),
                n_steps=config.get("n_steps", 128),
                batch_size=config.get("batch_size", 64),
                n_epochs=config.get("n_epochs", 4),
                gamma=config.get("gamma", 0.99),
                gae_lambda=config.get("gae_lambda", 0.95),
                clip_range=config.get("clip_range", 0.2),
                ent_coef=config.get("ent_coef", 0.01),
                verbose=1,
                seed=seed
            )
        else:
            print(f"Loading checkpoint from stage {stage - 1}: {prev_checkpoint}")
            model = RecurrentPPO.load(prev_checkpoint, env=env)

        model.learn(total_timesteps=timesteps, callback=callback)

        checkpoint_path = f"checkpoints/curriculum_stage_{stage}_{env_id}.zip"
        os.makedirs("checkpoints", exist_ok=True)
        model.save(checkpoint_path)
        print(f"Stage {stage} checkpoint saved to {checkpoint_path}")

        if run is not None:
            run.finish()

        env.close()

        # Evaluate stage checkpoint
        eval_results = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            env_id=env_id,
            n_episodes=20,
            start_seed=1000,
            is_recurrent=True,
            obs_mode=obs_mode,
            deterministic=False,
            verbose=True
        )

        # Write results to CSV
        with open(results_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                stage,
                env_id,
                timesteps,
                eval_results["mean_reward"],
                eval_results["std_reward"],
                eval_results["success_rate"],
                eval_results["mean_steps"],
                eval_results["mean_collisions"]
            ])

        prev_checkpoint = checkpoint_path

    print(f"\nCurriculum training complete! All results saved to {results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Curriculum Recurrent PPO Agent")
    parser.add_argument("--config", type=str, default="configs/curriculum.yaml", help="Path to config YAML")
    args = parser.parse_args()
    train_curriculum(args.config)
