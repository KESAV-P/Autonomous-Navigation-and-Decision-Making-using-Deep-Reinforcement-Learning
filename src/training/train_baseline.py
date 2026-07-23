"""
Baseline PPO Training Script.

Trains a standard non-recurrent MlpPolicy PPO agent on MiniGrid-Empty-8x8-v0.
Logs metrics to WandB and saves the trained checkpoint.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from src.envs.make_env import make_env


def train_baseline(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set reproducibility seeds
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_id = config.get("env_id", "MiniGrid-Empty-8x8-v0")
    obs_mode = config.get("obs_mode", "symbolic")
    
    # Initialize environment
    raw_env = make_env(env_id=env_id, seed=seed, obs_mode=obs_mode)
    env = Monitor(raw_env)

    # WandB Initialization (offline mode fallback if unauthenticated)
    wandb_project = config.get("wandb_project", "drl-nav-baseline")
    try:
        run = wandb.init(
            project=wandb_project,
            config=config,
            sync_tensorboard=True,
            mode=os.environ.get("WANDB_MODE", "offline")
        )
        callback = WandbCallback(verbose=2)
    except Exception as e:
        print(f"[WARN] WandB init failed: {e}. Proceeding without WandB callback.")
        run = None
        callback = None

    # Instantiate PPO model
    model = PPO(
        policy=config.get("policy", "MlpPolicy"),
        env=env,
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

    print(f"\nStarting PPO Baseline training for {config['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback
    )

    # Save model checkpoint
    checkpoint_path = config.get("checkpoint_path", "checkpoints/baseline_ppo.zip")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model.save(checkpoint_path)
    print(f"Checkpoint successfully saved to {checkpoint_path}")

    if run is not None:
        run.finish()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline PPO Model")
    parser.add_argument("--config", type=str, default="configs/baseline_ppo.yaml", help="Path to config YAML")
    args = parser.parse_args()
    train_baseline(args.config)
