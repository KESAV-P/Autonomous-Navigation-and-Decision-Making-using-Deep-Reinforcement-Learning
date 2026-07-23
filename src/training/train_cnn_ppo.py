"""
CNN PPO Training Script.

Trains a PPO agent with a custom 3-layer CNN Feature Extractor (NavCNNExtractor)
on MiniGrid-Empty-8x8-v0. Logs metrics to WandB and saves trained checkpoint.
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
from src.models.cnn_extractor import NavCNNExtractor


def train_cnn_ppo(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set reproducibility seeds
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_id = config.get("env_id", "MiniGrid-Empty-8x8-v0")
    obs_mode = config.get("obs_mode", "symbolic")
    features_dim = config.get("features_dim", 128)

    # Initialize environment
    raw_env = make_env(env_id=env_id, seed=seed, obs_mode=obs_mode)
    env = Monitor(raw_env)

    # WandB Initialization
    wandb_project = config.get("wandb_project", "drl-nav-cnn")
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

    # Custom policy kwargs with NavCNNExtractor
    policy_kwargs = dict(
        features_extractor_class=NavCNNExtractor,
        features_extractor_kwargs=dict(features_dim=features_dim)
    )

    # Instantiate PPO model with CnnPolicy and custom CNN extractor
    model = PPO(
        policy=config.get("policy", "CnnPolicy"),
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

    print(f"\nStarting CNN PPO training for {config['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback
    )

    # Save model checkpoint
    checkpoint_path = config.get("checkpoint_path", "checkpoints/cnn_ppo.zip")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model.save(checkpoint_path)
    print(f"Checkpoint successfully saved to {checkpoint_path}")

    if run is not None:
        run.finish()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN PPO Model")
    parser.add_argument("--config", type=str, default="configs/cnn_ppo.yaml", help="Path to config YAML")
    args = parser.parse_args()
    train_cnn_ppo(args.config)
