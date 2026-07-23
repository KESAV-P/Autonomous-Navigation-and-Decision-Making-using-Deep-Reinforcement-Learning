"""
Complex Environment Training Script.

Trains a RecurrentPPO (CNN + LSTM) agent using vectorized parallel environments and reward shaping
to master complex sequential skills (picking up keys, unlocking doors, navigating multi-room grids).
"""

import os
import argparse
import yaml
import torch
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from src.envs.make_env import make_env
from src.models.cnn_extractor import NavCNNExtractor
from src.evaluation.quick_eval import evaluate_checkpoint


def make_vector_envs(env_id: str, n_envs: int, seed: int, obs_mode: str, use_shaping: bool):
    def _thunk(rank):
        def _init():
            env = make_env(env_id=env_id, seed=seed + rank, obs_mode=obs_mode, use_shaping=use_shaping)
            return Monitor(env)
        return _init

    env_fns = [_thunk(i) for i in range(n_envs)]
    return DummyVecEnv(env_fns)


def train_complex_agent(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_id = config.get("env_id", "MiniGrid-DoorKey-8x8-v0")
    obs_mode = config.get("obs_mode", "symbolic")
    n_envs = config.get("n_envs", 8)
    use_shaping = config.get("use_shaping", True)
    features_dim = config.get("features_dim", 128)
    n_lstm_layers = config.get("n_lstm_layers", 1)
    lstm_hidden_size = config.get("lstm_hidden_size", 128)
    total_timesteps = config.get("total_timesteps", 500000)

    print(f"\n=======================================================")
    print(f"  Training Complex Agent on {env_id} ({total_timesteps} steps, {n_envs} envs)")
    print(f"=======================================================")

    vec_env = make_vector_envs(env_id, n_envs, seed, obs_mode, use_shaping)

    wandb_project = config.get("wandb_project", "drl-nav-complex")
    try:
        run = wandb.init(
            project=wandb_project,
            config=config,
            sync_tensorboard=True,
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

    model = RecurrentPPO(
        policy=config.get("policy", "CnnLstmPolicy"),
        env=vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("n_steps", 256),
        batch_size=config.get("batch_size", 128),
        n_epochs=config.get("n_epochs", 4),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        ent_coef=config.get("ent_coef", 0.02),
        verbose=1,
        seed=seed
    )

    print("Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    checkpoint_path = config.get("checkpoint_path", "checkpoints/complex_doorkey_agent.zip")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model.save(checkpoint_path)
    print(f"Successfully saved checkpoint to {checkpoint_path}")

    if run is not None:
        run.finish()

    vec_env.close()

    # Evaluate trained complex agent
    print("\nRunning evaluation on held-out test seeds...")
    evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        env_id=env_id,
        n_episodes=20,
        start_seed=1000,
        is_recurrent=True,
        obs_mode=obs_mode,
        deterministic=False,
        verbose=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Complex Recurrent PPO Agent")
    parser.add_argument("--config", type=str, default="configs/complex_training.yaml", help="Path to config YAML")
    args = parser.parse_args()
    train_complex_agent(args.config)
