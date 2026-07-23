"""
16x16 Larger Grid Training Script.

Trains a RecurrentPPO (CNN + LSTM) agent on larger 16x16 MiniGrid environments:
- MiniGrid-Empty-16x16-v0
- MiniGrid-DoorKey-16x16-v0

Uses transfer learning from the mastered 8x8 DoorKey checkpoint for rapid convergence.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import yaml
import torch
import numpy as np
import wandb
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
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


def train_16x16_agent(
    env_id: str = "MiniGrid-DoorKey-16x16-v0",
    pretrained_path: str = "checkpoints/complex_doorkey_agent.zip",
    save_path: str = "checkpoints/doorkey_16x16_agent.zip",
    total_timesteps: int = 300000,
    n_envs: int = 8,
    seed: int = 42,
    use_shaping: bool = True
):
    print(f"\n=======================================================")
    print(f"  Training 16x16 Larger Grid Agent: {env_id}")
    print(f"  Total Timesteps: {total_timesteps} | Envs: {n_envs}")
    print(f"  Pretrained Checkpoint: {pretrained_path}")
    print(f"=======================================================")

    vec_env = make_vector_envs(env_id, n_envs, seed, "symbolic", use_shaping)

    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path} for fine-tuning on 16x16 grid...")
        model = RecurrentPPO.load(pretrained_path, env=vec_env)
        model.seed = seed
    else:
        print("Pretrained checkpoint not found. Training from scratch...")
        policy_kwargs = dict(
            features_extractor_class=NavCNNExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            n_lstm_layers=1,
            lstm_hidden_size=128
        )
        model = RecurrentPPO(
            policy="CnnLstmPolicy",
            env=vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=128,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            seed=seed
        )

    print("Starting training on 16x16 grid...")
    model.learn(total_timesteps=total_timesteps)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nSaved 16x16 model checkpoint to {save_path}")

    vec_env.close()

    print("\nRunning evaluation on held-out 16x16 test seeds...")
    evaluate_checkpoint(
        checkpoint_path=save_path,
        env_id=env_id,
        n_episodes=10,
        start_seed=1000,
        is_recurrent=True,
        verbose=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 16x16 Larger Grid Agent")
    parser.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-16x16-v0")
    parser.add_argument("--pretrained", type=str, default="checkpoints/complex_doorkey_agent.zip")
    parser.add_argument("--save_path", type=str, default="checkpoints/doorkey_16x16_agent.zip")
    parser.add_argument("--timesteps", type=int, default=300000)

    args = parser.parse_args()
    train_16x16_agent(
        env_id=args.env_id,
        pretrained_path=args.pretrained,
        save_path=args.save_path,
        total_timesteps=args.timesteps
    )
