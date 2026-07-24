"""
Training Script for Warehouse AMR Navigation Agent.

Trains a RecurrentPPO (CNN + LSTM) agent on custom `Warehouse-Navigate-v0` environment.
Leverages transfer learning from the 16x16 DoorKey agent checkpoint for accelerated convergence.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import src.envs.warehouse_env  # Register Warehouse-Navigate-v0
from src.envs.make_env import make_env
from src.models.cnn_extractor import NavCNNExtractor
from src.evaluation.quick_eval import evaluate_checkpoint


def make_vector_envs(env_id: str, n_envs: int, seed: int, use_shaping: bool, noise_sigma: float):
    def _thunk(rank):
        def _init():
            env = make_env(
                env_id=env_id,
                seed=seed + rank,
                obs_mode="symbolic",
                use_shaping=use_shaping,
                noise_sigma=noise_sigma
            )
            return Monitor(env)
        return _init

    env_fns = [_thunk(i) for i in range(n_envs)]
    return DummyVecEnv(env_fns)


def train_warehouse_agent(
    env_id: str = "Warehouse-Navigate-v0",
    pretrained_path: str = "checkpoints/doorkey_16x16_agent.zip",
    save_path: str = "checkpoints/warehouse_agent.zip",
    total_timesteps: int = 200000,
    n_envs: int = 8,
    seed: int = 42,
    use_shaping: bool = True,
    noise_sigma: float = 0.05
):
    print(f"\n=======================================================")
    print(f"  Training Warehouse AMR Agent: {env_id}")
    print(f"  Total Timesteps: {total_timesteps} | Envs: {n_envs}")
    print(f"  Pretrained Checkpoint: {pretrained_path}")
    print(f"  Sensor Noise Sigma: {noise_sigma}")
    print(f"=======================================================")

    vec_env = make_vector_envs(env_id, n_envs, seed, use_shaping, noise_sigma)

    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path} for fine-tuning on Warehouse env...")
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

    print("Starting training on Warehouse AMR environment...")
    model.learn(total_timesteps=total_timesteps)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nSaved Warehouse AMR agent checkpoint to {save_path}")

    vec_env.close()

    print("\nEvaluating trained Warehouse AMR agent on held-out test seeds...")
    evaluate_checkpoint(
        checkpoint_path=save_path,
        env_id=env_id,
        n_episodes=10,
        start_seed=2000,
        is_recurrent=True,
        verbose=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Warehouse AMR Agent")
    parser.add_argument("--env_id", type=str, default="Warehouse-Navigate-v0")
    parser.add_argument("--pretrained", type=str, default="checkpoints/doorkey_16x16_agent.zip")
    parser.add_argument("--save_path", type=str, default="checkpoints/warehouse_agent.zip")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--noise_sigma", type=float, default=0.05)

    args = parser.parse_args()
    train_warehouse_agent(
        env_id=args.env_id,
        pretrained_path=args.pretrained,
        save_path=args.save_path,
        total_timesteps=args.timesteps,
        noise_sigma=args.noise_sigma
    )
