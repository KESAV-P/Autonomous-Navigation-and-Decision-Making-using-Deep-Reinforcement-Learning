"""
Environment factory module for MiniGrid environments.

Provides make_env() function to create Gymnasium-compliant MiniGrid environments
with configured observation modes ('symbolic' or 'rgb'), observation processing wrappers,
and collision counting wrappers.
"""

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from src.envs.wrappers import CollisionCounterWrapper


def make_env(env_id: str, seed: int = 0, obs_mode: str = "symbolic", render_mode: str = None) -> gym.Env:
    """
    Creates and wraps a MiniGrid environment.

    Args:
        env_id: The Gymnasium/MiniGrid environment ID (e.g. 'MiniGrid-Empty-8x8-v0').
        seed: Random seed for environment initialization.
        obs_mode: 'symbolic' (default MiniGrid encoding: 7x7x3) or 'rgb' (pixel observation).
        render_mode: Optional render mode ('rgb_array', 'human', etc.).

    Returns:
        gym.Env: Wrapped Gymnasium environment.
    """
    if render_mode is not None:
        env = gym.make(env_id, render_mode=render_mode)
    else:
        env = gym.make(env_id)

    if obs_mode == "rgb":
        env = RGBImgPartialObsWrapper(env)
    
    # ImgObsWrapper extracts the 'image' key from the observation dict into a plain array
    env = ImgObsWrapper(env)

    # Collision tracking
    env = CollisionCounterWrapper(env)

    env.reset(seed=seed)
    return env
