"""
Environment factory module for MiniGrid environments.

Provides make_env() function to create Gymnasium-compliant MiniGrid environments
with configured observation modes ('symbolic' or 'rgb'), observation processing wrappers,
collision counting wrappers, and optional reward shaping.
"""

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from src.envs.wrappers import CollisionCounterWrapper, RewardShapingWrapper


from src.envs.sensor_noise_wrapper import SensorNoiseWrapper


def make_env(
    env_id: str,
    seed: int = 0,
    obs_mode: str = "symbolic",
    render_mode: str = None,
    use_shaping: bool = False,
    noise_sigma: float = 0.0
) -> gym.Env:
    """
    Creates and wraps a MiniGrid or custom environment.

    Args:
        env_id: The Gymnasium/MiniGrid environment ID.
        seed: Random seed for environment initialization.
        obs_mode: 'symbolic' or 'rgb'.
        render_mode: Optional render mode ('rgb_array', 'human', etc.).
        use_shaping: Whether to apply RewardShapingWrapper.
        noise_sigma: Standard deviation for Gaussian observation noise.

    Returns:
        gym.Env: Wrapped Gymnasium environment.
    """
    if render_mode is not None:
        env = gym.make(env_id, render_mode=render_mode)
    else:
        env = gym.make(env_id)

    # Check if observation is a dict (MiniGrid style) or array (custom warehouse)
    if isinstance(env.observation_space, gym.spaces.Dict):
        if obs_mode == "rgb":
            env = RGBImgPartialObsWrapper(env)
        # ImgObsWrapper extracts the 'image' key from observation dict
        env = ImgObsWrapper(env)

    # Apply sensor noise wrapper if configured
    if noise_sigma > 0.0:
        env = SensorNoiseWrapper(env, sigma=noise_sigma)

    # Reward shaping if enabled
    if use_shaping:
        env = RewardShapingWrapper(env)

    # Collision tracking
    env = CollisionCounterWrapper(env)

    env.reset(seed=seed)
    return env
