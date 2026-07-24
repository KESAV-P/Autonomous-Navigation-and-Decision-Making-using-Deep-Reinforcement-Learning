"""
Sensor Noise Wrapper for MiniGrid / Gymnasium environments.

Simulates sensor noise (e.g. camera blur, lidar interference) by adding
zero-mean Gaussian noise to the observation space.
"""

import numpy as np
import gymnasium as gym


class SensorNoiseWrapper(gym.ObservationWrapper):
    """
    Gymnasium ObservationWrapper that adds Gaussian noise to observations.

    Args:
        env: The Gymnasium environment to wrap.
        sigma: Standard deviation of the zero-mean Gaussian noise (0.0 means clean).
    """

    def __init__(self, env: gym.Env, sigma: float = 0.0):
        super().__init__(env)
        self.sigma = float(sigma)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if self.sigma <= 0.0:
            return observation

        # Add Gaussian noise
        noise = np.random.normal(loc=0.0, scale=self.sigma, size=observation.shape)
        noisy_obs = observation.astype(np.float32) + noise

        # Clip values to ensure valid visual/feature range
        if np.issubdtype(observation.dtype, np.integer):
            low = self.observation_space.low.min()
            high = self.observation_space.high.max()
            noisy_obs = np.clip(noisy_obs, low, high).astype(observation.dtype)
        else:
            noisy_obs = np.clip(noisy_obs, 0.0, 1.0)

        return noisy_obs
