"""
Custom CNN Feature Extractor Module.

Implements NavCNNExtractor extending BaseFeaturesExtractor from stable_baselines3.
Processes grid / image observations with a 3-layer Convolutional Neural Network
and maps spatial features into a feature vector of specified dimension.
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NavCNNExtractor(BaseFeaturesExtractor):
    """
    Custom 3-layer CNN Feature Extractor for MiniGrid navigation tasks.

    Args:
        observation_space: Gymnasium observation space (Box).
        features_dim: Output dimension of the feature vector (default 128).
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        # 3-layer ConvNet
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute output shape dynamically by passing a dummy tensor
        with torch.no_grad():
            dummy_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
