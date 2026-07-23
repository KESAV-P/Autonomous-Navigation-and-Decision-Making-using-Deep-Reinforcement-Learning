"""
Custom Gymnasium wrappers for MiniGrid environments.

This module includes wrappers to track collisions and modify reward or observation structure as required.
"""

import gymnasium as gym
from minigrid.core.actions import Actions


class CollisionCounterWrapper(gym.Wrapper):
    """
    Gymnasium Wrapper that tracks when an agent attempts to move forward
    into a wall or obstacle and fails to change position.
    
    Increments info['collisions'] count on every collision.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.collisions = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.collisions = 0
        info['collisions'] = self.collisions
        return obs, info

    def step(self, action):
        unwrapped_env = self.env.unwrapped
        pos_before = getattr(unwrapped_env, 'agent_pos', None)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        pos_after = getattr(unwrapped_env, 'agent_pos', None)
        
        # Action 2 in MiniGrid is 'forward'
        is_forward = (action == Actions.forward or action == 2)
        if is_forward and pos_before is not None and pos_after is not None:
            if pos_before[0] == pos_after[0] and pos_before[1] == pos_after[1]:
                self.collisions += 1

        info['collisions'] = self.collisions
        return obs, reward, terminated, truncated, info
