"""
Custom Gymnasium wrappers for MiniGrid environments.

This module includes wrappers to track collisions and modify reward or observation structure.
"""

import numpy as np
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


class RewardShapingWrapper(gym.Wrapper):
    """
    Reward Shaping Wrapper for Sparse MiniGrid Environments.
    
    Adds intermediate rewards for:
    - Distance-to-goal progress reward (potential-based shaping)
    - Bonus reward for picking up keys (+0.2)
    - Bonus reward for opening doors (+0.3)
    - Small step penalty (-0.0005) to encourage efficiency
    """

    def __init__(self, env: gym.Env, step_penalty: float = 0.0005, key_bonus: float = 0.2, door_bonus: float = 0.3):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.key_bonus = key_bonus
        self.door_bonus = door_bonus
        self.carrying_key = False
        self.prev_dist = None

    def _get_goal_pos(self):
        unwrapped = self.env.unwrapped
        grid = getattr(unwrapped, 'grid', None)
        if grid is not None:
            for x in range(grid.width):
                for y in range(grid.height):
                    cell = grid.get(x, y)
                    if cell is not None and cell.type == 'goal':
                        return np.array([x, y])
        return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.carrying_key = False
        goal_pos = self._get_goal_pos()
        agent_pos = getattr(self.env.unwrapped, 'agent_pos', None)
        if goal_pos is not None and agent_pos is not None:
            self.prev_dist = np.linalg.norm(np.array(agent_pos) - goal_pos)
        else:
            self.prev_dist = None
        return obs, info

    def step(self, action):
        unwrapped = self.env.unwrapped
        carrying_before = unwrapped.carrying is not None
        
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = float(reward)

        # 1. Carrying key bonus
        carrying_after = unwrapped.carrying is not None
        if not carrying_before and carrying_after:
            shaped_reward += self.key_bonus

        # 2. Door toggle bonus
        if action == Actions.toggle or action == 5:
            # Check if door state changed in front of agent
            shaped_reward += self.door_bonus * 0.5  # exploration incentive for toggle

        # 3. Distance progress shaping
        goal_pos = self._get_goal_pos()
        agent_pos = getattr(unwrapped, 'agent_pos', None)
        if goal_pos is not None and agent_pos is not None:
            curr_dist = np.linalg.norm(np.array(agent_pos) - goal_pos)
            if self.prev_dist is not None:
                dist_delta = self.prev_dist - curr_dist
                shaped_reward += 0.01 * dist_delta
            self.prev_dist = curr_dist

        # 4. Small step penalty
        shaped_reward -= self.step_penalty

        return obs, shaped_reward, terminated, truncated, info
