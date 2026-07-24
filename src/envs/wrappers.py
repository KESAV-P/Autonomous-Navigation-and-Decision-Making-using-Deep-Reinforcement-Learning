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
    - One-time bonus reward for picking up key (+0.2)
    - One-time bonus reward for unlocking/opening door (+0.3)
    - Small step penalty (-0.0005) to encourage efficiency
    """

    def __init__(self, env: gym.Env, step_penalty: float = 0.0005, key_bonus: float = 0.2, door_bonus: float = 0.3):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.key_bonus = key_bonus
        self.door_bonus = door_bonus
        self.key_picked = False
        self.door_opened = False
        self.prev_dist = None

    def _get_goal_pos(self):
        unwrapped = self.env.unwrapped
        dock_pos = getattr(unwrapped, 'dock_pos', None)
        if dock_pos is not None:
            return np.array(dock_pos)

        grid = getattr(unwrapped, 'grid', None)
        if grid is not None and hasattr(grid, 'width'):
            for x in range(grid.width):
                for y in range(grid.height):
                    cell = grid.get(x, y)
                    if cell is not None and cell.type == 'goal':
                        return np.array([x, y])
        elif isinstance(grid, np.ndarray):
            goals = np.argwhere(grid[:, :, 0] == 8)  # OBJECT_GOAL = 8
            if len(goals) > 0:
                return np.array([goals[0][1], goals[0][0]])
        return None

    def _get_front_cell(self):
        unwrapped = self.env.unwrapped
        front_pos = getattr(unwrapped, 'front_pos', None)
        grid = getattr(unwrapped, 'grid', None)
        if front_pos is not None and grid is not None and hasattr(grid, 'get'):
            return grid.get(*front_pos)
        return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.key_picked = False
        self.door_opened = False
        goal_pos = self._get_goal_pos()
        agent_pos = getattr(self.env.unwrapped, 'agent_pos', None)
        if goal_pos is not None and agent_pos is not None:
            self.prev_dist = float(np.linalg.norm(np.array(agent_pos) - goal_pos))
        else:
            self.prev_dist = None
        return obs, info

    def step(self, action):
        unwrapped = self.env.unwrapped
        carrying_before = getattr(unwrapped, 'carrying', None) is not None
        
        # Check door state in front of agent before step
        front_before = self._get_front_cell()
        door_open_before = (front_before is not None and getattr(front_before, 'type', None) == 'door' and getattr(front_before, 'is_open', False))

        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = float(reward)

        # 1. One-time key pickup bonus (+0.2)
        has_cargo_after = getattr(unwrapped, 'has_cargo', False) or (getattr(unwrapped, 'carrying', None) is not None)
        if not self.key_picked and not carrying_before and has_cargo_after:
            shaped_reward += self.key_bonus
            self.key_picked = True

        # 2. One-time door open bonus (+0.3)
        front_after = self._get_front_cell()
        door_open_after = (front_after is not None and getattr(front_after, 'type', None) == 'door' and getattr(front_after, 'is_open', False))
        if not self.door_opened and not door_open_before and door_open_after:
            shaped_reward += self.door_bonus
            self.door_opened = True

        # 3. Distance progress shaping (potential-based delta)
        goal_pos = self._get_goal_pos()
        agent_pos = getattr(unwrapped, 'agent_pos', None)
        if goal_pos is not None and agent_pos is not None:
            curr_dist = np.linalg.norm(np.array(agent_pos) - goal_pos)
            if self.prev_dist is not None:
                dist_delta = self.prev_dist - curr_dist
                shaped_reward += 0.01 * dist_delta
            self.prev_dist = curr_dist

        # 4. Step penalty
        shaped_reward -= self.step_penalty

        return obs, shaped_reward, terminated, truncated, info
