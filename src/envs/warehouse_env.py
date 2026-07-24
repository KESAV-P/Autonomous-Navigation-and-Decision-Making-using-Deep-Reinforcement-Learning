"""
Custom Warehouse Autonomous Mobile Robot (AMR) Navigation Environment.

Implements a 16x16 warehouse layout with:
- Storage Zone with Shelves and Cargo Pickup location
- Aisle / Corridor passageways
- Loading Dock delivery target location
- Dynamic state tracking (Cargo status: Searching -> Picked Up -> Delivered)
- Compatible with MiniGrid observation format (7x7 local view, 3 symbolic channels)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


# Object Types (Matching MiniGrid channel 0 symbolic encoding)
OBJECT_EMPTY = 0
OBJECT_WALL = 1
OBJECT_FLOOR = 2
OBJECT_DOOR = 4
OBJECT_KEY = 5      # Repurposed as Cargo Item
OBJECT_BALL = 6     # Repurposed as Cargo Unit
OBJECT_GOAL = 8     # Repurposed as Delivery Dock

# Color Index (MiniGrid channel 1)
COLOR_RED = 0
COLOR_GREEN = 1
COLOR_BLUE = 2
COLOR_PURPLE = 3
COLOR_YELLOW = 4
COLOR_GREY = 5

# State Index (MiniGrid channel 2)
STATE_OPEN = 0
STATE_CLOSED = 1
STATE_LOCKED = 2


class WarehouseEnv(gym.Env):
    """
    16x16 AMR Warehouse Navigation & Cargo Delivery Environment.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(self, size=16, max_steps=400, render_mode=None):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # 7x7 local partial view, 3 symbolic channels (Object, Color, State)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(7, 7, 3), dtype=np.uint8)
        })

        # Actions: 0: turn left, 1: turn right, 2: move forward, 3: pickup, 4: drop, 5: toggle, 6: done
        self.action_space = spaces.Discrete(7)

        # Direction vectors: 0: Right (>), 1: Down (v), 2: Left (<), 3: Up (^)
        self.dir_vecs = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

        self.grid = None
        self.agent_pos = None
        self.agent_dir = None
        self.cargo_pos = None
        self.dock_pos = None
        self.has_cargo = False
        self.cargo_delivered = False
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.has_cargo = False
        self.cargo_delivered = False

        # Initialize 16x16 grid with floor (2, 0, 0)
        self.grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        self.grid[:, :, 0] = OBJECT_FLOOR

        # Outer walls
        self.grid[0, :, 0] = OBJECT_WALL
        self.grid[-1, :, 0] = OBJECT_WALL
        self.grid[:, 0, 0] = OBJECT_WALL
        self.grid[:, -1, 0] = OBJECT_WALL

        # Warehouse interior shelves (Storage Zone Left)
        for r in range(2, 6):
            self.grid[r, 3, 0] = OBJECT_WALL
            self.grid[r, 6, 0] = OBJECT_WALL
        for r in range(10, 14):
            self.grid[r, 3, 0] = OBJECT_WALL
            self.grid[r, 6, 0] = OBJECT_WALL

        # Aisle Wall dividing Storage (cols 1..7) and Dock (cols 9..14)
        for r in range(1, self.size - 1):
            if r not in [7, 8]:  # Passageway openings at middle
                self.grid[r, 8, 0] = OBJECT_WALL

        # Set colors for walls
        self.grid[self.grid[:, :, 0] == OBJECT_WALL, 1] = COLOR_GREY

        # Agent initial position (Storage entry)
        self.agent_pos = np.array([2, 1])
        self.agent_dir = 1  # Facing Down

        # Cargo position (Storage Zone shelf aisle)
        self.cargo_pos = np.array([5, 4])
        self.grid[self.cargo_pos[1], self.cargo_pos[0]] = [OBJECT_KEY, COLOR_YELLOW, 0]

        # Delivery Dock position (Dock Zone right)
        self.dock_pos = np.array([13, 13])
        self.grid[self.dock_pos[1], self.dock_pos[0]] = [OBJECT_GOAL, COLOR_GREEN, 0]

        obs = self._get_obs()
        info = {
            "has_cargo": self.has_cargo,
            "cargo_delivered": self.cargo_delivered,
            "mission": "Pick up cargo in Storage Zone and deliver to Loading Dock"
        }
        return obs, info

    def _get_obs(self):
        """Extracts 7x7 local view grid in front of the agent."""
        local_grid = np.zeros((7, 7, 3), dtype=np.uint8)
        local_grid[:, :, 0] = OBJECT_WALL
        local_grid[:, :, 1] = COLOR_GREY

        # Agent views 7 cells ahead and 3 cells to left/right
        fwd = self.dir_vecs[self.agent_dir]
        right = np.array([-fwd[1], fwd[0]])

        # Map local (rx, ry) -> global (gx, gy)
        for local_y in range(7):
            for local_x in range(7):
                offset_fwd = (6 - local_y)  # Local y=6 is agent pos, y=0 is 6 steps ahead
                offset_right = (local_x - 3)

                gx = self.agent_pos[0] + fwd[0] * offset_fwd + right[0] * offset_right
                gy = self.agent_pos[1] + fwd[1] * offset_fwd + right[1] * offset_right

                if 0 <= gx < self.size and 0 <= gy < self.size:
                    local_grid[local_y, local_x] = self.grid[gy, gx]

        return {"image": local_grid}

    def step(self, action):
        self.step_count += 1
        reward = -0.001  # Small step penalty to encourage speed
        terminated = False
        truncated = self.step_count >= self.max_steps

        # Action 0: Turn Left
        if action == 0:
            self.agent_dir = (self.agent_dir - 1) % 4
        # Action 1: Turn Right
        elif action == 1:
            self.agent_dir = (self.agent_dir + 1) % 4
        # Action 2: Move Forward
        elif action == 2:
            fwd = self.dir_vecs[self.agent_dir]
            target_pos = self.agent_pos + fwd
            if 0 <= target_pos[0] < self.size and 0 <= target_pos[1] < self.size:
                target_cell = self.grid[target_pos[1], target_pos[0], 0]
                if target_cell != OBJECT_WALL:
                    self.agent_pos = target_pos
                    # Distance shaping towards target (cargo or dock)
                    target_dest = self.dock_pos if self.has_cargo else self.cargo_pos
                    dist = float(np.linalg.norm(self.agent_pos - target_dest))
                    reward += 0.01 / (dist + 1.0)

        # Action 3: Pickup Cargo
        elif action == 3:
            fwd = self.dir_vecs[self.agent_dir]
            front_pos = self.agent_pos + fwd
            if np.array_equal(front_pos, self.cargo_pos) and not self.has_cargo:
                self.has_cargo = True
                self.grid[self.cargo_pos[1], self.cargo_pos[0]] = [OBJECT_FLOOR, 0, 0]
                reward += 0.5
                print(">>> [WarehouseEnv] Cargo Picked Up!")

        # Action 4: Drop / Deliver Cargo
        elif action == 4:
            fwd = self.dir_vecs[self.agent_dir]
            front_pos = self.agent_pos + fwd
            if self.has_cargo and np.array_equal(front_pos, self.dock_pos):
                self.cargo_delivered = True
                self.has_cargo = False
                reward += 1.0
                terminated = True
                print(">>> [WarehouseEnv] Cargo Delivered to Loading Dock! Mission Success!")

        obs = self._get_obs()
        info = {
            "has_cargo": self.has_cargo,
            "cargo_delivered": self.cargo_delivered,
            "agent_pos": self.agent_pos.tolist(),
            "collisions": 0
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render grid into RGB image frame for Pygame / visualization."""
        cell_size = 32
        img = np.ones((self.size * cell_size, self.size * cell_size, 3), dtype=np.uint8) * 240

        # Draw grid walls and items
        for y in range(self.size):
            for x in range(self.size):
                cell_type = self.grid[y, x, 0]
                r_start, r_end = y * cell_size, (y + 1) * cell_size
                c_start, c_end = x * cell_size, (x + 1) * cell_size

                if cell_type == OBJECT_WALL:
                    img[r_start:r_end, c_start:c_end] = [80, 80, 90]  # Dark Slate Wall
                elif cell_type == OBJECT_KEY:
                    img[r_start:r_end, c_start:c_end] = [240, 200, 20] # Bright Yellow Cargo
                elif cell_type == OBJECT_GOAL:
                    img[r_start:r_end, c_start:c_end] = [40, 200, 60]  # Green Loading Dock

        # Draw Agent
        ay, ax = self.agent_pos[1] * cell_size, self.agent_pos[0] * cell_size
        img[ay + 4:ay + cell_size - 4, ax + 4:ax + cell_size - 4] = [30, 140, 240]  # Blue Robot

        return img


# Register environment with Gymnasium
try:
    register(
        id="Warehouse-Navigate-v0",
        entry_point="src.envs.warehouse_env:WarehouseEnv",
        max_episode_steps=400,
    )
except Exception:
    pass  # Avoid re-registration error if reloaded
