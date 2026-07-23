# MiniGrid Environment Specification Report

## Overview
This document outlines the observation space, action space, reward dynamics, and wrapper specifications used throughout the Autonomous Navigation RL project.

## Observation Spaces

### 1. Symbolic Mode (`obs_mode='symbolic'`, Default for Training)
- **Wrapper**: `ImgObsWrapper` stripping dictionary observation to `image` tensor.
- **Shape**: `(7, 7, 3)`
- **Dtype**: `uint8`
- **Channels Encoding**:
  - Channel 0: Object ID (e.g. wall, floor, door, key, goal)
  - Channel 1: Object Color (e.g. red, green, blue)
  - Channel 2: Object State (open, closed, locked)
- **Field of View**: 7x7 partial view in front of the agent.

### 2. RGB Mode (`obs_mode='rgb'`, Used for Evaluation & GIF Rendering)
- **Wrapper**: `RGBImgPartialObsWrapper` + `ImgObsWrapper`.
- **Shape**: `(56, 56, 3)` (Pixel resolution for rendering).
- **Dtype**: `uint8`

## Action Space
- **Type**: `Discrete(7)`
- **Action Set**:
  - `0`: Turn Left
  - `1`: Turn Right
  - `2`: Move Forward
  - `3`: Pick Up Object
  - `4`: Drop Object
  - `5`: Toggle / Activate Object (e.g., open door)
  - `6`: Done / Finish Episode

## Reward Dynamics
- **Standard MiniGrid Reward**:
  $$\text{Reward} = 1 - 0.9 \times \left(\frac{\text{steps}}{\text{max\_steps}}\right)$$
  when reaching the green goal cell.
- **Reward Range**: $[0.0, 1.0]$ upon goal completion; $0.0$ if episode time limit (`max_steps`) is reached without reaching the goal.

## Custom Wrappers
- **`CollisionCounterWrapper`**:
  - Tracks forward movement attempts (`action == 2`) where `agent_pos` remains unchanged.
  - Adds `collisions` metric into `info['collisions']`.
