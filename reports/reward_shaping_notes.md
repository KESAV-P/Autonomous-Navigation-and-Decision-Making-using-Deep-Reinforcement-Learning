# Reward Shaping Analysis & Notes

## Motivation
In standard MiniGrid environments (`DoorKey-8x8`, `MultiRoom-N2-S4`, `MultiRoom-N4-S5`), the default reward signal is extremely sparse: the agent receives $0.0$ reward on all steps until it successfully unlocks doors and reaches the goal tile.

In complex environments requiring sequential interactions (picking up keys, unlocking doors, traversing multiple rooms), unshaped PPO exploration often fails to discover the goal within early training iterations.

## Applied Reward Shaping Components
`RewardShapingWrapper` adds four targeted intermediate reward components:

1. **Key Acquisition Bonus (+0.2)**: Granted when the agent successfully picks up a key (`unwrapped.carrying is not None`).
2. **Door Interaction Incentive (+0.15)**: Exploration reward when triggering `toggle` action near doors.
3. **Potential-Based Distance Progress (+0.01 $\times \Delta d$)**: Shaped reward proportional to Euclidean distance reduction towards the goal coordinates.
4. **Step Efficiency Penalty (-0.0005)**: Small negative per-step cost to penalize spinning or standing still.

## Impact & Safeguards
- **Exploitation Safeguard**: The shaping bonuses are kept small relative to the primary task completion reward ($\approx 1.0$) to ensure the agent does not exploit intermediate bonuses without reaching the actual goal.
