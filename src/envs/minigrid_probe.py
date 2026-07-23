"""
MiniGrid Environment Probe Module.

This script tests basic interaction with MiniGrid environments, verifying resetting,
stepping, observation shapes, dtypes, and reward outputs.
"""

import gymnasium as gym
import minigrid

def probe_minigrid():
    env_id = "MiniGrid-Empty-8x8-v0"
    print(f"Initializing probe for environment: {env_id}")
    
    env = gym.make(env_id)
    obs, info = env.reset(seed=42)
    
    print("\n--- Initial Reset ---")
    if isinstance(obs, dict):
        for k, v in obs.items():
            if hasattr(v, "shape"):
                print(f"Obs Key '{k}': shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"Obs Key '{k}': value={v}")
    else:
        print(f"Observation: shape={obs.shape}, dtype={obs.dtype}")
        
    print(f"Action Space: {env.action_space}")

    print("\n--- Stepping 5 Random Actions ---")
    for i in range(1, 6):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i} | Action: {action} | Reward: {reward:.4f} | Terminated: {terminated} | Truncated: {truncated}")
        
    env.close()
    print("\nMiniGrid probe completed successfully!")

if __name__ == "__main__":
    probe_minigrid()
