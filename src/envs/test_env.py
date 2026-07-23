"""
Environment and wrapper verification script for Phase 1.

Verifies Gymnasium API compliance of make_env and tests CollisionCounterWrapper
with a scripted wall-collision sequence.
"""

from src.envs.make_env import make_env
from minigrid.core.actions import Actions


def test_make_env_symbolic():
    env = make_env("MiniGrid-Empty-8x8-v0", seed=42, obs_mode="symbolic")
    obs, info = env.reset(seed=42)
    assert obs is not None, "Observation should not be None"
    assert len(obs.shape) == 3, f"Expected 3D tensor obs, got shape {obs.shape}"
    assert 'collisions' in info, "Collision count missing in info dictionary"
    print(f"[PASS] Symbolic environment reset OK. Obs shape: {obs.shape}, dtype: {obs.dtype}")
    env.close()


def test_make_env_rgb():
    env = make_env("MiniGrid-Empty-8x8-v0", seed=42, obs_mode="rgb")
    obs, info = env.reset(seed=42)
    assert obs is not None, "RGB observation should not be None"
    assert len(obs.shape) == 3, f"Expected 3D RGB image, got shape {obs.shape}"
    print(f"[PASS] RGB environment reset OK. Obs shape: {obs.shape}, dtype: {obs.dtype}")
    env.close()


def test_collision_counter():
    env = make_env("MiniGrid-Empty-8x8-v0", seed=42, obs_mode="symbolic")
    obs, info = env.reset(seed=42)
    initial_collisions = info['collisions']
    assert initial_collisions == 0, f"Expected 0 initial collisions, got {initial_collisions}"

    # In Empty-8x8, agent starts facing right (dir 0) at (1, 1).
    # Facing wall is far, so let's turn left (facing wall at y=0) and step forward.
    # Action 0 = turn left
    obs, reward, term, trunc, info = env.step(Actions.left)
    
    # Action 2 = forward (tries to step into outer wall at top)
    obs, reward, term, trunc, info = env.step(Actions.forward)
    col1 = info['collisions']
    assert col1 == 1, f"Expected collision count 1 after hitting wall, got {col1}"

    # Try moving forward into wall again
    obs, reward, term, trunc, info = env.step(Actions.forward)
    col2 = info['collisions']
    assert col2 == 2, f"Expected collision count 2 after second hit, got {col2}"

    print(f"[PASS] CollisionCounterWrapper verified. Collision count reached {col2}")
    env.close()


if __name__ == "__main__":
    test_make_env_symbolic()
    test_make_env_rgb()
    test_collision_counter()
    print("\nPhase 1 Environment and Wrapper tests passed successfully!")
