"""
Environment check utility module.

This script imports all required core dependencies for the DRL Autonomous Navigation project,
verifies that they are properly installed, and prints their versions.
"""

import sys

def check_environment():
    print(f"Python version: {sys.version}")
    
    libraries = [
        ("gymnasium", "gymnasium"),
        ("minigrid", "minigrid"),
        ("torch", "torch"),
        ("stable_baselines3", "stable_baselines3"),
        ("sb3_contrib", "sb3_contrib"),
        ("wandb", "wandb"),
        ("matplotlib", "matplotlib"),
        ("cv2", "opencv-python"),
        ("imageio", "imageio"),
        ("yaml", "pyyaml"),
    ]
    
    all_ok = True
    for mod_name, pkg_name in libraries:
        try:
            mod = __import__(mod_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK] {pkg_name} ({mod_name}) version: {version}")
        except ImportError as e:
            print(f"[FAIL] {pkg_name} ({mod_name}) failed to import: {e}")
            all_ok = False

    if hasattr(__import__("torch"), "cuda"):
        import torch
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    if all_ok:
        print("\nAll core dependencies verified successfully!")
    else:
        print("\nSome dependencies failed to load.")
        sys.exit(1)

if __name__ == "__main__":
    check_environment()
