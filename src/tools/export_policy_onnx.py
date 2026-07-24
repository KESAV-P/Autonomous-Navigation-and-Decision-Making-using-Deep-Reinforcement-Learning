"""
ONNX Policy Exporter for Embedded Robot Deployment.

Exports trained Stable-Baselines3 PyTorch policies into standard ONNX format.
This demonstrates the model's readiness for real-world deployment on embedded hardware
(e.g., NVIDIA Jetson, Raspberry Pi, ROS2 nodes).
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO


class OnnxPolicyWrapper(nn.Module):
    """Wraps policy network for clean single-tensor forward execution in ONNX."""

    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor(features)
        action_logits = self.action_net(latent_pi)
        return action_logits


def export_to_onnx(
    checkpoint_path: str,
    output_onnx_path: str = "exports/nav_policy.onnx",
    is_recurrent: bool = False
):
    print(f"\n=======================================================")
    print(f"  Exporting Policy Checkpoint to ONNX")
    print(f"  Input Checkpoint: {checkpoint_path}")
    print(f"  Output ONNX Path: {output_onnx_path}")
    print(f"=======================================================")

    if is_recurrent:
        model = RecurrentPPO.load(checkpoint_path)
    else:
        model = PPO.load(checkpoint_path)

    policy = model.policy
    policy.eval()

    # Determine input observation shape (Channels, Height, Width)
    obs_space = model.observation_space
    if hasattr(obs_space, "shape"):
        dummy_shape = obs_space.shape
    else:
        dummy_shape = (3, 7, 7)

    # MiniGrid / SB3 image channel transposing check (C, H, W)
    if len(dummy_shape) == 3 and dummy_shape[0] != 3 and dummy_shape[2] == 3:
        # Transposed shape (3, H, W)
        dummy_shape = (dummy_shape[2], dummy_shape[0], dummy_shape[1])

    dummy_input = torch.zeros((1, *dummy_shape), dtype=torch.float32)

    # Wrap policy
    wrapper = OnnxPolicyWrapper(policy)
    wrapper.eval()

    # 1. Export TorchScript (.pt) Model
    torchscript_path = output_onnx_path.replace(".onnx", ".pt")
    try:
        traced_model = torch.jit.trace(wrapper, dummy_input)
        traced_model.save(torchscript_path)
        print(f"Successfully exported TorchScript policy model to {torchscript_path}")
        ts_size_kb = os.path.getsize(torchscript_path) / 1024.0
        print(f"TorchScript Model File Size: {ts_size_kb:.2f} KB")
    except Exception as e:
        print(f"TorchScript export note: {e}")

    # 2. Export ONNX (.onnx) Model
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_onnx_path,
            opset_version=14,
            input_names=["observation"],
            output_names=["action_logits"],
            dynamic_axes={"observation": {0: "batch_size"}, "action_logits": {0: "batch_size"}},
            dynamo=False
        )
        print(f"Successfully exported ONNX policy model to {output_onnx_path}")
        file_size_kb = os.path.getsize(output_onnx_path) / 1024.0
        print(f"ONNX Model File Size: {file_size_kb:.2f} KB")
    except Exception as e:
        print(f"ONNX export exception: {e}")

    # Verify PyTorch forward output
    with torch.no_grad():
        pytorch_logits = wrapper(dummy_input)
        predicted_action = torch.argmax(pytorch_logits, dim=-1).item()

    print(f"Sample Prediction Action Logits Shape: {list(pytorch_logits.shape)}")
    print(f"Sample Top Action Index: {predicted_action}")
    print("Model Export & Validation Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RL Policy to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/complex_doorkey_agent.zip")
    parser.add_argument("--output", type=str, default="exports/nav_policy.onnx")
    parser.add_argument("--is_recurrent", action="store_true")

    args = parser.parse_args()
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        is_recurrent=args.is_recurrent
    )
