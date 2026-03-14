"""
Step 2: Random Rollouts
========================
Runs the OpenCabinet environment with random actions and saves a video.
This demonstrates that random actions are insufficient to solve the task,
motivating the need for demonstrations and policy learning.

Usage:
    python 02_random_rollouts.py [--num_rollouts 3] [--num_steps 100] [--video_path /tmp/cabinet_random.mp4]

Mac users: use mjpython instead of python for on-screen rendering.
"""

import argparse
import os
import sys

# Force osmesa (CPU offscreen renderer) on Linux/WSL2 -- EGL requires
# /dev/dri device access that is unavailable in WSL environments.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

import robocasa  # noqa: F401
from robocasa.utils.env_utils import create_env, run_random_rollouts
import gymnasium as gym


def main():
    parser = argparse.ArgumentParser(description="Run random rollouts on OpenCabinet")
    parser.add_argument("--num_rollouts", type=int, default=3, help="Number of episodes")
    parser.add_argument("--num_steps", type=int, default=100, help="Steps per episode")
    parser.add_argument(
        "--video_path",
        type=str,
        default="/tmp/cabinet_random_rollouts.mp4",
        help="Path to save the video",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Random Rollouts")
    print("=" * 60)

    # Create environment via the gymnasium interface
    # This is the standard way you'll interact with RoboCasa environments
    print("\nCreating gym environment...")
    env = gym.make(
        "robocasa/OpenCabinet",
        split="pretrain",
        seed=args.seed,
    )

    print(f"Observation space keys: {list(env.observation_space.spaces.keys())}")
    print(f"Action space keys:      {list(env.action_space.spaces.keys())}")

    # Run rollouts with random actions
    print(f"\nRunning {args.num_rollouts} rollouts of {args.num_steps} steps each...")
    print(f"Saving video to: {args.video_path}")

    info = run_random_rollouts(
        env,
        num_rollouts=args.num_rollouts,
        num_steps=args.num_steps,
        video_path=args.video_path,
    )

    successes = info.get("num_success_rollouts", 0)
    print(f"\nResults:")
    print(f"  Successes: {successes}/{args.num_rollouts}")
    print(f"  Success rate: {successes / args.num_rollouts * 100:.1f}%")
    print(f"\n  (A ~0% success rate with random actions is expected!)")
    print(f"  This is why we need demonstrations and policy learning.")

    env.close()
    print(f"\nVideo saved to: {args.video_path}")
    print("Done! Proceed to 03_teleop_collect_demos.py")


if __name__ == "__main__":
    main()
