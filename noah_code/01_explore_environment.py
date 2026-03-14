"""
Step 1: Explore the OpenCabinet Environment
=============================================
Creates the OpenCabinet environment and inspects its observation space,
action space, task description, and success criteria.

This script does NOT require a display -- all output is printed to the terminal.

Usage:
    python 01_explore_environment.py
"""

import os
import sys

# Force osmesa (CPU offscreen renderer) on Linux/WSL2 -- EGL requires
# /dev/dri device access that is unavailable in WSL environments.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import robocasa  # noqa: F401 - registers environments
from robocasa.utils.env_utils import create_env


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    # ----------------------------------------------------------------
    # 1. Create the environment
    # ----------------------------------------------------------------
    print_section("Creating OpenCabinet Environment")

    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=42,
        camera_widths=128,
        camera_heights=128,
    )
    obs = env.reset()
    ep_meta = env.get_ep_meta()

    print(f"Task:           OpenCabinet")
    print(f"Robot:          {env.robots[0].name}")
    print(f"Robot model:    {env.robots[0].robot_model.__class__.__name__}")
    print(f"Description:    {ep_meta.get('lang', '')}")
    print(f"Kitchen layout: {env.layout_id}")
    print(f"Kitchen style:  {env.style_id}")
    print(f"Horizon:        {env.horizon} steps ({env.horizon / env.control_freq:.1f}s)")
    print(f"Control freq:   {env.control_freq} Hz")

    # ----------------------------------------------------------------
    # 2. Inspect the observation space
    # ----------------------------------------------------------------
    print_section("Observation Space")

    print(f"{'Key':<45} {'Shape':>15} {'Dtype':>10} {'Range':>20}")
    print("-" * 95)
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, np.ndarray):
            shape = str(val.shape)
            dtype = str(val.dtype)
            vmin, vmax = val.min(), val.max()
            range_str = f"[{vmin:.3f}, {vmax:.3f}]"
        else:
            shape = str(type(val).__name__)
            dtype = "-"
            range_str = str(val)[:20]
        print(f"  {key:<43} {shape:>15} {dtype:>10} {range_str:>20}")

    # Highlight the key observations for policy learning
    print("\nKey observations for policy learning:")
    important_keys = [
        "robot0_gripper_qpos",
        "robot0_base_pos",
        "robot0_base_quat",
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
    ]
    for key in important_keys:
        if key in obs:
            print(f"  {key}: {obs[key]}")

    image_keys = [k for k in obs.keys() if k.endswith("_image")]
    print(f"\nCamera images ({len(image_keys)}):")
    for key in image_keys:
        print(f"  {key}: shape={obs[key].shape}, dtype={obs[key].dtype}")

    # ----------------------------------------------------------------
    # 3. Inspect the action space
    # ----------------------------------------------------------------
    print_section("Action Space")

    robot = env.robots[0]
    cc = robot.composite_controller
    print(f"Composite controller type: {cc.__class__.__name__}")
    print(f"Total action dim:          {env.action_dim}")
    print(f"\nAction components:")
    for part_name in cc.part_controllers:
        start, end = cc._action_split_indexes[part_name]
        dim = end - start
        low = cc.action_limits[0][start:end]
        high = cc.action_limits[1][start:end]
        print(f"  {part_name:<25} dim={dim:>2}  range=[{low[0]:.1f}, {high[0]:.1f}]")

    print("\nAction mapping for the PandaOmron:")
    print("  Indices  0-5:   Arm end-effector (3 pos + 3 rot)")
    print("  Index    6:     Gripper (-1=open, 1=close)")
    print("  Indices  7-9:   Base motion (forward, side, yaw)")
    print("  Index    10:    Torso lift")
    print("  Index    11:    Control mode toggle (-1=arm, 1=base)")

    # ----------------------------------------------------------------
    # 4. Test the success criteria
    # ----------------------------------------------------------------
    print_section("Success Criteria")

    success_before = env._check_success()
    print(f"Is cabinet open at start? {success_before}")
    print(f"(Expected: False -- the cabinet starts closed)")

    # Show what fixture we're working with
    if hasattr(env, "fxtr"):
        fxtr = env.fxtr
        print(f"\nTarget fixture: {fxtr.name}")
        print(f"Fixture class:  {fxtr.__class__.__name__}")
        print(f"Natural lang:   {fxtr.nat_lang}")

    # ----------------------------------------------------------------
    # 5. Take a single step
    # ----------------------------------------------------------------
    print_section("Single Step")

    action = np.zeros(env.action_dim)
    obs_next, reward, done, info = env.step(action)

    print(f"Reward:     {reward}")
    print(f"Done:       {done}")
    print(f"Info keys:  {list(info.keys())}")
    print(f"\nReward is sparse: 1.0 if cabinet is open, 0.0 otherwise.")

    # ----------------------------------------------------------------
    # 6. List other door-related tasks
    # ----------------------------------------------------------------
    print_section("Related Door/Cabinet Tasks in RoboCasa")

    from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS

    door_tasks = sorted(
        name
        for name in REGISTERED_KITCHEN_ENVS
        if any(
            kw in name.lower()
            for kw in ["door", "cabinet", "fridge", "oven", "microwave", "dishwasher"]
        )
    )
    print("Available door/cabinet manipulation tasks:")
    for task in door_tasks:
        print(f"  - {task}")

    env.close()
    print("\nDone! Proceed to 02_random_rollouts.py")


if __name__ == "__main__":
    main()
