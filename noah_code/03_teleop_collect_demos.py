"""
Step 3: Teleoperate the Robot to Collect Demonstrations
========================================================
Opens an interactive window where you control the PandaOmron robot
with your keyboard (or SpaceMouse) to open cabinet doors.

This gives you hands-on intuition for the task. Note: in normal mode
this script does NOT save demonstration data to disk. To get training
data, run 04_download_dataset.py to download the pre-collected dataset.

DAgger mode (--dagger):
    The trained policy drives the robot autonomously. Press movement
    keys to override the policy at any time. All (state, action) pairs
    are saved as parquet files compatible with the training pipeline in
    06_train_policy.py. This implements Dataset Aggregation (DAgger) —
    a simple way to improve a policy by collecting corrections.

Usage:
    # Mac users MUST use mjpython for the rendering window
    mjpython 03_teleop_collect_demos.py

    # Linux users
    python 03_teleop_collect_demos.py

    # Use spacemouse instead of keyboard
    python 03_teleop_collect_demos.py --device spacemouse

    # DAgger mode: policy drives, human overrides with keyboard
    mjpython 03_teleop_collect_demos.py --dagger --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt

    # DAgger with custom output directory
    mjpython 03_teleop_collect_demos.py --dagger --checkpoint best_policy.pt --save_dir data/dagger_round2/chunk-000

    Recording:
        Q       - Discard the current episode
"""

import os
import sys

# ── WSLg / XWayland GL setup — re-exec approach ─────────────────────────────
# On WSLg (Windows 11) the .bashrc often sets DISPLAY to a stale VcXsrv-style
# "IP:0" address, and Mesa's D3D12 GPU path fails with gladLoadGL on XWayland.
#
# Setting os.environ inside a running Python process is NOT reliable: C
# libraries (Mesa, GLFW) may read the env at dlopen() time which happens
# during import, before our code runs.  The only guaranteed fix is to restart
# the process with the correct vars already in the OS-level environment.
#
# We use os.execve() to atomically replace this process with an identical one
# that has the correct vars from the very start.  A sentinel env var prevents
# the new process from re-execing again.
if sys.platform == "linux" and "__TELEOP_DISPLAY_OK" not in os.environ:
    _env = dict(os.environ)
    _changed = False

    if _env.get("WAYLAND_DISPLAY"):
        # WSLg: force XWayland socket display and Mesa software renderer.
        if not _env.get("DISPLAY", "").startswith(":"):
            _env["DISPLAY"] = ":0"
            _changed = True
        if _env.get("GALLIUM_DRIVER") != "llvmpipe":
            _env["GALLIUM_DRIVER"] = "llvmpipe"
            _changed = True
        if _env.get("MESA_GL_VERSION_OVERRIDE") != "4.5":
            _env["MESA_GL_VERSION_OVERRIDE"] = "4.5"
            _changed = True

    if _changed:
        _env["__TELEOP_DISPLAY_OK"] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, _env)
    else:
        os.environ["__TELEOP_DISPLAY_OK"] = "1"
# ────────────────────────────────────────────────────────────────────────────

import argparse
import json
import time
from collections import deque
from copy import deepcopy

import numpy as np
import robocasa  # noqa: F401 - registers environments including OpenCabinet
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from policy_utils import (
    augment_lowdim_observation,
    extract_lowdim_state,
    load_checkpoint_policy,
    predict_policy_action,
    unwrap_reset,
    unwrap_step,
)


# ── DAgger helpers ───────────────────────────────────────────────────────


def save_trajectory_parquet(trajectory, save_dir, episode_index):
    """
    Save a list of {state, action} dicts as a parquet file.

    The output schema matches what CabinetDemoDataset in 06_train_policy.py
    expects: columns ``observation.state`` and ``action``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(save_dir, exist_ok=True)

    states = [step["state"].tolist() for step in trajectory]
    actions = [step["action"].tolist() for step in trajectory]

    table = pa.table(
        {
            "observation.state": states,
            "action": actions,
        }
    )

    path = os.path.join(save_dir, f"episode_{episode_index:06d}.parquet")
    pq.write_table(table, path)
    return path


def save_dagger_metadata(save_dir, checkpoint_path, checkpoint, policy_type):
    os.makedirs(save_dir, exist_ok=True)
    metadata = {
        "format": "dagger_precomputed_lowdim_v1",
        "checkpoint": checkpoint_path,
        "policy_type": policy_type,
        "state_dim": checkpoint["state_dim"],
        "action_dim": checkpoint["action_dim"],
        "state_keys": checkpoint.get("state_keys"),
        "seq_len": checkpoint.get("seq_len", 1),
    }
    metadata_path = os.path.join(save_dir, "dagger_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def collect_dagger_trajectory(
    env,
    device,
    model,
    checkpoint,
    torch_device,
    mirror_actions=True,
    policy_max_fr=30,
    human_max_fr=10,
):
    """
    Collect a single DAgger trajectory.

    The trained policy drives the robot. When the human presses movement
    keys, their input overrides the policy. All (state, action) pairs are
    recorded regardless of who was in control.

    Returns:
        (success, trajectory): success bool and list of {state, action} dicts.
    """
    import torch

    obs = unwrap_reset(env.reset())
    state_keys = checkpoint.get("state_keys")
    action_dim = checkpoint["action_dim"]
    history = deque(maxlen=checkpoint.get("seq_len", 1))
    active_handle_site = None

    ep_meta = env.get_ep_meta()
    lang = ep_meta.get("lang", None)
    if lang is not None:
        print(f"  Task: {lang}")

    task_completion_hold_count = -1
    device_input = device
    device_input.start_control()

    # Track gripper state
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Dummy step to initialize
    zero_action = np.zeros(env.action_dim)
    obs, _, _, _ = unwrap_step(env.step(zero_action))

    discard_traj = False
    trajectory = []
    step_count = 0
    human_took_over = False

    while True:
        start = time.time()

        active_robot = env.robots[device_input.active_robot]

        obs, active_handle_site = augment_lowdim_observation(
            obs=obs,
            env=env,
            state_keys=state_keys or [],
            active_handle_site=active_handle_site,
        )
        state = extract_lowdim_state(obs, state_keys=state_keys)

        # Get human input
        input_ac_dict = device_input.input2action(mirror_actions=mirror_actions)

        if input_ac_dict is None:
            discard_traj = True
            break

        action_dict = deepcopy(input_ac_dict)

        # Set arm actions based on controller type
        for arm in active_robot.arms:
            controller_input_type = active_robot.part_controllers[arm].input_type
            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]

        # Detect human activity: check if right_delta or base actions are non-zero
        human_active = False
        right_delta = input_ac_dict.get("right_delta", None)
        if right_delta is not None and np.any(right_delta != 0):
            human_active = True
        base_action = input_ac_dict.get("base", None)
        if base_action is not None and np.any(base_action != 0):
            human_active = True

        if human_active and not human_took_over:
            human_took_over = True
            print("\n  Human override detected. Policy control disabled for this episode.")

        if human_took_over:
            # Sticky handoff: after the first human intervention, the policy stays off.
            history.append(state)
            env_action = [
                robot.create_action_vector(all_prev_gripper_actions[i])
                for i, robot in enumerate(env.robots)
            ]
            env_action[device_input.active_robot] = (
                active_robot.create_action_vector(action_dict)
            )
            env_action = np.concatenate(env_action)
        else:
            # Policy drives using the same enriched lowdim features as eval.
            policy_action = predict_policy_action(
                model=model,
                checkpoint=checkpoint,
                obs=obs,
                history=history,
                device=torch_device,
            )

            # Pad/trim to environment action dimension
            env_dim = env.action_dim
            if len(policy_action) < env_dim:
                policy_action = np.pad(policy_action, (0, env_dim - len(policy_action)))
            elif len(policy_action) > env_dim:
                policy_action = policy_action[:env_dim]
            env_action = policy_action

        # Step the environment
        obs, _, _, _ = unwrap_step(env.step(env_action))

        # Record (state, action) — trim action to action_dim for training
        recorded_action = env_action[:action_dim]
        trajectory.append({"state": state, "action": recorded_action})

        # Status line every 10 steps
        step_count += 1
        if step_count % 10 == 0:
            who = "[HUMAN]" if human_took_over else "[policy]"
            print(f"\r  step {step_count:4d}  {who}  "
                  f"traj_len={len(trajectory)}", end="", flush=True)

        # Check for task completion (15 consecutive success steps)
        if task_completion_hold_count == 0:
            break

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 14
        else:
            task_completion_hold_count = -1

        # Frame rate limiting
        current_max_fr = human_max_fr if human_took_over else policy_max_fr
        if current_max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / current_max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # Clear the \r status line
    print()

    success = not discard_traj
    return success, trajectory


def collect_trajectory(env, device, mirror_actions=True, max_fr=30):
    """
    Collect a single teleoperation trajectory.

    This is a simplified version of RoboCasa's collect_human_trajectory
    that avoids the circular import in robocasa.scripts.collect_demos.

    Returns:
        success (bool): Whether the cabinet was opened during the episode.
    """
    env.reset()

    ep_meta = env.get_ep_meta()
    lang = ep_meta.get("lang", None)
    if lang is not None:
        print(f"  Task: {lang}")

    # Counter: task must be successful for 15 consecutive timesteps.
    # Counts down from 14 to 0, then breaks on the next check (= 15 steps total).
    task_completion_hold_count = -1
    device.start_control()
    nonzero_ac_seen = False

    # Track gripper state
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Do a dummy step to initialize
    zero_action = np.zeros(env.action_dim)
    env.step(zero_action)

    discard_traj = False

    while True:
        start = time.time()

        active_robot = env.robots[device.active_robot]

        # Get action from input device
        input_ac_dict = device.input2action(mirror_actions=mirror_actions)

        # None means the user pressed Q (reset signal)
        if input_ac_dict is None:
            discard_traj = True
            break

        action_dict = deepcopy(input_ac_dict)

        # Set arm actions based on controller type
        for arm in active_robot.arms:
            controller_input_type = active_robot.part_controllers[arm].input_type
            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]

        # Skip if no meaningful input yet (spacemouse idle)
        if not nonzero_ac_seen:
            is_empty = np.all(action_dict.get("right_delta", np.array([1])) == 0)
            if is_empty:
                continue
            nonzero_ac_seen = True

        # Build full action vector
        env_action = [
            robot.create_action_vector(all_prev_gripper_actions[i])
            for i, robot in enumerate(env.robots)
        ]
        env_action[device.active_robot] = active_robot.create_action_vector(
            action_dict
        )
        env_action = np.concatenate(env_action)

        # Step the environment
        obs, _, _, _ = env.step(env_action)

        # Check for task completion (15 consecutive success steps)
        if task_completion_hold_count == 0:
            break

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 14
        else:
            task_completion_hold_count = -1

        # Frame rate limiting
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    success = not discard_traj
    return success


def _check_display():
    """Exit early with helpful instructions if no X display is available."""
    display = os.environ.get("DISPLAY", "")
    wayland = os.environ.get("WAYLAND_DISPLAY", "")

    if wayland:
        # WSLg is running. It provides XWayland at :0, which both pynput
        # (X11-based) and MuJoCo's GLFW viewer need.
        if not display or not display.startswith(":"):
            os.environ["DISPLAY"] = ":0"
        # Force Mesa software rendering (llvmpipe) for the GLFW window.
        # WSLg's Mesa D3D12 GPU path often fails with gladLoadGL; llvmpipe
        # is slower but reliably provides OpenGL 4.5 for the viewer.
        os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
        os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.5")
        return

    if display:
        # Some X display is claimed — let MuJoCo's own error handling deal
        # with actual render failures.
        return

    # Nothing set at all.
    print("ERROR: This script requires a display (X server) for the MuJoCo viewer")
    print("       and keyboard input. No display environment variable is set.")
    print()
    print("Windows 11 (WSLg — recommended, no extra software needed):")
    print("  WSLg provides a built-in display. If DISPLAY is not set, try:")
    print("  export DISPLAY=:0")
    print()
    print("Windows 10 / VcXsrv (Maybe MacOS too?):")
    print("  1. Launch XLaunch, on 'Extra settings' uncheck 'Native opengl'")
    print("     and check 'Disable access control'")
    print("  2. export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0")
    print("  3. export LIBGL_ALWAYS_INDIRECT=0")
    print()
    print("Note: steps 01, 02, 04-07 do NOT require a display.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Teleoperate robot for OpenCabinet")
    parser.add_argument(
        "--layout", type=int, default=None, help="Kitchen layout ID (1-60)"
    )
    parser.add_argument(
        "--style", type=int, default=None, help="Kitchen style ID (1-60)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="keyboard",
        choices=["keyboard", "spacemouse"],
        help="Input device",
    )
    parser.add_argument(
        "--dagger",
        action="store_true",
        help="Enable DAgger mode: policy drives, human overrides with keyboard",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to policy checkpoint (.pt) — required with --dagger",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/dagger/chunk-000",
        help="Where to save DAgger trajectories (default: data/dagger/chunk-000)",
    )
    parser.add_argument(
        "--policy_max_fr",
        type=int,
        default=30,
        help="Control rate cap while the policy is driving in DAgger mode",
    )
    parser.add_argument(
        "--human_max_fr",
        type=int,
        default=10,
        help="Slower control rate cap after human takeover in DAgger mode",
    )
    args = parser.parse_args()

    if args.dagger and not args.checkpoint:
        parser.error("--checkpoint is required when using --dagger")

    # _check_display()

    print("=" * 60)
    print("  OpenCabinet - Teleoperation Demo Collection")
    print("=" * 60)
    print()

    # Create the environment
    config = {
        "env_name": "OpenCabinet",
        "robots": "PandaOmron",
        "controller_configs": load_composite_controller_config(robot="PandaOmron"),
        "layout_ids": args.layout,
        "style_ids": args.style,
        "translucent_robot": True,
    }

    print("Initializing environment (this may take a moment)...")
    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="robot0_frontview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
        use_object_obs=args.dagger,
    )

    env = VisualizationWrapper(env)

    # Initialize input device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)
    elif args.device == "spacemouse":
        import robocasa.macros as macros
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=4.0,
            rot_sensitivity=4.0,
            vendor_id=macros.SPACEMOUSE_VENDOR_ID,
            product_id=macros.SPACEMOUSE_PRODUCT_ID,
        )

    # ── DAgger mode setup ──────────────────────────────────────────────────
    if args.dagger:
        import torch

        if not os.path.exists(args.checkpoint):
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
            print("Train a policy first with:  python 06_train_policy.py")
            sys.exit(1)

        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, ckpt, policy_type = load_checkpoint_policy(args.checkpoint, torch_device)
        metadata_path = save_dagger_metadata(
            save_dir=args.save_dir,
            checkpoint_path=args.checkpoint,
            checkpoint=ckpt,
            policy_type=policy_type,
        )

        print(f"DAgger mode enabled")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Policy:     {policy_type}")
        print(f"  Epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f}")
        print(f"  State dim: {ckpt['state_dim']},  Action dim: {ckpt['action_dim']}")
        print(f"  Save dir:  {args.save_dir}")
        print(f"  Metadata:  {metadata_path}")
        print()
        print("The policy will drive the robot automatically.")
        print("Press movement keys once to take over. After that, policy control")
        print("stays off for the rest of the episode and the loop slows down.")
        print("Press Q to discard an episode.")
        print()

    # ── Episode loop ─────────────────────────────────────────────────────
    if not args.dagger:
        print("\nReady! Move the robot to open the cabinet door.")
        print("Press Q when done with each episode.\n")

    episode = 0
    saved_count = 0
    try:
        while True:
            episode += 1
            print(f"--- Episode {episode} ---")

            if args.dagger:
                success, trajectory = collect_dagger_trajectory(
                    env,
                    device,
                    model,
                    ckpt,
                    torch_device,
                    mirror_actions=True,
                    policy_max_fr=args.policy_max_fr,
                    human_max_fr=args.human_max_fr,
                )
                if success and trajectory:
                    path = save_trajectory_parquet(
                        trajectory, args.save_dir, saved_count
                    )
                    saved_count += 1
                    print(f"  Result: saved {len(trajectory)} steps -> {path}")
                else:
                    print(f"  Result: Discarded")
            else:
                success = collect_trajectory(
                    env, device, mirror_actions=True, max_fr=30
                )
                status = "SUCCESS" if success else "Discarded"
                print(f"  Result: {status}")

            print()
    except KeyboardInterrupt:
        if args.dagger:
            print(f"\nDAgger collection ended. Saved {saved_count} episodes "
                  f"to {args.save_dir}")
        else:
            print("\nTeleoperation ended.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
