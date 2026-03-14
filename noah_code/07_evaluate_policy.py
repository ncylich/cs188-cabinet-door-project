"""
Step 7: Evaluate a Trained Policy
=================================
Runs a trained OpenCabinet policy in simulation and reports success rate across
multiple episodes. Supports the temporal lowdim transformer checkpoints from
06_train_policy.py and the older simple MLP checkpoints.

Usage:
    python 07_evaluate_policy.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --num_rollouts 50
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --split target
"""

import argparse
import os
import sys
from collections import deque

# Force osmesa (CPU offscreen renderer) on Linux/WSL2.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

import robocasa  # noqa: F401
import robosuite
from robosuite.controllers import load_composite_controller_config

from policy_utils import (
    augment_lowdim_observation,
    load_checkpoint_policy,
    print_section,
    predict_policy_action,
    unwrap_reset,
    unwrap_step,
)


def load_policy(checkpoint_path, device):
    model, checkpoint, policy_type = load_checkpoint_policy(checkpoint_path, device)

    print(f"Loaded policy from: {checkpoint_path}")
    print(f"  Policy type: {policy_type}")
    print(f"  Trained for {checkpoint['epoch']} epochs")
    print(f"  State dim: {checkpoint['state_dim']}, Action dim: {checkpoint['action_dim']}")

    if policy_type == "temporal_bc_transformer":
        print(f"  Validation loss: {checkpoint.get('val_loss', checkpoint.get('loss', 0.0)):.6f}")
    else:
        print(f"  Loss: {checkpoint['loss']:.6f}")

    return model, checkpoint


def predict_action(model, checkpoint, obs, history, device):
    return predict_policy_action(model, checkpoint, obs, history, device)


def create_eval_env(split, seed, enable_rendering):
    obj_instance_split = None
    layout_ids = None
    style_ids = None
    layout_and_style_ids = None

    if split == "target":
        obj_instance_split = "target"
        layout_and_style_ids = list(zip(range(1, 11), range(1, 11)))
    elif split == "pretrain":
        obj_instance_split = "pretrain"
        layout_ids = -2
        style_ids = -2
    else:
        raise ValueError(f"Unsupported split: {split}")

    return robosuite.make(
        env_name="OpenCabinet",
        robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_widths=256,
        camera_heights=256,
        has_renderer=False,
        has_offscreen_renderer=enable_rendering,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
    )


def run_evaluation(
    model,
    checkpoint,
    num_rollouts,
    max_steps,
    split,
    video_path,
    seed,
):
    import imageio

    device = next(model.parameters()).device

    env = create_eval_env(split=split, seed=seed, enable_rendering=bool(video_path))

    video_writer = None
    if video_path:
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)

    results = {
        "successes": [],
        "episode_lengths": [],
        "rewards": [],
    }

    seq_len = checkpoint.get("seq_len", 1)

    for ep in range(num_rollouts):
        history = deque(maxlen=seq_len)
        active_handle_site = None
        obs = unwrap_reset(env.reset())
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")

        ep_reward = 0.0
        success = False
        done = False

        for step in range(max_steps):
            obs, active_handle_site = augment_lowdim_observation(
                obs=obs,
                env=env,
                state_keys=checkpoint.get("state_keys", []),
                active_handle_site=active_handle_site,
            )
            action = predict_action(model, checkpoint, obs, history, device)

            env_action_dim = env.action_dim
            if len(action) < env_action_dim:
                action = np.pad(action, (0, env_action_dim - len(action)))
            elif len(action) > env_action_dim:
                action = action[:env_action_dim]

            obs, reward, done, info = unwrap_step(env.step(action))
            ep_reward += reward

            if video_writer is not None:
                frame = env.sim.render(
                    height=512, width=768, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(frame)

            success = bool(info.get("success", False) or env._check_success())
            if success or done:
                break

        results["successes"].append(success)
        results["episode_lengths"].append(step + 1)
        results["rewards"].append(ep_reward)

        status = "SUCCESS" if success else "FAIL"
        print(
            f"  Episode {ep + 1:3d}/{num_rollouts}: {status:7s} "
            f"(steps={step + 1:4d}, reward={ep_reward:.1f}) "
            f'layout={env.layout_id}, style={env.style_id}, task="{lang}"'
        )

    if video_writer:
        video_writer.close()

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained OpenCabinet policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policy checkpoint (.pt file)",
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=20, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max_steps", type=int, default=500, help="Max steps per episode"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["pretrain", "target"],
        help="Kitchen scene split to evaluate on",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save evaluation video (optional)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    print("=" * 60)
    print("  OpenCabinet - Policy Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, checkpoint = load_policy(args.checkpoint, device)

    print_section(f"Evaluating on {args.split} split ({args.num_rollouts} episodes)")
    results = run_evaluation(
        model=model,
        checkpoint=checkpoint,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        split=args.split,
        video_path=args.video_path,
        seed=args.seed,
    )

    print_section("Evaluation Results")
    num_success = sum(results["successes"])
    success_rate = num_success / args.num_rollouts * 100
    avg_length = np.mean(results["episode_lengths"])
    avg_reward = np.mean(results["rewards"])

    print(f"  Split:          {args.split}")
    print(f"  Episodes:       {args.num_rollouts}")
    print(f"  Successes:      {num_success}/{args.num_rollouts}")
    print(f"  Success rate:   {success_rate:.1f}%")
    print(f"  Avg ep length:  {avg_length:.1f} steps")
    print(f"  Avg reward:     {avg_reward:.3f}")

    if args.video_path:
        print(f"\n  Video saved to: {args.video_path}")


if __name__ == "__main__":
    main()
