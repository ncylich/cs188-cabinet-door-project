# Cabinet Door Opening Robot

## Overview

This repository contains a RoboCasa `OpenCabinet` project built around two use
cases:

1. A compact starter workflow for exploring the environment, downloading data,
   and training a simple baseline.
2. The final project experiments, including the BC U-Net and diffusion U-Net
   workflows documented below.

### The robot

We use the **PandaOmron** mobile manipulator -- a Franka Panda 7-DOF arm
mounted on an Omron wheeled base with a torso lift joint. This is the default
and best-supported robot in RoboCasa.

---

## Installation

Run the install script (works on **macOS** and **WSL/Linux**):

```bash
./install.sh
```

This will:
- Create a Python virtual environment (`.venv`)
- Clone and install robosuite and robocasa
- Install all Python dependencies (PyTorch, numpy, matplotlib, etc.)
- Download RoboCasa kitchen assets (~10 GB)

After installation, activate the environment:

```bash
source .venv/bin/activate
```

Then verify everything works:

```bash
cd cabinet_door_project
python 00_verify_installation.py
```

> **macOS note:** Scripts that open a rendering window (03, 05) require
> `mjpython` instead of `python`. The install script will remind you of this.

---

## Project Structure

```
cabinet_door_project/
  00_verify_installation.py
  01_explore_environment.py
  02_random_rollouts.py
  03_teleop_collect_demos.py
  04_download_dataset.py
  05_playback_demonstrations.py
  06_train_policy.py
  07_evaluate_policy.py
  08_visualize_policy_rollout.py
  prepare_dataset.py             # Build oracle / handle feature caches
  validate_best.py               # Reproduce the final pretrain BC U-Net run
  bc_handle.py                   # Handle-aware BC / diffusion training entrypoint
  compare_bc_vs_diffusion.py     # Matched BC vs diffusion experiments
  ablation_sweep.py              # Feature / method sweep code
  preprocess_all_states.py       # Precompute oracle state features
  generate_door_positions.py     # Replay-based door geometry extraction
  diffusion_policy/              # Core training / inference / evaluation package
  tests/                         # Regression tests for the diffusion policy code
  configs/
    diffusion_policy.yaml
install.sh
README.md
```

---

## Starter Workflow

For the lightweight baseline path:

```bash
python 00_verify_installation.py
python 01_explore_environment.py
python 04_download_dataset.py
python 05_playback_demonstrations.py
python 06_train_policy.py
python 07_evaluate_policy.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt
```

Use `03_teleop_collect_demos.py` if you want to collect demonstrations
manually. On macOS, scripts that open a renderer window should be run with
`mjpython`.

---

## Reproducing The Main U-Net Experiments

If you want to reproduce the main U-Net workflows used in this project, use
the commands below instead of the starter `06_train_policy.py` script. These
commands assume you already ran `./install.sh`, activated `.venv`, and are
working from `cabinet_door_project/`.

### Step A: Prepare The Oracle / Handle Features

Download the dataset, then build the cached oracle features used by the final
U-Net experiments:

```bash
python 04_download_dataset.py
python prepare_dataset.py --n_workers 4
python prepare_dataset.py --validate_only
```

This produces the artifacts consumed by the final experiments under
`/tmp/diffusion_policy_checkpoints/`, including:

- `door_positions.npz`
- `door_quats.npz`
- `preprocessed_all_states.pt`
- `handle_cache/`

### Step B: Reproduce The Final Pretrain BC U-Net Result

Run:

```bash
python validate_best.py
```

What this does:

- Retrains the final **F3 BC U-Net** on the 107-demo pretrain dataset
- Uses the 22-dim handle-aware state:
  `proprio + handle_pos + handle_to_eef`
- Evaluates **100 episodes** with the relaxed one-door-open success criterion

Expected output:

- Checkpoint written to `/tmp/diffusion_policy_checkpoints/best_f3_bc_unet.pt`
- Final summary line close to:

```text
FINAL RESULT: 44/100 success (44.0%)
```

Wall-clock time depends heavily on GPU / CPU count. Training is short; the
100-episode evaluation is the expensive part.

### Step C: Reproduce The Checked-In Diffusion U-Net Run

Run:

```bash
python bc_handle.py \
  --arch unet \
  --horizon 16 \
  --n_obs_steps 2 \
  --n_action_steps 8 \
  --epochs 300 \
  --patience 40 \
  --ddpm_steps 100 \
  --ddim_steps 10 \
  --n_eps 20 \
  --max_steps 500
```

What this does:

- Trains the repository's **handle-aware diffusion U-Net**
- Uses the 44-dim handle-relative oracle state from `bc_handle.py`
- Evaluates **20 episodes** in the same pretrain-only setting

Outputs:

- Checkpoint written to `/tmp/diffusion_policy_checkpoints/bc_unet_best.pt`
- Eval summary printed at the end as:

```text
Result: X/20 (Y.Y%)
```

To re-run eval only from the saved checkpoint:

```bash
python bc_handle.py \
  --arch unet \
  --eval_only \
  --checkpoint /tmp/diffusion_policy_checkpoints/bc_unet_best.pt \
  --n_eps 20 \
  --max_steps 500
```

### Notes

- The BC U-Net path above is the reproducible path for the **44/100**
  pretrain result reported in the final writeup.
- The diffusion U-Net command above reproduces the **checked-in pretrain
  diffusion U-Net pipeline** in this repo.

---

## Key Concepts

### The OpenCabinet Task

- **Goal**: Open a kitchen cabinet door
- **Fixture**: `HingeCabinet` (a cabinet with hinged doors)
- **Initial state**: Cabinet door is closed; robot is positioned nearby
- **Success**: `fixture.is_open(env)` returns `True`
- **Horizon**: 500 timesteps at 20 Hz control frequency (25 seconds)
- **Scene variety**: 2,500+ kitchen layouts/styles for generalization

### Observation Space (PandaOmron)

| Key | Shape | Description |
|-----|-------|-------------|
| `robot0_agentview_left_image` | (256, 256, 3) | Left shoulder camera |
| `robot0_agentview_right_image` | (256, 256, 3) | Right shoulder camera |
| `robot0_eye_in_hand_image` | (256, 256, 3) | Wrist-mounted camera |
| `robot0_gripper_qpos` | (2,) | Gripper finger positions |
| `robot0_base_pos` | (3,) | Base position (x, y, z) |
| `robot0_base_quat` | (4,) | Base orientation quaternion |
| `robot0_base_to_eef_pos` | (3,) | End-effector pos relative to base |
| `robot0_base_to_eef_quat` | (4,) | End-effector orientation relative to base |

### Action Space (PandaOmron)

| Key | Dim | Description |
|-----|-----|-------------|
| `end_effector_position` | 3 | Delta (dx, dy, dz) for the end-effector |
| `end_effector_rotation` | 3 | Delta rotation (axis-angle) |
| `gripper_close` | 1 | 0 = open, 1 = close |
| `base_motion` | 4 | (forward, side, yaw, torso) |
| `control_mode` | 1 | 0 = arm control, 1 = base control |

### Dataset Format (LeRobot)

Datasets are stored in LeRobot format:
```
dataset/
  meta/           # Episode metadata (task descriptions, camera info)
  videos/         # MP4 videos from each camera
  data/           # Parquet files with actions, states, rewards
  extras/         # Per-episode metadata
```

---

## Troubleshooting

I'll continually update this section as students find bugs in the system. Please, let me know if you encounter issues!

| Problem | Solution |
|---------|----------|
| `MuJoCo version must be 3.3.1` | `pip install mujoco==3.3.1` |
| `numpy version must be 2.2.5` | `pip install numpy==2.2.5` |
| Rendering crashes on Mac | Use `mjpython` instead of `python` |
| `GLFW error` on headless server | Set `export MUJOCO_GL=egl` or `osmesa` |
| Out of GPU memory during training | Reduce batch size in `configs/diffusion_policy.yaml` |
| Kitchen assets not found | Run `python -m robocasa.scripts.download_kitchen_assets` |

---

## References

- [RoboCasa Paper & Website](https://robocasa.ai/)
- [RoboCasa GitHub](https://github.com/robocasa/robocasa)
- [robosuite Documentation](https://robosuite.ai/)
- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
