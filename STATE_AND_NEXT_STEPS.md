# State Transfer Document — OpenCabinet Diffusion Policy Project

**Date:** 2026-03-14
**Hardware so far:** NVIDIA A100-SXM4-80GB, CUDA 12.4
**Target hardware:** 4× GPU machine (use same codebase, same venv)

---

## 1. Quick-Start on New Machine

```bash
./install.sh                          # Creates .venv, installs deps, downloads assets (~10GB)
source .venv/bin/activate
cd cabinet_door_project
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa   # headless Linux required
python 00_verify_installation.py      # sanity check

# Primary training script (all recent work):
python bc_handle.py --help

# Re-run the current best eval (5% success rate):
python bc_handle.py --eval_only --checkpoint /tmp/diffusion_policy_checkpoints/bc_handle_best.pt --n_eps 50
```

---

## 2. Task Description

**Task:** `OpenCabinet` in RoboCasa — PandaOmron mobile manipulator (7-DOF Franka arm on Omron wheeled base) must open a kitchen cabinet door in MuJoCo simulation. Each eval episode spawns a random kitchen from 2500+ layout variants. Success = any door joint ≥ 90% open within 500 timesteps.

**Published benchmark baseline:** 30–60% pretrain success (RoboCasa paper, using MimicGen data).

**Action space (12-dim, HybridMobileBase composite controller):**
```
env format:  [eef_pos(3), eef_rot(3), gripper(1), base_motion(3), torso(1), base_mode(1)]
             dims: 0:3        3:6        6           7:10           10         11

LeRobot format (dataset): [base_motion(3), torso(1), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]
                           dims: 0:3          3         4               5:8        8:11         11
```
Gripper and base_mode/control_mode are binary: raw values `{-1=open/off, +1=close/on}`.

---

## 3. Critical Bug Found and Fixed (ROOT CAUSE OF ALL 0% RESULTS)

### ⚠️ Bug: Gripper Binarization Threshold (FIXED)
**File:** `diffusion_policy/evaluation.py` → `dataset_action_to_env_action()`

```python
# BROKEN (was):
env_action[6] = -1.0 if dataset_action[11] < 0.5 else 1.0

# FIXED (now):
env_action[6] = -1.0 if dataset_action[11] < 0.0 else 1.0
```

**Why it was broken:** Raw action space is `{-1=open, +1=close}` with training data mean ≈ -0.643 (82% open). Any neutral model prediction (0 in normalized space) denormalizes to -0.643, which is < 0.5 → gripper always stays open. The robot could approach the handle but never grasp it.

**Fix:** Threshold changed from 0.5 → 0.0 (the midpoint of the raw `{-1, +1}` space). Same fix applied to base_mode dim (line 42, same file).

**Validation:** After fix, `bc_handle_best.pt` gets **5% success (1/20)** — matches expected baseline. Before fix: 0/20.

**Impact:** ALL previous evaluation results (Phases 1–12) were collected with this bug. Any 0% success rates are **invalid/unreliable**. The robot was physically approaching the handle (distances of 5–15cm) but the gripper never closed.

---

## 4. Current Best Checkpoint

| File | Arch | State dim | Val loss | Eval (fixed threshold) |
|------|------|-----------|----------|------------------------|
| `/tmp/diffusion_policy_checkpoints/bc_handle_best.pt` | BCTransformer | 44-dim oracle | ~best | **5% (1/20)** |
| `/tmp/diffusion_policy_checkpoints/bc_unet_best.pt` | 1D UNet diffusion | 44-dim oracle | 0.0534 | UNKNOWN — re-eval needed |
| `/tmp/diffusion_policy_checkpoints/bc_split_grip_best.pt` | Split arm+gripper | 44-dim oracle | ? | UNKNOWN — re-eval needed |

**44-dim oracle state** (computed by `preprocess_all_states.py`, cached at `/tmp/diffusion_policy_checkpoints/feature_cache/`):
```
proprio(16): base_pos(3) + base_quat(4) + base_to_eef_pos(3) + base_to_eef_quat(4) + gripper_qpos(2)
door_pos(3) + door_quat(4)           — cabinet door position/orientation (oracle)
eef_pos(3) + eef_quat(4)            — global EEF position/orientation
door_to_eef_pos(3) + door_to_eef_quat(4) — relative door-to-EEF transform
gripper_dist(1)                      — distance between gripper fingers
handle_pos(3)                        — handle world position (oracle)
handle_to_eef_pos(3)                 — handle-to-EEF displacement (oracle)
```

---

## 5. All Experiments — Full History

> **IMPORTANT:** Experiments from Phases 1–12 were run with the broken gripper threshold (0.5 on denormalized values). Any success-rate results of 0% are unreliable. Training losses and distance-reduction metrics are still valid.

### Phase 1: Low-Dim Backbone (training loss only, no env eval)
| Exp | Model | Loss | Notes |
|-----|-------|------|-------|
| H1-MLP | V1 Diffusion MLP (2.9M) | 0.065 | Baseline |
| H1-UNet | V2 Diffusion UNet (30.1M) | 0.002 | 33× better than MLP |
| H1-Transformer | V3 Diffusion Transformer (6.4M) | 0.026 | |

**Conclusion:** Models learn to fit training data. Low-dim policy (16-dim proprio only) insufficient for eval.

### Phase 2: Visuomotor — ImageNet spatial softmax
| Exp | Encoder | Loss | Eval | Notes |
|-----|---------|------|------|-------|
| H4 | ImageNet ResNet18 + spatial softmax | 0.011 | ~40% distance reduction | Best visual encoder found |
| H5 | R3M ResNet18 frozen | 0.015 | ~7% distance reduction | Worse than ImageNet — global pool loses spatial info |

**Conclusion:** Vision needed but frozen global-pool encoders insufficient. Spatial softmax better.

### Phases 3–11: Various Ablations
All ran with broken gripper threshold → **success rates meaningless**. Distance metrics still valid.
- Orientation features: marginally helpful
- Binary gripper head: tested but effect unclear due to bug
- Reward shaping: no effect
- Action horizon tuning: horizon=16, n_action_steps=8 used throughout

### Phase 12: Oracle State Experiments (INVALIDATED BY GRIPPER BUG)
These were run with `bc_handle.py` using 44-dim oracle state. All got 0/20 with broken threshold.

| Exp | Description | Success (broken) | Status after fix |
|-----|-------------|-----------------|-----------------|
| 12a | bc_handle_best.pt, relaxed (any door) criterion | ~~0/20~~ | **5% (1/20) ✓** |
| 12b | U-Net diffusion, 44-dim oracle | ~~0/20~~ | **NEEDS RE-EVAL** |
| 12c | Split arm+gripper arch | ~~0/20~~ | **NEEDS RE-EVAL** |

---

## 6. Remaining Known Bugs (Not Yet Fixed)

### Bug A: `binary_gripper` path stores un-binarized prob (bc_handle.py:1096)
When training with `--binary_gripper` flag, the eval worker stores raw sigmoid probability (∈ [0,1]) in `raw_act[action_dim-1]` without converting to {-1, +1}:
```python
# BROKEN:
raw_act[action_dim-1] = pred_n[action_dim-1]   # grip_prob ∈ [0,1]

# FIX:
raw_act[action_dim-1] = 1.0 if pred_n[action_dim-1] >= 0.5 else -1.0
```
**Impact:** With the threshold-0.0 fix, any prob ∈ [0,1] is ≥ 0 → gripper always closes. **Does NOT affect `bc_handle_best.pt`** (trained without `--binary_gripper`).

### Bug B: Eval diversity limited to 4 kitchens
Fast-reset eval: each of 4 workers resets to the SAME initial kitchen state for all 5 of its episodes. Only 4 distinct kitchen layouts tested per 20-episode eval. Real diversity requires 20 separate env.reset() calls. This is intentional for speed but should be noted.

### Bug C: MimicGen (`mg`) dataset not publicly available
The `mg` dataset is registered in the dataset registry metadata but has **no entry in `box_links_ds.json`** (the file that maps paths to Box.com download URLs). The human dataset key `pretrain/atomic/OpenCabinet/20250819/lerobot.tar` exists; the mg key `pretrain/atomic/OpenCabinet/20250819/mg/demo/2025-08-20-21-54-43/lerobot.tar` does not. The MimicGen data is not publicly hosted yet through this mechanism. Human demos (107 episodes) are all we have for now. Check the RoboCasa GitHub for updates.

---

## 7. Architecture Details (bc_handle.py)

### BCTransformer (default `--arch transformer`)
- Input: seq_len=16 observations of state_dim, padded causal history
- Encoder: multi-head attention (d_model=256, n_heads=8, n_layers=4)
- Output: action_dim predictions via linear head
- Training: Huber loss, AdamW, cosine LR schedule

### UNetNoiseNet (`--arch unet`)
- Condition: n_obs_steps=2 obs concatenated → projected to 256-dim via FiLM
- Denoiser: 1D Conv U-Net, channels=(256, 512, 1024), down/up blocks
- Training: DDPM noise prediction on action horizons (horizon=16, n_obs_steps=2)
- Eval: DDIM 10-step denoising, receding horizon (execute n_action_steps=8, replan)

### Split-gripper (`--arch split_gripper`)
- Arm: BCTransformer → 11 continuous dims (no gripper)
- Gripper: GripperMLP(state_dim, hidden=128) → 1 binary logit, BCE loss
- Independent training, no shared gradients

---

## 8. Key Files

```
cabinet_door_project/
├── bc_handle.py                    # PRIMARY: data loading, training, eval (1339 lines)
├── preprocess_all_states.py        # Builds 44-dim oracle state cache
├── diffusion_policy/
│   ├── evaluation.py              # *** FIXED: gripper threshold 0.5→0.0 ***
│   ├── scheduler.py               # DDPM/DDIM scheduler (verified correct)
│   ├── models/unet.py             # 1D UNet with FiLM conditioning
│   ├── models/noise_net.py        # MLP/Transformer denoiser
│   ├── data.py                    # DiffusionPolicyDataset (used by diffusion_policy/ pipeline only)
│   └── inference.py               # DiffusionPolicyInference wrapper
```

**Dataset:** LeRobot format at path from `get_ds_path("OpenCabinet", source="human")` = `robocasa/datasets/v1.0/pretrain/atomic/OpenCabinet/20250819/lerobot/` (107 episodes, 37k timesteps)

**Checkpoints:** `/tmp/diffusion_policy_checkpoints/` (ephemeral! copy to persistent storage)

**Oracle state cache:** `/tmp/diffusion_policy_checkpoints/feature_cache/` (saves ~5min per run)

---

## 9. Immediate Next Steps (Priority Order)

### Step 1: Re-evaluate unet and split_gripper checkpoints (quick, ~5min each)
```bash
python bc_handle.py --eval_only --checkpoint /tmp/diffusion_policy_checkpoints/bc_unet_best.pt --n_eps 20 --n_eval_workers 4
python bc_handle.py --eval_only --checkpoint /tmp/diffusion_policy_checkpoints/bc_split_grip_best.pt --n_eps 20 --n_eval_workers 4
```

### Step 2: Get MimicGen data (5000 demos vs 107) — BLOCKED
The `mg` dataset is registered in the metadata but has **no download URL** in `box_links_ds.json`. The file only contains human and target splits for OpenCabinet. Options:
- Check RoboCasa GitHub issues/releases for when mg data will be released
- Check Hugging Face (`lerobot/robocasa` or `utaustin-robotics`) for alternative hosting
- Generate MimicGen data locally (requires MimicGen install + compute — contact course staff)
- Train on the existing 107 human demos and accept the data limitation

### Step 3: Retrain bc_handle.py on MimicGen data (key experiment)
```bash
python preprocess_all_states.py --dataset_source mg   # build oracle state for mg dataset
python bc_handle.py --arch transformer --max_epochs 300 --batch_size 512 --lr 1e-3
```
With 5000 demos (46× more data), expect training loss to drop further and eval to improve substantially.

### Step 4: Run longer eval (50 episodes) on bc_handle_best.pt for reliable estimate
Current 5% from 20 episodes has wide confidence interval. 50 episodes gives better estimate.
```bash
python bc_handle.py --eval_only --checkpoint /tmp/diffusion_policy_checkpoints/bc_handle_best.pt --n_eps 50 --n_eval_workers 4
```

### Step 5: Fix binary_gripper bug (if using that arch)
In `bc_handle.py` line 1096, change:
```python
raw_act[action_dim-1] = pred_n[action_dim-1]
# to:
raw_act[action_dim-1] = 1.0 if pred_n[action_dim-1] >= 0.5 else -1.0
```

---

## 10. Training Commands Reference

```bash
# Standard transformer (current best arch):
python bc_handle.py --arch transformer --max_epochs 200 --batch_size 128 --lr 3e-4

# U-Net diffusion:
python bc_handle.py --arch unet --horizon 16 --n_obs_steps 2 --n_action_steps 8 --max_epochs 200

# Split arm+gripper:
python bc_handle.py --arch split_gripper --max_epochs 200

# Eval only:
python bc_handle.py --eval_only --checkpoint PATH --n_eps 20 --n_eval_workers 4

# Key flags:
# --seq_len 16          observation history window
# --d_model 256         transformer width
# --n_layers 4          transformer depth
# --patience 20         early stopping patience
# --n_eval_workers 4    parallel eval workers (each gets 1 kitchen, runs N/workers eps)
```

---

## 11. Code Review Findings (Two Agents, 2026-03-14)

### Confirmed Bugs (Fixed)
- **evaluation.py:39,42** — gripper and base_mode thresholds 0.5→0.0 ✅ FIXED

### Confirmed Bugs (Not Yet Fixed)
- **bc_handle.py:1096** — binary_gripper eval path stores grip_prob ∈ [0,1] as raw action instead of converting to {-1,+1}. Affects `--binary_gripper` arch only.

### False Positives (Verified OK)
- obs_deque in unet path: HAS `maxlen=n_obs_steps` (line 1019) — doesn't grow unbounded ✓
- build_seq_tensors "causality violation": standard BC alignment (obs[t] → predict act[t]) ✓
- Split-gripper static masking (lines 1063-1064): boolean indexing correct ✓
- DDPM add_noise / DDIM denoise math: verified correct ✓
- Quaternion math in preprocess_all_states.py: verified correct ✓
- Train/val split: episode-level, no leakage ✓
- EEF position computation (evaluation.py): Rodrigues rotation correct ✓

### Low-Priority Issues
- Episode reset (bc_handle.py:1009-1015): fast-reset via `set_state_from_flattened` doesn't restore controller state — controller.reset() called manually, may have stale targets for first step
- `hdist` metric uses state from step before success (minor reporting issue)
- Eval covers only 4 distinct kitchens per 20-episode run (by design for speed)

---

## 12. Environment Notes

```bash
# Always activate venv first:
source /home/noahcylich/cs188-cabinet-door-project/.venv/bin/activate

# Headless Linux required:
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

# Run from this directory:
cd cabinet_door_project/

# Dataset path (auto-resolved):
# get_ds_path("OpenCabinet", source="human") = robocasa/datasets/...
```

