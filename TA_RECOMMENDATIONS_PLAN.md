# TA Recommendations: Implementation Plan

Based on the TA's guidance to get a functional model without training full video diffusion.

**TA's three core recommendations:**
1. Augment state with cabinet handle position, EEF-to-handle relative position, and hinge angle
2. Use the 1D Convolutional U-Net backbone (~15M params) instead of MLP
3. Count one cabinet door open as a success at eval time (keep two-door demos in training)

---

## Part A: Code Changes

### Change 1: Switch Default Backbone to U-Net

**File:** `cabinet_door_project/diffusion_policy/config.py` — line 18

`UNetNoiseNet` already exists in `models/unet.py` with FiLM conditioning and skip connections. `build_model()` in `training.py` already handles the `"unet"` case. This is a one-line change.

```python
# Before
backbone: str = "mlp"

# After
backbone: str = "unet"
```

---

### Change 2: Integrate Handle Augmentation into Training Pipeline

**Files:** `config.py`, `data.py`, `evaluation.py`

**Background:** The infrastructure already exists:
- `preprocess_all_states.py` has `build_handle_cache()` which replays episodes in sim to extract per-timestep handle site positions
- `ablation_sweep.py` already reads `door_obj_to_robot0_eef_pos` and `door_obj_pos` from env obs at eval time
- `evaluation.py` already has `get_handle_pos_from_env()` and `compute_eef_pos_from_obs()`

**New features to add to state (16-dim → 23-dim):**

| Feature | Dim | Description |
|---|---|---|
| `handle_pos` | 3 | Handle site world position (from sim replay / env obs) |
| `handle_to_eef` | 3 | `eef_pos - handle_pos` — relative position, dynamic each step |
| `hinge_angle` | 1 | Cabinet hinge joint angle from sim |

**Step 2a — `config.py`: Add augmentation flag**
```python
use_handle_augmentation: bool = False   # set True for 23-dim state
```
When `True`, `state_dim` should be set to `23` at construction time. Consider making `state_dim` auto-computed from this flag rather than manually set.

**Step 2b — `data.py`: Load augmented state**

After the TA's script runs (output format TBD — likely a new `.npz`/`.pt` file or extended parquet columns), add a loader that concatenates handle data onto the 16-dim proprio:

```python
def load_augmented_episodes(dataset_path: str, handle_cache_dir: str) -> list[dict]:
    """Load episodes with handle_pos, handle_to_eef, hinge_angle appended to state."""
    episodes = load_episodes(dataset_path)
    for ep in episodes:
        eid = ep['episode_index']
        handle_data = np.load(f"{handle_cache_dir}/episode_{eid:06d}.npy")  # (T, 7)
        ep['states'] = np.concatenate([ep['states'], handle_data], axis=-1)
    return episodes
```

Alternatively, update the `build_obs_tensor()` path in `ablation_sweep.py` to include `'handle_pos'` and `'handle_to_eef'` feature names once they are added to `preprocessed_all_states.pt`.

**Step 2c — `evaluation.py`: Match training features at eval time**

Update `extract_state()` to optionally append handle features using the already-existing helper functions:

```python
def extract_state(obs: dict, use_handle_aug: bool = False,
                  env=None, active_site=None) -> np.ndarray:
    parts = [obs[k].flatten() for k in STATE_KEYS_ORDERED]  # 16-dim base
    if use_handle_aug and env is not None:
        eef_pos = compute_eef_pos_from_obs(obs)
        handle_pos, active_site = get_handle_pos_from_env(env, active_site, eef_pos)
        hinge_angle = _get_hinge_angle(env)           # new helper, see below
        parts.append(handle_pos)                      # 3-dim
        parts.append(eef_pos - handle_pos)            # 3-dim
        parts.append(np.array([hinge_angle], dtype=np.float32))  # 1-dim
    return np.concatenate(parts).astype(np.float32), active_site
```

Add `_get_hinge_angle(env)` to read the hinge joint qpos from `env.sim`. The exact joint name depends on which fixture is loaded — iterate `env.sim.model.joint_names` and find joints containing `"hinge"` associated with the active cabinet.

Update `run_rollouts()` to pass `env` and track `active_site` across steps.

**Step 2d — Normalization stats**

The current `load_stats()` reads `meta/stats.json` (16-dim stats). For augmented state:
- Compute per-feature mean/std from the full training set (see `build_obs_tensor()` in `ablation_sweep.py` for the pattern)
- Save to a new file, e.g. `meta/stats_augmented.json` or inline in `preprocessed_all_states.pt`
- Update `DiffusionPolicyDataset.__init__()` to use extended stats when `config.use_handle_augmentation=True`

**Dependency:** These changes depend on the format of the TA's augmentation script output. Wait for the script before writing step 2b/2d. Steps 2c can be prototyped now using `get_handle_pos_from_env()` which already works at eval time.

---

### Change 3: One-Door Success Criterion at Eval

**Files:** `evaluation.py` and `ablation_sweep.py`

The current code calls `env._check_success()` which requires all doors open. The TA says to require only one.

**Add to `evaluation.py`:**
```python
def check_one_door_success(env) -> bool:
    """Return True if at least one cabinet door is sufficiently open (~17 degrees)."""
    try:
        for joint_name in env.sim.model.joint_names:
            if 'hinge' in joint_name.lower():
                jid = env.sim.model.joint_name2id(joint_name)
                qposadr = env.sim.model.jnt_qposadr[jid]
                angle = abs(float(env.sim.data.qpos[qposadr]))
                if angle > 0.3:  # radians; tune based on what env._check_success uses
                    return True
        return False
    except Exception:
        return env._check_success()  # fallback if joint introspection fails
```

> **Note:** The threshold `0.3 rad` (~17°) is a starting guess. Check what angle `env._check_success()` uses internally in the RoboCasa source and align with that. Could also use `reward > 0` as a proxy — each door opening gives a positive reward signal.

**Replace in `evaluation.py` line 212:**
```python
# Before
if env._check_success():

# After
if check_one_door_success(env):
```

**Replace in `ablation_sweep.py` line 419:**
```python
# Before
if env._check_success():

# After
from diffusion_policy.evaluation import check_one_door_success
if check_one_door_success(env):
```

Keep two-door demonstration episodes in the training data — the TA explicitly says not to remove them.

---

### Change 4: Data Normalization (Verification)

**File:** `data.py`

The `Normalizer` class already clamps std to `1e-8` minimum. Verify that the augmented features (especially `hinge_angle`) don't have near-zero variance that would cause issues. The `ablation_sweep.py` uses `.clamp(min=1e-6)` — slightly more conservative. Consider aligning `STD_CLAMP_MIN = 1e-6` in `data.py`.

---

## Part B: Ablations to Run

Run in order — each round informs the next. `ablation_sweep.py` already has all the infrastructure (BC models, diffusion training, parallel eval workers, summary printing).

---

### Round 1: Feature Set Selection
**Architecture:** BC + U-Net (fast, ~10 min each)
**Purpose:** Find which handle features matter before committing to slower diffusion training.

| Config ID | State Dim | Features |
|---|---|---|
| F1_baseline_16d | 16 | proprio only (current baseline) |
| F2_+handle_pos_19d | 19 | proprio + handle_pos |
| F3_+rel_pos_22d | 22 | proprio + handle_pos + handle_to_eef |
| F4_+hinge_23d | 23 | proprio + handle_pos + handle_to_eef + hinge_angle |
| F5_+door_obj_25d | 25 | F4 + door_obj_pos (static door position) |
| F6_rel_only_19d | 19 | proprio + handle_to_eef only (skip absolute handle) |

**Decision criterion:** Best success rate and dist_reduction (% reduction in handle-to-eef distance across eval rollouts).

Expected winner: F3 or F4 per TA guidance.

---

### Round 2: Backbone / Method Comparison
**Features:** Best set from Round 1
**Purpose:** Confirm that Diffusion+UNet outperforms alternatives.

| Config ID | Backbone | Method |
|---|---|---|
| R2_BC_UNet | 1D Conv U-Net | BC (direct action prediction) |
| R2_Diff_UNet | 1D Conv U-Net | **Diffusion** ← TA recommended |
| R2_BC_MLP | MLP | BC |
| R2_Diff_MLP | MLP | Diffusion |
| R2_BC_Transformer | Transformer | BC |
| R2_Diff_Transformer | Transformer | Diffusion |

**Decision logic (already in `ablation_sweep.py`):**
- If BC is >3x faster and diffusion only wins by <5% dist_reduction → scale up BC
- Otherwise → scale up Diffusion+UNet

---

### Round 3: Scale Up (~2 hours)
**Config:** Winner from Round 2, best features from Round 1

Scale settings vs. Round 2:

| Parameter | Round 2 | Round 3 |
|---|---|---|
| U-Net channels | `(64, 128, 256)` | `(128, 256, 512)` |
| d_model (Transformer) | 128 | 256 |
| Max epochs | 100 | 3000 (early stop) |
| Early stop patience | 30 | 200 |
| Batch size | 128 | 256 |
| Eval episodes | 16 | 50 |
| Eval split | pretrain | pretrain + target |

---

### Round 4 (Optional): DAgger
Only run if Round 3 success rate is below ~20% on pretrain split.

Procedure:
1. Run trained policy in sim for 100 episodes; record trajectories
2. Flag timesteps where policy is about to fail (heuristic: handle-to-eef distance increasing)
3. Collect human/oracle corrections for those states
4. Mix corrections with original demos (50/50) and retrain from scratch or fine-tune

---

## Dependency Order

```
TA augmentation script arrives
        │
        ├─► Change 1 (U-Net backbone)      ← can do NOW
        ├─► Change 3 (one-door success)    ← can do NOW
        │
        └─► Change 2 (handle augmentation)
                │
                ├─► Run preprocess_all_states.py with build_handle_cache()
                ├─► Update data.py stats loading
                └─► Run ablation_sweep.py
                        │
                        ├─► Round 1 (feature selection, ~1 hr)
                        ├─► Round 2 (method comparison, ~2 hrs)
                        └─► Round 3 (scale up, ~2 hrs)
```

Changes 1 and 3 are independent of the TA's script and should be done first.
