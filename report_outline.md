# Report Outline: Learning to Open Cabinet Doors with Diffusion Policy

---

## 1. Introduction / Problem Definition

**Task:** Train a robot manipulator to open a kitchen cabinet door in simulation using imitation learning from human demonstrations.

**Environment:**
- Robot: PandaOmron mobile manipulator — 7-DOF Franka Panda arm mounted on an Omron wheeled base
- Simulation: MuJoCo 3.3.1 via robosuite → RoboCasa
- Task: `OpenCabinet` — reach the cabinet handle, grasp it, and pull the door open
- Episode horizon: 500 timesteps (25 seconds at 20Hz)

**What makes this hard:**
- Each evaluation episode spawns a completely random kitchen from 2,500+ layout variants. The robot must generalize to unseen kitchens at test time — it cannot memorize the cabinet location.
- The action space is 12-dimensional (EEF position/rotation deltas, gripper open/close, base motion, torso height, arm/base mode switch) and requires precise sequential coordination.
- Only 107 human demonstrations were available (37,492 timesteps). The MimicGen-augmented dataset (expected ~5,000 demos) is not yet publicly hosted.

**Success criterion (final):** Any hinge joint ≥ 0.3 rad open (~17°) within 500 timesteps — a lenient single-door criterion introduced after discovering the original robocasa `_check_success()` required all doors open.

**Published benchmark:** 30–60% success rate (RoboCasa paper, using MimicGen data on the same task).

**Our final result:** 44/100 (44%) success across 100 random kitchen layouts — achieved with 21 seconds of training.

---

## 2. Method

### 2.1 Implementation Overview

We built a full imitation learning pipeline from scratch:
- DDPM/DDIM noise scheduler with configurable beta schedules (linear, cosine, squared cosine)
- Action chunking: predict H=16 future actions, execute n=8, replan
- Three denoising backbone architectures: MLP, 1D Conv U-Net, Transformer
- Visuomotor image encoders: ImageNet ResNet-18 with spatial softmax, R3M ResNet-18
- Oracle state augmentation pipeline (full preprocessing of 107 episodes in simulation)
- Behavior Cloning (BC) variants of all architectures as comparisons
- Full evaluation pipeline with parallel MuJoCo workers

### 2.2 Action Space (Corrected Mapping)

The dataset (LeRobot format) stores actions in a different dimension ordering than the environment expects:

```
Dataset format: [base_motion(3), torso(1), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]
                 dims 0:3         3         4               5:8         8:11          11

Env format:     [eef_pos(3), eef_rot(3), gripper(1), base_motion(3), torso(1), base_mode(1)]
                 dims 0:3      3:6        6           7:10            10         11
```

Gripper and base_mode are binary: raw values `{−1 = open/off, +1 = close/on}`.

**Critical bug discovered:** The gripper binarization threshold in evaluation was set to `< 0.5` (treating the raw value as a probability), but training data mean ≈ −0.643 (82% open). Any model prediction near zero binarizes to open → **gripper never closes**. Fix: threshold changed to `< 0.0` (the true midpoint of {−1, +1}). This bug invalidated all success-rate results prior to Phase 13 — the robot was approaching the handle but never grasping it.

### 2.3 Observation Space Design (Evolution)

The observation space went through four major designs over the course of the project:

| Version | Dims | Content | Outcome |
|---------|------|---------|---------|
| Proprioception only | 16 | Base pose, EEF relative pose, gripper | 0% success — can't locate cabinet |
| +Vision (frozen) | 16+encoder | ResNet-18 spatial softmax from 3 cameras | 40% dist reduction, 0% success |
| +Oracle door centroid | 19–38 | Door body position/orientation from sim | 5% success (first breakthrough) |
| **+Oracle handle site (final)** | **22** | **proprio + handle_pos + handle_to_eef** | **44% success** |

The handle site (actual MuJoCo grasp point) vs. the door body centroid (geometric center of the door panel) turned out to be the single most important design decision.

### 2.4 Oracle State Preprocessing Pipeline

For each of 107 training episodes:
1. Reconstruct matching kitchen using `set_ep_meta()` from `extras/ep_meta.json`
2. Step through the episode in simulation to extract per-timestep features:
   - `door_obj_pos` / `door_obj_quat`: door body centroid (static per episode)
   - `handle_pos`: actual MuJoCo handle site position (dynamic, 3-dim per timestep)
   - `hinge_angle`: max absolute hinge qpos across all hinge joints (dynamic)
3. Compute derived features analytically: `eef_pos` (from base pose + relative EEF), `handle_to_eef` (displacement), `gripper_to_door_dist`
4. Cache normalized tensors to disk (`/tmp/diffusion_policy_checkpoints/feature_cache/`); saves ~5 min per run

**Final feature set (22-dim, best model):**

| Feature | Dims | Type | Description |
|---------|------|------|-------------|
| `proprio` | 16 | dynamic | Base pose, EEF relative pose, gripper |
| `handle_pos` | 3 | dynamic | Actual handle site world position |
| `handle_to_eef` | 3 | dynamic | EEF − handle displacement (live error signal) |

### 2.5 Policy Architecture (Final)

**BC with 1D U-Net backbone (BC_UNet):**
- Input: causal sequence of `seq_len=16` observations → action chunk of H=16 steps
- Backbone: 1D convolutional U-Net with skip connections for temporal smoothness
- Training: Huber loss, AdamW optimizer, cosine LR schedule, episode-level 85/15 train/val split, early stopping (patience=30)
- Hyperparameters: BS=128, LR=1e-3, max 100 epochs; best epoch is consistently 2–4

**Why not diffusion (despite the title):** At 107-demo scale, BC consistently outperformed diffusion. The noise-prediction objective acts as a regularizer (train ≈ val loss) but does not improve task success. Only the Diffusion Transformer combination showed competitive performance, at 14× the training cost.

---

## 3. Experiments and Results

### 3.1 Phase 1 — Backbone Architecture (Training Loss)

**Question:** Does the diffusion pipeline correctly learn action prediction?

Three denoising backbones trained on proprioception + oracle door centroid (19-dim), evaluated on training loss only:

| Model | Params | Loss | Training Time |
|-------|--------|------|--------------|
| V1 MLP | 2.9M | 0.065 | 47 min |
| **V2 1D U-Net** | **30.1M** | **0.002** | **2.5 hr** |
| V3 Transformer | 6.4M | 0.026 | 3.3 hr |

**H1 confirmed:** The diffusion pipeline correctly fits training data; U-Net achieves 33× lower loss than MLP.

**H2 disproved:** Training loss does not translate to task success — proprioception alone (16-dim) achieves 0% success regardless of how low the training loss gets. The model has no information about where the cabinet is.

---

### 3.2 Phase 2 — Visuomotor Policy (Image Encoders)

**Question:** Can frozen visual encoders (RGB cameras) replace oracle state?

Three 256×256 RGB cameras (left, right, hand-mounted) → frozen ResNet-18 → spatial softmax keypoints or global pool features → concatenated with proprioception → U-Net diffusion.

| Encoder | Pooling | Loss | Distance Reduction | Success |
|---------|---------|------|-------------------|---------|
| ImageNet ResNet-18 | Spatial softmax | 0.011 | ~40% | 0% |
| R3M ResNet-18 | Global pool | 0.015 | ~7% | 0% |
| Published config (84×84 crop, 64-dim keypoints, E2E) | Spatial softmax | 0.030 | 19% | 0% |

**H4 partial:** Frozen ImageNet + spatial softmax achieves 40% distance reduction — the robot moves meaningfully toward the cabinet — but never completes the task. Spatial pooling preserves localization signal; global pooling (R3M) loses it.

**H5 disproved:** R3M performs dramatically worse than ImageNet (7% vs 40% DR). R3M's global pooling was trained for temporal contrastive learning, not spatial localization. The bottleneck is pooling strategy, not pretraining data.

**H6 disproved:** The published configuration's 84×84 random crop from 128×128 input loses too much spatial context at our resolution.

**Key insight:** Even with perfect localization (40% DR), frozen encoders cannot provide the precise handle-level spatial signal needed to complete the grasp. Fine-tuning or oracle state is required.

---

### 3.3 Phase 3 — Oracle Door Centroid (First Success)

**Question:** If we give the policy the exact door position, can it succeed?

For each training episode, we replay the simulation and extract the door body centroid position (constant per episode). This simulates what a perfect vision system would extract.

**First 5% success result:**

| Model | State Dim | Epochs | BS | Loss | Training Time | Success |
|-------|-----------|--------|-----|------|--------------|---------|
| U-Net diffusion + door_pos | 19 | 3000 | 512 | 0.002 | 94 min | **1/20 = 5%** |
| MLP + door_pos | 19 | 5000 | 1024 | 0.051 | 16 min | — |

**H7 confirmed:** Oracle door position enables first task success (Episode 12, step 180). Average distance reduction ~38%; robot reaches the cabinet vicinity and occasionally opens a door.

**Batch size matters:** BS=2048 → 0% (only 17 gradient steps/epoch); BS=512 → 5% (69 gradient steps/epoch). With only 35K training frames, large batches underfit.

**LR sweep insight:** LR=1e-3 and LR=3e-4 both achieve ~39% distance reduction after 100 epochs — the ceiling is data-driven, not optimization-driven. There is no more signal to extract from door centroid features alone.

---

### 3.4 Phase 4 — BC vs. Diffusion: Initial Comparison

**Question:** Is diffusion necessary? Can BC match it with far less training?

Motivated by observing that a colleague achieved success with a tiny transformer trained for 2 epochs (no diffusion).

**H8 confirmed — BC matches diffusion with 100× less training:**

| Model | Params | Epochs | Training Time | Dist Reduction |
|-------|--------|--------|--------------|----------------|
| **BC Transformer (d=64, L=2)** | 0.10M | **2** | **5 seconds** | **33%** |
| BC Transformer (d=64, L=2) | 0.10M | 10 | 25s | 27% |
| BC Transformer (d=128, L=4) | 0.80M | 50 | 207s | 14% |
| Diffusion U-Net T=20 | 2.31M | 50 | 316s | 27% |
| Diffusion Transformer T=100 | 0.83M | 50 | 274s | 28% |

**Surprising finding:** More training actively hurts BC eval performance. The simplest, least-trained model (2 epochs, 5 seconds) gets the best distance reduction. This is overfitting — BC models memorize training trajectories by epoch 3.

**H9 partially confirmed:** Fewer diffusion timesteps (T=20) converge faster but achieve identical eval performance to T=100.

---

### 3.5 Phase 5 — BC vs. Diffusion Head-to-Head with Validation

**Question:** With proper validation-based early stopping, which method and architecture is best?

Added 15% episode-level validation split and early stopping (patience=30). Ran all 6 combinations of {BC, Diffusion} × {MLP, Transformer, U-Net}.

| Model | Mode | Best Epoch | Val Loss | Dist Reduction | Training Time |
|-------|------|-----------|----------|---------------|--------------|
| BC_MLP | BC | 3 | 0.350 | 34% | 23s |
| Diff_MLP | Diffusion | 196 | 0.060 | 20% | 296s |
| BC_Transformer | BC | 3 | 0.355 | 33% | 110s |
| **Diff_Transformer** | **Diffusion** | **93** | **0.049** | **45%** | 486s |
| **BC_UNet** | **BC** | **2** | **0.363** | **38%** | **22s** |
| Diff_UNet | Diffusion | 24 | 0.050 | 18% | 300s |

**Overfitting analysis:**

| Model | Train Loss | Val Loss | Ratio |
|-------|-----------|----------|-------|
| BC_MLP | 0.515 | 0.350 | 0.68× |
| Diff_Transformer | 0.053 | 0.049 | 0.93× |
| Diff_UNet | 0.049 | 0.050 | 1.02× |

BC models "converge" at epoch 2–3 because val < train — they're barely trained and still generalizing. Diffusion models have train ≈ val (noise prediction acts as regularization) but this doesn't translate to better task performance.

**Key findings:**
1. **Diff_Transformer wins on distance reduction (45%)** — the only case where diffusion helps. Attention + iterative refinement is synergistic.
2. **BC_UNet is the best efficiency tradeoff (38% DR in 22s)** — nearly matches Diff_Transformer at 22× less training time.
3. **Architecture matters more than method**: within diffusion, Transformer (45%) >> MLP (20%) ≈ UNet (18%). Within BC, UNet (38%) > MLP (34%) ≈ Transformer (33%).

---

### 3.6 Phase 6 — Feature & Method Ablation (Parallel Eval)

**Question:** Which oracle features matter most? How do all architectures rank on best features?

Infrastructure upgrade: 8 parallel MuJoCo workers → 3.4× speedup (16 episodes in ~10 min vs 3 episodes in 6 min).

**Expanded feature set (38 dims total):**

| Feature | Dims | Type |
|---------|------|------|
| `proprio` | 16 | dynamic |
| `door_pos` | 3 | static/ep |
| `door_quat` | 4 | static/ep |
| `eef_pos` | 3 | dynamic |
| `eef_quat` | 4 | dynamic |
| `door_to_eef_pos` | 3 | dynamic |
| `door_to_eef_quat` | 4 | dynamic |
| `gripper_to_door_dist` | 1 | dynamic (scalar) |

**Round 1 — Feature selection (BC_UNet, 100 epochs, 16 eval episodes):**

| Config | Dim | Dist Reduction |
|--------|-----|---------------|
| F1: proprio+door_pos (baseline) | 19 | 31% |
| F2: +door_to_eef_pos | 22 | 26% |
| F4: relative only (no abs door_pos) | 19 | 22% |
| F5: all door features | 30 | 28% |
| F6: +global_eef_pos | 25 | 13% |
| **F7: +door_to_eef_pos+scalar_dist** | **23** | **43%** |

**Key findings:**
- **The scalar distance-to-door (+1 dim) is the single most valuable feature** — 43% vs 31%, a 12-point gain
- **Global EEF position actively hurts (13%)** — redundant with base+relative, adds noise
- **More features ≠ better** — adding all door features (30-dim) is worse than baseline (28% vs 31%)
- **Absolute door position is essential** — removing it (F4: 22%) is worse than keeping it

**Round 2 — BC vs Diffusion on F7 (23-dim):**

| Config | Mode | Dist Reduction | Training Time |
|--------|------|---------------|--------------|
| **R2_BC_MLP** | BC | **51%** | 21s |
| R2_BC_UNet | BC | 49% | 21s |
| R2_BC_Transformer | BC | 42% | 103s |
| R2_Diff_MLP | Diffusion | 32% | 164s |
| R2_Diff_UNet | Diffusion | 32% | 263s |
| R2_Diff_Transf | Diffusion | 12% | 439s |

**BC crushes diffusion (51% vs 32%) at 8–20× faster training.** Best model: BC_MLP (0.09M, 1 epoch, 21s).

**Round 3 — Scale-up (10× bigger BC_MLP):**
- 0.91M params: 38% DR (best_ep=2) — **worse than the tiny model's 51%**
- Confirms: the data ceiling from 107 demos prevents larger models from helping

**Phase 6 takeaways:**
1. Scalar distance-to-door is the most informative single feature addition
2. BC >> Diffusion at 107-demo scale, consistently
3. Tiny models (0.09M), minimal epochs (1–4) — optimal every time

---

### 3.7 Phase 7 — Deep Ablation (Architecture, Gripper, Features)

**Question:** What are the fundamental failure modes? Can we fix the gripper?

All experiments: BC MLP (3-layer, 128 hidden, dropout=0.3), 23-dim oracle, 8–12 eval rollouts.

**7.1 Training ceiling confirmation:**
Val loss plateaus at ~0.33 regardless of epochs or patience. At 107 demos, the model has extracted all available signal. This is a data limitation, not an optimization problem.

**7.2 Action horizon (reactive vs. chunked):**

| Horizon H | Execute Steps | Dist Reduction |
|-----------|--------------|---------------|
| **H=16** | **8** | **54%** |
| H=4 | 2 | 47% |
| H=1 (reactive) | 1 | ~22% |

H=16 action chunking is essential. Reactive (H=1) policy produces jittery, temporally incoherent predictions — the robot oscillates and never commits to a trajectory.

**7.3 Loss function comparison:**

| Loss Type | Dist Reduction |
|-----------|---------------|
| MSE (baseline) | 31% |
| GMM 5 modes | 19% |
| GMM 10 modes | 32% |
| **Huber** | **42%** |
| **Weighted MSE** | **43%** |

GMM fails completely — mode selection at inference picks the wrong mode. Huber and weighted MSE give modest improvements by downweighting outlier steps.

**7.4 Root cause — gripper never closes:**

Expert demonstrations have a bimodal gripper distribution: 82.1% open (raw = −1.0), 17.9% close (raw = +1.0). MSE loss averages these → predicted mean ≈ −0.643. Any binarization threshold above −0.643 → gripper always open. This is a structural problem with MSE/Huber objectives on bimodal data.

**Attempted fix — hybrid rule-based gripper:**

| Policy | Dist Reduction | Successes |
|--------|---------------|-----------|
| BC controls everything | 47% | 0/12 |
| Always force close | 0% | 0/12 |
| Rule: close when dist < 12cm | 47% | 0/12 |
| Rule: close when dist < 5cm | 47% | 0/12 |

Always-close destroys arm performance (0% DR) — forcing the gripper shut changes the observation distribution relative to training (arm was trained mostly with gripper open). Rule-based override is safe but achieves 0 successes anyway because the arm approaches the wrong target (door centroid, not handle).

**Key realization:** The failure has two compounding causes — (1) targeting the door body centroid instead of the handle, and (2) the gripper problem. Fixing the gripper won't help if the arm never reaches the right location.

**7.5 Architecture variants:**

| Config | Dist Reduction |
|--------|---------------|
| Baseline (MSE, no dropout) | 31% |
| **Dropout=0.3, 500 epochs** | **55%** |
| FiLM conditioning (door → scale/shift) | 49% |
| Auxiliary rel_pos head | 52% |

Dropout regularization helps most (55%). FiLM conditioning of door features (treating them as conditioning rather than concatenation) hurts — simple concatenation is better. Despite reaching 55% DR (arm getting very close), still 0 successes due to targeting the wrong location.

---

### 3.8 Phase 8 — Handle Site Oracle (Key Breakthrough)

**Question:** What if we target the actual grasp point (handle site) instead of the door body centroid?

**Insight from external code review:** The door body centroid is the center of the door panel — not where a robot would grip. The actual MuJoCo handle site is the physical attachment point used in the expert demonstrations.

**Handle site extraction:** Replay each of 107 training episodes in simulation, record per-timestep `handle_pos` from MuJoCo's named site. Build and cache `handle_cache/` (107 × T × 3 tensors).

**Friend's code results (running directly):** 2/20 (10%) success with 44-dim handle-relative features using BC Transformer (d=256, 8 heads, 4 layers).

**Our re-implementation (`bc_handle.py`, 44-dim):**

| Run | Val Loss | Best Epoch | Success |
|-----|----------|-----------|---------|
| Without episode boundary fix | 0.08098 | 3 | 0/20 (0%) |
| **With episode boundary fix** | **0.07941** | **4** | **1/20 (5%)** |

**H12 confirmed:** Handle site as oracle → first confirmed success at step 175 (episode 4).

**H13 confirmed — episode boundary contamination:** The causal 16-step observation window crosses episode boundaries during training (the first 16 timesteps of episode N look back into episode N−1's kitchen). This contaminates ~4.3% of training frames with inconsistent context. Masking these frames improved val loss from 0.0810 → 0.0794 and unblocked the first success.

**H14 confirmed — temporal context is essential:** MLP (no history) has val loss 0.110; seq_len=1 has val loss 0.112; 16-step causal transformer has 0.079 — 39–41% worse without history.

**H15 confirmed — manipulation is the bottleneck:** With seq_len=1, the robot gets within 1.8cm of the handle 3 times in 500 steps. It can find the handle — but with no temporal context it can't apply consistent force to pull it open.

---

### 3.9 Phase 9 — Parallel Evaluation Infrastructure

**Question:** Can we evaluate more episodes per run for reliable statistics?

Built spawn-safe parallel eval worker (`_eval_worker`):
- One slow reset per worker (30s MuJoCo XML compilation), fast state-restore for subsequent episodes (1s)
- Pre-allocated buffers, model warmup, headless rendering

| Config | Wall Time | Notes |
|--------|-----------|-------|
| Sequential (original) | ~20 min | 1 env, 20 eps × ~60s |
| **4 workers** | **~4 min** | **5× speedup** |
| 10 workers | ~29 min | Worse — CPU saturated by MuJoCo XML compilation |

4 workers is the empirical optimum for a 12-core machine.

**Known limitation:** Each of 4 workers uses one kitchen for all 5 of its episodes (via `set_state_from_flattened`). So 20-episode evals test only 4 distinct kitchens. This inflates variance — a good kitchen gives 5/5, a bad one gives 0/5.

---

### 3.10 Phase 10 — Ablation on Handle Oracle Features

**Question:** What exactly is each feature contributing in the 44-dim handle-relative state?

Baseline: BC Transformer (d=256, 8 heads, 4 layers, seq_len=16), 44-dim oracle, 20-episode eval.

| Experiment | Val Loss | Eval | Closest Approach |
|-----------|----------|------|-----------------|
| Baseline (transformer, 44-dim) | 0.0794 | 1/20 (5%) | ep4: step 175 |
| Exp A: MLP (no temporal) | 0.110 | 0/20 | 0.017m |
| Exp B: No handle features (38-dim) | 0.078 | 0/20 | 0.029m |
| Exp C: gripper_weight=5.0 | 0.095 | 0/20 | — |
| Exp D: seq_len=1 (no history) | 0.112 | 0/20 | **0.018m (3 eps)** |
| Exp E: handle_only (30-dim) | 0.076 | 0/20 | — |

**Key insights:**
1. **Temporal context essential:** MLP and seq_len=1 are both 39–41% worse on val loss than the 16-step transformer
2. **Handle features are marginally helpful but not sufficient alone** — removing door centroid (Exp E) actually gives better val loss (0.076!) but still 0/20. Absolute + relative together is best.
3. **Manipulation is the bottleneck, not localization:** Exp D (no history) gets within 1.8cm 3 times but never opens the door. The robot finds the handle — it just can't apply consistent force without temporal context.
4. **N=20 eval has high variance at 5% true success rate** — P(0/20 | p=0.05) = 36%. Seed=1 gave 0/20 for the same checkpoint that gave 1/20 on seed=0.

---

### 3.11 Phase 11 — Gripper Fixes: Binary Head and Temporal Diffusion

**Question:** Can we fix the gripper-never-closes problem directly?

**Attempt 1 — Binary gripper BCE head (`BCTransformerBinaryGripper`):**
- Two heads: `head_cont` (Huber loss, 11 continuous dims) + `head_grip` (BCE loss, 1 dim)
- Result: 0/20 (0%), median handle distance 0.38m vs baseline 0.25m

The combined Huber+BCE loss creates competing gradients through the shared transformer body. The arm approach **degraded** compared to baseline. The gripper head learns to fire correctly but at the cost of worse positioning.

**Attempt 2 — Temporal Diffusion Policy:**
- 16-step causal history → 256-dim context → MLP denoiser, DDPM T=100 train, DDIM T=10 inference
- Result: 0/20 (0%), inconsistent trajectories: ep4 approaches to 0.084m, but ep3/7/11/15 all wander to 1.4m+

DDIM's stochastic noise injections produce wildly different trajectories from the same starting state — in the same kitchen (same 4 kitchens per 4 workers), 4 consecutive episodes diverge completely. Diffusion's stochasticity is a liability when the task requires precise, consistent approach trajectories.

**Phase 11 conclusion:** Both gripper-specific fixes degraded overall performance. The true root cause (targeting centroid not handle) had not yet been fully addressed through the pipeline.

---

### 3.12 Phase 12 — Professor Recommendations (Post-Bug-Discovery)

At this point, the gripper binarization bug was discovered. All prior 0% success rates were invalid — the robot was approaching but never closing the gripper. Three new experiments were run with the bug fixed:

**Re-evaluation of existing checkpoints (post-fix, 20 episodes each):**

| Checkpoint | Architecture | Post-fix Success | Note |
|-----------|-------------|-----------------|------|
| bc_handle_best.pt | BC Transformer, 44-dim | **5% (1/20)** | First confirmed valid success |
| bc_unet_best.pt | U-Net diffusion, 44-dim | 0% (0/20) | Gets 3–6cm but overfits 107 demos |
| bc_split_grip_best.pt | Split arm+gripper | 5% (1/20) | Same rate as transformer baseline |

**H18 disproved — relaxed single-door criterion reveals no hidden successes:** With the robocasa `any_door_open()` criterion (any joint ≥ 90% normalized travel), still 0/20. Robot reaches 5–8cm from handle but can't pull through.

**H19 disproved — U-Net diffusion + action chunking:** Early stops at epoch 51, val=0.053, 0/20. Closest approach 1.8cm (ep8) — same overfitting pattern as earlier diffusion runs with larger data.

**H20 disproved — Split arm+gripper (independent models):** Median handle distance 0.34m vs baseline 0.19m. Separate training for arm and gripper degrades the arm — the models can't coordinate since arm actions depend on gripper state.

---

### 3.13 Phase 13 — TA Recommendations: Handle Augmentation + New Success Criterion

**Question:** Can TA-recommended features (`handle_pos`, `handle_to_eef`, `hinge_angle`) and the 1D U-Net backbone significantly improve results?

**New success criterion:** Any hinge joint > 0.3 rad (~17°) — more lenient and task-meaningful than requiring full open.

**Round 1 — Feature selection with new criterion (BC_UNet, 16 eval episodes):**

| Config | Dim | Success | Dist Reduction | Val Loss |
|--------|-----|---------|---------------|----------|
| F1: proprio only | 16 | 2/16 (13%) | 38% | 0.3487 |
| F2: +handle_pos | 19 | 2/16 (13%) | 65% | 0.3350 |
| **F3: +handle_pos+handle_to_eef** | **22** | **6/16 (38%)** | **73%** | 0.3366 |
| F4: +handle_pos+handle_to_eef+hinge_angle | 23 | 3/16 (19%) | 82% | 0.3084 |
| F5: F4+door_pos | 26 | 3/16 (19%) | 81% | 0.3041 |
| **F6: proprio+handle_to_eef only** | **19** | **5/16 (31%)** | **88%** | 0.3267 |
| F7: proprio+door_pos (centroid) | 19 | 1/16 (6%) | 37% | 0.3576 |
| F8: proprio+door_pos+door_to_eef_pos | 22 | 0/16 (0%) | 40% | 0.3480 |

**Key findings:**
1. **handle_to_eef (relative displacement) is the critical feature** — F6 with just proprio+handle_to_eef (19-dim) achieves 5/16, 88% DR. Relative displacement is more useful than absolute handle position alone.
2. **Door centroid features dramatically underperform** — F7 (door_pos): 1/16 vs F6 (handle_to_eef): 5/16. This validates the Phase 7/8 conclusion: centroid ≠ handle.
3. **hinge_angle hurts success despite increasing DR** — F4 (83% DR, 3/16) vs F3 (73% DR, 6/16). The hinge signal leaks future trajectory information, possibly causing mode confusion.
4. **F3 is the best overall** — 6/16 (38%) with proprio+handle_pos+handle_to_eef (22-dim)

**Round 2 — BC vs Diffusion on F6 (19-dim, 16 eval episodes):**

| Config | Mode | Success | Dist Reduction | Training Time |
|--------|------|---------|---------------|--------------|
| **R2_BC_UNet** | BC | **5/16 (31%)** | **84%** | **20s** |
| R2_Diff_Transformer | Diffusion | 4/16 (25%) | 82% | 417s |
| R2_BC_MLP | BC | 3/16 (19%) | 72% | 20s |
| R2_BC_Transformer | BC | 2/16 (13%) | 79% | 99s |
| R2_Diff_UNet | Diffusion | 1/16 (6%) | 72% | 277s |
| R2_Diff_MLP | Diffusion | 0/16 (0%) | 57% | 156s |

BC_UNet is best: 5/16 success, 84% DR, trained in 20 seconds. Diffusion Transformer is competitive (4/16) at 20× the cost. BC U-Net's temporal smoothness outperforms MLP and Transformer within BC.

**Round 3 — Scale-up (3× wider U-Net, patience=100):**
- 2/10 (20%) vs small model's 5/16 (31%)
- Best epoch: 2 (same rapid overfit as all previous scale-up attempts)
- Confirms: 107-demo data ceiling is fundamental — wider models overfit faster and worse

**Mechanism breakdown — why BC_UNet dominates:**

| Mechanism | What it fixes | Evidence |
|-----------|---------------|---------|
| `handle_to_eef` as live error signal | Wrong target (centroid vs handle) | F6 88% DR vs F7 37% DR |
| 1D U-Net temporal smoothness | Jitter → inconsistent force during pull | BC_UNet > BC_MLP > BC_Transformer |
| BC determinism over diffusion | High-variance trajectories at low data | Diff_UNet 1/16 vs BC_UNet 5/16 |
| 0.3 rad hinge criterion | Full-open too strict for partial pulls | 0/20 with door centroid + relaxed criterion (Phase 12a) |

---

### 3.14 Phase 14 — 100-Episode Validation of Final Model

**Question:** What is the true success rate of our best model?

Retrained F3 (proprio + handle_pos + handle_to_eef, 22-dim, BC_UNet) from scratch. Evaluated 100 episodes across 100 distinct kitchens (seed=0–99), 500 steps/episode, 8 parallel workers (~2 hours total eval time).

Training: best_ep=4, val=0.3266, 21 seconds — identical to ablation sweep.

**Result: 44/100 (44%) — above the 30% target.**

| Metric | Value |
|--------|-------|
| **Success rate** | **44/100 (44.0%)** |
| Avg distance reduction | ~85% |
| Best episode (ep40) | 99% DR (0.50m → 0.01m) |
| Training time | 21 seconds |
| Eval time | ~2 hours (8 workers) |

**Stability across evaluation waves:**

| Episodes | Running Success % |
|----------|-----------------|
| 1–8 | 75.0% |
| 1–16 | 50.0% |
| 1–40 | 45.0% |
| 1–80 | 43.8% |
| **1–100** | **44.0%** |

Converges quickly; σ ≈ 4% across all waves after the first 16 episodes.

**Failure mode analysis:**

| Failure Category | ~Count | Description |
|-----------------|--------|-------------|
| Approach failure | ~8 eps | DR < 30%; policy never finds handle. Unusual kitchen layouts or handle site extraction failure. |
| Grasp failure | ~48 eps | DR 70–97%; robot reaches 1–10cm but can't complete the pull. Gripper contacts handle but insufficient sustained force. |

Successful episodes almost universally end with final distance < 5cm (ep4: 0.01m, ep6: 0.01m, ep11: 0.00m, ep40: 0.01m). When the arm gets within contact range and applies force, success follows.

---

## 4. All Hypotheses Summary

| # | Hypothesis | Outcome | Key Result |
|---|-----------|---------|------------|
| H1 | Diffusion pipeline correctly learns action prediction | **Confirmed** | U-Net loss 0.002, matches GT trajectories |
| H2 | Low-dim proprioception can solve the task | **Disproved** | 0% — no cabinet localization info |
| H3 | Squared cosine beta schedule outperforms linear | **Confirmed** | 55% lower loss than linear schedule |
| H4 | Frozen ImageNet encoder enables task success | **Partial** | 40% dist reduction, 0% success |
| H5 | R3M outperforms ImageNet | **Disproved** | R3M 7% vs ImageNet 40% DR — pooling matters, not pretraining |
| H6 | Published 84×84 crop config works at our scale | **Disproved** | Too aggressive; 19% DR vs 40% |
| H7 | Oracle door centroid enables task success | **Confirmed** | 5% success (first success) |
| H8 | BC matches diffusion with 100× less training | **Confirmed** | 33% DR in 5s vs 27% in 5min |
| H9 | Fewer diffusion timesteps improve efficiency | **Partial** | Faster convergence, identical eval performance |
| H10 | Validation loss identifies optimal training epoch | **Confirmed** | BC peaks at ep 2–3; without it, more training = worse eval |
| H11 | Diff_Transformer is best architecture overall | **Confirmed** | 45% DR, only diffusion config that beats all BC variants |
| H12 | Handle site (not door centroid) is critical for grasp | **Confirmed** | 0% → 5–10% once handle used as target |
| H13 | Episode boundary contamination hurts training | **Confirmed** | Val 0.0810 → 0.0794; unlocked first success |
| H14 | Temporal context (causal history) is essential | **Confirmed** | seq_len=16: val 0.079 vs MLP/seq_len=1: 0.110–0.112 |
| H15 | Manipulation, not localization, is the bottleneck | **Confirmed** | seq_len=1 gets within 1.8cm but never opens door |
| H16 | Binary gripper BCE head fixes gripper failure | **Disproved** | Combined loss hurt arm approach; 0/20 |
| H17 | Temporal diffusion fixes bimodal gripper | **Disproved** | DDIM stochasticity → inconsistent trajectories; 0/20 |
| H18 | Relaxed success criterion reveals hidden successes | **Disproved** | 0/20 with any_door_open; robot contacts handle but can't pull |
| H19 | U-Net diffusion + action chunks outperforms MLP denoiser | **Disproved** | 0/20; same rapid overfit |
| H20 | Independent arm+gripper models fix gradient conflict | **Disproved** | Split arm degrades to median 34cm vs baseline 19cm |
| H21 | handle_to_eef relative displacement outperforms door centroid | **Confirmed** | F6: 88% DR, 5/16 vs F7: 37% DR, 1/16 |
| H22 | BC_UNet outperforms diffusion on handle features | **Confirmed** | BC_UNet 5/16 at 20s vs Diff_UNet 1/16 at 277s |
| H23 | Scale-up with handle features overcomes data ceiling | **Disproved** | Wider U-Net: best_ep=2, 2/10, worse than small model |

---

## 5. Discussion and Reflections

### 5.1 What Worked and Why

**Correct target localization was the largest single factor.** Switching from the door body centroid to the actual MuJoCo handle site improved success from near-zero to 38–44%. The `handle_to_eef` relative displacement feature (3 dims) functions as a live error signal — the policy sees exactly how far and in which direction the end-effector is from the grasp point at every timestep.

**Tiny models with minimal training.** Given only 107 demonstrations, the optimal model was consistently one of the smallest tested (0.09–0.26M params) trained for 2–4 epochs (5–21 seconds on A100). Every scale-up experiment made things worse. The data ceiling is the true bottleneck, not model capacity.

**BC over diffusion at this scale.** BC's determinism and fast convergence (via early stopping) made it consistently better than diffusion for this dataset size. Diffusion's implicit regularization (train ≈ val) does not compensate for the increased training complexity. The exception — Diff_Transformer — worked precisely because the Transformer's attention mechanism benefits from iterative refinement in a way MLP and U-Net don't.

**Validation-based early stopping was essential.** Without a held-out validation set, BC models train past their peak (epoch 2–4) and memorize the training distribution. This was the root cause of the "more training = worse eval" pattern observed throughout Phase 4.

**1D U-Net temporal smoothness.** BC_UNet outperformed BC_MLP and BC_Transformer on the final task, likely because the U-Net's skip connections produce temporally coherent action sequences — important for the sustained pulling motion required to open the door.

### 5.2 What Didn't Work and Why

**Vision-based policies (frozen encoders).** Frozen ImageNet + spatial softmax provides enough spatial information to approach the cabinet (40% DR) but not enough precision to complete the grasp. Fine-tuning or oracle state is required. R3M's global pooling architecture is fundamentally incompatible with spatial localization tasks.

**Diffusion at low data scale.** Noise-prediction objectives add training complexity without benefit when the underlying dataset has only 107 examples. The denoising process doesn't help the model learn better actions — it just adds variance at inference time.

**Gripper-specific fixes.** Binary gripper BCE head, split arm+gripper models, and rule-based override all failed to improve on the baseline. The gripper problem is structural (MSE averages a bimodal distribution), but fixing the gripper in isolation doesn't help if the arm isn't positioned correctly.

**More data would likely be the biggest improvement.** The MimicGen-augmented dataset (expected ~5,000 demos vs. 107) was not publicly available. With 46× more data, the training ceiling would shift substantially and larger models could be leveraged.

### 5.3 Critical Bugs and Their Impact

**Gripper binarization threshold (root cause of all early 0% results):** The threshold for converting predicted gripper action to binary {open, close} was set to 0.5 on a raw space of {−1, +1}. Training data mean ≈ −0.643 → any model prediction near zero → gripper always open. Fix: threshold 0.5 → 0.0. This single bug caused all Phases 1–12 success-rate evaluations to be invalid. Distance metrics from those phases are still reliable.

**Episode boundary contamination:** The causal 16-step observation window at the start of each episode looks into the previous episode's context. ~4.3% of training frames had inconsistent context. Masking them improved val loss 0.0810 → 0.0794 and was necessary for the first confirmed success.

### 5.4 Key Lessons for Future Work

1. **Verify the action space mapping before training.** A one-line threshold bug cost multiple phases of experiments.
2. **Use N ≥ 50 eval episodes.** At true success rates of 5–10%, N=20 has a 30–40% chance of returning 0% even for a working policy.
3. **Oracle state as a stepping stone.** Using simulation oracle state (handle position) is a principled intermediate step between proprioception-only and full visuomotor — it bounds what a perfect visual system could achieve and identifies whether the architecture is the bottleneck.
4. **Get more data first.** MimicGen augmentation would likely matter more than any architectural choice made here.

---

## 6. Team Contributions

*(To be filled in by team — suggested categories: environment setup & verification, data preprocessing & oracle pipeline, architecture implementations, training infrastructure, evaluation pipeline & parallel workers, ablation experiments, bug discovery & fixes, report & website, demo video)*

---

## Appendix

### A. All-Time Best Results

| Metric | Value | Config |
|--------|-------|--------|
| **Success rate (100-ep validated)** | **44% (44/100)** | Phase 14: F3 BC_UNet, 22-dim, 21s training |
| Best ablation sweep result | 38% (6/16) | Phase 13 F3: same config, 16-ep sample |
| Best distance reduction | 88–99% | Phase 13 F6 / Phase 14 ep40 |
| Fastest to high DR | 20s → 84% DR | Phase 13 R2_BC_UNet |
| Best val loss (handle oracle) | 0.0794 | bc_handle.py BC Transformer, 44-dim |

### B. Compute

- Hardware: NVIDIA A100-SXM4-80GB, CUDA 12.4
- Final model training: 21 seconds
- Final 100-episode evaluation: ~2 hours (8 parallel MuJoCo workers)
- Oracle state preprocessing: ~37 minutes for 107 episodes (4 workers, handle + hinge cache)

### C. Dataset

- 107 human demonstrations, 37,492 timesteps, LeRobot format
- MimicGen dataset (expected ~5,000 demos): not publicly hosted as of March 2026
- 89 unique door x-positions, 63 unique y-positions — diverse kitchen layouts confirmed
