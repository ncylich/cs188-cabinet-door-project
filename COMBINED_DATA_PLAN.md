# Combined Data Training Plan

## Background & Current State

### Working Results (from WORKING_RESULTS.md, other machine)

**Phase 13 winner:** BC_UNet (direct behavior cloning, NOT diffusion) on 22-dim features
(F3: proprio + handle_pos + handle_to_eef) → **6/16 = 38% success** in 53s training.

**Current bc_handle.py best (this machine):** Diffusion U-Net on 44-dim full features →
**7/50 = 14%** (seed42, ddim_samples=4, success_threshold=0.30).

### New Data Available

- **Pretrain split** (proper data): 107 demos, 37,492 frames — layouts 11–60, styles 12–60
- **Target split** (new data): 500 demos, 184,024 frames — layouts 1–10, styles 1–10
- **Key finding: ZERO overlap** between splits in layout and style IDs. These are genuinely
  different kitchen configurations designed for generalization testing, not more of the same.

### Distribution Analysis

| | Transfers to pretrain eval | Doesn't Transfer |
|---|---|---|
| Handle-relative manipulation (grasp + pull) | ✓ physics/handle features are relative | |
| Base navigation to cabinet | | ✗ cabinet is at different positions per layout |
| Cabinet appearance | | ✗ styles 1-10 vs 12-60 |

---

## Step 0: Git Housekeeping (DO FIRST)

```bash
# 1. Commit current work on this machine
git add -A
git commit -m "add combined data preprocessing, target dataset download, eval results"

# 2. Fetch and merge main (has Phase 13 BC_UNet code from other machine)
git fetch origin
git merge origin/main

# Resolve any conflicts, then:
git push origin main
```

---

## Step 1: Reproduce Main's Baseline (Before Any New Training)

After merging, verify the Phase 13 BC_UNet results are reproducible on this machine.
This is the ground truth we're trying to beat.

```bash
# Re-run the bc_handle.py baseline with the same config as Phase 13 winner (F3, 22-dim)
# Check what the merged main uses for BC_UNet — likely arch=transformer or a new flag
python bc_handle.py --arch unet --feat_subset handle_only \
    --seed 0 --epochs 200 --patience 30 \
    --save_path ../checkpoints/bc_unet_main_reproduced.pt

# Evaluate
python bc_handle.py --eval_only --checkpoint ../checkpoints/bc_unet_main_reproduced.pt \
    --arch unet --n_eps 20 --n_eval_workers 4 --success_threshold 0.30
```

**Target**: Reproduce ~38% (6/16) success from WORKING_RESULTS.md Phase 13 F3.
If we can't reproduce it, stop and debug before combining data.

---

## Step 2: Data Mixing Ablations (Requires preprocess_target.py First)

Run `preprocess_target.py` (~60-90 min) to build the handle cache for 500 target episodes.

### Mix A: Heterogeneous / Uniform

Standard concatenation — randomly shuffle all 607 demos together, train normally.
The model sees pretrain and target episodes interleaved throughout all epochs.

```bash
python bc_handle.py --arch unet --seed 42 --combined_data \
    --save_path ../checkpoints/bc_unet_mixA_seed42.pt \
    --epochs 200 --patience 30 --batch_size 256
```

**Hypothesis:** More diverse data → better generalization. Val loss should drop below 0.051 ceiling.
**Risk:** Target kitchens (layouts 1-10) are out-of-distribution for our eval (layouts 11-60),
so model may learn layout-specific navigation that doesn't transfer.

---

### Mix B: New Data First, Original Data Last (Sequential / Fine-tuning)

Two-phase training:
- Phase 1: Train on target data only (500 demos) until convergence
- Phase 2: Fine-tune on pretrain data only (107 demos) until convergence

The idea: learn general manipulation from target data, then specialize to pretrain kitchen configs.

**Implementation needed in bc_handle.py:**
```bash
# Phase 1: train on target only
python bc_handle.py --arch unet --seed 42 --use_target_only \
    --save_path ../checkpoints/bc_unet_mixB_phase1.pt \
    --epochs 100 --patience 20

# Phase 2: fine-tune on pretrain only, starting from phase 1 weights
python bc_handle.py --arch unet --seed 42 \
    --checkpoint ../checkpoints/bc_unet_mixB_phase1.pt --finetune \
    --save_path ../checkpoints/bc_unet_mixB_final.pt \
    --epochs 100 --patience 20
```

**Hypothesis:** Pretraining on 500 diverse demos teaches manipulation primitives; fine-tuning
on 107 pretrain demos specializes to the eval distribution. Similar to ImageNet pretraining → task fine-tuning.
**Risk:** Catastrophic forgetting during phase 2 if learning rate is too high.

---

### Mix C: Biased Curriculum (Target-Heavy → Pretrain-Only)

Train with a sampling schedule that starts with mostly target demos, then smoothly shifts
to all-pretrain by the end of training. No hard phase boundary — gradual curriculum.

**Sampling schedule:**
- Epoch 1: 90% target, 10% pretrain
- Epoch ~50: 50% target, 50% pretrain
- Epoch ~100: 10% target, 90% pretrain
- Epoch ~150+: 100% pretrain (fine-tuning)

**Implementation needed in bc_handle.py:**
The DataLoader's sampler weights change each epoch based on a linear or cosine schedule.

```bash
python bc_handle.py --arch unet --seed 42 --curriculum_data \
    --curriculum_warmup_epochs 100 \
    --save_path ../checkpoints/bc_unet_mixC_seed42.pt \
    --epochs 200 --patience 30
```

**Hypothesis:** Curriculum avoids the cold-start problem (107 demos is sparse) while ending
in the correct distribution. Best of both worlds — diverse early learning, specialized late.
**Risk:** Most complex to implement; schedule tuning is a hyperparameter.

---

## Implementation Checklist

| Feature | Status | Notes |
|---------|--------|-------|
| `preprocess_target.py` | Written, not run | ~60-90min, needs user approval |
| `--combined_data` flag | ✓ Done | Mix A (uniform) |
| `--use_target_only` flag | ✗ Needed | Mix B phase 1 |
| `--finetune` flag | ✗ Needed | Mix B phase 2 (load checkpoint, continue training) |
| `--curriculum_data` flag | ✗ Needed | Mix C (epoch-based sampling weights) |

---

## Evaluation Protocol

All experiments evaluated with:
```bash
python bc_handle.py --eval_only --checkpoint <path> --arch unet \
    --n_eps 50 --n_eval_workers 4 --success_threshold 0.30
```

**Baseline to beat**: seed42 diffusion = 7/50 = 14%
**Target from Phase 13**: BC_UNet F3 = 6/16 = 38% (different arch/loss, direct BC)

---

## Decision Tree

```
Step 0: Git commit + merge
    ↓
Step 1: Reproduce Phase 13 BC_UNet baseline
    ↓
    ├── Can reproduce (~38%)? → Proceed to Step 2
    └── Cannot reproduce? → Debug first, then continue
    ↓
Step 2: Run preprocess_target.py (60-90 min)
    ↓
Run A, B, C ablations in parallel if resources allow
    ↓
Compare all results:
    Best mix → 50-ep confirmation eval
    If ≥25% success → Try multi-seed ensemble of best mix config
```

---

## Key Risk: Distribution Shift

The target split (layouts 1-10) is the evaluation split for the RoboCasa benchmark's
generalization test — it's intentionally different from pretrain. Training on it:
- MAY help: robot sees more handle manipulation examples, learns better grasping
- MAY hurt: model learns navigation patterns that don't transfer to layouts 11-60

Mix C (curriculum ending in pretrain-only) is the most principled approach to manage this risk.
Mix B (fine-tune) is the standard transfer learning solution.
Mix A (uniform) is the simplest and should be run first as a sanity check.
