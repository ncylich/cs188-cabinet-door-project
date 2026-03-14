#!/bin/bash
# Run all key evaluations sequentially with OMP_NUM_THREADS=2
set -e
source /home/noahcylich/cs188-cabinet-door-project/.venv/bin/activate
cd /home/noahcylich/cs188-cabinet-door-project/cabinet_door_project
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=2

CKPT_DIR="../checkpoints"
LOG_DIR=".."
N_EPS=20
N_WORKERS=4
SEED=0

run_eval() {
    local name="$1"
    local ckpt="$2"
    local extra="$3"
    echo ""
    echo "=== Running eval: $name ==="
    python bc_handle.py --eval_only --checkpoint "$ckpt" \
        --arch unet --n_eps $N_EPS --n_eval_workers $N_WORKERS --seed $SEED \
        $extra \
        > "$LOG_DIR/eval_${name}.log" 2>&1
    local sr=$(grep "Result:" "$LOG_DIR/eval_${name}.log" | tail -1)
    echo "  $sr"
}

run_transformer_eval() {
    local name="$1"
    local ckpt="$2"
    echo ""
    echo "=== Running transformer eval: $name ==="
    python bc_handle.py --eval_only --checkpoint "$ckpt" \
        --arch transformer --n_eps $N_EPS --n_eval_workers $N_WORKERS --seed $SEED \
        > "$LOG_DIR/eval_${name}.log" 2>&1
    local sr=$(grep "Result:" "$LOG_DIR/eval_${name}.log" | tail -1)
    echo "  $sr"
}

# 1. Verify transformer baseline
run_transformer_eval "transformer_base" "$CKPT_DIR/bc_transformer_best.pt"

# 2. UNet seed1 (previously got 5%)
run_eval "unet_seed1_v2" "$CKPT_DIR/bc_unet_seed1_best.pt"

# 3. UNet with obs_noise augmentation
run_eval "unet_noise01" "$CKPT_DIR/bc_unet_noise01_seed77.pt"

# 4. UNet with n_obs_steps=4
run_eval "unet_nobs4" "$CKPT_DIR/bc_unet_nobs4_seed42.pt" "--n_obs_steps 4"

# 5. Combined (noise + nobs4)
run_eval "unet_combined" "$CKPT_DIR/bc_unet_combined_seed99.pt" "--n_obs_steps 4"

# 6. Seed42 with ddim_samples=4
run_eval "seed42_ddim4" "$CKPT_DIR/bc_unet_seed42_best.pt" "--ddim_samples 4"

# 7. Seed7 (best val=0.05112)
run_eval "unet_seed7_v2" "$CKPT_DIR/bc_unet_seed7_best.pt"

# 8. Seed50 (val=0.05178)
run_eval "unet_seed50_v2" "$CKPT_DIR/bc_unet_seed50_best.pt"

# 9. Seed3 (val=0.05142)
run_eval "unet_seed3_v2" "$CKPT_DIR/bc_unet_seed3_best.pt"

# 10. Seed42 with success_threshold=0.30 (TA-recommended lenient criterion)
run_eval "seed42_lenienth" "$CKPT_DIR/bc_unet_seed42_best.pt" "--success_threshold 0.30"

echo ""
echo "=== ALL EVALS DONE ==="
echo "Results summary:"
for f in "$LOG_DIR"/eval_*.log; do
    name=$(basename $f .log | sed 's/eval_//')
    sr=$(grep "Result:" $f | tail -1)
    echo "  $name: $sr"
done
