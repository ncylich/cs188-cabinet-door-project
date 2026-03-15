#!/bin/bash
# Run Mix A, B, C ablations after preprocess_target.py completes.
# Usage: bash run_ablations.sh 2>&1 | tee ablations.log

set -e
source /home/noahcylich/cs188-cabinet-door-project/.venv/bin/activate
cd /home/noahcylich/cs188-cabinet-door-project/cabinet_door_project
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=2

TARGET_PT="/tmp/diffusion_policy_checkpoints/preprocessed_target_states.pt"
PRETRAIN_PT="/tmp/diffusion_policy_checkpoints/preprocessed_all_states.pt"
CKPT_DIR="/tmp/diffusion_policy_checkpoints"
LOG_DIR="/home/noahcylich/cs188-cabinet-door-project"

# Wait for both data files
echo "=== Waiting for both preprocessed data files ==="
while [ ! -f "$TARGET_PT" ] || [ ! -f "$PRETRAIN_PT" ]; do
    tgt=$([ -f "$TARGET_PT" ] && echo "OK" || echo "missing")
    pre=$([ -f "$PRETRAIN_PT" ] && echo "OK" || echo "missing")
    echo "  $(date): target=$tgt pretrain=$pre, sleeping 5min..."
    sleep 300
done
echo "=== Both data files ready! Starting ablations ==="

EVAL_FLAGS="--arch unet --feat_subset f3 --eval_only --n_eps 50 --n_eval_workers 4 --success_threshold 0.30"
TRAIN_FLAGS="--arch unet --feat_subset f3 --epochs 200 --patience 30 --batch_size 256 --lr 1e-3"

# ── Mix A: Uniform (pretrain + target, randomly interleaved) ──────────────────
echo ""
echo "=== MIX A: Uniform combined data ==="
python bc_handle.py $TRAIN_FLAGS --combined_data \
    --save_path "$CKPT_DIR/mix_a_f3_unet.pt" \
    2>&1 | tee "$LOG_DIR/train_mixA.log"

echo "--- Evaluating Mix A ---"
python bc_handle.py $EVAL_FLAGS \
    --checkpoint "$CKPT_DIR/mix_a_f3_unet.pt" \
    2>&1 | tee "$LOG_DIR/eval_mixA.log"

# ── Mix B Phase 1: Train on target only ───────────────────────────────────────
echo ""
echo "=== MIX B Phase 1: Target-only pretraining ==="
python bc_handle.py $TRAIN_FLAGS --use_target_only \
    --save_path "$CKPT_DIR/mix_b_phase1.pt" \
    2>&1 | tee "$LOG_DIR/train_mixB_phase1.log"

# ── Mix B Phase 2: Fine-tune on pretrain only ─────────────────────────────────
echo ""
echo "=== MIX B Phase 2: Fine-tune on pretrain ==="
python bc_handle.py $TRAIN_FLAGS --checkpoint "$CKPT_DIR/mix_b_phase1.pt" \
    --epochs 100 --patience 20 --lr 3e-4 \
    --save_path "$CKPT_DIR/mix_b_final.pt" \
    2>&1 | tee "$LOG_DIR/train_mixB_phase2.log"

echo "--- Evaluating Mix B ---"
python bc_handle.py $EVAL_FLAGS \
    --checkpoint "$CKPT_DIR/mix_b_final.pt" \
    2>&1 | tee "$LOG_DIR/eval_mixB.log"

# ── Mix C: Curriculum (decay target 1→0 over 100 epochs) ─────────────────────
echo ""
echo "=== MIX C: Curriculum combined data ==="
python bc_handle.py $TRAIN_FLAGS --combined_data --curriculum_epochs 100 \
    --save_path "$CKPT_DIR/mix_c_f3_unet.pt" \
    2>&1 | tee "$LOG_DIR/train_mixC.log"

echo "--- Evaluating Mix C ---"
python bc_handle.py $EVAL_FLAGS \
    --checkpoint "$CKPT_DIR/mix_c_f3_unet.pt" \
    2>&1 | tee "$LOG_DIR/eval_mixC.log"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "ABLATION RESULTS SUMMARY"
echo "========================================"
for mix in A B C; do
    log="$LOG_DIR/eval_mix${mix}.log"
    if [ -f "$log" ]; then
        result=$(grep -E "success|RESULT" "$log" | tail -3)
        echo "Mix $mix: $result"
    fi
done
echo "========================================"
