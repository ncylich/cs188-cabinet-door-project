#!/bin/bash
# Wait for the current training to finish, then run ablations
set -e

# The training python process PID
TRAIN_PID=$(pgrep -f "python -u -c.*unet_19dim_3k_bs2048" | head -1)
echo "Waiting for training process PID $TRAIN_PID to finish..."

if [ -n "$TRAIN_PID" ]; then
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep 30
        echo "$(date): Training still running... ($(tail -1 /tmp/oracle_1hr_log.txt))"
    done
fi

echo "$(date): Training complete! Final log:"
cat /tmp/oracle_1hr_log.txt
echo ""
echo "Starting ablations..."

cd /home/noahcylich/cs188-cabinet-door-project/cabinet_door_project
source /home/noahcylich/cs188-cabinet-door-project/.venv/bin/activate
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}

python -u run_ablations.py 2>&1 | tee /tmp/ablation_log.txt
