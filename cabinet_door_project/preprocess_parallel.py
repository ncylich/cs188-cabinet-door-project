"""Full preprocessing pipeline with parallelized door-position extraction.

Runs:
  1. Parallel door_positions.npz / door_quats.npz  (spawn one env per worker)
  2. preprocess_all_states.preprocess_all()         (already parallelized)
  3. preprocess_all_states.extend_preprocessed()    (adds handle_pos, handle_to_eef)

Usage:
    python preprocess_parallel.py [--workers N]
"""
import os
import sys
import gzip
import json
import time
import argparse
import logging
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

SAVE_DIR     = Path("/tmp/diffusion_policy_checkpoints")
LEROBOT_ROOT = Path(
    "/home/noahcylich/cs188-cabinet-door-project/robocasa/datasets"
    "/v1.0/pretrain/atomic/OpenCabinet/20250819/lerobot"
)
POS_PATH  = SAVE_DIR / "door_positions.npz"
QUAT_PATH = SAVE_DIR / "door_quats.npz"


# ── Parallel door-position worker (one env per process) ───────────────────────

def _door_worker(args):
    """Spawn-safe worker: process a chunk of episodes, return list of (ep_id, pos, quat)."""
    ep_dirs_str, worker_id = args

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    import gzip as _gzip, json as _json
    import numpy as _np
    from pathlib import Path as _Path
    import robocasa  # noqa
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    env = robosuite.make(
        env_name="OpenCabinet",
        robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_depths=False,
        seed=0,
        obj_instance_split="pretrain",
        layout_ids=-2,
        style_ids=-2,
    )

    results = []
    try:
        for ep_dir_str in ep_dirs_str:
            ep_dir = _Path(ep_dir_str)
            ep_id = int(ep_dir.name.split("_")[-1])

            with open(ep_dir / "ep_meta.json") as f:
                ep_meta = _json.load(f)
            with _gzip.open(ep_dir / "model.xml.gz", "rb") as f:
                model_xml = f.read().decode("utf-8")

            if hasattr(env, "set_ep_meta"):
                env.set_ep_meta(ep_meta)
            env.reset()
            env.reset_from_xml_string(env.edit_model_xml(model_xml))
            env.sim.reset()
            obs = env.reset()

            results.append((
                ep_id,
                obs["door_obj_pos"].flatten().astype(_np.float32),
                obs["door_obj_quat"].flatten().astype(_np.float32),
            ))
    finally:
        env.close()

    return results


def generate_door_positions_parallel(n_workers=8):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if POS_PATH.exists() and QUAT_PATH.exists():
        logger.info("door_positions.npz and door_quats.npz already exist — skipping.")
        return

    ep_dirs = sorted(LEROBOT_ROOT.glob("extras/episode_*"))
    if not ep_dirs:
        raise FileNotFoundError(f"No episode dirs under {LEROBOT_ROOT / 'extras'}")

    n = len(ep_dirs)
    logger.info(f"Extracting door positions for {n} episodes with {n_workers} workers...")

    # Split episodes into chunks, one chunk per worker
    chunk_size = (n + n_workers - 1) // n_workers
    chunks = [
        ([str(ep_dirs[i]) for i in range(start, min(start + chunk_size, n))], wid)
        for wid, start in enumerate(range(0, n, chunk_size))
    ]

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    t0 = time.time()

    with ctx.Pool(n_workers) as pool:
        all_results = pool.map(_door_worker, chunks)

    door_positions, door_quats = {}, {}
    for chunk_results in all_results:
        for ep_id, pos, quat in chunk_results:
            door_positions[str(ep_id)] = pos
            door_quats[str(ep_id)]     = quat

    np.savez(POS_PATH,  **door_positions)
    np.savez(QUAT_PATH, **door_quats)
    logger.info(f"Saved {len(door_positions)} entries in {time.time()-t0:.0f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for door position extraction")
    args = parser.parse_args()

    # Step 1: parallel door position extraction
    generate_door_positions_parallel(n_workers=args.workers)

    # Step 2: compute all derived features (already parallel internally)
    logger.info("\n=== Step 2: preprocess_all_states ===")
    from preprocess_all_states import preprocess_all
    preprocess_all(save_dir=str(SAVE_DIR))

    # Step 3: add handle_pos, handle_to_eef
    logger.info("\n=== Step 3: extend with handle features ===")
    from preprocess_all_states import extend_preprocessed
    extend_preprocessed(save_path=str(SAVE_DIR / "preprocessed_all_states.pt"))

    logger.info("\n=== Preprocessing complete ===")
    logger.info(f"Output: {SAVE_DIR / 'preprocessed_all_states.pt'}")


if __name__ == "__main__":
    main()
