"""Extract per-episode door_obj_pos and door_obj_quat from the simulation.

These are static per-episode values (the door position/orientation doesn't
change between resets of the same episode). They're needed by
preprocess_all_states.py to compute door-relative features.

Both door_positions.npz and door_quats.npz are saved together since they're
extracted in the same env.reset() call.

Usage:
    python generate_door_positions.py              # generate both npz files
    python generate_door_positions.py --validate   # validate existing files only
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SAVE_DIR = Path('/tmp/diffusion_policy_checkpoints')
POS_PATH  = SAVE_DIR / 'door_positions.npz'
QUAT_PATH = SAVE_DIR / 'door_quats.npz'


def get_lerobot_root() -> Path:
    from diffusion_policy.data import get_dataset_path
    return Path(get_dataset_path())


def _make_env():
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    import robocasa  # noqa
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    return robosuite.make(
        env_name='OpenCabinet',
        robots='PandaOmron',
        controller_configs=load_composite_controller_config(robot='PandaOmron'),
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_depths=False,
        seed=0,
        obj_instance_split='pretrain',
        layout_ids=-2,
        style_ids=-2,
    )


def generate(force=False):
    """Extract door_obj_pos and door_obj_quat for every episode.

    Saves:
        door_positions.npz  — {str(episode_id): (3,) float32}
        door_quats.npz      — {str(episode_id): (4,) float32}
    """
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if POS_PATH.exists() and QUAT_PATH.exists() and not force:
        print(f"Both files already exist (use --force to regenerate):\n"
              f"  {POS_PATH}\n  {QUAT_PATH}")
        return

    lerobot_root = get_lerobot_root()
    ep_dirs = sorted(lerobot_root.glob('extras/episode_*'))
    if not ep_dirs:
        raise FileNotFoundError(
            f"No episode dirs found under {lerobot_root / 'extras'}. "
            "Run 04_download_dataset.py first."
        )

    print(f"Extracting door positions/quats for {len(ep_dirs)} episodes...")
    t0 = time.time()

    env = _make_env()
    door_positions = {}
    door_quats = {}

    try:
        for ep_dir in ep_dirs:
            ep_id = int(ep_dir.name.split('_')[-1])

            with open(ep_dir / 'ep_meta.json') as f:
                ep_meta = json.load(f)
            with gzip.open(ep_dir / 'model.xml.gz', 'rb') as f:
                model_xml = f.read().decode('utf-8')

            if hasattr(env, 'set_ep_meta'):
                env.set_ep_meta(ep_meta)
            obs = env.reset()
            env.reset_from_xml_string(env.edit_model_xml(model_xml))
            env.sim.reset()
            # Re-read obs after full reset so the door body is placed correctly
            obs = env.reset()

            door_positions[str(ep_id)] = obs['door_obj_pos'].flatten().astype(np.float32)
            door_quats[str(ep_id)]     = obs['door_obj_quat'].flatten().astype(np.float32)

            if (ep_id + 1) % 20 == 0 or ep_id == len(ep_dirs) - 1:
                print(f"  {ep_id + 1}/{len(ep_dirs)} episodes done "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)
    finally:
        env.close()

    np.savez(POS_PATH,  **door_positions)
    np.savez(QUAT_PATH, **door_quats)
    print(f"Saved {len(door_positions)} entries to:\n  {POS_PATH}\n  {QUAT_PATH}")
    print(f"Total time: {time.time() - t0:.1f}s")

    _validate_files(door_positions, door_quats, len(ep_dirs))


def validate():
    """Validate existing door_positions.npz and door_quats.npz."""
    ok = True

    for path, name, expected_shape in [
        (POS_PATH,  'door_positions.npz', (3,)),
        (QUAT_PATH, 'door_quats.npz',     (4,)),
    ]:
        if not path.exists():
            print(f"MISSING: {path}")
            ok = False
            continue

        data = np.load(path)
        keys = sorted(data.keys(), key=int)
        n = len(keys)

        # Shape check
        bad_shape = [k for k in keys if data[k].shape != expected_shape]
        if bad_shape:
            print(f"FAIL {name}: wrong shape for episodes {bad_shape[:5]}")
            ok = False

        # Continuity: episode IDs should be 0..n-1
        expected_ids = set(str(i) for i in range(n))
        missing = expected_ids - set(keys)
        extra   = set(keys) - expected_ids
        if missing:
            print(f"FAIL {name}: missing episode IDs {sorted(missing, key=int)[:10]}")
            ok = False
        if extra:
            print(f"WARN {name}: unexpected episode IDs {sorted(extra, key=int)[:5]}")

        # NaN/Inf
        has_nan = any(np.any(~np.isfinite(data[k])) for k in keys)
        if has_nan:
            print(f"FAIL {name}: contains NaN or Inf values")
            ok = False

        # Plausibility for door positions: z should be ~1.3–1.6 m (cabinet height)
        if name == 'door_positions.npz':
            zvals = np.array([data[k][2] for k in keys])
            if not (1.0 < zvals.min() and zvals.max() < 2.0):
                print(f"WARN {name}: z-values outside expected range [1.0, 2.0]: "
                      f"min={zvals.min():.3f} max={zvals.max():.3f}")
            else:
                print(f"OK  {name}: {n} episodes, z in [{zvals.min():.3f}, {zvals.max():.3f}]")

        # Plausibility for quats: should be unit quaternions
        if name == 'door_quats.npz':
            norms = np.array([np.linalg.norm(data[k]) for k in keys])
            max_err = np.abs(norms - 1.0).max()
            if max_err > 1e-3:
                print(f"FAIL {name}: quaternions not unit (max norm error {max_err:.6f})")
                ok = False
            else:
                print(f"OK  {name}: {n} episodes, max quat norm error {max_err:.6f}")

    if ok:
        print("All checks passed.")
    return ok


def _validate_files(door_positions, door_quats, n_expected):
    """Quick inline validation after generation."""
    assert len(door_positions) == n_expected, \
        f"Expected {n_expected} episodes, got {len(door_positions)}"
    assert len(door_quats) == len(door_positions), "Mismatch between pos/quat counts"

    zvals = np.array([v[2] for v in door_positions.values()])
    assert 1.0 < zvals.min() and zvals.max() < 2.0, \
        f"Door z-positions out of range: {zvals.min():.3f}–{zvals.max():.3f}"

    norms = np.array([np.linalg.norm(v) for v in door_quats.values()])
    assert np.abs(norms - 1.0).max() < 1e-3, \
        f"Non-unit quaternions (max norm error {np.abs(norms - 1.0).max():.6f})"

    print(f"Validation passed: {n_expected} episodes, "
          f"z in [{zvals.min():.3f}, {zvals.max():.3f}], "
          f"max quat error {np.abs(norms - 1.0).max():.2e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true',
                        help='Validate existing files without regenerating')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if files already exist')
    args = parser.parse_args()

    if args.validate:
        ok = validate()
        sys.exit(0 if ok else 1)
    else:
        generate(force=args.force)
