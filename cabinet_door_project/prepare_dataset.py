"""End-to-end dataset preparation pipeline.

Stages (each is skipped if output already exists):
  1. Check dataset downloaded
  2. Generate door_positions.npz + door_quats.npz
  3. Preprocess all states → preprocessed_all_states.pt
  4. Build handle position cache (parallel sim replay)
  5. Validate all outputs

Usage:
    python prepare_dataset.py                  # run all stages, skip completed
    python prepare_dataset.py --force          # re-run all stages
    python prepare_dataset.py --validate_only  # validate existing outputs only
    python prepare_dataset.py --n_workers 6    # parallel workers for handle cache
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S', force=True)
log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

SAVE_DIR     = Path('/tmp/diffusion_policy_checkpoints')
HANDLE_CACHE = SAVE_DIR / 'handle_cache'


# ── helpers ────────────────────────────────────────────────────────────────

def _banner(msg):
    log.info('─' * 60)
    log.info(msg)
    log.info('─' * 60)


def _done(t0):
    log.info(f'  done ({time.time() - t0:.1f}s)')


# ── stage 1: dataset ────────────────────────────────────────────────────────

def check_dataset():
    from diffusion_policy.data import get_dataset_path
    try:
        path = get_dataset_path()
        ep_dirs = list(Path(path).glob('extras/episode_*'))
        if not ep_dirs:
            raise FileNotFoundError("extras/ is empty")
        log.info(f'  Dataset OK: {len(ep_dirs)} episodes at {path}')
        return path, len(ep_dirs)
    except FileNotFoundError as e:
        log.error(f'  Dataset not found: {e}')
        log.error('  Run: python 04_download_dataset.py')
        sys.exit(1)


# ── stage 2: door positions / quats ─────────────────────────────────────────

def stage_door_positions(force=False):
    pos_path  = SAVE_DIR / 'door_positions.npz'
    quat_path = SAVE_DIR / 'door_quats.npz'
    if pos_path.exists() and quat_path.exists() and not force:
        log.info(f'  Skipping: {pos_path.name} and {quat_path.name} already exist')
        return
    from generate_door_positions import generate
    t0 = time.time()
    generate(force=force)
    _done(t0)


# ── stage 3: preprocess all states ──────────────────────────────────────────

def stage_preprocess(force=False):
    out = SAVE_DIR / 'preprocessed_all_states.pt'
    if out.exists() and not force:
        log.info(f'  Skipping: {out.name} already exists')
        return
    from preprocess_all_states import preprocess_all
    t0 = time.time()
    preprocess_all(save_dir=str(SAVE_DIR))
    _done(t0)


# ── stage 4: handle cache ────────────────────────────────────────────────────

def stage_handle_cache(n_workers=4, force=False):
    preprocessed = SAVE_DIR / 'preprocessed_all_states.pt'
    if not preprocessed.exists():
        log.error('  preprocessed_all_states.pt missing — run stage 3 first')
        sys.exit(1)

    data = torch.load(preprocessed, weights_only=False)
    ep_bounds = data['ep_boundaries']
    n_eps = len(ep_bounds)

    cached = sum(
        1 for eid, _, _ in ep_bounds
        if (HANDLE_CACHE / f'episode_{int(eid):06d}.npy').exists()
    )

    if cached == n_eps and not force:
        log.info(f'  Skipping: handle cache complete ({cached}/{n_eps} episodes)')
        return

    if force:
        import shutil
        if HANDLE_CACHE.exists():
            shutil.rmtree(HANDLE_CACHE)
        log.info('  Cleared existing handle cache (--force)')

    log.info(f'  Building handle cache: {cached}/{n_eps} already done, '
             f'{n_workers} workers...')

    # Import build_handle_cache from bc_handle (same function, avoids duplication)
    from bc_handle import build_handle_cache
    t0 = time.time()
    build_handle_cache(data['features']['eef_pos'], ep_bounds, n_workers=n_workers)
    _done(t0)


# ── stage 5: validation ──────────────────────────────────────────────────────

def validate_all(n_eps_expected):
    all_ok = True

    # 2. door positions / quats
    from generate_door_positions import validate as validate_doors
    log.info('  Validating door_positions.npz and door_quats.npz...')
    if not validate_doors():
        all_ok = False

    # 3. preprocessed_all_states.pt
    from preprocess_all_states import validate_preprocessed
    log.info('  Validating preprocessed_all_states.pt...')
    if not validate_preprocessed(path=str(SAVE_DIR / 'preprocessed_all_states.pt')):
        all_ok = False

    # 4. handle cache
    log.info('  Validating handle cache...')
    if not validate_handle_cache(n_eps_expected):
        all_ok = False

    return all_ok


def validate_handle_cache(n_eps_expected):
    ok = True
    preprocessed = SAVE_DIR / 'preprocessed_all_states.pt'
    if not preprocessed.exists():
        log.error('  FAIL: preprocessed_all_states.pt missing')
        return False

    data      = torch.load(preprocessed, weights_only=False)
    ep_bounds = data['ep_boundaries']
    eef_pos   = data['features']['eef_pos'].numpy()

    missing, bad_shape, bad_values = [], [], []
    for eid, start, end in ep_bounds:
        path = HANDLE_CACHE / f'episode_{int(eid):06d}.npy'
        if not path.exists():
            missing.append(int(eid))
            continue
        arr = np.load(path)
        T = int(end) - int(start)
        if arr.shape != (T, 3):
            bad_shape.append((int(eid), arr.shape, (T, 3)))
            continue
        # Handle positions should be finite and within ~2m of EEF
        if not np.isfinite(arr).all():
            bad_values.append((int(eid), 'non-finite'))
            continue
        max_dist = np.linalg.norm(arr - eef_pos[int(start):int(end)], axis=-1).max()
        if max_dist > 2.0:
            bad_values.append((int(eid), f'max dist from EEF = {max_dist:.2f}m'))

    if missing:
        log.error(f'  FAIL handle cache: {len(missing)} missing episodes: {missing[:10]}')
        ok = False
    if bad_shape:
        log.error(f'  FAIL handle cache: wrong shape in {len(bad_shape)} episodes: '
                  f'{bad_shape[:3]}')
        ok = False
    if bad_values:
        log.error(f'  FAIL handle cache: bad values in {len(bad_values)} episodes: '
                  f'{bad_values[:3]}')
        ok = False

    n_cached = len(ep_bounds) - len(missing)
    if ok:
        log.info(f'  OK  handle_cache: {n_cached}/{len(ep_bounds)} episodes, '
                 f'all shapes/values valid')
    return ok


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')

    parser = argparse.ArgumentParser()
    parser.add_argument('--force',         action='store_true',
                        help='Re-run all stages even if outputs exist')
    parser.add_argument('--validate_only', action='store_true',
                        help='Only validate existing outputs, do not generate')
    parser.add_argument('--n_workers',     type=int, default=4,
                        help='Parallel workers for handle cache (default 4)')
    args = parser.parse_args()

    t_total = time.time()

    _banner('Stage 1: Check dataset')
    _, n_eps = check_dataset()

    if args.validate_only:
        _banner('Validation only')
        ok = validate_all(n_eps)
        log.info('All checks passed.' if ok else 'Validation FAILED.')
        sys.exit(0 if ok else 1)

    _banner('Stage 2: Door positions + quaternions')
    stage_door_positions(force=args.force)

    _banner('Stage 3: Preprocess all states')
    stage_preprocess(force=args.force)

    _banner('Stage 4: Handle position cache')
    stage_handle_cache(n_workers=args.n_workers, force=args.force)

    _banner('Stage 5: Validate all outputs')
    ok = validate_all(n_eps)

    log.info('─' * 60)
    total = time.time() - t_total
    if ok:
        log.info(f'Pipeline complete in {total:.0f}s. Ready to train.')
    else:
        log.error(f'Pipeline finished with errors after {total:.0f}s.')
        sys.exit(1)


if __name__ == '__main__':
    main()
