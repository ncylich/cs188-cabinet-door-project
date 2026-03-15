"""Parallel version of preprocess_target.py — uses N workers for door position extraction.

Outputs same files as preprocess_target.py:
  /tmp/diffusion_policy_checkpoints/preprocessed_target_states.pt
  /tmp/diffusion_policy_checkpoints/handle_cache_target/episode_*.npy

Workers: each spawns its own robosuite env, processes a disjoint chunk of episodes.
Door positions cached per-episode to door_pos_cache/ so work survives restarts.
"""
import gzip
import json
import logging
import numpy as np
import os
import sys
import time
import torch
from multiprocessing import Pool, cpu_count
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')

TARGET_LEROBOT_ROOT = Path(
    '/home/noahcylich/cs188-cabinet-door-project/robocasa/datasets'
    '/v1.0/target/atomic/OpenCabinet/20250813/lerobot'
)
SAVE_DIR         = Path('/tmp/diffusion_policy_checkpoints')
DOOR_POS_CACHE   = SAVE_DIR / 'door_pos_cache_target'
HANDLE_CACHE_TARGET = SAVE_DIR / 'handle_cache_target'
OUTPUT_PATH      = SAVE_DIR / 'preprocessed_target_states.pt'
N_WORKERS        = 16

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── Quaternion math ──────────────────────────────────────────────────────────

def quat_conjugate(q):
    q = np.asarray(q, dtype=np.float32)
    conj = q.copy(); conj[..., :3] *= -1
    return conj

def quat_multiply(q1, q2):
    q1, q2 = np.asarray(q1, np.float32), np.asarray(q2, np.float32)
    x1, y1, z1, w1 = q1[...,0], q1[...,1], q1[...,2], q1[...,3]
    x2, y2, z2, w2 = q2[...,0], q2[...,1], q2[...,2], q2[...,3]
    return np.stack([
        w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2, w1*w2-x1*x2-y1*y2-z1*z2,
    ], axis=-1)

def quat_rotate_vector(q, v):
    q = np.asarray(q, np.float32); v = np.asarray(v, np.float32)
    v_quat = np.zeros(v.shape[:-1]+(4,), dtype=np.float32); v_quat[...,:3] = v
    return quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))[...,:3]

def compute_derived_features(proprio_seq, door_pos, door_quat):
    T = len(proprio_seq)
    base_pos = proprio_seq[:, 0:3]; base_quat = proprio_seq[:, 3:7]
    base_to_eef_pos = proprio_seq[:, 7:10]; base_to_eef_quat = proprio_seq[:, 10:14]
    eef_pos = base_pos + quat_rotate_vector(base_quat, base_to_eef_pos)
    eef_quat = quat_multiply(base_quat, base_to_eef_quat)
    door_pos_t = np.tile(door_pos, (T,1)); door_quat_t = np.tile(door_quat, (T,1))
    door_to_eef_pos = eef_pos - door_pos_t
    door_to_eef_quat = quat_multiply(quat_conjugate(door_quat_t), eef_quat)
    gripper_to_door_dist = np.linalg.norm(door_to_eef_pos, axis=-1, keepdims=True)
    return {
        'proprio': proprio_seq, 'door_pos': door_pos_t, 'door_quat': door_quat_t,
        'eef_pos': eef_pos, 'eef_quat': eef_quat, 'door_to_eef_pos': door_to_eef_pos,
        'door_to_eef_quat': door_to_eef_quat, 'gripper_to_door_dist': gripper_to_door_dist,
    }


# ── Spawn-safe worker: extract door positions for a chunk of episodes ────────

def _door_pos_worker(args):
    ep_ids, extras_root_str, cache_dir_str = args
    extras_root = Path(extras_root_str)
    cache_dir   = Path(cache_dir_str)

    os.environ['MUJOCO_GL'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

    import robocasa  # noqa
    import robosuite as _rs
    from robosuite.controllers import load_composite_controller_config

    env = _rs.make(
        env_name='OpenCabinet', robots='PandaOmron',
        controller_configs=load_composite_controller_config(robot='PandaOmron'),
        has_renderer=False, has_offscreen_renderer=False, ignore_done=True,
        use_object_obs=True, use_camera_obs=False, camera_depths=False,
        seed=0, obj_instance_split='pretrain', layout_ids=-2, style_ids=-2,
    )

    results = {}
    try:
        for ep_id in ep_ids:
            pos_path  = cache_dir / f'episode_{ep_id:06d}_pos.npy'
            quat_path = cache_dir / f'episode_{ep_id:06d}_quat.npy'
            if pos_path.exists() and quat_path.exists():
                results[ep_id] = (np.load(pos_path), np.load(quat_path))
                continue
            ep_dir   = extras_root / f'episode_{ep_id:06d}'
            meta_path = ep_dir / 'ep_meta.json'
            xml_path  = ep_dir / 'model.xml.gz'
            if not meta_path.exists() or not xml_path.exists():
                results[ep_id] = (np.zeros(3, np.float32), np.array([0,0,0,1], np.float32))
                continue
            with open(meta_path) as f:
                ep_meta = json.load(f)
            with gzip.open(xml_path, 'rb') as f:
                model_xml = f.read().decode('utf-8')
            if hasattr(env, 'set_ep_meta'):
                env.set_ep_meta(ep_meta)
            env.reset()
            env.reset_from_xml_string(env.edit_model_xml(model_xml))
            env.sim.reset()
            obs = env.reset()
            dp = obs['door_obj_pos'].flatten().astype(np.float32)
            dq = obs['door_obj_quat'].flatten().astype(np.float32)
            np.save(pos_path,  dp)
            np.save(quat_path, dq)
            results[ep_id] = (dp, dq)
    finally:
        env.close()
    return results


# ── Spawn-safe worker: build handle cache for a chunk of episodes ────────────

def _handle_worker(args):
    ep_ids_bounds, extras_root_str, handle_cache_str, eef_pos_all = args
    extras_root  = Path(extras_root_str)
    handle_cache = Path(handle_cache_str)

    os.environ['MUJOCO_GL'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

    import robocasa  # noqa
    import robosuite as _rs
    from robosuite.controllers import load_composite_controller_config
    from diffusion_policy.evaluation import get_handle_site_names

    env = _rs.make(
        env_name='OpenCabinet', robots='PandaOmron',
        controller_configs=load_composite_controller_config(robot='PandaOmron'),
        has_renderer=False, has_offscreen_renderer=False, ignore_done=True,
        use_object_obs=True, use_camera_obs=False, camera_depths=False,
        seed=0, obj_instance_split='pretrain', layout_ids=-2, style_ids=-2,
    )

    handle_results = {}
    try:
        for ep_id, s, e in ep_ids_bounds:
            ep_id = int(ep_id); s = int(s); e = int(e); T = e - s
            cache_path = handle_cache / f'episode_{ep_id:06d}.npy'
            if cache_path.exists():
                handle_results[ep_id] = np.load(cache_path)[:T]
                continue
            ep_dir = extras_root / f'episode_{ep_id:06d}'
            sim_states_path = ep_dir / 'states.npz'
            if not sim_states_path.exists():
                handle_results[ep_id] = np.zeros((T, 3), np.float32)
                continue
            sim_states = np.load(sim_states_path)['states']
            with open(ep_dir / 'ep_meta.json') as f:
                ep_meta = json.load(f)
            with gzip.open(ep_dir / 'model.xml.gz', 'rb') as f:
                model_xml = f.read().decode('utf-8')
            if hasattr(env, 'set_ep_meta'):
                env.set_ep_meta(ep_meta)
            env.reset()
            xml = env.edit_model_xml(model_xml)
            env.reset_from_xml_string(xml)
            env.sim.reset()
            sites = get_handle_site_names(env)
            ep_handle_pos = np.zeros((T, 3), np.float32)
            active_site = sites[0] if sites else None
            for idx in range(min(T, len(sim_states))):
                env.sim.set_state_from_flattened(sim_states[idx])
                env.sim.forward()
                if not sites:
                    continue
                eef = eef_pos_all[s + idx]
                if len(sites) > 1:
                    active_site = min(sites, key=lambda st: np.linalg.norm(
                        env.sim.data.site_xpos[env.sim.model.site_name2id(st)] - eef))
                sid = env.sim.model.site_name2id(active_site)
                ep_handle_pos[idx] = env.sim.data.site_xpos[sid]
            np.save(cache_path, ep_handle_pos)
            handle_results[ep_id] = ep_handle_pos
    finally:
        env.close()
    return handle_results


# ── Main ─────────────────────────────────────────────────────────────────────

def preprocess_target_parallel():
    import multiprocessing as mp
    ctx = mp.get_context('spawn')

    t0 = time.time()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    DOOR_POS_CACHE.mkdir(parents=True, exist_ok=True)
    HANDLE_CACHE_TARGET.mkdir(parents=True, exist_ok=True)

    # Load episodes from parquet
    logger.info('Loading target episodes...')
    import pyarrow.parquet as pq
    chunk_dir = TARGET_LEROBOT_ROOT / 'data' / 'chunk-000'
    parquet_files = sorted(chunk_dir.glob('episode_*.parquet'))
    episodes = []
    for pf in parquet_files:
        df = pq.read_table(str(pf)).to_pandas()
        states  = np.stack(df['observation.state'].values).astype(np.float32)
        actions = np.stack(df['action'].values).astype(np.float32)
        ep_idx  = int(df['episode_index'].iloc[0])
        episodes.append({'states': states, 'actions': actions, 'episode_index': ep_idx})
    logger.info(f'Loaded {len(episodes)} target episodes')

    ep_ids = [ep['episode_index'] for ep in episodes]
    extras_root = str(TARGET_LEROBOT_ROOT / 'extras')

    # ── Phase 1: Door positions (parallel) ───────────────────────────────────
    logger.info(f'Extracting door positions with {N_WORKERS} workers...')
    t1 = time.time()

    # Check cache
    uncached = [eid for eid in ep_ids
                if not (DOOR_POS_CACHE / f'episode_{eid:06d}_pos.npy').exists()]
    logger.info(f'  {len(ep_ids) - len(uncached)} already cached, {len(uncached)} to process')

    if uncached:
        chunk_size = max(1, len(uncached) // N_WORKERS)
        chunks = [uncached[i:i+chunk_size] for i in range(0, len(uncached), chunk_size)]
        worker_args = [(chunk, extras_root, str(DOOR_POS_CACHE)) for chunk in chunks]
        with ctx.Pool(len(chunks)) as pool:
            results_list = pool.map(_door_pos_worker, worker_args)
        for r in results_list:
            pass  # results saved to disk by workers
        logger.info(f'  Door positions done in {time.time()-t1:.0f}s')

    # Load all from cache
    door_positions = {}; door_quats = {}
    for eid in ep_ids:
        pos_path  = DOOR_POS_CACHE / f'episode_{eid:06d}_pos.npy'
        quat_path = DOOR_POS_CACHE / f'episode_{eid:06d}_quat.npy'
        if pos_path.exists() and quat_path.exists():
            door_positions[eid] = np.load(pos_path)
            door_quats[eid]     = np.load(quat_path)
        else:
            door_positions[eid] = np.zeros(3, np.float32)
            door_quats[eid]     = np.array([0,0,0,1], np.float32)

    logger.info(f'Door positions loaded for {len(door_positions)} episodes')

    # ── Build features ────────────────────────────────────────────────────────
    all_features = {}
    all_actions  = []
    ep_boundaries = []
    total_frames  = 0

    for ep in episodes:
        ep_id   = ep['episode_index']
        proprio = ep['states'].astype(np.float32)
        actions = ep['actions'].astype(np.float32)
        T       = len(actions)
        dp = door_positions.get(ep_id, np.zeros(3, np.float32))
        dq = door_quats.get(ep_id,     np.array([0,0,0,1], np.float32))
        features = compute_derived_features(proprio, dp, dq)
        start = total_frames; total_frames += T
        ep_boundaries.append((ep_id, start, start + T))
        for name, arr in features.items():
            all_features.setdefault(name, []).append(arr)
        all_actions.append(actions)

    for name in all_features:
        all_features[name] = np.concatenate(all_features[name], axis=0)
    all_actions_np  = np.concatenate(all_actions, axis=0)
    ep_boundaries   = np.array(ep_boundaries)
    eef_pos_all     = all_features['eef_pos']

    logger.info(f'Total frames: {total_frames}, episodes: {len(ep_boundaries)}')

    # ── Phase 2: Handle cache (parallel) ─────────────────────────────────────
    n_cached = sum(1 for eid, s, e in ep_boundaries
                   if (HANDLE_CACHE_TARGET / f'episode_{int(eid):06d}.npy').exists())
    logger.info(f'Handle cache: {n_cached}/{len(ep_boundaries)} already cached')

    uncached_eps = [(int(eid), int(s), int(e)) for eid, s, e in ep_boundaries
                    if not (HANDLE_CACHE_TARGET / f'episode_{int(eid):06d}.npy').exists()]

    handle_pos_all = np.zeros((total_frames, 3), np.float32)

    if uncached_eps:
        logger.info(f'Building handle cache for {len(uncached_eps)} episodes with {N_WORKERS} workers...')
        t2 = time.time()
        chunk_size = max(1, len(uncached_eps) // N_WORKERS)
        chunks = [uncached_eps[i:i+chunk_size] for i in range(0, len(uncached_eps), chunk_size)]
        worker_args = [(chunk, extras_root, str(HANDLE_CACHE_TARGET), eef_pos_all)
                       for chunk in chunks]
        with ctx.Pool(len(chunks)) as pool:
            results_list = pool.map(_handle_worker, worker_args)
        logger.info(f'  Handle cache done in {time.time()-t2:.0f}s')

    # Load all handle positions from cache
    for eid, s, e in ep_boundaries:
        eid = int(eid); s = int(s); e = int(e); T = e - s
        cache_path = HANDLE_CACHE_TARGET / f'episode_{eid:06d}.npy'
        if cache_path.exists():
            arr = np.load(cache_path)
            handle_pos_all[s:e] = arr[:T]

    # ── Build final obs tensor ─────────────────────────────────────────────────
    FEAT_NAMES = [
        'proprio', 'door_pos', 'door_quat', 'eef_pos', 'eef_quat',
        'door_to_eef_pos', 'door_to_eef_quat', 'gripper_to_door_dist',
    ]
    feats_t      = {k: torch.from_numpy(v) for k, v in all_features.items()}
    handle_pos_t     = torch.from_numpy(handle_pos_all)
    handle_to_eef_t  = feats_t['eef_pos'] - handle_pos_t

    obs_full = torch.cat(
        [feats_t[n] for n in FEAT_NAMES] + [handle_pos_t, handle_to_eef_t],
        dim=-1,
    ).float()
    logger.info(f'obs_full shape: {obs_full.shape}  (expected 44-dim)')

    # ── Save ───────────────────────────────────────────────────────────────────
    # Write to a temp file first, then atomically rename so the sequential
    # version (if still running) doesn't race on the same path.
    tmp_path = OUTPUT_PATH.with_suffix('.pt.tmp')
    torch.save({
        'features':    feats_t,
        'actions':     torch.from_numpy(all_actions_np),
        'ep_boundaries': ep_boundaries,
        'obs_full':    obs_full,
    }, tmp_path)
    tmp_path.rename(OUTPUT_PATH)
    logger.info(f'Saved to {OUTPUT_PATH}')
    logger.info(f'Total time: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    preprocess_target_parallel()
