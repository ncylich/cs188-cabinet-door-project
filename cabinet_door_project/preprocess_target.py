"""Preprocess TARGET split OpenCabinet dataset (500 demos) into the same format as pretrain.

Outputs: /tmp/diffusion_policy_checkpoints/preprocessed_target_states.pt
          /tmp/diffusion_policy_checkpoints/handle_cache_target/episode_*.npy

Usage:
    python preprocess_target.py

After running, use --combined_data flag in bc_handle.py training to include both splits.
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
SAVE_DIR = Path('/tmp/diffusion_policy_checkpoints')
HANDLE_CACHE_TARGET = SAVE_DIR / 'handle_cache_target'
OUTPUT_PATH = SAVE_DIR / 'preprocessed_target_states.pt'

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── Quaternion math ─────────────────────────────────────────────────────────

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


# ── Load target episodes from parquet ───────────────────────────────────────

def load_target_episodes():
    import pyarrow.parquet as pq
    chunk_dir = TARGET_LEROBOT_ROOT / 'data' / 'chunk-000'
    parquet_files = sorted(chunk_dir.glob('episode_*.parquet'))
    logger.info(f'Found {len(parquet_files)} target parquet files')
    episodes = []
    for pf in parquet_files:
        df = pq.read_table(str(pf)).to_pandas()
        states = np.stack(df['observation.state'].values).astype(np.float32)
        actions = np.stack(df['action'].values).astype(np.float32)
        ep_idx = int(df['episode_index'].iloc[0])
        episodes.append({'states': states, 'actions': actions, 'episode_index': ep_idx})
    return episodes


# ── Extract door positions for target episodes ───────────────────────────────

def extract_door_positions_target(episodes):
    """Extract door positions/quats by replaying each episode in sim."""
    import robocasa  # noqa
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    extras_root = TARGET_LEROBOT_ROOT / 'extras'
    env = robosuite.make(
        env_name='OpenCabinet', robots='PandaOmron',
        controller_configs=load_composite_controller_config(robot='PandaOmron'),
        has_renderer=False, has_offscreen_renderer=False, ignore_done=True,
        use_object_obs=True, use_camera_obs=False, camera_depths=False,
        seed=0, obj_instance_split='pretrain', layout_ids=-2, style_ids=-2,
    )

    ep_ids = [ep['episode_index'] for ep in episodes]
    door_positions = {}; door_quats = {}
    t0 = time.time()

    try:
        for ep_id in ep_ids:
            ep_dir = extras_root / f'episode_{ep_id:06d}'
            meta_path = ep_dir / 'ep_meta.json'
            xml_path = ep_dir / 'model.xml.gz'
            if not meta_path.exists() or not xml_path.exists():
                logger.warning(f'Missing extras for episode {ep_id}')
                door_positions[ep_id] = np.zeros(3, dtype=np.float32)
                door_quats[ep_id] = np.array([0,0,0,1], dtype=np.float32)
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
            obs = env.reset()  # second reset places door correctly
            door_positions[ep_id] = obs['door_obj_pos'].flatten().astype(np.float32)
            door_quats[ep_id] = obs['door_obj_quat'].flatten().astype(np.float32)
            if (ep_id + 1) % 50 == 0:
                logger.info(f'  Door positions: {ep_id+1}/{len(ep_ids)} ({time.time()-t0:.0f}s)')
    finally:
        env.close()
    return door_positions, door_quats


# ── Build handle cache for target episodes ───────────────────────────────────

def build_handle_cache_target(episodes, eef_pos_all, ep_boundaries):
    """Build handle position cache for target episodes."""
    from diffusion_policy.evaluation import get_handle_site_names
    import robocasa  # noqa
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    HANDLE_CACHE_TARGET.mkdir(parents=True, exist_ok=True)

    ep_bounds = {int(eid): (int(s), int(e)) for eid, s, e in ep_boundaries}
    extras_root = TARGET_LEROBOT_ROOT / 'extras'
    N = len(eef_pos_all)
    handle_pos_all = np.zeros((N, 3), dtype=np.float32)

    # Check how many are already cached
    n_cached = sum(1 for eid in ep_bounds
                   if (HANDLE_CACHE_TARGET / f'episode_{int(eid):06d}.npy').exists())
    if n_cached == len(ep_bounds):
        logger.info(f'Handle cache complete ({n_cached}/{len(ep_bounds)} episodes cached).')
        for eid, (s, e) in ep_bounds.items():
            p = HANDLE_CACHE_TARGET / f'episode_{int(eid):06d}.npy'
            arr = np.load(p); handle_pos_all[s:e] = arr[:e-s]
        return handle_pos_all

    env = robosuite.make(
        env_name='OpenCabinet', robots='PandaOmron',
        controller_configs=load_composite_controller_config(robot='PandaOmron'),
        has_renderer=False, has_offscreen_renderer=False, ignore_done=True,
        use_object_obs=True, use_camera_obs=False, camera_depths=False,
        seed=0, obj_instance_split='pretrain', layout_ids=-2, style_ids=-2,
    )

    episode_dirs = sorted(extras_root.glob('episode_*'))
    logger.info(f'Building handle cache for {len(episode_dirs)} target episodes...')

    try:
        for ep_dir in episode_dirs:
            ep_id = int(ep_dir.name.split('_')[-1])
            if ep_id not in ep_bounds:
                continue
            s, e = ep_bounds[ep_id]; T = e - s
            cache_path = HANDLE_CACHE_TARGET / f'episode_{ep_id:06d}.npy'
            if cache_path.exists():
                arr = np.load(cache_path); handle_pos_all[s:e] = arr[:T]
                continue
            sim_states_path = ep_dir / 'states.npz'
            if not sim_states_path.exists():
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
            ep_handle_pos = np.zeros((T, 3), dtype=np.float32)
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
            handle_pos_all[s:e] = ep_handle_pos
            logger.info(f'  Cached episode {ep_id} ({s}-{e})')
    finally:
        env.close()

    return handle_pos_all


# ── Main ─────────────────────────────────────────────────────────────────────

def preprocess_target():
    t0 = time.time()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    HANDLE_CACHE_TARGET.mkdir(parents=True, exist_ok=True)

    logger.info('Loading target episodes...')
    episodes = load_target_episodes()
    logger.info(f'Loaded {len(episodes)} target episodes')

    logger.info('Extracting door positions for target episodes...')
    door_positions, door_quats = extract_door_positions_target(episodes)

    # Build features for each episode
    all_features = {}
    all_actions = []
    ep_boundaries = []
    total_frames = 0

    for ep in episodes:
        ep_id = ep['episode_index']
        proprio = ep['states'].astype(np.float32)
        actions = ep['actions'].astype(np.float32)
        T = len(actions)
        dp = door_positions.get(ep_id, np.zeros(3, dtype=np.float32))
        dq = door_quats.get(ep_id, np.array([0,0,0,1], dtype=np.float32))
        features = compute_derived_features(proprio, dp, dq)
        start = total_frames; total_frames += T
        ep_boundaries.append((ep_id, start, start + T))
        for name, arr in features.items():
            all_features.setdefault(name, []).append(arr)
        all_actions.append(actions)

    for name in all_features:
        all_features[name] = np.concatenate(all_features[name], axis=0)
    all_actions_np = np.concatenate(all_actions, axis=0)
    ep_boundaries = np.array(ep_boundaries)

    logger.info(f'Total frames: {total_frames}, episodes: {len(ep_boundaries)}')

    # Build handle cache
    eef_pos_all = all_features['eef_pos']
    logger.info('Building handle cache (requires MuJoCo simulation)...')
    handle_pos = build_handle_cache_target(episodes, eef_pos_all, ep_boundaries)

    # Compute final 44-dim obs
    FEAT_NAMES = [
        'proprio', 'door_pos', 'door_quat', 'eef_pos', 'eef_quat',
        'door_to_eef_pos', 'door_to_eef_quat', 'gripper_to_door_dist',
    ]
    feats_t = {k: torch.from_numpy(v) for k, v in all_features.items()}
    handle_pos_t = torch.from_numpy(handle_pos)
    handle_to_eef_t = feats_t['eef_pos'] - handle_pos_t

    obs_full = torch.cat(
        [feats_t[n] for n in FEAT_NAMES] + [handle_pos_t, handle_to_eef_t],
        dim=-1,
    ).float()
    logger.info(f'obs_full shape: {obs_full.shape}  (expected 44-dim)')

    save_dict = {
        'features': feats_t,
        'actions': torch.from_numpy(all_actions_np),
        'ep_boundaries': ep_boundaries,
        'obs_full': obs_full,  # precomputed 44-dim obs (with handle features)
    }
    torch.save(save_dict, OUTPUT_PATH)
    logger.info(f'Saved target preprocessed data to {OUTPUT_PATH}')
    logger.info(f'Total time: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    preprocess_target()
