"""Preprocess dataset to compute ALL derived state dimensions.

Computes from existing 16-dim proprio + static door positions:
- Global EEF position/quaternion (from base + relative)
- Door-to-EEF relative position (dynamic, changes every step)
- Door-to-EEF relative quaternion (dynamic)

Uses multiprocessing for parallel episode processing.
"""
import numpy as np
import torch
import time
import os
import sys
import logging
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

from diffusion_policy.data import get_dataset_path, load_episodes


# ========== Quaternion math (numpy, batch-friendly) ==========

def quat_conjugate(q):
    """Conjugate of quaternion(s). q = [x, y, z, w] → [-x, -y, -z, w]"""
    q = np.asarray(q, dtype=np.float32)
    conj = q.copy()
    conj[..., :3] *= -1
    return conj


def quat_multiply(q1, q2):
    """Hamilton product q1 * q2. Convention: [x, y, z, w]"""
    q1, q2 = np.asarray(q1, np.float32), np.asarray(q2, np.float32)
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], axis=-1)


def quat_rotate_vector(q, v):
    """Rotate vector v by quaternion q. Convention: [x, y, z, w]"""
    q = np.asarray(q, np.float32)
    v = np.asarray(v, np.float32)
    # v as pure quaternion [vx, vy, vz, 0]
    v_quat = np.zeros(v.shape[:-1] + (4,), dtype=np.float32)
    v_quat[..., :3] = v
    # q * v * q_conj
    result = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
    return result[..., :3]


# ========== State dimension definitions ==========

# Original 16-dim proprio layout
PROPRIO_SLICES = {
    'base_pos':      (0, 3),    # robot0_base_pos
    'base_quat':     (3, 7),    # robot0_base_quat
    'base_to_eef_pos': (7, 10), # robot0_base_to_eef_pos
    'base_to_eef_quat': (10, 14), # robot0_base_to_eef_quat
    'gripper_qpos':  (14, 16),  # robot0_gripper_qpos
}

# Feature groups we can compute
FEATURE_GROUPS = {
    'proprio': 16,           # base 16-dim proprioception
    'door_pos': 3,           # door_obj_pos (static per ep)
    'door_quat': 4,          # door_obj_quat (static per ep)
    'eef_pos': 3,            # global EEF position (computed)
    'eef_quat': 4,           # global EEF quaternion (computed)
    'door_to_eef_pos': 3,    # relative position door→EEF (dynamic!)
    'door_to_eef_quat': 4,   # relative quaternion door→EEF (dynamic!)
    'gripper_to_door_dist': 1, # scalar distance (dynamic!)
}


def compute_derived_features(proprio_seq, door_pos, door_quat):
    """Compute all derived features for a sequence of proprio states.

    Args:
        proprio_seq: (T, 16) array of proprioception states
        door_pos: (3,) static door position for this episode
        door_quat: (4,) static door quaternion for this episode

    Returns:
        dict of feature_name → (T, dim) arrays
    """
    T = len(proprio_seq)

    base_pos = proprio_seq[:, 0:3]
    base_quat = proprio_seq[:, 3:7]
    base_to_eef_pos = proprio_seq[:, 7:10]
    base_to_eef_quat = proprio_seq[:, 10:14]

    # Global EEF position: base_pos + rotate(base_quat, base_to_eef_pos)
    eef_pos = base_pos + quat_rotate_vector(base_quat, base_to_eef_pos)

    # Global EEF quaternion: base_quat * base_to_eef_quat
    eef_quat = quat_multiply(base_quat, base_to_eef_quat)

    # Door-to-EEF relative position (vector from door to EEF in world frame)
    door_pos_tiled = np.tile(door_pos, (T, 1))
    door_to_eef_pos = eef_pos - door_pos_tiled

    # Door-to-EEF relative quaternion: conj(door_quat) * eef_quat
    door_quat_tiled = np.tile(door_quat, (T, 1))
    door_to_eef_quat = quat_multiply(quat_conjugate(door_quat_tiled), eef_quat)

    # Scalar distance
    gripper_to_door_dist = np.linalg.norm(door_to_eef_pos, axis=-1, keepdims=True)

    return {
        'proprio': proprio_seq,
        'door_pos': door_pos_tiled,
        'door_quat': door_quat_tiled,
        'eef_pos': eef_pos,
        'eef_quat': eef_quat,
        'door_to_eef_pos': door_to_eef_pos,
        'door_to_eef_quat': door_to_eef_quat,
        'gripper_to_door_dist': gripper_to_door_dist,
    }


def process_episode(args):
    """Process a single episode (for multiprocessing)."""
    ep_idx, states, actions, door_pos, door_quat = args
    proprio = states.astype(np.float32)
    features = compute_derived_features(proprio, door_pos, door_quat)
    return ep_idx, features, actions.astype(np.float32)


def preprocess_all(save_dir="/tmp/diffusion_policy_checkpoints"):
    """Preprocess all episodes and save comprehensive state data."""
    t0 = time.time()

    # Load existing data
    episodes = load_episodes(get_dataset_path())
    dp_data = np.load(os.path.join(save_dir, "door_positions.npz"))
    door_positions = {int(k): v.astype(np.float32) for k, v in dp_data.items()}

    # Load door quaternions (try multiple cached locations)
    dq_path = os.path.join(save_dir, "door_quats.npz")
    if not os.path.exists(dq_path):
        dq_path = os.path.join(save_dir, "door_quaternions.npz")
    if os.path.exists(dq_path):
        dq_data = np.load(dq_path)
        door_quats = {int(k): v.astype(np.float32) for k, v in dq_data.items()}
    else:
        logger.info("Extracting door quaternions from environment...")
        door_quats = extract_door_quaternions(episodes, save_dir)

    # Prepare args for parallel processing
    args_list = []
    for ep in episodes:
        eid = ep['episode_index']
        dp = door_positions[eid]
        dq = door_quats.get(eid, np.array([0, 0, 0, 1], dtype=np.float32))
        args_list.append((eid, ep['states'], ep['actions'], dp, dq))

    # Parallel processing
    n_workers = min(cpu_count(), 8)
    logger.info(f"Processing {len(args_list)} episodes with {n_workers} workers...")

    with Pool(n_workers) as pool:
        results = pool.map(process_episode, args_list)

    # Sort by episode index
    results.sort(key=lambda x: x[0])

    # Collect all features
    all_features = {}
    all_actions = []
    ep_boundaries = []  # (start_idx, end_idx) per episode

    total_frames = 0
    for ep_idx, features, actions in results:
        start = total_frames
        T = len(actions)
        total_frames += T
        ep_boundaries.append((ep_idx, start, start + T))

        for name, arr in features.items():
            if name not in all_features:
                all_features[name] = []
            all_features[name].append(arr)
        all_actions.append(actions)

    # Concatenate
    for name in all_features:
        all_features[name] = np.concatenate(all_features[name], axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    ep_boundaries = np.array(ep_boundaries)

    # Compute normalization stats per feature group
    stats = {}
    for name, arr in all_features.items():
        stats[f'{name}_mean'] = arr.mean(axis=0).astype(np.float32)
        stats[f'{name}_std'] = arr.std(axis=0).astype(np.float32)
        stats[f'{name}_std'] = np.maximum(stats[f'{name}_std'], 1e-6)  # avoid div by zero

    stats['action_mean'] = all_actions.mean(axis=0).astype(np.float32)
    stats['action_std'] = np.maximum(all_actions.std(axis=0).astype(np.float32), 1e-6)

    # Save everything
    out_path = os.path.join(save_dir, "preprocessed_all_states.pt")
    save_dict = {
        'features': {k: torch.from_numpy(v) for k, v in all_features.items()},
        'actions': torch.from_numpy(all_actions),
        'ep_boundaries': ep_boundaries,
        'stats': {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                  for k, v in stats.items()},
        'feature_dims': {k: v.shape[-1] for k, v in all_features.items()},
    }
    torch.save(save_dict, out_path)

    elapsed = time.time() - t0
    logger.info(f"Preprocessed {total_frames} frames from {len(results)} episodes in {elapsed:.1f}s")
    logger.info(f"Feature dimensions: {save_dict['feature_dims']}")
    logger.info(f"Saved to {out_path}")

    return save_dict


def extract_door_quaternions(episodes, save_dir):
    """Extract door_obj_quat for each episode from the environment."""
    import json
    import robocasa
    from robocasa.utils.env_utils import create_env as _create_env

    ds_path = get_dataset_path()
    meta_path = os.path.join(ds_path, "extras", "ep_meta.json")
    with open(meta_path) as f:
        ep_meta_all = json.load(f)

    env = _create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=0,
        split="pretrain",
        camera_widths=256,
        camera_heights=256,
    )

    door_quats = {}
    for ep in episodes:
        eid = ep['episode_index']
        meta_key = f"episode_{eid}"
        if meta_key in ep_meta_all:
            ep_meta = ep_meta_all[meta_key]
            env.set_ep_meta(ep_meta)
        obs = env.reset()
        dq = obs['door_obj_quat'].flatten().astype(np.float32)
        door_quats[eid] = dq

    env.close()

    # Save for future use
    np.savez(os.path.join(save_dir, "door_quaternions.npz"),
             **{str(k): v for k, v in door_quats.items()})
    logger.info(f"Saved door quaternions for {len(door_quats)} episodes")

    return door_quats


def build_handle_cache(lerobot_root, eef_pos_all, ep_boundaries,
                       cache_dir="/tmp/diffusion_policy_checkpoints/handle_cache"):
    """Replay all episodes in sim to extract per-timestep handle site positions.

    Args:
        lerobot_root: path to lerobot dataset root (contains extras/)
        eef_pos_all: (N, 3) array of global EEF positions (from preprocessed_all_states.pt)
        ep_boundaries: (num_eps, 3) array of (episode_id, start, end)
        cache_dir: directory to cache per-episode handle positions

    Returns:
        handle_pos_all: (N, 3) float32 array of handle world positions
    """
    import gzip
    import json
    from pathlib import Path
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa

    from diffusion_policy.evaluation import get_handle_site_names

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    lerobot_root = Path(lerobot_root)
    if isinstance(eef_pos_all, torch.Tensor):
        eef_pos_np = eef_pos_all.numpy()
    else:
        eef_pos_np = np.asarray(eef_pos_all, dtype=np.float32)

    N = len(eef_pos_np)
    handle_pos_all = np.zeros((N, 3), dtype=np.float32)

    ep_bounds = {int(eid): (int(start), int(end)) for eid, start, end in ep_boundaries}

    # Create a headless replay env (no cameras, no rendering)
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

    episode_dirs = sorted(lerobot_root.glob("extras/episode_*"))
    logger.info(f"Building handle cache for {len(episode_dirs)} episodes → {cache_dir}")

    try:
        for ep_dir in tqdm(episode_dirs, desc="Handle cache"):
            ep_id = int(ep_dir.name.split("_")[-1])
            if ep_id not in ep_bounds:
                continue

            start, end = ep_bounds[ep_id]
            T = end - start
            cache_path = cache_dir / f"episode_{ep_id:06d}.npy"

            if cache_path.exists():
                cached = np.load(cache_path)
                handle_pos_all[start:end] = cached[:T]
                continue

            # Load episode assets
            sim_states = np.load(ep_dir / "states.npz")["states"]
            with open(ep_dir / "ep_meta.json") as f:
                ep_meta = json.load(f)
            with gzip.open(ep_dir / "model.xml.gz", "rb") as f:
                model_xml = f.read().decode("utf-8")

            # Restore episode kitchen/fixture layout
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
                eef = eef_pos_np[start + idx]
                if len(sites) > 1:
                    active_site = min(
                        sites,
                        key=lambda s: np.linalg.norm(
                            env.sim.data.site_xpos[env.sim.model.site_name2id(s)] - eef
                        ),
                    )
                sid = env.sim.model.site_name2id(active_site)
                ep_handle_pos[idx] = env.sim.data.site_xpos[sid]

            np.save(cache_path, ep_handle_pos)
            handle_pos_all[start:end] = ep_handle_pos

    finally:
        env.close()

    logger.info(f"Handle cache complete. Mean handle dist from EEF: "
                f"{np.linalg.norm(handle_pos_all - eef_pos_np, axis=-1).mean():.3f}m")
    return handle_pos_all


if __name__ == "__main__":
    preprocess_all()
