import json
import os
import gzip
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

PROPRIO_STATE_KEYS = [
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "robot0_gripper_qpos",
]

OBJECT_AWARE_STATE_KEYS = [
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "robot0_gripper_qpos",
    "door_obj_pos",
    "door_obj_quat",
    "door_obj_to_robot0_eef_pos",
    "door_obj_to_robot0_eef_quat",
]

HANDLE_AWARE_STATE_KEYS = OBJECT_AWARE_STATE_KEYS + [
    "door_handle_pos",
    "door_handle_quat",
    "door_handle_to_robot0_eef_pos",
    "door_handle_to_robot0_eef_quat",
]

STATE_KEY_DIMS = {
    "robot0_base_pos": 3,
    "robot0_base_quat": 4,
    "robot0_base_to_eef_pos": 3,
    "robot0_base_to_eef_quat": 4,
    "robot0_gripper_qpos": 2,
    "door_obj_pos": 3,
    "door_obj_quat": 4,
    "door_obj_to_robot0_eef_pos": 3,
    "door_obj_to_robot0_eef_quat": 4,
    "door_handle_pos": 3,
    "door_handle_quat": 4,
    "door_handle_to_robot0_eef_pos": 3,
    "door_handle_to_robot0_eef_quat": 4,
}

ACTION_KEY_ORDERING_ENV = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def resolve_lerobot_root(dataset_path):
    dataset_path = Path(dataset_path)
    if (dataset_path / "data").exists() and (dataset_path / "meta").exists():
        return dataset_path
    lerobot_root = dataset_path / "lerobot"
    if (lerobot_root / "data").exists() and (lerobot_root / "meta").exists():
        return lerobot_root
    raise FileNotFoundError(
        f"Could not find a LeRobot dataset under {dataset_path}. "
        "Run 04_download_dataset.py first."
    )


def get_dataset_path(task="OpenCabinet", source="human"):
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path

    path = get_ds_path(task, source=source)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset for task={task}, source={source} not found. "
            "Run 04_download_dataset.py first."
        )
    return path


def load_modality_dict(lerobot_root):
    meta_modality = Path(lerobot_root) / "meta" / "modality.json"
    if meta_modality.exists():
        with open(meta_modality, "r") as f:
            return json.load(f)

    fallback = (
        Path(__file__).resolve().parent.parent
        / "robocasa"
        / "robocasa"
        / "models"
        / "assets"
        / "groot_dataset_assets"
        / "PandaOmron_modality.json"
    )
    with open(fallback, "r") as f:
        return json.load(f)


def find_parquet_files(lerobot_root):
    parquet_files = sorted(Path(lerobot_root).glob("data/chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {lerobot_root / 'data'}")
    return parquet_files


def infer_dataset_split(lerobot_root):
    lerobot_root = Path(lerobot_root)
    if "pretrain" in lerobot_root.parts:
        return "pretrain"
    if "target" in lerobot_root.parts:
        return "target"
    return "pretrain"


def reorder_lerobot_actions(action_lerobot, modality_dict):
    action_info = modality_dict["action"]
    reordered = np.zeros_like(action_lerobot, dtype=np.float32)
    for key, (env_start, env_end) in ACTION_KEY_ORDERING_ENV.items():
        le_start = action_info[key]["start"]
        le_end = action_info[key]["end"]
        reordered[:, env_start:env_end] = action_lerobot[:, le_start:le_end]
    return reordered


def load_episode_arrays(parquet_path, modality_dict):
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["observation.state", "action"])
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    actions = reorder_lerobot_actions(actions, modality_dict)
    return states, actions


def find_precomputed_parquet_files(dataset_path):
    dataset_path = Path(dataset_path)
    if dataset_path.is_file():
        return [dataset_path]

    parquet_files = sorted(dataset_path.rglob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_path}")
    return parquet_files


def load_precomputed_episode_arrays(parquet_path):
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["observation.state", "action"])
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    return states, actions


def get_state_keys(state_mode):
    if state_mode == "proprio":
        return list(PROPRIO_STATE_KEYS)
    if state_mode == "door_relative":
        return list(OBJECT_AWARE_STATE_KEYS)
    if state_mode == "handle_relative":
        return list(HANDLE_AWARE_STATE_KEYS)
    raise ValueError(f"Unsupported state_mode: {state_mode}")


def build_feature_cache_dir(lerobot_root, state_keys, feature_cache_dir=None):
    if feature_cache_dir is not None:
        cache_dir = Path(feature_cache_dir)
    else:
        cache_id = hashlib.sha1(
            (
                f"{Path(lerobot_root).resolve()}|"
                f"{','.join(state_keys)}|feature_cache_v1"
            ).encode("utf-8")
        ).hexdigest()[:12]
        cache_dir = Path("/tmp/cabinet_policy_feature_cache") / cache_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def create_replay_env(split, seed=0):
    import robocasa  # noqa: F401
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    obj_instance_split = None
    layout_ids = None
    style_ids = None
    layout_and_style_ids = None

    if split == "target":
        obj_instance_split = "target"
        layout_and_style_ids = list(zip(range(1, 11), range(1, 11)))
    elif split == "pretrain":
        obj_instance_split = "pretrain"
        layout_ids = -2
        style_ids = -2

    return robosuite.make(
        env_name="OpenCabinet",
        robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
    )


def get_raw_observation(env):
    if hasattr(env, "_get_observations"):
        return env._get_observations(force_update=True)
    return env._get_observation()


def state_keys_require_handle_features(state_keys):
    return any(key.startswith("door_handle_") for key in state_keys)


def get_fixture_handle_site_names(env):
    fxtr = getattr(env, "fxtr", None)
    if fxtr is None:
        return []

    site_names = []
    for attr_name in ("handle_name", "left_handle_name", "right_handle_name"):
        if not hasattr(fxtr, attr_name):
            continue
        handle_name = getattr(fxtr, attr_name)
        if not isinstance(handle_name, str) or not handle_name:
            continue

        candidates = []
        if handle_name.endswith("_handle"):
            candidates.append(f"{handle_name[:-len('_handle')]}_default_site")
        if handle_name.endswith("_main"):
            candidates.append(f"{handle_name[:-len('_main')]}_default_site")
        candidates.append(f"{handle_name}_default_site")

        for candidate in candidates:
            try:
                env.sim.model.site_name2id(candidate)
            except Exception:
                continue
            site_names.append(candidate)
            break

    deduped = []
    for site_name in site_names:
        if site_name not in deduped:
            deduped.append(site_name)
    return deduped


def augment_lowdim_observation(obs, env, state_keys, active_handle_site=None):
    if not state_keys_require_handle_features(state_keys):
        return obs, active_handle_site

    enriched_obs = dict(obs)
    enriched_obs["door_handle_pos"] = np.zeros(3, dtype=np.float32)
    enriched_obs["door_handle_quat"] = np.zeros(4, dtype=np.float32)
    enriched_obs["door_handle_to_robot0_eef_pos"] = np.zeros(3, dtype=np.float32)
    enriched_obs["door_handle_to_robot0_eef_quat"] = np.zeros(4, dtype=np.float32)

    handle_sites = get_fixture_handle_site_names(env)
    if not handle_sites:
        return enriched_obs, None

    if active_handle_site not in handle_sites:
        eef_pos = np.asarray(
            enriched_obs.get("robot0_eef_pos", np.zeros(3, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if eef_pos.shape[0] >= 3:
            active_handle_site = min(
                handle_sites,
                key=lambda site_name: np.linalg.norm(
                    env.sim.data.site_xpos[env.sim.model.site_name2id(site_name)] - eef_pos[:3]
                ),
            )
        else:
            active_handle_site = handle_sites[0]

    handle_site_id = env.sim.model.site_name2id(active_handle_site)
    handle_pos = np.asarray(env.sim.data.site_xpos[handle_site_id], dtype=np.float32)

    from robosuite.utils import transform_utils as T

    handle_rot = env.sim.data.site_xmat[handle_site_id].reshape(3, 3)
    handle_quat = T.mat2quat(handle_rot).astype(np.float32)

    enriched_obs["door_handle_pos"] = handle_pos
    enriched_obs["door_handle_quat"] = handle_quat

    if "robot0_eef_pos" not in enriched_obs or "robot0_eef_quat" not in enriched_obs:
        return enriched_obs, active_handle_site

    eef_pos = np.asarray(enriched_obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(enriched_obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
    if eef_pos.shape[0] < 3 or eef_quat.shape[0] < 4:
        return enriched_obs, active_handle_site

    handle_pose = T.pose2mat((handle_pos, handle_quat))
    world_pose_in_gripper = T.pose_inv(T.pose2mat((eef_pos[:3], eef_quat[:4])))
    rel_pose = T.pose_in_A_to_pose_in_B(handle_pose, world_pose_in_gripper)
    rel_pos, rel_quat = T.mat2pose(rel_pose)
    enriched_obs["door_handle_to_robot0_eef_pos"] = np.asarray(
        rel_pos, dtype=np.float32
    )
    enriched_obs["door_handle_to_robot0_eef_quat"] = np.asarray(
        rel_quat, dtype=np.float32
    )
    return enriched_obs, active_handle_site


def restore_env_from_episode(env, model_xml, ep_meta):
    if hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)
    elif hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)

    env.reset()
    xml = env.edit_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()


def restore_observation_from_flattened_state(env, state):
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()
    return get_raw_observation(env)


def get_episode_assets(lerobot_root, episode_id):
    episode_dir = Path(lerobot_root) / "extras" / f"episode_{episode_id:06d}"
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode extras not found: {episode_dir}")

    states = np.load(episode_dir / "states.npz")["states"]
    with open(episode_dir / "ep_meta.json", "r") as f:
        ep_meta = json.load(f)
    with gzip.open(episode_dir / "model.xml.gz", "rb") as f:
        model_xml = f.read().decode("utf-8")
    return states, ep_meta, model_xml


def extract_replay_episode_states(
    lerobot_root,
    episode_id,
    state_keys,
    replay_env,
    cache_dir,
):
    cache_path = cache_dir / f"episode_{episode_id:06d}.npz"
    if cache_path.exists():
        return np.load(cache_path)["states"].astype(np.float32)

    raw_states, ep_meta, model_xml = get_episode_assets(lerobot_root, episode_id)
    features = np.zeros((len(raw_states), sum(STATE_KEY_DIMS[k] for k in state_keys)), dtype=np.float32)
    restore_env_from_episode(replay_env, model_xml, ep_meta)
    active_handle_site = None

    for idx, state in enumerate(raw_states):
        obs = restore_observation_from_flattened_state(replay_env, state)
        obs, active_handle_site = augment_lowdim_observation(
            obs=obs,
            env=replay_env,
            state_keys=state_keys,
            active_handle_site=active_handle_site,
        )
        features[idx] = extract_lowdim_state(obs, state_keys=state_keys)

    np.savez_compressed(cache_path, states=features)
    return features


def load_lowdim_episodes(
    dataset_path,
    max_episodes=None,
    state_mode="proprio",
    feature_cache_dir=None,
):
    lerobot_root = resolve_lerobot_root(dataset_path)
    modality_dict = load_modality_dict(lerobot_root)
    parquet_files = find_parquet_files(lerobot_root)
    if max_episodes is not None:
        parquet_files = parquet_files[:max_episodes]

    state_keys = get_state_keys(state_mode)
    replay_env = None
    cache_dir = None
    if state_mode != "proprio":
        replay_env = create_replay_env(split=infer_dataset_split(lerobot_root))
        cache_dir = build_feature_cache_dir(
            lerobot_root=lerobot_root,
            state_keys=state_keys,
            feature_cache_dir=feature_cache_dir,
        )

    episodes = []
    iterator = parquet_files
    if state_mode != "proprio":
        from tqdm import tqdm

        iterator = tqdm(parquet_files, desc=f"Loading {state_mode} states")
    try:
        for parquet_path in iterator:
            states, actions = load_episode_arrays(parquet_path, modality_dict)
            episode_id = int(parquet_path.stem.split("_")[-1])

            if state_mode != "proprio":
                states = extract_replay_episode_states(
                    lerobot_root=lerobot_root,
                    episode_id=episode_id,
                    state_keys=state_keys,
                    replay_env=replay_env,
                    cache_dir=cache_dir,
                )

            if len(states) != len(actions):
                min_len = min(len(states), len(actions))
                states = states[:min_len]
                actions = actions[:min_len]

            episodes.append(
                {
                    "episode_id": episode_id,
                    "states": states.astype(np.float32),
                    "actions": actions.astype(np.float32),
                    "length": len(actions),
                }
            )
    finally:
        if replay_env is not None:
            replay_env.close()

    return episodes, lerobot_root, state_keys, cache_dir


def load_precomputed_lowdim_episodes(
    dataset_path,
    expected_state_dim=None,
    expected_action_dim=None,
    episode_id_offset=0,
):
    parquet_files = find_precomputed_parquet_files(dataset_path)
    episodes = []

    for idx, parquet_path in enumerate(parquet_files):
        states, actions = load_precomputed_episode_arrays(parquet_path)

        if states.ndim != 2:
            raise ValueError(f"Expected 2D states in {parquet_path}, got {states.shape}")
        if actions.ndim != 2:
            raise ValueError(f"Expected 2D actions in {parquet_path}, got {actions.shape}")

        if expected_state_dim is not None and states.shape[1] != expected_state_dim:
            raise ValueError(
                f"State dim mismatch for {parquet_path}: got {states.shape[1]}, "
                f"expected {expected_state_dim}. "
                "Collect DAgger data with a checkpoint that matches your current state_mode."
            )
        if expected_action_dim is not None and actions.shape[1] != expected_action_dim:
            raise ValueError(
                f"Action dim mismatch for {parquet_path}: got {actions.shape[1]}, "
                f"expected {expected_action_dim}."
            )

        if len(states) != len(actions):
            min_len = min(len(states), len(actions))
            states = states[:min_len]
            actions = actions[:min_len]

        episodes.append(
            {
                "episode_id": episode_id_offset + idx,
                "states": states.astype(np.float32),
                "actions": actions.astype(np.float32),
                "length": len(actions),
            }
        )

    return episodes


def split_episodes(episodes, val_fraction=0.1, seed=0):
    if len(episodes) < 2 or val_fraction <= 0.0:
        return episodes, []

    rng = np.random.default_rng(seed)
    indices = np.arange(len(episodes))
    rng.shuffle(indices)

    num_val = max(1, int(round(len(indices) * val_fraction)))
    num_val = min(num_val, len(indices) - 1)

    val_ids = set(indices[:num_val].tolist())
    train_episodes = [episodes[i] for i in range(len(episodes)) if i not in val_ids]
    val_episodes = [episodes[i] for i in range(len(episodes)) if i in val_ids]
    return train_episodes, val_episodes


def compute_normalization_stats(episodes):
    states = np.concatenate([ep["states"] for ep in episodes], axis=0)
    actions = np.concatenate([ep["actions"] for ep in episodes], axis=0)

    state_mean = states.mean(axis=0).astype(np.float32)
    state_std = np.clip(states.std(axis=0), 1e-3, None).astype(np.float32)
    action_mean = actions.mean(axis=0).astype(np.float32)
    action_std = np.clip(actions.std(axis=0), 1e-3, None).astype(np.float32)

    static_action_mask = (actions.std(axis=0) < 1e-5).astype(np.bool_)
    static_action_values = action_mean.copy()

    return {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "static_action_mask": static_action_mask,
        "static_action_values": static_action_values,
    }


def normalize_states(states, state_mean, state_std):
    return (states - state_mean) / state_std


def normalize_actions(actions, action_mean, action_std):
    return (actions - action_mean) / action_std


def denormalize_actions(actions, action_mean, action_std):
    return actions * action_std + action_mean


def extract_lowdim_state(obs, state_keys=None):
    if state_keys is None:
        state_keys = PROPRIO_STATE_KEYS
    state_parts = []
    for key in state_keys:
        value = obs.get(key, None)
        if value is None:
            state_parts.append(np.zeros(STATE_KEY_DIMS[key], dtype=np.float32))
            continue
        state_parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(state_parts, axis=0).astype(np.float32)


def build_history_batch(history, seq_len, state_mean, state_std, device):
    state_dim = len(state_mean)
    seq = np.zeros((1, seq_len, state_dim), dtype=np.float32)
    padding_mask = np.ones((1, seq_len), dtype=bool)
    history = list(history)[-seq_len:]
    if history:
        hist_arr = np.stack(history).astype(np.float32)
        seq[0, -len(history) :] = hist_arr
        padding_mask[0, -len(history) :] = False
    seq = normalize_states(seq, state_mean.reshape(1, 1, -1), state_std.reshape(1, 1, -1))
    seq[padding_mask] = 0.0
    states = torch.from_numpy(seq).to(device)
    mask = torch.from_numpy(padding_mask).to(device)
    return states, mask


class EpisodeSequenceDataset(Dataset):
    def __init__(
        self,
        episodes,
        seq_len,
        state_mean,
        state_std,
        action_mean,
        action_std,
    ):
        self.episodes = episodes
        self.seq_len = seq_len
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.index = []
        for ep_idx, episode in enumerate(episodes):
            for step_idx in range(episode["length"]):
                self.index.append((ep_idx, step_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, step_idx = self.index[idx]
        episode = self.episodes[ep_idx]
        states = episode["states"]
        actions = episode["actions"]

        start_idx = max(0, step_idx - self.seq_len + 1)
        state_hist = states[start_idx : step_idx + 1]

        seq = np.zeros((self.seq_len, states.shape[-1]), dtype=np.float32)
        padding_mask = np.ones(self.seq_len, dtype=bool)
        seq[-len(state_hist) :] = state_hist
        padding_mask[-len(state_hist) :] = False

        seq = normalize_states(seq, self.state_mean, self.state_std)
        seq[padding_mask] = 0.0
        target_action = normalize_actions(
            actions[step_idx], self.action_mean, self.action_std
        )

        return (
            torch.from_numpy(seq),
            torch.from_numpy(padding_mask),
            torch.from_numpy(target_action.astype(np.float32)),
        )


class TemporalBCTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        seq_len,
        d_model=256,
        n_heads=8,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(state_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, action_dim),
        )

        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, states, padding_mask):
        x = self.input_proj(states) + self.pos_embed[:, : states.shape[1]]
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        lengths = (~padding_mask).sum(dim=1).clamp(min=1) - 1
        pooled = x[torch.arange(x.shape[0], device=x.device), lengths]
        pooled = self.final_norm(pooled)
        return self.action_head(pooled)


class SimpleMLPPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


def build_policy_from_checkpoint(checkpoint, device):
    policy_type = checkpoint.get("policy_type", "simple_mlp")
    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]

    if policy_type == "temporal_bc_transformer":
        model = TemporalBCTransformer(**checkpoint["model_kwargs"]).to(device)
    elif policy_type == "simple_mlp":
        model = SimpleMLPPolicy(state_dim, action_dim).to(device)
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, policy_type


def load_checkpoint_policy(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, policy_type = build_policy_from_checkpoint(checkpoint, device)
    return model, checkpoint, policy_type


def unwrap_reset(reset_result):
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def unwrap_step(step_result):
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
        return obs, reward, done, info
    return step_result


def predict_policy_action(model, checkpoint, obs, history, device):
    policy_type = checkpoint.get("policy_type", "simple_mlp")
    state_keys = checkpoint.get("state_keys", PROPRIO_STATE_KEYS)

    if policy_type == "temporal_bc_transformer":
        current_state = extract_lowdim_state(obs, state_keys=state_keys)
        history.append(current_state)
        states, padding_mask = build_history_batch(
            history=history,
            seq_len=checkpoint["seq_len"],
            state_mean=np.asarray(checkpoint["state_mean"], dtype=np.float32),
            state_std=np.asarray(checkpoint["state_std"], dtype=np.float32),
            device=device,
        )
        with torch.no_grad():
            pred_action = model(states, padding_mask).cpu().numpy()[0]
        action = denormalize_actions(
            pred_action,
            np.asarray(checkpoint["action_mean"], dtype=np.float32),
            np.asarray(checkpoint["action_std"], dtype=np.float32),
        )
        static_mask = np.asarray(
            checkpoint.get("static_action_mask", np.zeros_like(action, dtype=bool)),
            dtype=bool,
        )
        static_values = np.asarray(
            checkpoint.get("static_action_values", np.zeros_like(action)),
            dtype=np.float32,
        )
        action[static_mask] = static_values[static_mask]
        return np.clip(action, -1.0, 1.0)

    state_dim = checkpoint["state_dim"]
    state = extract_lowdim_state(obs, state_keys=state_keys)
    if len(state) < state_dim:
        state = np.pad(state, (0, state_dim - len(state)))
    elif len(state) > state_dim:
        state = state[:state_dim]

    with torch.no_grad():
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        action = model(state_tensor).cpu().numpy().squeeze(0)
    return np.clip(action, -1.0, 1.0)
