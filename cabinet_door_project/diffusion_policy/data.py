import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from diffusion_policy.config import DiffusionConfig

logger = logging.getLogger(__name__)

STD_CLAMP_MIN = 1e-8


def get_dataset_path() -> str:
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path
    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        raise FileNotFoundError("Dataset not found. Run 04_download_dataset.py first.")
    return path


def load_stats(dataset_path: str) -> dict:
    stats_path = os.path.join(dataset_path, "meta", "stats.json")
    with open(stats_path) as f:
        raw = json.load(f)
    return {
        "state_mean": np.array(raw["observation.state"]["mean"], dtype=np.float32),
        "state_std": np.array(raw["observation.state"]["std"], dtype=np.float32),
        "action_mean": np.array(raw["action"]["mean"], dtype=np.float32),
        "action_std": np.array(raw["action"]["std"], dtype=np.float32),
    }


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(np.maximum(std, STD_CLAMP_MIN)).float()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


def load_episodes(dataset_path: str) -> list[dict]:
    chunk_dir = os.path.join(dataset_path, "data", "chunk-000")
    parquet_files = sorted(Path(chunk_dir).glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {chunk_dir}")

    episodes = []
    for pf in parquet_files:
        df = pq.read_table(str(pf)).to_pandas()
        states = np.stack(df["observation.state"].values).astype(np.float32)
        actions = np.stack(df["action"].values).astype(np.float32)
        episode_idx = int(df["episode_index"].iloc[0])
        frame_indices = df["frame_index"].values.astype(np.int64)
        episodes.append({
            "states": states,
            "actions": actions,
            "episode_index": episode_idx,
            "frame_indices": frame_indices,
        })

    logger.info("Loaded %d episodes, %d total frames",
                len(episodes), sum(len(e["states"]) for e in episodes))
    return episodes


VIDEO_CAMERA_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]


def load_video_frames(dataset_path: str, episode_idx: int, camera_key: str) -> np.ndarray:
    import imageio.v3 as iio
    video_path = os.path.join(
        dataset_path, "videos", "chunk-000", camera_key,
        f"episode_{episode_idx:06d}.mp4",
    )
    return iio.imread(video_path, plugin="pyav")


def precompute_image_features(
    dataset_path: str,
    encoder: "torch.nn.Module",
    device: "torch.device",
    batch_size: int = 64,
) -> dict[int, torch.Tensor]:
    import imageio.v3 as iio
    from torchvision.transforms import Resize

    resize = Resize(224, antialias=True)
    encoder = encoder.to(device).eval()
    episodes_dir = os.path.join(dataset_path, "data", "chunk-000")
    parquet_files = sorted(Path(episodes_dir).glob("episode_*.parquet"))

    features_by_episode: dict[int, torch.Tensor] = {}

    for pf in parquet_files:
        df = pq.read_table(str(pf)).to_pandas()
        ep_idx = int(df["episode_index"].iloc[0])
        ep_len = len(df)

        camera_frames = []
        for cam_key in VIDEO_CAMERA_KEYS:
            frames = load_video_frames(dataset_path, ep_idx, cam_key)
            camera_frames.append(frames[:ep_len])

        all_features = []
        for start in range(0, ep_len, batch_size):
            end = min(start + batch_size, ep_len)
            cam_features = []
            for cam_idx in range(len(VIDEO_CAMERA_KEYS)):
                batch_np = camera_frames[cam_idx][start:end]
                batch_t = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(device)
                batch_t = batch_t.float() / 255.0
                batch_t = resize(batch_t)
                with torch.no_grad():
                    feat = encoder.encoders[cam_idx](batch_t)
                cam_features.append(feat)
            combined = torch.cat(cam_features, dim=-1)
            all_features.append(combined.cpu())

        features_by_episode[ep_idx] = torch.cat(all_features, dim=0)
        if (ep_idx + 1) % 20 == 0:
            logger.info("Precomputed features for %d/%d episodes", ep_idx + 1, len(parquet_files))

    logger.info("Precomputed image features for all %d episodes", len(parquet_files))
    return features_by_episode


class DiffusionPolicyDataset(Dataset):
    def __init__(self, config: DiffusionConfig, dataset_path: Optional[str] = None):
        dataset_path = dataset_path or config.dataset_path
        self.config = config
        self.horizon = config.horizon
        self.n_obs_steps = config.n_obs_steps

        stats = load_stats(dataset_path)
        self.state_normalizer = Normalizer(stats["state_mean"], stats["state_std"])
        self.action_normalizer = Normalizer(stats["action_mean"], stats["action_std"])

        self.episodes = load_episodes(dataset_path)

        self.samples: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            ep_len = len(ep["states"])
            n_valid = max(0, ep_len - self.horizon - self.n_obs_steps + 1)
            for i in range(n_valid):
                self.samples.append((ep_idx, i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ep_idx, start = self.samples[idx]
        ep = self.episodes[ep_idx]

        obs = torch.from_numpy(
            ep["states"][start:start + self.n_obs_steps].copy()
        )
        actions = torch.from_numpy(
            ep["actions"][start + self.n_obs_steps - 1:
                          start + self.n_obs_steps - 1 + self.horizon].copy()
        )

        obs = self.state_normalizer.normalize(obs)
        actions = self.action_normalizer.normalize(actions)

        return obs, actions


class VisuomotorDataset(Dataset):
    def __init__(
        self,
        config: DiffusionConfig,
        image_features: dict[int, torch.Tensor],
        dataset_path: Optional[str] = None,
    ):
        dataset_path = dataset_path or config.dataset_path
        self.config = config
        self.horizon = config.horizon
        self.n_obs_steps = config.n_obs_steps

        stats = load_stats(dataset_path)
        self.state_normalizer = Normalizer(stats["state_mean"], stats["state_std"])
        self.action_normalizer = Normalizer(stats["action_mean"], stats["action_std"])

        self.episodes = load_episodes(dataset_path)
        self.image_features = image_features

        self.samples: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            ep_len = len(ep["states"])
            n_valid = max(0, ep_len - self.horizon - self.n_obs_steps + 1)
            for i in range(n_valid):
                self.samples.append((ep_idx, i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_idx, start = self.samples[idx]
        ep = self.episodes[ep_idx]
        ep_episode_index = ep["episode_index"]

        obs = torch.from_numpy(ep["states"][start:start + self.n_obs_steps].copy())
        actions = torch.from_numpy(
            ep["actions"][start + self.n_obs_steps - 1:
                          start + self.n_obs_steps - 1 + self.horizon].copy()
        )

        img_feats = self.image_features[ep_episode_index]
        obs_img = img_feats[start:start + self.n_obs_steps]

        obs = self.state_normalizer.normalize(obs)
        actions = self.action_normalizer.normalize(actions)

        return obs, obs_img, actions
