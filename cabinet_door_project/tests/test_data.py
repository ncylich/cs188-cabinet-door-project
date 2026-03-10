import numpy as np
import torch
import pytest

from diffusion_policy.data import (
    DiffusionPolicyDataset,
    load_episodes,
    load_stats,
    Normalizer,
)
from diffusion_policy.config import DiffusionConfig


class TestEpisodeBoundaries:
    def test_samples_never_cross_episodes(self, dataset):
        for i in range(0, len(dataset), max(1, len(dataset) // 50)):
            ep_idx, start = dataset.samples[i]
            ep = dataset.episodes[ep_idx]
            ep_len = len(ep["states"])
            n_obs = dataset.n_obs_steps
            h = dataset.horizon
            assert start + n_obs <= ep_len
            assert start + n_obs - 1 + h <= ep_len


class TestNormalization:
    def test_roundtrip(self, dataset_path):
        stats = load_stats(dataset_path)
        norm = Normalizer(stats["state_mean"], stats["state_std"])
        x = torch.randn(10, 16)
        recovered = norm.denormalize(norm.normalize(x))
        torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)

    def test_roundtrip_action(self, dataset_path):
        stats = load_stats(dataset_path)
        norm = Normalizer(stats["action_mean"], stats["action_std"])
        x = torch.randn(10, 12)
        recovered = norm.denormalize(norm.normalize(x))
        torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)

    def test_statistics_approx_standard(self, dataset):
        all_obs = []
        all_actions = []
        indices = np.random.default_rng(42).choice(len(dataset), size=min(2000, len(dataset)), replace=False)
        for i in indices:
            obs, actions = dataset[int(i)]
            all_obs.append(obs)
            all_actions.append(actions)

        obs_tensor = torch.stack(all_obs).reshape(-1, 16)
        act_tensor = torch.stack(all_actions).reshape(-1, 12)

        obs_mean = obs_tensor.mean(dim=0)
        obs_std = obs_tensor.std(dim=0)
        act_mean = act_tensor.mean(dim=0)
        act_std = act_tensor.std(dim=0)

        # Dims with zero variance in raw data will have ~0 std after normalization too
        nonzero_obs = obs_std > 0.01
        nonzero_act = act_std > 0.01

        assert obs_mean[nonzero_obs].abs().max() < 0.5
        assert (obs_std[nonzero_obs] - 1.0).abs().max() < 0.5
        assert act_mean[nonzero_act].abs().max() < 0.5
        assert (act_std[nonzero_act] - 1.0).abs().max() < 0.5


class TestSampleShapes:
    def test_shapes(self, dataset):
        obs, actions = dataset[0]
        assert obs.shape == (dataset.n_obs_steps, 16)
        assert actions.shape == (dataset.horizon, 12)

    def test_dtypes(self, dataset):
        obs, actions = dataset[0]
        assert obs.dtype == torch.float32
        assert actions.dtype == torch.float32


class TestDatasetLength:
    def test_length_matches_formula(self, dataset):
        expected = 0
        for ep in dataset.episodes:
            ep_len = len(ep["states"])
            expected += max(0, ep_len - dataset.horizon - dataset.n_obs_steps + 1)
        assert len(dataset) == expected


class TestActionRanges:
    def test_raw_actions_in_range(self, dataset):
        for ep in dataset.episodes:
            assert ep["actions"].min() >= -1.0 - 1e-6
            assert ep["actions"].max() <= 1.0 + 1e-6


class TestNoNaN:
    def test_no_nan_or_inf(self, dataset):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(dataset), size=min(100, len(dataset)), replace=False)
        for i in indices:
            obs, actions = dataset[int(i)]
            assert torch.isfinite(obs).all(), f"Non-finite in obs at index {i}"
            assert torch.isfinite(actions).all(), f"Non-finite in actions at index {i}"
