import os
import sys
import tempfile

import numpy as np
import torch
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from diffusion_policy.evaluation import (
    dataset_action_to_env_action,
    env_action_to_dataset_action,
    extract_state,
    create_env,
    STATE_KEYS_ORDERED,
    STATE_DIMS,
)
from diffusion_policy.config import DiffusionConfig
from diffusion_policy.data import Normalizer, load_stats
from diffusion_policy.inference import DiffusionPolicyInference
from diffusion_policy.scheduler import DDPMScheduler
from diffusion_policy.training import build_model


class TestActionReorderRoundtrip:
    def test_unique_values(self):
        # ds: [base(3), reserve(1), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]
        dataset_action = np.array([0.1, 0.2, 0.3, 0.0, -1.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, -0.5])
        env_action = dataset_action_to_env_action(dataset_action)
        # env: [right(6), right_gripper(1), base(3), torso(1), base_mode(1)]
        np.testing.assert_array_almost_equal(env_action[0:3], [0.5, 0.6, 0.7])   # eef_pos
        np.testing.assert_array_almost_equal(env_action[3:6], [0.8, 0.9, 1.0])   # eef_rot
        assert env_action[6] == -1.0  # gripper: -0.5 < 0.5 → -1.0
        np.testing.assert_array_almost_equal(env_action[7:10], [0.1, 0.2, 0.3])  # base
        assert env_action[10] == 0.0  # torso
        assert env_action[11] == -1.0  # base_mode: -1.0 < 0.5 → -1.0


class TestActionReorderGripperBinary:
    def test_close(self):
        dataset_action = np.zeros(12)
        dataset_action[11] = 1.0  # close
        env_action = dataset_action_to_env_action(dataset_action)
        assert env_action[6] == 1.0

    def test_open(self):
        dataset_action = np.zeros(12)
        dataset_action[11] = -1.0  # open
        env_action = dataset_action_to_env_action(dataset_action)
        assert env_action[6] == -1.0


class TestControlModeAsBaseMode:
    def test_arm_mode(self):
        dataset_action = np.zeros(12)
        dataset_action[4] = -1.0  # arm mode
        env_action = dataset_action_to_env_action(dataset_action)
        assert env_action[11] == -1.0  # base_mode flag
        assert env_action.shape == (12,)


class TestEnvStepAcceptsActions:
    def test_action_accepted(self):
        env = create_env(split="pretrain", seed=0)
        try:
            obs = env.reset()
            dataset_action = np.zeros(12, dtype=np.float64)
            dataset_action[4] = -1.0  # arm mode
            env_action = dataset_action_to_env_action(dataset_action)
            obs, reward, done, info = env.step(env_action)
            assert obs is not None
        finally:
            env.close()


class TestStateExtraction:
    def test_matches_dataset_format(self):
        env = create_env(split="pretrain", seed=0)
        try:
            obs = env.reset()
            state = extract_state(obs)
            assert state.shape == (16,)
            assert state.dtype == np.float32

            idx = 0
            for key, dim in zip(STATE_KEYS_ORDERED, STATE_DIMS):
                expected = obs[key].flatten().astype(np.float32)
                np.testing.assert_array_almost_equal(state[idx:idx + dim], expected)
                idx += dim
        finally:
            env.close()


class TestFullEpisodeNoCrash:
    def test_100_steps(self, dataset_path, device):
        config = DiffusionConfig(
            dataset_path=dataset_path,
            backbone="mlp",
            hidden_dim=64,
            n_layers=2,
            num_diffusion_steps=10,
            num_inference_steps=10,
            n_action_steps=8,
        )
        model = build_model(config).to(device)
        scheduler = DDPMScheduler(num_train_steps=10, beta_schedule="cosine")
        stats = load_stats(dataset_path)
        state_norm = Normalizer(stats["state_mean"], stats["state_std"])
        action_norm = Normalizer(stats["action_mean"], stats["action_std"])
        pipeline = DiffusionPolicyInference(config, model, scheduler, state_norm, action_norm, device)

        from diffusion_policy.evaluation import run_rollouts
        results = run_rollouts(pipeline, num_rollouts=1, max_steps=100, split="pretrain", seed=0)
        assert len(results["successes"]) == 1
        assert len(results["episode_lengths"]) == 1
        assert np.isfinite(results["rewards"][0])


class TestMetricsComputed:
    def test_metrics_are_finite(self, dataset_path, device):
        config = DiffusionConfig(
            dataset_path=dataset_path,
            backbone="mlp",
            hidden_dim=64,
            n_layers=2,
            num_diffusion_steps=10,
            num_inference_steps=10,
            n_action_steps=8,
        )
        model = build_model(config).to(device)
        scheduler = DDPMScheduler(num_train_steps=10, beta_schedule="cosine")
        stats = load_stats(dataset_path)
        state_norm = Normalizer(stats["state_mean"], stats["state_std"])
        action_norm = Normalizer(stats["action_mean"], stats["action_std"])
        pipeline = DiffusionPolicyInference(config, model, scheduler, state_norm, action_norm, device)

        from diffusion_policy.evaluation import run_rollouts
        results = run_rollouts(pipeline, num_rollouts=3, max_steps=50, split="pretrain", seed=0)
        success_rate = sum(results["successes"]) / 3
        avg_length = np.mean(results["episode_lengths"])
        avg_reward = np.mean(results["rewards"])
        assert np.isfinite(success_rate)
        assert np.isfinite(avg_length)
        assert np.isfinite(avg_reward)
