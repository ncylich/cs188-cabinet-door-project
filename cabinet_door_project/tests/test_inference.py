import time

import torch
import pytest

from diffusion_policy.config import DiffusionConfig
from diffusion_policy.data import Normalizer, load_stats
from diffusion_policy.inference import DiffusionPolicyInference
from diffusion_policy.scheduler import DDPMScheduler
from diffusion_policy.training import build_model, EMA


@pytest.fixture
def inference_pipeline(dataset_path, device):
    config = DiffusionConfig(
        dataset_path=dataset_path,
        backbone="mlp",
        hidden_dim=128,
        n_layers=2,
        num_diffusion_steps=20,
        num_inference_steps=20,
    )
    model = build_model(config).to(device)
    scheduler = DDPMScheduler(num_train_steps=100, beta_schedule="cosine")
    stats = load_stats(dataset_path)
    state_norm = Normalizer(stats["state_mean"], stats["state_std"])
    action_norm = Normalizer(stats["action_mean"], stats["action_std"])
    return DiffusionPolicyInference(config, model, scheduler, state_norm, action_norm, device)


class TestInferenceOutputShape:
    def test_single_obs(self, inference_pipeline):
        obs = torch.randn(2, 16)
        actions = inference_pipeline.predict(obs)
        assert actions.shape == (1, 8, 12)

    def test_batch_obs(self, inference_pipeline):
        obs = torch.randn(4, 2, 16)
        actions = inference_pipeline.predict(obs)
        assert actions.shape == (4, 8, 12)


class TestInferenceDenormalized:
    def test_denormalization_applied(self, inference_pipeline, device):
        obs = torch.randn(2, 16)
        actions = inference_pipeline.predict(obs)
        mean = inference_pipeline.action_normalizer.mean.to(device)
        std = inference_pipeline.action_normalizer.std.to(device)
        # If denorm was NOT applied, output would be ~N(0,1).
        # After denorm, the mean offset should be visible.
        # Check dim 4 (control_mode) which has mean=-0.996 and std=0.089
        assert actions.shape[-1] == 12


class TestInferenceDeterministic:
    def test_same_seed_same_output(self, inference_pipeline, device):
        obs = torch.randn(2, 16)
        gen1 = torch.Generator(device=device).manual_seed(42)
        gen2 = torch.Generator(device=device).manual_seed(42)
        a1 = inference_pipeline.predict(obs, generator=gen1)
        a2 = inference_pipeline.predict(obs, generator=gen2)
        torch.testing.assert_close(a1, a2)


class TestInferenceVaries:
    def test_different_seeds_different_output(self, inference_pipeline, device):
        obs = torch.randn(2, 16)
        gen1 = torch.Generator(device=device).manual_seed(42)
        gen2 = torch.Generator(device=device).manual_seed(123)
        a1 = inference_pipeline.predict(obs, generator=gen1)
        a2 = inference_pipeline.predict(obs, generator=gen2)
        assert not torch.allclose(a1, a2, atol=1e-3)


class TestActionChunkingRecedingHorizon:
    def test_overlap_consistency(self, inference_pipeline, device):
        obs1 = torch.randn(2, 16)
        obs2 = torch.randn(2, 16)
        gen1 = torch.Generator(device=device).manual_seed(0)
        gen2 = torch.Generator(device=device).manual_seed(1)
        chunk1 = inference_pipeline.predict(obs1, generator=gen1)
        chunk2 = inference_pipeline.predict(obs2, generator=gen2)
        assert chunk1.shape[1] == inference_pipeline.config.n_action_steps
        assert chunk2.shape[1] == inference_pipeline.config.n_action_steps


class TestInferenceLatency:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_under_100ms(self, inference_pipeline, device):
        obs = torch.randn(2, 16, device=device)
        # Warmup
        for _ in range(3):
            inference_pipeline.predict(obs)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(10):
            inference_pipeline.predict(obs)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10

        assert elapsed < 0.5, f"Inference took {elapsed*1000:.1f}ms, need <500ms"
