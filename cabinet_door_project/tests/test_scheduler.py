import torch
import torch.nn as nn
import pytest

from diffusion_policy.scheduler import DDPMScheduler


class TestForwardNoise:
    def test_t0_nearly_unchanged(self):
        scheduler = DDPMScheduler(num_train_steps=100, beta_schedule="cosine")
        x_0 = torch.randn(32, 64)
        noise = torch.randn_like(x_0)
        t = torch.zeros(32, dtype=torch.long)
        x_t = scheduler.add_noise(x_0, noise, t)
        diff = (x_t - x_0).abs().mean()
        assert diff < 0.1, f"At t=0, diff={diff} should be near 0"

    def test_tmax_nearly_pure_noise(self):
        scheduler = DDPMScheduler(num_train_steps=100, beta_schedule="cosine")
        x_0 = torch.ones(1000, 64) * 5.0
        noise = torch.randn_like(x_0)
        t = torch.full((1000,), 99, dtype=torch.long)
        x_t = scheduler.add_noise(x_0, noise, t)
        assert x_t.mean().abs() < 1.0
        assert (x_t.std() - 1.0).abs() < 0.5

    def test_noise_monotonic(self):
        scheduler = DDPMScheduler(num_train_steps=100, beta_schedule="cosine")
        x_0 = torch.randn(100, 64)
        noise = torch.randn_like(x_0)
        diffs = []
        for t_val in range(0, 100, 10):
            t = torch.full((100,), t_val, dtype=torch.long)
            x_t = scheduler.add_noise(x_0, noise, t)
            diffs.append((x_t - x_0).pow(2).mean().item())
        for i in range(len(diffs) - 1):
            assert diffs[i] < diffs[i + 1], f"Noise not monotonic: {diffs[i]} >= {diffs[i+1]}"


class TestBetaSchedule:
    @pytest.mark.parametrize("schedule", ["linear", "cosine", "squared_cosine"])
    def test_bounds(self, schedule):
        scheduler = DDPMScheduler(num_train_steps=100, beta_schedule=schedule)
        assert (scheduler.betas > 0).all()
        assert (scheduler.betas < 1).all()
        assert (scheduler.alphas_cumprod > 0).all()
        assert (scheduler.alphas_cumprod < 1).all()
        diffs = scheduler.alphas_cumprod[1:] - scheduler.alphas_cumprod[:-1]
        assert (diffs <= 0).all(), "alphas_cumprod not monotonically decreasing"


class TestReverseStep:
    def test_shape(self):
        scheduler = DDPMScheduler(num_train_steps=100)
        x_t = torch.randn(8, 64)
        model_output = torch.randn(8, 64)
        result = scheduler.step(model_output, 50, x_t)
        assert result.shape == x_t.shape


class TestFullDenoise:
    def test_recovers_signal(self):
        torch.manual_seed(42)
        scheduler = DDPMScheduler(num_train_steps=100, beta_schedule="cosine")
        dim = 16

        class TinyDenoiser(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim + dim + 64, 256),
                    nn.SiLU(),
                    nn.Linear(256, 256),
                    nn.SiLU(),
                    nn.Linear(256, dim),
                )
                self.time_embed = nn.Sequential(
                    nn.Linear(1, 64),
                    nn.SiLU(),
                    nn.Linear(64, 64),
                )

            def forward(self, x: torch.Tensor, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                t_emb = self.time_embed(t.float().unsqueeze(-1) / 100.0)
                return self.net(torch.cat([x, context, t_emb], dim=-1))

        model = TinyDenoiser()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        signal = torch.sin(torch.linspace(0, 2 * 3.14159, dim)).unsqueeze(0).expand(64, -1)
        context = torch.zeros(64, dim)

        for _ in range(500):
            noise = torch.randn_like(signal)
            t = torch.randint(0, 100, (64,))
            x_t = scheduler.add_noise(signal, noise, t)
            pred_noise = model(x_t, context, t)
            loss = nn.functional.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x_T = torch.randn(64, dim)
            ctx = torch.zeros(64, dim)
            recovered = scheduler.denoise_ddpm(model, x_T, ctx)
            mse_recovered = (recovered - signal).pow(2).mean().item()
            mse_noise = (x_T - signal).pow(2).mean().item()

        assert mse_recovered < mse_noise, f"Denoised MSE {mse_recovered} >= noise MSE {mse_noise}"


class TestDDIM:
    def test_fewer_steps(self):
        torch.manual_seed(42)
        scheduler = DDPMScheduler(num_train_steps=100, beta_schedule="cosine")
        dim = 16

        class TinyDenoiser(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim + dim + 64, 256), nn.SiLU(),
                    nn.Linear(256, 256), nn.SiLU(),
                    nn.Linear(256, dim),
                )
                self.time_embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))

            def forward(self, x, context, t):
                t_emb = self.time_embed(t.float().unsqueeze(-1) / 100.0)
                return self.net(torch.cat([x, context, t_emb], dim=-1))

        model = TinyDenoiser()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        signal = torch.sin(torch.linspace(0, 2 * 3.14159, dim)).unsqueeze(0).expand(64, -1)
        context = torch.zeros(64, dim)

        for _ in range(500):
            noise = torch.randn_like(signal)
            t = torch.randint(0, 100, (64,))
            x_t = scheduler.add_noise(signal, noise, t)
            pred_noise = model(x_t, context, t)
            loss = nn.functional.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            gen = torch.Generator().manual_seed(123)
            x_T = torch.randn(64, dim, generator=gen)

            gen_ddpm = torch.Generator().manual_seed(456)
            ddpm_result = scheduler.denoise_ddpm(model, x_T.clone(), context, generator=gen_ddpm)

            gen_ddim = torch.Generator().manual_seed(456)
            ddim_result = scheduler.denoise_ddim(model, x_T.clone(), context, num_inference_steps=16, generator=gen_ddim)

            mse_ddpm = (ddpm_result - signal).pow(2).mean().item()
            mse_ddim = (ddim_result - signal).pow(2).mean().item()

        assert mse_ddim < mse_ddpm + 0.1, f"DDIM MSE {mse_ddim} much worse than DDPM {mse_ddpm}"
