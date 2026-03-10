import os
import tempfile
import copy

import torch
import pytest

from diffusion_policy.config import DiffusionConfig
from diffusion_policy.training import (
    build_model,
    build_scheduler,
    EMA,
    get_cosine_schedule_with_warmup,
    save_checkpoint,
    load_checkpoint,
)


@pytest.fixture
def small_config(dataset_path):
    return DiffusionConfig(
        dataset_path=dataset_path,
        backbone="mlp",
        hidden_dim=128,
        n_layers=2,
        batch_size=32,
        num_epochs=1,
        num_workers=0,
        use_amp=False,
    )


@pytest.fixture
def small_model_and_scheduler(small_config):
    model = build_model(small_config)
    scheduler = build_scheduler(small_config)
    return model, scheduler


class TestLossDecreases:
    def test_loss_drops_in_50_steps(self, dataset, small_config, device):
        from torch.utils.data import DataLoader

        model = build_model(small_config).to(device)
        scheduler = build_scheduler(small_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

        losses = []
        model.train()
        step = 0
        for obs, actions in loader:
            obs, actions = obs.to(device), actions.to(device)
            noise = torch.randn_like(actions)
            t = torch.randint(0, scheduler.num_train_steps, (obs.shape[0],), device=device)
            noisy = scheduler.add_noise(actions, noise, t)
            pred = model(noisy, obs, t)
            loss = torch.nn.functional.mse_loss(pred, noise.reshape(pred.shape))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
            if step >= 50:
                break

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"


class TestEMA:
    def test_ema_diverges(self, small_config, device):
        model = build_model(small_config).to(device)
        ema = EMA(model, decay=0.99)

        for _ in range(100):
            for p in model.parameters():
                p.data.add_(torch.randn_like(p) * 0.1)
            ema.update(model)

        for name, param in model.named_parameters():
            assert not torch.allclose(param.data, ema.shadow[name].to(device), atol=1e-6), \
                f"EMA identical to model for {name}"

    def test_ema_smoother(self, small_config, device):
        model = build_model(small_config).to(device)
        ema = EMA(model, decay=0.99)

        prev_model = {n: p.data.clone() for n, p in model.named_parameters()}
        prev_ema = {n: v.clone() for n, v in ema.shadow.items()}

        model_diffs = []
        ema_diffs = []
        for _ in range(50):
            for p in model.parameters():
                p.data.add_(torch.randn_like(p) * 0.1)
            ema.update(model)

            model_diff = sum((p.data - prev_model[n]).pow(2).sum() for n, p in model.named_parameters())
            ema_diff = sum((ema.shadow[n].to(device) - prev_ema[n].to(device)).pow(2).sum() for n in ema.shadow)
            model_diffs.append(model_diff.item())
            ema_diffs.append(ema_diff.item())

            prev_model = {n: p.data.clone() for n, p in model.named_parameters()}
            prev_ema = {n: v.clone() for n, v in ema.shadow.items()}

        assert sum(ema_diffs) < sum(model_diffs), "EMA not smoother than model"


class TestCheckpoint:
    def test_save_load_roundtrip(self, small_config, device):
        model = build_model(small_config).to(device)
        ema = EMA(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lr_sched = get_cosine_schedule_with_warmup(optimizer, 10, 100)

        x = torch.randn(4, 16 * 12, device=device)
        obs = torch.randn(4, 2 * 16, device=device)
        t = torch.randint(0, 100, (4,), device=device)

        with torch.no_grad():
            original_out = model(x, obs, t).clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(path, model, ema, optimizer, lr_sched, small_config, 10, 100, 0.5)

            model2 = build_model(small_config).to(device)
            load_checkpoint(path, model2)
            with torch.no_grad():
                loaded_out = model2(x, obs, t)
            torch.testing.assert_close(original_out, loaded_out)
        finally:
            os.unlink(path)


class TestLRSchedule:
    def test_warmup_and_decay(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        warmup = 100
        total = 1000
        sched = get_cosine_schedule_with_warmup(optimizer, warmup, total)

        lrs = []
        for step in range(total):
            lrs.append(optimizer.param_groups[0]["lr"])
            sched.step()

        # Warmup: monotonically non-decreasing
        for i in range(1, warmup):
            assert lrs[i] >= lrs[i - 1] - 1e-10, f"LR decreased during warmup at step {i}"

        # After warmup target LR reached
        assert abs(lrs[warmup] - 1e-4) < 1e-6

        # Decay: monotonically non-increasing after warmup
        for i in range(warmup + 1, total):
            assert lrs[i] <= lrs[i - 1] + 1e-10, f"LR increased after warmup at step {i}"

        # Near zero at end
        assert lrs[-1] < 1e-6


class TestAMPNoNaN:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_no_nan_with_amp(self, dataset, small_config, device):
        from torch.utils.data import DataLoader

        small_config.use_amp = True
        model = build_model(small_config).to(device)
        scheduler = build_scheduler(small_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

        model.train()
        step = 0
        for obs, actions in loader:
            obs, actions = obs.to(device), actions.to(device)
            noise = torch.randn_like(actions)
            t = torch.randint(0, scheduler.num_train_steps, (obs.shape[0],), device=device)
            noisy = scheduler.add_noise(actions, noise, t)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(noisy, obs, t)
                loss = torch.nn.functional.mse_loss(pred, noise.reshape(pred.shape))

            assert not torch.isnan(loss), f"NaN loss at step {step}"
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for p in model.parameters():
                assert torch.isfinite(p.data).all(), f"Non-finite params at step {step}"

            step += 1
            if step >= 50:
                break
