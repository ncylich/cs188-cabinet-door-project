import torch
import pytest

from diffusion_policy.models.mlp import MLPNoiseNet


@pytest.fixture
def model():
    return MLPNoiseNet(action_dim=12, state_dim=16, horizon=16, n_obs_steps=2, hidden_dim=512, n_layers=4)


class TestOutputShape:
    def test_flat_input(self, model):
        batch = 8
        noisy = torch.randn(batch, 16 * 12)
        obs = torch.randn(batch, 2 * 16)
        t = torch.randint(0, 100, (batch,))
        out = model(noisy, obs, t)
        assert out.shape == (batch, 16 * 12)

    def test_3d_input(self, model):
        batch = 8
        noisy = torch.randn(batch, 16, 12)
        obs = torch.randn(batch, 2, 16)
        t = torch.randint(0, 100, (batch,))
        out = model(noisy, obs, t)
        assert out.shape == (batch, 16 * 12)


class TestTimestepSensitivity:
    def test_different_timesteps_different_outputs(self, model):
        model.eval()
        x = torch.randn(4, 16 * 12)
        obs = torch.randn(4, 2 * 16)
        t1 = torch.tensor([0, 0, 0, 0])
        t2 = torch.tensor([50, 50, 50, 50])
        with torch.no_grad():
            out1 = model(x, obs, t1)
            out2 = model(x, obs, t2)
        assert not torch.allclose(out1, out2, atol=1e-6)


class TestObservationSensitivity:
    def test_different_obs_different_outputs(self, model):
        model.eval()
        x = torch.randn(4, 16 * 12)
        obs1 = torch.randn(4, 2 * 16)
        obs2 = torch.randn(4, 2 * 16)
        t = torch.tensor([10, 10, 10, 10])
        with torch.no_grad():
            out1 = model(x, obs1, t)
            out2 = model(x, obs2, t)
        assert not torch.allclose(out1, out2, atol=1e-6)


class TestGradientFlow:
    def test_all_params_have_gradients(self, model):
        x = torch.randn(4, 16 * 12)
        obs = torch.randn(4, 2 * 16)
        t = torch.randint(0, 100, (4,))
        out = model(x, obs, t)
        loss = out.pow(2).mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


class TestSingleBatchOverfit:
    def test_overfit(self):
        model = MLPNoiseNet(action_dim=12, state_dim=16, horizon=16, n_obs_steps=2, hidden_dim=512, n_layers=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(16, 16 * 12)
        obs = torch.randn(16, 2 * 16)
        t = torch.randint(0, 100, (16,))
        target = torch.randn(16, 16 * 12)

        for _ in range(200):
            pred = model(x, obs, t)
            loss = torch.nn.functional.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = torch.nn.functional.mse_loss(model(x, obs, t), target).item()
        assert final_loss < 0.01, f"Could not overfit single batch: loss={final_loss}"
