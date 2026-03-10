import math

import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepMLP(nn.Module):
    def __init__(self, sinusoidal_dim: int, hidden_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbedding(sinusoidal_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MLPNoiseNet(nn.Module):
    def __init__(
        self,
        action_dim: int = 12,
        state_dim: int = 16,
        horizon: int = 16,
        n_obs_steps: int = 2,
        hidden_dim: int = 512,
        n_layers: int = 4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.flat_action_dim = horizon * action_dim
        obs_input_dim = n_obs_steps * state_dim

        self.time_embed = TimestepMLP(128, hidden_dim)

        self.input_proj = nn.Linear(self.flat_action_dim + obs_input_dim, hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden_dim, self.flat_action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        obs_context: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_actions.dim() == 3:
            batch = noisy_actions.shape[0]
            noisy_actions = noisy_actions.reshape(batch, -1)
        if obs_context.dim() == 3:
            obs_context = obs_context.reshape(obs_context.shape[0], -1)

        t_emb = self.time_embed(timesteps)
        x = self.input_proj(torch.cat([noisy_actions, obs_context], dim=-1))
        x = x + self.time_proj(t_emb)

        for block in self.blocks:
            x = block(x)

        return self.output_proj(x)
