import torch
import torch.nn as nn

from diffusion_policy.models.mlp import TimestepMLP


class FiLMBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.scale_proj = nn.Linear(cond_dim, channels)
        self.shift_proj = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.scale_proj(cond).unsqueeze(-1)
        shift = self.shift_proj(cond).unsqueeze(-1)
        return x * (1 + scale) + shift


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.film = FiLMBlock(out_channels, cond_dim)
        self.act = nn.SiLU()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.film(h, cond)
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.residual(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.res = ResBlock1D(in_ch, out_ch, cond_dim)
        self.down = nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.res(x, cond)
        return self.down(h), h


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlock1D(in_ch + skip_ch, out_ch, cond_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = x[..., :skip.shape[-1]]
        return self.res(torch.cat([x, skip], dim=1), cond)


class UNetNoiseNet(nn.Module):
    def __init__(
        self,
        action_dim: int = 12,
        state_dim: int = 16,
        horizon: int = 16,
        n_obs_steps: int = 2,
        channels: tuple[int, ...] = (256, 512, 1024),
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        obs_dim = n_obs_steps * state_dim

        cond_dim = 256
        self.time_embed = TimestepMLP(128, cond_dim)
        self.obs_proj = nn.Linear(obs_dim, cond_dim)

        self.input_conv = nn.Conv1d(action_dim, channels[0], 1)

        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downs.append(DownBlock(channels[i], channels[i + 1], cond_dim))

        self.mid = ResBlock1D(channels[-1], channels[-1], cond_dim)

        self.ups = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.ups.append(UpBlock(channels[i], channels[i], channels[i - 1], cond_dim))

        self.output_conv = nn.Conv1d(channels[0], action_dim, 1)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        obs_context: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_actions.dim() == 2:
            batch = noisy_actions.shape[0]
            noisy_actions = noisy_actions.reshape(batch, self.horizon, self.action_dim)
        if obs_context.dim() == 3:
            obs_context = obs_context.reshape(obs_context.shape[0], -1)

        t_emb = self.time_embed(timesteps)
        obs_emb = self.obs_proj(obs_context)
        cond = t_emb + obs_emb

        # (batch, horizon, action_dim) -> (batch, action_dim, horizon)
        x = noisy_actions.permute(0, 2, 1)
        x = self.input_conv(x)

        skips = []
        for down in self.downs:
            x, skip = down(x, cond)
            skips.append(skip)

        x = self.mid(x, cond)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, cond)

        x = self.output_conv(x)
        # (batch, action_dim, horizon) -> (batch, horizon, action_dim)
        return x.permute(0, 2, 1)
