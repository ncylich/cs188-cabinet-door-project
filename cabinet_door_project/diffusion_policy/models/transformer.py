import math

import torch
import torch.nn as nn

from diffusion_policy.models.mlp import TimestepMLP


class TransformerNoiseNet(nn.Module):
    def __init__(
        self,
        action_dim: int = 12,
        state_dim: int = 16,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_layers: int = 8,
        n_heads: int = 4,
        d_model: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.d_model = d_model

        self.time_embed = TimestepMLP(128, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.pos_embed = nn.Embedding(n_obs_steps + horizon, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        obs_context: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch = noisy_actions.shape[0]
        if noisy_actions.dim() == 2:
            noisy_actions = noisy_actions.reshape(batch, self.horizon, self.action_dim)
        if obs_context.dim() == 2:
            obs_context = obs_context.reshape(batch, self.n_obs_steps, -1)

        t_emb = self.time_embed(timesteps).unsqueeze(1)

        state_tokens = self.state_proj(obs_context)
        action_tokens = self.action_proj(noisy_actions)

        seq = torch.cat([state_tokens, action_tokens], dim=1)
        seq_len = seq.shape[1]
        positions = torch.arange(seq_len, device=seq.device)
        seq = seq + self.pos_embed(positions).unsqueeze(0) + t_emb

        out = self.transformer(seq)
        action_out = out[:, self.n_obs_steps:]
        return self.output_proj(action_out)
