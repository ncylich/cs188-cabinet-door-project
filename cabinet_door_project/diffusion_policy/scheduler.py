import math

import torch
import torch.nn as nn


def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps)


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, num_steps, num_steps + 1)
    alphas_cumprod = torch.cos((t / num_steps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0, 0.999)


def squared_cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, num_steps, num_steps + 1)
    alphas_cumprod = torch.cos((t / num_steps + s) / (1 + s) * math.pi * 0.5) ** 4
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0, 0.999)


SCHEDULE_REGISTRY = {
    "linear": linear_beta_schedule,
    "cosine": cosine_beta_schedule,
    "squared_cosine": squared_cosine_beta_schedule,
}


class DDPMScheduler:
    def __init__(
        self,
        num_train_steps: int = 100,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.num_train_steps = num_train_steps

        schedule_fn = SCHEDULE_REGISTRY[beta_schedule]
        if beta_schedule == "linear":
            betas = schedule_fn(num_train_steps, beta_start, beta_end)
        else:
            betas = schedule_fn(num_train_steps)

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            betas * (1.0 - torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]]))
            / (1.0 - self.alphas_cumprod)
        )

    def add_noise(
        self, x_0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps.cpu()].to(x_0.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps.cpu()].to(x_0.device)
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        x_t: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        beta = self.betas[timestep].to(x_t.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timestep].to(x_t.device)
        sqrt_recip_alpha = self.sqrt_recip_alphas[timestep].to(x_t.device)

        pred_mean = sqrt_recip_alpha * (x_t - beta / sqrt_one_minus_alpha * model_output)

        if timestep == 0:
            return pred_mean

        variance = self.posterior_variance[timestep].to(x_t.device)
        noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
        return pred_mean + torch.sqrt(variance) * noise

    def ddim_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        x_t: torch.Tensor,
        next_timestep: int,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        alpha_t = self.alphas_cumprod[timestep].to(x_t.device)
        alpha_next = (
            self.alphas_cumprod[next_timestep].to(x_t.device)
            if next_timestep >= 0
            else torch.tensor(1.0, device=x_t.device)
        )

        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
        pred_x0 = pred_x0.clamp(-10, 10)

        sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_next - sigma**2, min=0)) * model_output

        x_next = torch.sqrt(alpha_next) * pred_x0 + dir_xt
        if eta > 0 and next_timestep >= 0:
            noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
            x_next = x_next + sigma * noise
        return x_next

    def get_ddim_timesteps(self, num_inference_steps: int) -> list[int]:
        step_ratio = self.num_train_steps // num_inference_steps
        timesteps = list(range(0, self.num_train_steps, step_ratio))
        return list(reversed(timesteps))

    def denoise_ddpm(
        self,
        model: nn.Module,
        x_T: torch.Tensor,
        context: torch.Tensor,
        generator: torch.Generator | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        x = x_T
        original_shape = x.shape
        for t in reversed(range(self.num_train_steps)):
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            noise_pred = model(x, context, t_batch, **model_kwargs).reshape(original_shape)
            x = self.step(noise_pred, t, x, generator=generator)
        return x

    def denoise_ddim(
        self,
        model: nn.Module,
        x_T: torch.Tensor,
        context: torch.Tensor,
        num_inference_steps: int = 16,
        generator: torch.Generator | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        timesteps = self.get_ddim_timesteps(num_inference_steps)
        x = x_T
        original_shape = x.shape
        for i, t in enumerate(timesteps):
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            noise_pred = model(x, context, t_batch, **model_kwargs).reshape(original_shape)
            next_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x = self.ddim_step(noise_pred, t, x, next_t, generator=generator)
        return x
