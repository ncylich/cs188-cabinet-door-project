import torch
import torch.nn as nn

from diffusion_policy.config import DiffusionConfig
from diffusion_policy.data import Normalizer, load_stats
from diffusion_policy.scheduler import DDPMScheduler
from diffusion_policy.training import build_model, build_scheduler, EMA, load_checkpoint


class DiffusionPolicyInference:
    def __init__(
        self,
        config: DiffusionConfig,
        model: nn.Module,
        scheduler: DDPMScheduler,
        state_normalizer: Normalizer,
        action_normalizer: Normalizer,
        device: torch.device,
    ):
        self.config = config
        self.model = model
        self.scheduler = scheduler
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.device = device
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device | None = None) -> "DiffusionPolicyInference":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        dataset_path = config.dataset_path

        model = build_model(config).to(device)
        ema = EMA(model, decay=config.ema_decay)
        load_checkpoint(checkpoint_path, model, ema)
        ema.apply(model)

        scheduler = build_scheduler(config)
        stats = load_stats(dataset_path)
        state_norm = Normalizer(stats["state_mean"], stats["state_std"])
        action_norm = Normalizer(stats["action_mean"], stats["action_std"])

        return cls(config, model, scheduler, state_norm, action_norm, device)

    @torch.no_grad()
    def predict(
        self,
        obs_context: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if obs_context.dim() == 2:
            obs_context = obs_context.unsqueeze(0)

        obs_context = obs_context.to(self.device)
        obs_norm = self.state_normalizer.normalize(obs_context)

        batch_size = obs_norm.shape[0]
        action_shape = (batch_size, self.config.horizon, self.config.action_dim)
        x_T = torch.randn(action_shape, device=self.device, dtype=torch.float32, generator=generator)

        if self.config.num_inference_steps < self.config.num_diffusion_steps:
            denoised = self.scheduler.denoise_ddim(
                self.model, x_T, obs_norm,
                num_inference_steps=self.config.num_inference_steps,
                generator=generator,
            )
        else:
            denoised = self.scheduler.denoise_ddpm(
                self.model, x_T, obs_norm,
                generator=generator,
            )

        denoised = denoised.reshape(batch_size, self.config.horizon, self.config.action_dim)
        actions = self.action_normalizer.denormalize(denoised)
        return actions[:, :self.config.n_action_steps]
