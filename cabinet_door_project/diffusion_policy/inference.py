from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as T

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
        image_encoder: Optional[nn.Module] = None,
    ):
        self.config = config
        self.model = model
        self.scheduler = scheduler
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.device = device
        self.image_encoder = image_encoder
        self.model.eval()
        if self.image_encoder is not None:
            self.image_encoder.eval()
        self._resize = T.Resize(224, antialias=True)

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

        image_encoder = None
        if config.backbone == "visuomotor":
            from diffusion_policy.models.vision import MultiCameraEncoder
            image_encoder = MultiCameraEncoder(
                num_cameras=len(config.image_keys),
                feature_dim=config.image_encoder_feature_dim,
                freeze=True,
                encoder_type=getattr(config, "encoder_type", "spatial_resnet"),
                r3m_model_size=getattr(config, "r3m_model_size", "resnet18"),
            ).to(device).eval()

        return cls(config, model, scheduler, state_norm, action_norm, device, image_encoder)

    def encode_images(self, images: list[torch.Tensor]) -> torch.Tensor:
        processed = []
        for img in images:
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.float()
            if img.max() <= 1.0:
                img = img * 255.0
            processed.append(img)
        with torch.no_grad():
            return self.image_encoder(processed)

    @torch.no_grad()
    def predict(
        self,
        obs_context: torch.Tensor,
        generator: torch.Generator | None = None,
        image_features: Optional[torch.Tensor] = None,
        images: Optional[list[list[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        if obs_context.dim() == 2:
            obs_context = obs_context.unsqueeze(0)

        obs_context = obs_context.to(self.device)
        obs_norm = self.state_normalizer.normalize(obs_context)

        if images is not None and self.image_encoder is not None:
            feats_per_step = []
            for step_images in images:
                cam_tensors = [img.to(self.device) for img in step_images]
                feats_per_step.append(self.encode_images(cam_tensors))
            image_features = torch.stack(feats_per_step, dim=1)

        if image_features is not None:
            image_features = image_features.to(self.device)

        model_kwargs = {}
        if image_features is not None:
            model_kwargs["image_features"] = image_features

        batch_size = obs_norm.shape[0]
        action_shape = (batch_size, self.config.horizon, self.config.action_dim)
        x_T = torch.randn(action_shape, device=self.device, dtype=torch.float32, generator=generator)

        if self.config.num_inference_steps < self.config.num_diffusion_steps:
            denoised = self.scheduler.denoise_ddim(
                self.model, x_T, obs_norm,
                num_inference_steps=self.config.num_inference_steps,
                generator=generator,
                **model_kwargs,
            )
        else:
            denoised = self.scheduler.denoise_ddpm(
                self.model, x_T, obs_norm,
                generator=generator,
                **model_kwargs,
            )

        denoised = denoised.reshape(batch_size, self.config.horizon, self.config.action_dim)
        actions = self.action_normalizer.denormalize(denoised)
        return actions[:, :self.config.n_action_steps]
