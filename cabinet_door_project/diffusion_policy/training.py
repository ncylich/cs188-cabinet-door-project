import copy
import logging
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusion_policy.config import DiffusionConfig
from diffusion_policy.data import DiffusionPolicyDataset
from diffusion_policy.scheduler import DDPMScheduler

logger = logging.getLogger(__name__)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters()}

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            self.shadow[name].lerp_(param.data, 1 - self.decay)

    def apply(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict) -> None:
        for k, v in state_dict.items():
            self.shadow[k] = v.clone()


def build_model(config: DiffusionConfig) -> nn.Module:
    if config.backbone == "mlp":
        from diffusion_policy.models.mlp import MLPNoiseNet
        return MLPNoiseNet(
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            horizon=config.horizon,
            n_obs_steps=config.n_obs_steps,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
        )
    elif config.backbone == "unet":
        from diffusion_policy.models.unet import UNetNoiseNet
        return UNetNoiseNet(
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            horizon=config.horizon,
            n_obs_steps=config.n_obs_steps,
        )
    elif config.backbone == "transformer":
        from diffusion_policy.models.transformer import TransformerNoiseNet
        return TransformerNoiseNet(
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            horizon=config.horizon,
            n_obs_steps=config.n_obs_steps,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_model=config.d_model,
        )
    elif config.backbone == "visuomotor":
        from diffusion_policy.models.transformer import VisuomotorTransformerNoiseNet
        return VisuomotorTransformerNoiseNet(
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            image_feature_dim=config.image_encoder_feature_dim * len(config.image_keys),
            horizon=config.horizon,
            n_obs_steps=config.n_obs_steps,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_model=config.d_model,
        )
    raise ValueError(f"Unknown backbone: {config.backbone}")


def build_scheduler(config: DiffusionConfig) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_steps=config.num_diffusion_steps,
        beta_schedule=config.beta_schedule,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: str,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: DiffusionConfig,
    epoch: int,
    global_step: int,
    loss: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    ema: Optional[EMA] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    return checkpoint


def _preload_dataset_to_gpu(dataset: DiffusionPolicyDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    all_obs = []
    all_actions = []
    for i in range(len(dataset)):
        obs, actions = dataset[i]
        all_obs.append(obs)
        all_actions.append(actions)
    return torch.stack(all_obs).to(device), torch.stack(all_actions).to(device)


def train(config: DiffusionConfig, dataset: Optional[DiffusionPolicyDataset] = None) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset is None:
        dataset = DiffusionPolicyDataset(config)

    use_gpu_preload = not config.use_images and device.type == "cuda"

    if use_gpu_preload:
        logger.info("Pre-loading %d samples to GPU for maximum throughput", len(dataset))
        all_obs, all_actions = _preload_dataset_to_gpu(dataset, device)
        n_samples = all_obs.shape[0]
        n_batches_per_epoch = n_samples // config.batch_size
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
            drop_last=True,
        )
        n_batches_per_epoch = len(dataloader)

    model = build_model(config).to(device)
    if config.compile_model:
        model = torch.compile(model)

    noise_scheduler = build_scheduler(config)
    ema = EMA(model, decay=config.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    total_steps = config.num_epochs * n_batches_per_epoch
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")
    global_step = 0

    logger.info("Training %s backbone for %d epochs (%d steps)", config.backbone, config.num_epochs, total_steps)
    start_time = time.time()

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        if use_gpu_preload:
            perm = torch.randperm(n_samples, device=device)
            for batch_start in range(0, n_samples - config.batch_size + 1, config.batch_size):
                idx = perm[batch_start:batch_start + config.batch_size]
                obs = all_obs[idx]
                actions = all_actions[idx]

                noise = torch.randn_like(actions)
                timesteps = torch.randint(0, noise_scheduler.num_train_steps, (obs.shape[0],), device=device)
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

                with torch.amp.autocast("cuda", enabled=config.use_amp, dtype=torch.bfloat16):
                    noise_pred = model(noisy_actions, obs, timesteps)
                    loss = nn.functional.mse_loss(noise_pred, noise.reshape(noise_pred.shape))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                ema.update(model)

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1
        else:
            for obs, actions in dataloader:
                obs = obs.to(device)
                actions = actions.to(device)

                noise = torch.randn_like(actions)
                timesteps = torch.randint(0, noise_scheduler.num_train_steps, (obs.shape[0],), device=device)
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

                with torch.amp.autocast("cuda", enabled=config.use_amp, dtype=torch.bfloat16):
                    noise_pred = model(noisy_actions, obs, timesteps)
                    loss = nn.functional.mse_loss(noise_pred, noise.reshape(noise_pred.shape))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                ema.update(model)

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            logger.info(
                "Epoch %d/%d  loss=%.6f  lr=%.2e  time=%.1fs",
                epoch + 1, config.num_epochs, avg_loss,
                optimizer.param_groups[0]["lr"], elapsed,
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                os.path.join(config.checkpoint_dir, "best.pt"),
                model, ema, optimizer, lr_scheduler, config, epoch, global_step, avg_loss,
            )

    save_checkpoint(
        os.path.join(config.checkpoint_dir, "final.pt"),
        model, ema, optimizer, lr_scheduler, config, epoch, global_step, avg_loss,
    )

    elapsed = time.time() - start_time
    logger.info("Training complete in %.1fs. Best loss: %.6f", elapsed, best_loss)
    return os.path.join(config.checkpoint_dir, "best.pt")


def train_visuomotor(config: DiffusionConfig) -> str:
    from diffusion_policy.data import VisuomotorDataset, precompute_image_features
    from diffusion_policy.models.vision import MultiCameraEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Precomputing image features with %s encoder", config.encoder_type)
    image_encoder = MultiCameraEncoder(
        num_cameras=len(config.image_keys),
        feature_dim=config.image_encoder_feature_dim,
        freeze=True,
        encoder_type=config.encoder_type,
        r3m_model_size=config.r3m_model_size,
    )
    image_features = precompute_image_features(
        config.dataset_path, image_encoder, device, batch_size=64,
    )

    dataset = VisuomotorDataset(config, image_features)
    img_feat_dim = config.image_encoder_feature_dim * len(config.image_keys)

    all_obs, all_img, all_actions = [], [], []
    for i in range(len(dataset)):
        obs, img, actions = dataset[i]
        all_obs.append(obs)
        all_img.append(img)
        all_actions.append(actions)
    all_obs = torch.stack(all_obs).to(device)
    all_img = torch.stack(all_img).to(device)
    all_actions = torch.stack(all_actions).to(device)
    n_samples = all_obs.shape[0]

    logger.info("Loaded %d visuomotor samples to GPU (img features: %d-dim)", n_samples, img_feat_dim)

    model = build_model(config).to(device)
    noise_scheduler = build_scheduler(config)
    ema = EMA(model, decay=config.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    n_batches_per_epoch = n_samples // config.batch_size
    total_steps = config.num_epochs * n_batches_per_epoch
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")
    global_step = 0

    logger.info("Training visuomotor backbone for %d epochs (%d steps)", config.num_epochs, total_steps)
    start_time = time.time()

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(n_samples, device=device)
        for batch_start in range(0, n_samples - config.batch_size + 1, config.batch_size):
            idx = perm[batch_start:batch_start + config.batch_size]
            obs = all_obs[idx]
            img = all_img[idx]
            actions = all_actions[idx]

            noise = torch.randn_like(actions)
            timesteps = torch.randint(0, noise_scheduler.num_train_steps, (obs.shape[0],), device=device)
            noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

            with torch.amp.autocast("cuda", enabled=config.use_amp, dtype=torch.bfloat16):
                noise_pred = model(noisy_actions, obs, timesteps, image_features=img)
                loss = nn.functional.mse_loss(noise_pred, noise.reshape(noise_pred.shape))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            logger.info(
                "Epoch %d/%d  loss=%.6f  lr=%.2e  time=%.1fs",
                epoch + 1, config.num_epochs, avg_loss,
                optimizer.param_groups[0]["lr"], elapsed,
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                os.path.join(config.checkpoint_dir, "best.pt"),
                model, ema, optimizer, lr_scheduler, config, epoch, global_step, avg_loss,
            )

    save_checkpoint(
        os.path.join(config.checkpoint_dir, "final.pt"),
        model, ema, optimizer, lr_scheduler, config, epoch, global_step, avg_loss,
    )

    elapsed = time.time() - start_time
    logger.info("Training complete in %.1fs. Best loss: %.6f", elapsed, best_loss)
    return os.path.join(config.checkpoint_dir, "best.pt")
