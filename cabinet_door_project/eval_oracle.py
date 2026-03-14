#!/usr/bin/env python
"""Evaluate oracle-state diffusion policy checkpoints.

Usage:
    python eval_oracle.py --checkpoint /tmp/diffusion_policy_checkpoints/unet_19dim_3k_bs2048/best.pt --num_rollouts 5
"""
import argparse
import logging
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

# Import our modules
from diffusion_policy.config import DiffusionConfig
from diffusion_policy.models.unet import UNetNoiseNet
from diffusion_policy.models.mlp import MLPNoiseNet
from diffusion_policy.scheduler import DDPMScheduler
from diffusion_policy.training import build_scheduler, EMA
from diffusion_policy.evaluation import (
    extract_state, dataset_action_to_env_action, create_env, STATE_KEYS_ORDERED
)
from diffusion_policy.data import Normalizer


def load_oracle_checkpoint(path, device):
    """Load checkpoint saved by our oracle training script."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    # Handle both new format (state_dim key) and old format (config object)
    # Default architecture params
    unet_channels = None
    mlp_hidden_dim = None
    mlp_n_layers = None

    if 'config' in ckpt:
        cfg = ckpt['config']
        state_dim = cfg.state_dim
        backbone = cfg.backbone if hasattr(cfg, 'backbone') else 'unet'
        horizon = cfg.horizon
        n_obs_steps = cfg.n_obs_steps
        n_action_steps = cfg.n_action_steps
    else:
        state_dim = ckpt['state_dim']
        backbone = ckpt.get('backbone', 'unet')
        horizon = ckpt.get('horizon', 16)
        n_obs_steps = ckpt.get('n_obs_steps', 2)
        n_action_steps = ckpt.get('n_action_steps', 8)
        unet_channels = ckpt.get('unet_channels', (256, 512, 1024))
        mlp_hidden_dim = ckpt.get('mlp_hidden_dim', 512)
        mlp_n_layers = ckpt.get('mlp_n_layers', 4)

    # Auto-detect backbone if not stored
    if backbone == 'unet':
        first_key = list(ckpt['model_state_dict'].keys())[0]
        if 'downs' not in first_key and 'time_embed' not in first_key and 'input_conv' not in first_key:
            # Check more carefully
            has_conv = any('conv' in k for k in ckpt['model_state_dict'].keys())
            if not has_conv:
                backbone = 'mlp'

    # Build model with correct architecture params
    if backbone == 'unet':
        channels = tuple(unet_channels) if unet_channels else (256, 512, 1024)
        model = UNetNoiseNet(action_dim=12, state_dim=state_dim, horizon=horizon,
                             n_obs_steps=n_obs_steps, channels=channels)
    else:
        h_dim = mlp_hidden_dim if mlp_hidden_dim else 512
        n_lay = mlp_n_layers if mlp_n_layers else 4
        model = MLPNoiseNet(action_dim=12, state_dim=state_dim, horizon=horizon,
                            n_obs_steps=n_obs_steps, hidden_dim=h_dim, n_layers=n_lay)

    model.load_state_dict(ckpt['model_state_dict'])

    # Apply EMA weights
    ema = EMA(model, decay=0.9999)
    ema.load_state_dict(ckpt['ema_state_dict'])
    ema.apply(model)

    model = model.to(device).eval()

    # Build normalizer - handle both new and old checkpoint formats
    if 'obs_mean' in ckpt:
        obs_mean = torch.tensor(ckpt['obs_mean'], dtype=torch.float32).to(device) if not isinstance(ckpt['obs_mean'], torch.Tensor) else ckpt['obs_mean'].float().to(device)
        obs_std = torch.tensor(ckpt['obs_std'], dtype=torch.float32).to(device) if not isinstance(ckpt['obs_std'], torch.Tensor) else ckpt['obs_std'].float().to(device)
        act_mean = torch.tensor(ckpt['act_mean'], dtype=torch.float32).to(device) if not isinstance(ckpt['act_mean'], torch.Tensor) else ckpt['act_mean'].float().to(device)
        act_std = torch.tensor(ckpt['act_std'], dtype=torch.float32).to(device) if not isinstance(ckpt['act_std'], torch.Tensor) else ckpt['act_std'].float().to(device)
    else:
        # Old format: load stats from the preprocessed data files
        # Try to find the right preprocessed file based on state_dim
        import os
        preproc_path = f'/tmp/diffusion_policy_checkpoints/preprocessed_{state_dim}dim.pt'
        if os.path.exists(preproc_path):
            preproc = torch.load(preproc_path, weights_only=False)
            obs_mean = torch.tensor(preproc['obs_mean'], dtype=torch.float32).to(device)
            obs_std = torch.tensor(preproc['obs_std'], dtype=torch.float32).to(device)
            act_mean = torch.tensor(preproc['act_mean'], dtype=torch.float32).to(device)
            act_std = torch.tensor(preproc['act_std'], dtype=torch.float32).to(device)
        else:
            raise ValueError(f"No preprocessed stats found for state_dim={state_dim}. "
                           f"Expected file: {preproc_path}")

    # Build scheduler
    scheduler = build_scheduler(DiffusionConfig(beta_schedule='squared_cosine'))

    return model, scheduler, obs_mean, obs_std, act_mean, act_std, state_dim, horizon, n_obs_steps, n_action_steps


def extract_oracle_state(obs, state_dim):
    """Extract proprioception + oracle door info from env observation."""
    # 16-dim proprioception
    parts = []
    for key in STATE_KEYS_ORDERED:
        parts.append(obs[key].flatten())
    proprio = np.concatenate(parts).astype(np.float32)

    if state_dim == 19:
        # Add door_obj_pos (3-dim)
        door_pos = obs['door_obj_pos'].flatten().astype(np.float32)
        return np.concatenate([proprio, door_pos])
    elif state_dim == 23:
        # Add door_obj_pos (3-dim) + door_obj_quat (4-dim)
        door_pos = obs['door_obj_pos'].flatten().astype(np.float32)
        door_quat = obs['door_obj_quat'].flatten().astype(np.float32)
        return np.concatenate([proprio, door_pos, door_quat])
    else:
        return proprio


@torch.no_grad()
def predict_actions(model, scheduler, obs_context, obs_mean, obs_std, act_mean, act_std, horizon, device):
    """Run DDIM denoising to predict actions."""
    if obs_context.dim() == 2:
        obs_context = obs_context.unsqueeze(0)
    obs_context = obs_context.to(device)

    # Normalize observations
    obs_norm = (obs_context - obs_mean) / (obs_std + 1e-8)

    batch_size = obs_norm.shape[0]
    action_shape = (batch_size, horizon, 12)
    x_T = torch.randn(action_shape, device=device, dtype=torch.float32)

    # Use DDIM with 16 steps for speed
    denoised = scheduler.denoise_ddim(model, x_T, obs_norm, num_inference_steps=16)
    denoised = denoised.reshape(batch_size, horizon, 12)

    # Denormalize actions
    actions = denoised * (act_std + 1e-8) + act_mean
    return actions


def run_eval(checkpoint_path, num_rollouts=5, max_steps=500, split='pretrain', seed=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Loading checkpoint: %s", checkpoint_path)
    model, scheduler, obs_mean, obs_std, act_mean, act_std, state_dim, horizon, n_obs_steps, n_action_steps = \
        load_oracle_checkpoint(checkpoint_path, device)
    logger.info("Model loaded: state_dim=%d, horizon=%d, n_obs_steps=%d, n_action_steps=%d",
                state_dim, horizon, n_obs_steps, n_action_steps)

    logger.info("Creating env...")
    env = create_env(split=split, seed=seed)

    successes = []
    distances = []

    for ep in range(num_rollouts):
        obs = env.reset()
        obs_history = deque(maxlen=n_obs_steps)
        state = extract_oracle_state(obs, state_dim)
        obs_history.append(state)

        success = False
        action_queue = deque()
        init_dist = np.linalg.norm(obs['door_obj_to_robot0_eef_pos'])
        min_dist = init_dist

        t0 = time.time()
        for step in range(max_steps):
            if len(action_queue) == 0:
                while len(obs_history) < n_obs_steps:
                    obs_history.appendleft(obs_history[0])

                obs_context = torch.from_numpy(np.stack(list(obs_history), axis=0)).float()
                predicted = predict_actions(model, scheduler, obs_context, obs_mean, obs_std, act_mean, act_std, horizon, device)

                for i in range(n_action_steps):
                    action_queue.append(predicted[0, i].cpu().numpy())

            dataset_action = action_queue.popleft()
            env_action = dataset_action_to_env_action(dataset_action)
            env_action = np.clip(env_action, -1.0, 1.0)

            obs, reward, done, info = env.step(env_action)
            state = extract_oracle_state(obs, state_dim)
            obs_history.append(state)

            dist = np.linalg.norm(obs['door_obj_to_robot0_eef_pos'])
            min_dist = min(min_dist, dist)

            if env._check_success():
                success = True
                break

        elapsed = time.time() - t0
        dist_reduction = (init_dist - min_dist) / init_dist * 100
        successes.append(success)
        distances.append({'init': init_dist, 'min': min_dist, 'reduction': dist_reduction})

        status = "SUCCESS" if success else "FAIL"
        logger.info("Episode %d/%d: %s (steps=%d, init_dist=%.3f, min_dist=%.3f, reduction=%.0f%%, time=%.1fs)",
                     ep + 1, num_rollouts, status, step + 1, init_dist, min_dist, dist_reduction, elapsed)

    env.close()

    success_rate = sum(successes) / num_rollouts
    avg_reduction = np.mean([d['reduction'] for d in distances])
    logger.info("=" * 60)
    logger.info("RESULTS: %d/%d = %.0f%% success, avg distance reduction = %.0f%%",
                sum(successes), num_rollouts, success_rate * 100, avg_reduction)
    logger.info("=" * 60)

    return {'success_rate': success_rate, 'successes': successes, 'distances': distances}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num_rollouts', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--split', default='pretrain')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_eval(args.checkpoint, args.num_rollouts, args.max_steps, args.split, args.seed)
