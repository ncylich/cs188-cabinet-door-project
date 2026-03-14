"""Ensemble evaluation: average predictions from multiple BC/UNet checkpoints.

Supports transformer (arch=transformer) and UNet diffusion (arch=unet) models.
For UNet models, DDIM denoising is run per model and action chunks are averaged.

Usage:
    python ensemble_eval.py --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt --n_eps 20 --n_eval_workers 4
"""
import argparse
import multiprocessing as mp
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from bc_handle import (
    create_env, extract_state, dataset_action_to_env_action,
    build_model, _any_door_open, GRIPPER_DIM,
    STATE_DIM, FEATURE_CONFIGS,
)


def _ensemble_worker(args):
    ckpt_paths, ep_indices, seq_len, max_steps, split, base_seed, ddim_samples, success_threshold = args

    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    # Load all checkpoints
    models_info = []
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        arch = ckpt.get('arch', 'transformer')
        feat_indices = ckpt.get('feat_indices', None)
        binary_gripper = ckpt.get('binary_gripper', False)
        chunk_size = ckpt.get('chunk_size', 1)
        mkw = ckpt['model_kwargs']
        obs_mean = ckpt['obs_mean'].cpu().numpy()
        obs_std = ckpt['obs_std'].cpu().numpy()
        act_mean = ckpt['act_mean'].cpu().numpy()
        act_std = ckpt['act_std'].cpu().numpy()
        static_mask = ckpt['static_mask'].numpy()
        static_vals = ckpt['static_vals'].numpy()
        model_state_dim = mkw['state_dim']
        action_dim = mkw['action_dim']
        unet_ch = ckpt.get('unet_channels', (256, 512, 1024))
        horizon = ckpt.get('horizon', 16)
        n_obs_steps = ckpt.get('n_obs_steps', 2)
        n_action_steps = ckpt.get('n_action_steps', 8)
        ddim_steps = ckpt.get('ddim_steps', 10)
        # Remove keys from mkw that are passed explicitly to avoid duplicate keyword args
        mkw_clean = {k: v for k, v in mkw.items()
                     if k not in ('horizon', 'n_obs_steps')}
        model = build_model(arch, binary_gripper=binary_gripper,
                            unet_channels=unet_ch, chunk_size=chunk_size,
                            horizon=horizon, n_obs_steps=n_obs_steps, **mkw_clean)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        models_info.append(dict(
            model=model, arch=arch, feat_indices=feat_indices,
            binary_gripper=binary_gripper, chunk_size=chunk_size,
            obs_mean=obs_mean, obs_std=obs_std,
            act_mean=act_mean, act_std=act_std,
            static_mask=static_mask, static_vals=static_vals,
            model_state_dim=model_state_dim, action_dim=action_dim,
            horizon=horizon, n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps, ddim_steps=ddim_steps,
        ))

    # Use first model's action dim (all should be same)
    action_dim = models_info[0]['action_dim']

    # Determine if we have any UNet models
    has_unet = any(mi['arch'] == 'unet' for mi in models_info)
    has_transformer = any(mi['arch'] == 'transformer' for mi in models_info)

    # For UNet models, load DDPMScheduler
    if has_unet:
        from diffusion_policy.scheduler import DDPMScheduler
        diff_schedulers = {}
        for j, mi in enumerate(models_info):
            if mi['arch'] == 'unet':
                diff_schedulers[j] = DDPMScheduler(num_train_steps=100, beta_schedule='squared_cosine')

    env = create_env(split=split, seed=base_seed + ep_indices[0])
    obs = env.reset()
    init_flat = env.sim.get_state().flatten()

    results = []

    for i, ep_i in enumerate(ep_indices):
        if i > 0:
            env.sim.set_state_from_flattened(init_flat)
            env.sim.forward()
            for robot in env.robots:
                ctrl = getattr(robot, 'composite_controller', None) or getattr(robot, 'controller', None)
                if ctrl is not None and hasattr(ctrl, 'reset'):
                    ctrl.reset()
            obs = env._get_observations()

        active_site = None
        # Per-model observation histories (for transformer: seq, for UNet: obs_deque)
        histories = [deque(maxlen=seq_len) for _ in models_info]
        obs_deques = [deque(maxlen=mi['n_obs_steps']) for mi in models_info]
        success = False
        last_state_full = None
        state_full, active_site = extract_state(obs, env, active_site)

        # Shared action queue (populated when empty, using ensemble prediction)
        shared_action_queue = deque()
        grip_hold_remaining = 0
        GRIP_HOLD = 30

        for step in range(max_steps):
            last_state_full = state_full

            if len(shared_action_queue) == 0:
                # Collect predictions from all models
                raw_preds = []

                for j, mi in enumerate(models_info):
                    state = state_full[mi['feat_indices']] if mi['feat_indices'] is not None else state_full
                    state_n = (state - mi['obs_mean']) / mi['obs_std']

                    if mi['arch'] == 'unet':
                        # UNet: run DDIM denoising to get action chunk
                        obs_deques[j].append(state_n)
                        odq = obs_deques[j]
                        while len(odq) < mi['n_obs_steps']:
                            odq.appendleft(odq[0])
                        obs_ctx = torch.from_numpy(
                            np.concatenate(list(odq))).unsqueeze(0)  # (1, n_obs*D)
                        if ddim_samples > 1:
                            preds = []
                            for _ in range(ddim_samples):
                                x_T = torch.randn(1, mi['horizon'], mi['action_dim'])
                                with torch.no_grad():
                                    p = diff_schedulers[j].denoise_ddim(
                                        mi['model'], x_T, obs_ctx,
                                        num_inference_steps=mi['ddim_steps'])
                                preds.append(p[0].numpy())
                            pred_np = np.mean(preds, axis=0)
                        else:
                            x_T = torch.randn(1, mi['horizon'], mi['action_dim'])
                            with torch.no_grad():
                                pred_h = diff_schedulers[j].denoise_ddim(
                                    mi['model'], x_T, obs_ctx,
                                    num_inference_steps=mi['ddim_steps'])  # (1,H,A)
                            pred_np = pred_h[0].numpy()  # (H, A)
                        # Denormalize all steps in chunk
                        chunk_raw = pred_np * mi['act_std'] + mi['act_mean']
                        chunk_raw[:, mi['static_mask']] = mi['static_vals'][mi['static_mask']]
                        chunk_raw = np.clip(chunk_raw, -1.0, 1.0)
                        raw_preds.append(('unet', chunk_raw, mi))  # (H, A)
                    else:
                        # Transformer: single-step prediction
                        histories[j].append(state_n)
                        L = len(histories[j])
                        seq_buf = np.zeros((seq_len, mi['model_state_dim']), dtype=np.float32)
                        seq_buf[-L:] = np.stack(list(histories[j]))
                        mask_buf = np.ones(seq_len, dtype=bool)
                        mask_buf[-L:] = False
                        seq_t = torch.from_numpy(seq_buf).unsqueeze(0)
                        mask_t = torch.from_numpy(mask_buf).unsqueeze(0)
                        with torch.no_grad():
                            pred_n = mi['model'](seq_t, mask_t).numpy()[0]
                        raw = pred_n * mi['act_std'] + mi['act_mean']
                        raw[mi['static_mask']] = mi['static_vals'][mi['static_mask']]
                        raw = np.clip(raw, -1.0, 1.0)
                        raw_preds.append(('transformer', raw, mi))

                # Average predictions across models
                # For UNet: average the n_action_steps actions from each chunk
                # For transformer: use single action
                # Strategy: expand transformers to match UNet chunk steps
                n_exec = max(
                    (mi['n_action_steps'] if mi['arch'] == 'unet' else 1)
                    for mi in models_info
                )

                avg_actions = []
                for h in range(n_exec):
                    step_preds = []
                    for ptype, pred_data, mi in raw_preds:
                        if ptype == 'unet':
                            step_preds.append(pred_data[min(h, pred_data.shape[0]-1)])
                        else:
                            step_preds.append(pred_data)
                    avg_raw = np.mean(step_preds, axis=0)
                    avg_raw[models_info[0]['static_mask']] = models_info[0]['static_vals'][models_info[0]['static_mask']]
                    avg_raw = np.clip(avg_raw, -1.0, 1.0)
                    shared_action_queue.append(avg_raw)

            raw_act = shared_action_queue.popleft()

            # Gripper hysteresis
            if raw_act[GRIPPER_DIM] >= 0.0:
                grip_hold_remaining = GRIP_HOLD
            elif grip_hold_remaining > 0:
                raw_act = raw_act.copy()
                raw_act[GRIPPER_DIM] = 1.0
                grip_hold_remaining -= 1

            env_act = np.clip(dataset_action_to_env_action(raw_act), -1.0, 1.0)
            obs, _, _, _ = env.step(env_act)

            if _any_door_open(env, th=success_threshold):
                success = True
                state_full, active_site = extract_state(obs, env, active_site)
                break

            state_full, active_site = extract_state(obs, env, active_site)

        hdist = np.linalg.norm(last_state_full[-3:]) if last_state_full is not None else 0.0
        results.append((ep_i, success, step + 1, hdist))

    env.close()
    return results


def ensemble_evaluate(ckpt_paths, n_eps=20, n_workers=4, seq_len=16,
                      max_steps=500, split='pretrain', seed=0,
                      ddim_samples=1, success_threshold=0.90):
    ep_indices = list(range(n_eps))
    batches = [ep_indices[i::n_workers] for i in range(n_workers)]
    batches = [b for b in batches if b]

    if not batches:
        print('  No episodes to evaluate (n_eps=0).', flush=True)
        return 0.0

    worker_args = [
        (ckpt_paths, batch, seq_len, max_steps, split, seed, ddim_samples, success_threshold)
        for batch in batches
    ]
    print(f'  Launching {len(batches)} ensemble workers ({len(ckpt_paths)} models each)...', flush=True)
    ctx = mp.get_context('spawn')
    with ctx.Pool(len(batches)) as pool:
        all_results = pool.map(_ensemble_worker, worker_args)

    flat = sorted([item for batch_res in all_results for item in batch_res],
                  key=lambda x: x[0])
    successes = []
    for ep_i, success, steps, hdist in flat:
        successes.append(success)
        print(f'  Ep {ep_i+1:2d}/{n_eps}: {"OK" if success else "X "}  '
              f'steps={steps:3d}  handle_dist={hdist:.3f}m', flush=True)

    sr = sum(successes) / n_eps
    print(f'\nResult: {sum(successes)}/{n_eps} ({sr*100:.1f}%)', flush=True)
    return sr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--n_eps', type=int, default=20)
    parser.add_argument('--n_eval_workers', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--split', default='pretrain')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ddim_samples', type=int, default=1)
    parser.add_argument('--success_threshold', type=float, default=0.90)
    args = parser.parse_args()

    ensemble_evaluate(
        ckpt_paths=args.checkpoints,
        n_eps=args.n_eps,
        n_workers=args.n_eval_workers,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        split=args.split,
        seed=args.seed,
        ddim_samples=args.ddim_samples,
        success_threshold=args.success_threshold,
    )
