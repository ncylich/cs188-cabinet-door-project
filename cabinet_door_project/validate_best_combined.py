"""BC_UNet F3 trained on combined pretrain+target data (607 demos), eval 100 episodes."""
import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
import logging
from copy import deepcopy
from collections import deque

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

SAVE_DIR = "/tmp/diffusion_policy_checkpoints"
CKPT_PATH = os.path.join(SAVE_DIR, "best_f3_bc_unet_combined.pt")
# F3 indices within the 44-dim obs_full saved by preprocess_target_parallel.py
# Layout: proprio(0:16) ... handle_pos(38:41) handle_to_eef(41:44)
TARGET_F3_INDICES = list(range(16)) + list(range(38, 44))
N_EVAL = 100
N_WORKERS = 8
MAX_STEPS = 500
FEATURE_NAMES = ['proprio', 'handle_pos', 'handle_to_eef']
HORIZON = 16
N_OBS = 2
N_ACTION_STEPS = 8


# ========== Model ==========

class BCUNet(nn.Module):
    def __init__(self, action_dim, state_dim, horizon, n_obs_steps, channels=(64, 128, 256)):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        obs_in = state_dim * n_obs_steps
        self.obs_proj = nn.Linear(obs_in, channels[0])
        self.decoder = nn.Sequential(
            nn.Linear(channels[0], channels[1]), nn.ReLU(),
            nn.Linear(channels[1], channels[2]), nn.ReLU(),
            nn.Linear(channels[2], horizon * action_dim),
        )

    def forward(self, obs):
        x = obs.reshape(obs.shape[0], -1)
        x = torch.relu(self.obs_proj(x))
        return self.decoder(x).reshape(-1, self.horizon, self.action_dim)


# ========== Eval worker (must be module-level for spawn) ==========

def _eval_worker(args):
    (worker_id, seed, max_steps, feature_names,
     model_path, obs_mean_np, obs_std_np, act_mean_np, act_std_np,
     horizon, n_obs, n_action_steps) = args

    os.environ['MUJOCO_GL'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

    import torch as _torch
    import numpy as _np
    from collections import deque as _deque
    from diffusion_policy.evaluation import (
        create_env, dataset_action_to_env_action, STATE_KEYS_ORDERED,
        get_handle_pos_from_env, check_one_door_success,
    )

    ckpt = _torch.load(model_path, weights_only=False, map_location='cpu')
    model = ckpt['model']
    model.eval()

    sn_mean = _torch.from_numpy(obs_mean_np).float()
    sn_std  = _torch.from_numpy(obs_std_np).float()
    an_mean = _torch.from_numpy(act_mean_np).float()
    an_std  = _torch.from_numpy(act_std_np).float()

    env = create_env(split='pretrain', seed=seed)
    obs = env.reset()

    _active_site = [None]

    def build_obs(obs, env):
        parts = []
        _eef_pos = None
        _handle_pos = None
        for name in feature_names:
            if name == 'proprio':
                parts.append(_np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED]))
            elif name == 'handle_pos':
                if _eef_pos is None:
                    _eef_pos = obs.get('robot0_eef_pos', _np.zeros(3)).flatten().astype(_np.float32)
                hp, _active_site[0] = get_handle_pos_from_env(env, _active_site[0], _eef_pos)
                _handle_pos = hp
                parts.append(hp)
            elif name == 'handle_to_eef':
                if _eef_pos is None:
                    _eef_pos = obs.get('robot0_eef_pos', _np.zeros(3)).flatten().astype(_np.float32)
                if _handle_pos is None:
                    hp, _active_site[0] = get_handle_pos_from_env(env, _active_site[0], _eef_pos)
                    _handle_pos = hp
                parts.append((_eef_pos - _handle_pos).astype(_np.float32))
        return _np.concatenate(parts).astype(_np.float32)

    aug = build_obs(obs, env)
    oh = _deque([aug] * n_obs, maxlen=n_obs)
    aq = _deque()
    success = False

    _eef0 = obs.get('robot0_eef_pos', _np.zeros(3)).flatten().astype(_np.float32)
    _hp0, _ = get_handle_pos_from_env(env, None, _eef0)
    init_dist = float(_np.linalg.norm(_eef0 - _hp0))
    min_dist = init_dist

    for step in range(max_steps):
        if not aq:
            oc = _torch.from_numpy(_np.stack(list(oh))).float().unsqueeze(0)
            oc = (oc - sn_mean) / sn_std
            with _torch.no_grad():
                acts = model(oc).reshape(1, horizon, 12) * an_std + an_mean
            for i in range(min(n_action_steps, horizon)):
                aq.append(acts[0, i].numpy())

        env_act = dataset_action_to_env_action(aq.popleft())
        env_act = _np.clip(env_act, -1.0, 1.0)
        obs, reward, done, info = env.step(env_act)
        aug = build_obs(obs, env)
        oh.append(aug)

        _eef_cur = obs.get('robot0_eef_pos', _np.zeros(3)).flatten().astype(_np.float32)
        _hp_cur, _active_site[0] = get_handle_pos_from_env(env, _active_site[0], _eef_cur)
        min_dist = min(min_dist, float(_np.linalg.norm(_eef_cur - _hp_cur)))

        if check_one_door_success(env):
            success = True
            break

    env.close()
    dr = (1 - min_dist / max(init_dist, 1e-6)) * 100
    return worker_id, success, init_dist, min_dist, dr


# ========== Main ==========

def main():
    from diffusion_policy.training import get_cosine_schedule_with_warmup

    device = torch.device("cuda")

    # Load pretrain data
    _pt_path = os.path.join(SAVE_DIR, "preprocessed_all_states.pt")
    data = torch.load(_pt_path, weights_only=False)
    if 'handle_pos' not in data['features']:
        from preprocess_all_states import extend_preprocessed
        data = extend_preprocessed(save_path=_pt_path)

    features_data = data['features']
    actions = data['actions']
    ep_bounds = data['ep_boundaries']

    obs_parts = [features_data[name] for name in FEATURE_NAMES]
    obs_all = torch.cat(obs_parts, dim=-1)  # (N_pretrain, 22)

    # Load target data and append (slice F3 from 44-dim obs_full)
    _tgt_path = os.path.join(SAVE_DIR, "preprocessed_target_states.pt")
    tdata = torch.load(_tgt_path, weights_only=False)
    tobs = tdata['obs_full'][:, TARGET_F3_INDICES].float()  # (N_target, 22)
    tact = tdata['actions'].float()
    tep  = tdata['ep_boundaries']
    offset = len(obs_all)
    tep_offset = tep.copy(); tep_offset[:, 1] += offset; tep_offset[:, 2] += offset
    obs_all   = torch.cat([obs_all, tobs], dim=0)
    actions   = torch.cat([actions, tact], dim=0)
    ep_bounds = np.concatenate([ep_bounds, tep_offset], axis=0)
    logger.info(f"Combined: {len(ep_bounds)} episodes, {len(obs_all):,} frames")

    state_dim = obs_all.shape[-1]
    logger.info(f"State dim: {state_dim} features: {FEATURE_NAMES}")

    # Train/val split (episode-level)
    rng = np.random.RandomState(42)
    n_eps = len(ep_bounds)
    perm = rng.permutation(n_eps)
    n_val = max(1, int(n_eps * 0.15))
    val_eps = set(perm[:n_val])

    train_idxs, val_idxs = [], []
    for i, (eid, start, end) in enumerate(ep_bounds):
        idxs = list(range(int(start), int(end)))
        if i in val_eps:
            val_idxs.extend(idxs)
        else:
            train_idxs.extend(idxs)

    train_obs_flat = obs_all[train_idxs]
    train_act_flat = actions[train_idxs]
    val_obs_flat   = obs_all[val_idxs]
    val_act_flat   = actions[val_idxs]

    obs_mean = train_obs_flat.mean(0); obs_std = train_obs_flat.std(0).clamp(min=1e-6)
    act_mean = train_act_flat.mean(0); act_std = train_act_flat.std(0).clamp(min=1e-6)

    train_obs_flat = ((train_obs_flat - obs_mean) / obs_std).to(device)
    train_act_flat = ((train_act_flat - act_mean) / act_std).to(device)
    val_obs_flat   = ((val_obs_flat   - obs_mean) / obs_std).to(device)
    val_act_flat   = ((val_act_flat   - act_mean) / act_std).to(device)

    # Build chunked dataset matching ablation_sweep.build_chunked_dataset
    def build_chunked(obs_t, act_t, horizon, n_obs):
        obs_list, act_list = [], []
        N = len(obs_t)
        for j in range(max(0, N - horizon - n_obs + 1)):
            obs_list.append(obs_t[j:j + n_obs])
            act_list.append(act_t[j + n_obs - 1:j + n_obs - 1 + horizon])
        return torch.stack(obs_list), torch.stack(act_list)

    train_obs, train_act = build_chunked(train_obs_flat, train_act_flat, HORIZON, N_OBS)
    val_obs,   val_act   = build_chunked(val_obs_flat,   val_act_flat,   HORIZON, N_OBS)
    # train_obs: (N_chunks, N_OBS, state_dim), train_act: (N_chunks, HORIZON, 12)

    logger.info(f"Train chunks: {len(train_obs)}, Val chunks: {len(val_obs)}")

    # Build model
    model = BCUNet(action_dim=12, state_dim=state_dim, horizon=HORIZON,
                   n_obs_steps=N_OBS, channels=(64, 128, 256)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"BCUNet params: {n_params:,}")

    # Train
    bs, lr, max_epochs, patience = 128, 1e-3, 100, 30
    ns = len(train_obs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    nbpe = max(1, ns // bs)
    lr_s = get_cosine_schedule_with_warmup(opt, min(10, max_epochs // 10), max_epochs * nbpe)

    best_val = float('inf')
    best_state = None
    wait = 0
    best_epoch = 0
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(ns, device=device)
        el, nb = 0.0, 0
        for b in range(0, ns - bs + 1, bs):
            idx = perm[b:b + bs]
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                pred = model(train_obs[idx])
                loss = nn.functional.mse_loss(pred, train_act[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); lr_s.step()
            el += loss.item(); nb += 1

        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            val_loss = nn.functional.mse_loss(model(val_obs), val_act).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Ep {epoch+1} train={el/max(nb,1):.4f} val={val_loss:.4f} "
                        f"best_ep={best_epoch} t={time.time()-t0:.0f}s")

    model.load_state_dict(best_state)
    logger.info(f"Training done: best_ep={best_epoch} val={best_val:.4f} in {time.time()-t0:.0f}s")

    # Save checkpoint
    torch.save({'model': deepcopy(model).cpu(),
                'obs_mean': obs_mean, 'obs_std': obs_std,
                'act_mean': act_mean, 'act_std': act_std,
                'state_dim': state_dim, 'feature_names': FEATURE_NAMES,
                'best_epoch': best_epoch, 'val_loss': best_val},
               CKPT_PATH)
    logger.info(f"Checkpoint saved to {CKPT_PATH}")

    # Eval 100 episodes
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING {N_EVAL}-EPISODE VALIDATION")
    logger.info(f"{'='*60}")

    import multiprocessing as mp
    ctx = mp.get_context('spawn')

    model_tmp_path = os.path.join(SAVE_DIR, "_validate_model_tmp.pt")
    torch.save({'model': deepcopy(model).cpu()}, model_tmp_path)

    obs_mean_np = obs_mean.numpy()
    obs_std_np  = obs_std.numpy()
    act_mean_np = act_mean.numpy()
    act_std_np  = act_std.numpy()

    worker_args = [
        (i, i, MAX_STEPS, FEATURE_NAMES,
         model_tmp_path, obs_mean_np, obs_std_np, act_mean_np, act_std_np,
         HORIZON, N_OBS, N_ACTION_STEPS)
        for i in range(N_EVAL)
    ]

    all_results = []
    n_waves = (N_EVAL + N_WORKERS - 1) // N_WORKERS
    t_eval = time.time()
    for wave in range(n_waves):
        wave_args = worker_args[wave * N_WORKERS:(wave + 1) * N_WORKERS]
        logger.info(f"  Wave {wave+1}/{n_waves} ({len(wave_args)} eps)...")
        with ctx.Pool(len(wave_args)) as pool:
            wave_results = pool.map(_eval_worker, wave_args)
        all_results.extend(wave_results)
        wave_successes = sum(1 for r in wave_results if r[1])
        running_total = sum(1 for r in all_results if r[1])
        eps_done = len(all_results)
        logger.info(f"    Wave {wave+1}: {wave_successes}/{len(wave_args)} | "
                    f"Running: {running_total}/{eps_done} ({100*running_total/eps_done:.1f}%)")

    os.remove(model_tmp_path)

    all_results.sort(key=lambda x: x[0])
    successes = sum(1 for r in all_results if r[1])
    dist_reds = [r[4] for r in all_results]

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULT: {successes}/{N_EVAL} success ({100*successes/N_EVAL:.1f}%)")
    logger.info(f"Avg dist reduction: {np.mean(dist_reds):.1f}%")
    logger.info(f"Total eval time: {time.time()-t_eval:.0f}s")
    logger.info(f"{'='*60}")
    logger.info(f"\nPer-episode breakdown:")
    for wid, succ, id_, md, dr in all_results:
        s = 'OK' if succ else 'X'
        logger.info(f"  Ep{wid+1:3d}: {s}  d={id_:.2f}->{md:.2f}  DR={dr:.0f}%")

    target = 30
    if successes >= target:
        logger.info(f"\n✓ PASSED: {successes} >= {target} (target)")
    else:
        logger.info(f"\n✗ FAILED: {successes} < {target} (target)")


if __name__ == '__main__':
    main()
