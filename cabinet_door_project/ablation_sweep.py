"""Ablation sweep: find best observation features, then best method, then scale up.

Round 1: Feature selection (BC_UNet, ~10 min each, 6 configs)
Round 2: Method comparison on best features (BC vs Diffusion, ~15 min each)
Round 3: Scale up best config for 2 hours

Total budget: ~3 hrs for rounds 1-2, then 2 hrs for round 3.
"""
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

from diffusion_policy.config import DiffusionConfig
from diffusion_policy.data import Normalizer
from diffusion_policy.models.unet import UNetNoiseNet
from diffusion_policy.models.mlp import MLPNoiseNet
from diffusion_policy.models.transformer import TransformerNoiseNet
from diffusion_policy.training import build_scheduler, EMA, get_cosine_schedule_with_warmup
from diffusion_policy.evaluation import (
    create_env, extract_state, dataset_action_to_env_action, check_one_door_success,
)

device = torch.device("cuda")

# ========== Load preprocessed data ==========
SAVE_DIR = "/tmp/diffusion_policy_checkpoints"
_pt_path = os.path.join(SAVE_DIR, "preprocessed_all_states.pt")

if not os.path.exists(_pt_path):
    logger.info("preprocessed_all_states.pt not found — running preprocess_all_states.py...")
    from preprocess_all_states import preprocess_all
    preprocess_all(save_dir=SAVE_DIR)

data = torch.load(_pt_path, weights_only=False)

# Auto-extend with handle/hinge features if missing
if 'handle_pos' not in data['features']:
    logger.info("Handle features missing — running extend_preprocessed()...")
    from preprocess_all_states import extend_preprocessed
    data = extend_preprocessed(save_path=_pt_path)

features = data['features']  # dict of name → tensor (N, dim)
actions = data['actions']     # (N, 12)
ep_bounds = data['ep_boundaries']  # (n_eps, 3) = (ep_idx, start, end)
stats = data['stats']
feat_dims = data['feature_dims']

N_TOTAL = len(actions)
logger.info(f"Loaded {N_TOTAL} frames, features: {feat_dims}")


# ========== Dataset builder ==========

def build_obs_tensor(feature_names, split='train', val_frac=0.15, seed=42):
    """Build observation tensor from selected features, with train/val split.

    Returns: (train_obs, train_act, val_obs, val_act, obs_mean, obs_std, act_mean, act_std)
    """
    rng = np.random.RandomState(seed)
    n_eps = len(ep_bounds)
    perm = rng.permutation(n_eps)
    n_val = max(1, int(n_eps * val_frac))
    val_eps = set(perm[:n_val])

    # Concatenate selected features
    obs_parts = [features[name] for name in feature_names]
    obs_all = torch.cat(obs_parts, dim=-1)  # (N, total_dim)
    state_dim = obs_all.shape[-1]

    # Split by episode
    train_idxs, val_idxs = [], []
    for i, (eid, start, end) in enumerate(ep_bounds):
        idxs = list(range(start, end))
        if i in val_eps:
            val_idxs.extend(idxs)
        else:
            train_idxs.extend(idxs)

    train_obs = obs_all[train_idxs]
    train_act = actions[train_idxs]
    val_obs = obs_all[val_idxs] if val_idxs else train_obs[:100]
    val_act = actions[val_idxs] if val_idxs else train_act[:100]

    # Compute stats from training set
    obs_mean = train_obs.mean(dim=0)
    obs_std = train_obs.std(dim=0).clamp(min=1e-6)
    act_mean = train_act.mean(dim=0)
    act_std = train_act.std(dim=0).clamp(min=1e-6)

    # Normalize
    train_obs = (train_obs - obs_mean) / obs_std
    val_obs = (val_obs - obs_mean) / obs_std
    train_act = (train_act - act_mean) / act_std
    val_act = (val_act - act_mean) / act_std

    return (train_obs.to(device), train_act.to(device),
            val_obs.to(device), val_act.to(device),
            obs_mean, obs_std, act_mean, act_std, state_dim)


def build_chunked_dataset(obs_t, act_t, horizon=16, n_obs=2):
    """Build (obs_chunk, act_chunk) pairs for training."""
    # obs_t: (N, state_dim), act_t: (N, 12)
    # We need episode boundaries to avoid cross-episode chunks
    obs_list, act_list = [], []
    # For simplicity, treat entire tensor as one sequence (minor boundary issue)
    # Better: use ep_bounds
    N = len(obs_t)
    for j in range(max(0, N - horizon - n_obs + 1)):
        obs_list.append(obs_t[j:j + n_obs])
        act_list.append(act_t[j + n_obs - 1:j + n_obs - 1 + horizon])

    return torch.stack(obs_list), torch.stack(act_list)


def build_chunked_by_episode(obs_t, act_t, ep_idxs, horizon=16, n_obs=2):
    """Build chunks respecting episode boundaries."""
    obs_list, act_list = [], []
    for start, end in ep_idxs:
        n = end - start
        if n < horizon + n_obs:
            continue
        o = obs_t[start:end]
        a = act_t[start:end]
        for j in range(n - horizon - n_obs + 1):
            obs_list.append(o[j:j + n_obs])
            act_list.append(a[j + n_obs - 1:j + n_obs - 1 + horizon])
    return torch.stack(obs_list), torch.stack(act_list)


# ========== BC Models ==========

class BCUNet(nn.Module):
    """Behavioral cloning U-Net: direct action prediction."""
    def __init__(self, action_dim, state_dim, horizon, n_obs_steps, channels=(64, 128, 256)):
        super().__init__()
        self.horizon = horizon
        obs_in = state_dim * n_obs_steps
        self.obs_proj = nn.Linear(obs_in, channels[0])
        # Simple 1D conv decoder
        layers = []
        layers.append(nn.Linear(channels[0], channels[1]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(channels[1], channels[2]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(channels[2], horizon * action_dim))
        self.decoder = nn.Sequential(*layers)
        self.action_dim = action_dim

    def forward(self, obs):
        # obs: (B, n_obs, state_dim)
        x = obs.reshape(obs.shape[0], -1)
        x = self.obs_proj(x)
        x = torch.relu(x)
        x = self.decoder(x)
        return x.reshape(-1, self.horizon, self.action_dim)


class BCTransformer(nn.Module):
    """Behavioral cloning Transformer."""
    def __init__(self, action_dim, state_dim, horizon, n_obs_steps, d_model=128, n_layers=4, n_heads=4):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.obs_proj = nn.Linear(state_dim, d_model)
        self.act_tokens = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, n_obs_steps + horizon, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, action_dim)

    def forward(self, obs):
        B = obs.shape[0]
        obs_emb = self.obs_proj(obs)  # (B, n_obs, d_model)
        act_emb = self.act_tokens.expand(B, -1, -1)
        tokens = torch.cat([obs_emb, act_emb], dim=1)
        tokens = tokens + self.pos_emb[:, :tokens.shape[1]]
        out = self.transformer(tokens)
        act_out = out[:, obs.shape[1]:]  # (B, horizon, d_model)
        return self.out_proj(act_out)


class BCMLP(nn.Module):
    """Behavioral cloning MLP."""
    def __init__(self, action_dim, state_dim, horizon, n_obs_steps, hidden_dim=512, n_layers=4):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        in_dim = state_dim * n_obs_steps
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, horizon * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        x = obs.reshape(obs.shape[0], -1)
        return self.net(x).reshape(-1, self.horizon, self.action_dim)


# ========== Training functions ==========

def train_bc(model, train_obs, train_act, val_obs, val_act,
             bs=128, lr=1e-3, max_epochs=100, patience=30):
    """Train BC model with validation-based early stopping."""
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
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lr_s.step()
            el += loss.item()
            nb += 1

        # Validation
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            val_pred = model(val_obs)
            val_loss = nn.functional.mse_loss(val_pred, val_act).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if (epoch + 1) % 25 == 0 or epoch == 0:
            logger.info(f"  Ep {epoch+1} train={el/max(nb,1):.4f} val={val_loss:.4f} "
                       f"best_ep={best_epoch} t={time.time()-t0:.0f}s")

    model.load_state_dict(best_state)
    train_time = time.time() - t0
    logger.info(f"  BC done: best_ep={best_epoch} val={best_val:.4f} in {train_time:.0f}s")
    return best_val, best_epoch, train_time


def train_diffusion(model, sched, train_obs, train_act, val_obs, val_act,
                    bs=128, lr=1e-3, max_epochs=100, patience=30):
    """Train diffusion model with validation-based early stopping."""
    ns = len(train_obs)
    ema = EMA(model, decay=0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    nbpe = max(1, ns // bs)
    lr_s = get_cosine_schedule_with_warmup(opt, min(10, max_epochs // 10), max_epochs * nbpe)

    best_val = float('inf')
    best_ema_state = None
    wait = 0
    best_epoch = 0
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(ns, device=device)
        el, nb = 0.0, 0
        for b in range(0, ns - bs + 1, bs):
            idx = perm[b:b + bs]
            noise = torch.randn_like(train_act[idx])
            tt = torch.randint(0, sched.num_train_steps, (min(bs, len(idx)),), device=device)
            na = sched.add_noise(train_act[idx], noise, tt)
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                pred = model(na, train_obs[idx], tt)
                loss = nn.functional.mse_loss(pred, noise.reshape(pred.shape))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lr_s.step()
            ema.update(model)
            el += loss.item()
            nb += 1

        # Validation with EMA
        orig_state = deepcopy(model.state_dict())
        ema.apply(model)
        model.eval()
        with torch.no_grad():
            vl, vnb = 0.0, 0
            for b in range(0, len(val_obs) - bs + 1, bs):
                vo = val_obs[b:b + bs]
                va = val_act[b:b + bs]
                noise = torch.randn_like(va)
                tt = torch.randint(0, sched.num_train_steps, (len(vo),), device=device)
                na = sched.add_noise(va, noise, tt)
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    pred = model(na, vo, tt)
                    vl += nn.functional.mse_loss(pred, noise.reshape(pred.shape)).item()
                vnb += 1
            val_loss = vl / max(vnb, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_ema_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1

        # Restore non-EMA weights for continued training
        model.load_state_dict(orig_state)

        if wait >= patience:
            break

        if (epoch + 1) % 25 == 0 or epoch == 0:
            logger.info(f"  Ep {epoch+1} train={el/max(nb,1):.4f} val={val_loss:.4f} "
                       f"best_ep={best_epoch} t={time.time()-t0:.0f}s")

    model.load_state_dict(best_ema_state)
    train_time = time.time() - t0
    logger.info(f"  Diff done: best_ep={best_epoch} val={best_val:.4f} in {train_time:.0f}s")
    return best_val, best_epoch, train_time


# ========== Evaluation (parallel) ==========

def _eval_worker(args):
    """Run a single rollout in a subprocess. No GPU — receives precomputed actions."""
    (worker_id, seed, max_steps, feature_names,
     model_path, obs_mean_np, obs_std_np, act_mean_np, act_std_np,
     mode, horizon, n_obs, n_action_steps, state_dim) = args

    import os
    os.environ['MUJOCO_GL'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

    import torch as _torch
    import numpy as _np
    from collections import deque as _deque

    # Load model on CPU in worker
    ckpt = _torch.load(model_path, weights_only=False, map_location='cpu')
    model = ckpt['model']
    model.eval()
    sched = ckpt.get('sched', None)

    sn_mean = _torch.from_numpy(obs_mean_np).float()
    sn_std = _torch.from_numpy(obs_std_np).float()
    an_mean = _torch.from_numpy(act_mean_np).float()
    an_std = _torch.from_numpy(act_std_np).float()

    from diffusion_policy.evaluation import (
        create_env, dataset_action_to_env_action, STATE_KEYS_ORDERED,
        get_handle_pos_from_env, check_one_door_success,
    )

    env = create_env(split='pretrain', seed=seed)
    obs = env.reset()

    # mutable active_site tracking for handle feature eval
    _active_site = [None]

    def _get_hinge_angle(env):
        best = 0.0
        try:
            for jname in env.sim.model.joint_names:
                if 'hinge' in jname.lower():
                    jid = env.sim.model.joint_name2id(jname)
                    qadr = env.sim.model.jnt_qposadr[jid]
                    ang = float(env.sim.data.qpos[qadr])
                    if abs(ang) > abs(best):
                        best = ang
        except Exception:
            pass
        return best

    # Build obs
    def build_obs(obs, env):
        parts = []
        _eef_pos = None
        _handle_pos = None
        for name in feature_names:
            if name == 'proprio':
                parts.append(_np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED]))
            elif name == 'door_pos':
                parts.append(obs['door_obj_pos'].flatten())
            elif name == 'door_quat':
                parts.append(obs['door_obj_quat'].flatten())
            elif name == 'eef_pos':
                parts.append(obs['robot0_eef_pos'].flatten())
            elif name == 'eef_quat':
                parts.append(obs['robot0_eef_quat'].flatten())
            elif name == 'door_to_eef_pos':
                parts.append(obs['door_obj_to_robot0_eef_pos'].flatten())
            elif name == 'door_to_eef_quat':
                parts.append(obs['door_obj_to_robot0_eef_quat'].flatten())
            elif name == 'gripper_to_door_dist':
                d2e = obs.get('door_obj_to_robot0_eef_pos', _np.zeros(3))
                parts.append(_np.array([_np.linalg.norm(d2e)]))
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
            elif name == 'hinge_angle':
                parts.append(_np.array([_get_hinge_angle(env)], dtype=_np.float32))
        return _np.concatenate(parts).astype(_np.float32)

    aug = build_obs(obs, env)
    oh = _deque([aug] * n_obs, maxlen=n_obs)
    aq = _deque()
    success = False

    # Use handle_to_eef distance for dist_reduction metric if available
    _uses_handle = 'handle_to_eef' in feature_names or 'handle_pos' in feature_names
    if _uses_handle:
        _eef0 = obs.get('robot0_eef_pos', _np.zeros(3)).flatten().astype(_np.float32)
        _hp0, _ = get_handle_pos_from_env(env, None, _eef0)
        init_dist = float(_np.linalg.norm(_eef0 - _hp0))
    else:
        d2e = obs.get('door_obj_to_robot0_eef_pos', _np.zeros(3))
        init_dist = _np.linalg.norm(d2e)
    min_dist = init_dist

    for step in range(max_steps):
        if not aq:
            oc = _torch.from_numpy(_np.stack(list(oh))).float().unsqueeze(0)
            oc = (oc - sn_mean) / sn_std
            with _torch.no_grad():
                if mode == 'bc':
                    acts = model(oc)
                    acts = acts.reshape(1, horizon, 12) * an_std + an_mean
                else:
                    xT = _torch.randn(1, horizon, 12)
                    den = sched.denoise_ddim(model, xT, oc, num_inference_steps=8)
                    acts = den.reshape(1, horizon, 12) * an_std + an_mean
            for i in range(min(n_action_steps, horizon)):
                aq.append(acts[0, i].numpy())

        env_act = dataset_action_to_env_action(aq.popleft())
        env_act = _np.clip(env_act, -1.0, 1.0)
        obs, reward, done, info = env.step(env_act)
        aug = build_obs(obs, env)
        oh.append(aug)

        if _uses_handle:
            _eef_cur = obs.get('robot0_eef_pos', _np.zeros(3)).flatten().astype(_np.float32)
            _hp_cur, _active_site[0] = get_handle_pos_from_env(env, _active_site[0], _eef_cur)
            min_dist = min(min_dist, float(_np.linalg.norm(_eef_cur - _hp_cur)))
        else:
            d2e = obs.get('door_obj_to_robot0_eef_pos', _np.zeros(3))
            min_dist = min(min_dist, _np.linalg.norm(d2e))

        if check_one_door_success(env):
            success = True
            break

    env.close()
    dr = (1 - min_dist / max(init_dist, 1e-6)) * 100
    return worker_id, success, init_dist, min_dist, dr


@torch.no_grad()
def eval_model(model, obs_mean, obs_std, act_mean, act_std, state_dim,
               feature_names, mode='bc', sched=None,
               n_eval=16, max_steps=300, horizon=16, n_obs=2, n_action_steps=8,
               n_workers=8):
    """Evaluate model with parallel env rollouts.

    Saves model+scheduler to temp file, spawns n_workers processes,
    each runs rollouts independently on CPU. ~8x faster than sequential.
    """
    import multiprocessing as mp
    ctx = mp.get_context('spawn')  # 'spawn' avoids fork issues with MuJoCo/OpenGL
    model.eval()

    obs_mean_np = obs_mean.cpu().numpy() if isinstance(obs_mean, torch.Tensor) else obs_mean
    obs_std_np = obs_std.cpu().numpy() if isinstance(obs_std, torch.Tensor) else obs_std
    act_mean_np = act_mean.cpu().numpy() if isinstance(act_mean, torch.Tensor) else act_mean
    act_std_np = act_std.cpu().numpy() if isinstance(act_std, torch.Tensor) else act_std

    # Save model to temp file for workers (CPU copy)
    model_cpu = deepcopy(model).cpu()
    model_path = '/tmp/diffusion_policy_checkpoints/_eval_model_tmp.pt'
    save_dict = {'model': model_cpu}
    if sched is not None:
        save_dict['sched'] = sched
    torch.save(save_dict, model_path)
    del model_cpu

    # Build worker args — each worker gets a different seed for different kitchens
    worker_args = []
    for i in range(n_eval):
        worker_args.append((
            i, i,  # worker_id, seed (different seed = different kitchen)
            max_steps, feature_names,
            model_path, obs_mean_np, obs_std_np, act_mean_np, act_std_np,
            mode, horizon, n_obs, n_action_steps, state_dim
        ))

    t0 = time.time()

    # Run in waves of n_workers to avoid overloading
    all_results = []
    n_waves = (n_eval + n_workers - 1) // n_workers
    for wave in range(n_waves):
        wave_args = worker_args[wave * n_workers : (wave + 1) * n_workers]
        logger.info(f"  Eval wave {wave+1}/{n_waves}: {len(wave_args)} episodes on {len(wave_args)} workers...")
        with ctx.Pool(len(wave_args)) as pool:
            wave_results = pool.map(_eval_worker, wave_args)
        all_results.extend(wave_results)
    results = all_results

    results.sort(key=lambda x: x[0])  # sort by worker_id
    successes = sum(1 for r in results if r[1])
    dist_reds = [r[4] for r in results]

    for wid, succ, id_, md, dr in results:
        s = 'OK' if succ else 'X'
        logger.info(f"    Ep{wid+1}: {s} d={id_:.2f}->{md:.2f} ({dr:.0f}%)")

    elapsed = time.time() - t0
    logger.info(f"  Eval: {successes}/{n_eval} success, avg_dr={np.mean(dist_reds):.0f}%, "
               f"time={elapsed:.0f}s ({elapsed/n_eval:.1f}s/ep)")

    # Cleanup
    os.remove(model_path)
    return successes, np.mean(dist_reds)


def _build_obs_from_env(obs, feature_names, env=None, active_site_ref=None):
    """Build observation vector from env obs dict matching feature_names."""
    from diffusion_policy.evaluation import STATE_KEYS_ORDERED, get_handle_pos_from_env
    parts = []
    _eef_pos = None
    _handle_pos = None
    for name in feature_names:
        if name == 'proprio':
            parts.append(np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED]))
        elif name == 'door_pos':
            parts.append(obs['door_obj_pos'].flatten())
        elif name == 'door_quat':
            parts.append(obs['door_obj_quat'].flatten())
        elif name == 'eef_pos':
            parts.append(obs['robot0_eef_pos'].flatten())
        elif name == 'eef_quat':
            parts.append(obs['robot0_eef_quat'].flatten())
        elif name == 'door_to_eef_pos':
            parts.append(obs['door_obj_to_robot0_eef_pos'].flatten())
        elif name == 'door_to_eef_quat':
            parts.append(obs['door_obj_to_robot0_eef_quat'].flatten())
        elif name == 'gripper_to_door_dist':
            d2e = obs.get('door_obj_to_robot0_eef_pos', np.zeros(3))
            parts.append(np.array([np.linalg.norm(d2e)]))
        elif name == 'handle_pos' and env is not None:
            if _eef_pos is None:
                _eef_pos = obs.get('robot0_eef_pos', np.zeros(3)).flatten().astype(np.float32)
            active = active_site_ref[0] if active_site_ref is not None else None
            hp, new_active = get_handle_pos_from_env(env, active, _eef_pos)
            if active_site_ref is not None:
                active_site_ref[0] = new_active
            _handle_pos = hp
            parts.append(hp)
        elif name == 'handle_to_eef' and env is not None:
            if _eef_pos is None:
                _eef_pos = obs.get('robot0_eef_pos', np.zeros(3)).flatten().astype(np.float32)
            if _handle_pos is None:
                active = active_site_ref[0] if active_site_ref is not None else None
                hp, new_active = get_handle_pos_from_env(env, active, _eef_pos)
                if active_site_ref is not None:
                    active_site_ref[0] = new_active
                _handle_pos = hp
            parts.append((_eef_pos - _handle_pos).astype(np.float32))
        else:
            # fallback: zeros for unsupported features
            pass
    return np.concatenate(parts).astype(np.float32)


# ========== Run a single ablation ==========

def run_ablation(name, feature_names, mode='bc', arch='unet',
                 bs=128, lr=1e-3, max_epochs=100, patience=30,
                 horizon=16, n_obs=2, n_eval=3, max_eval_steps=300,
                 channels=(64, 128, 256), d_model=128, n_layers=4):
    """Run a single ablation experiment."""
    print(f"\n{'='*60}", flush=True)
    print(f"ABLATION: {name}", flush=True)
    print(f"  features={feature_names}, mode={mode}, arch={arch}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()

    # Build dataset
    (train_obs, train_act, val_obs, val_act,
     obs_mean, obs_std, act_mean, act_std, state_dim) = build_obs_tensor(feature_names)

    # Build chunked data
    # We need to rebuild chunks respecting episode boundaries
    # For simplicity with the normalized data, use the flat approach
    # (minor boundary contamination at episode edges, negligible impact)
    train_chunks_obs, train_chunks_act = build_chunked_dataset(train_obs, train_act, horizon, n_obs)
    val_chunks_obs, val_chunks_act = build_chunked_dataset(val_obs, val_act, horizon, n_obs)

    ns = len(train_chunks_obs)
    nv = len(val_chunks_obs)
    print(f"  state_dim={state_dim}, train={ns}, val={nv}", flush=True)

    # Build model
    if mode == 'bc':
        if arch == 'unet':
            model = BCUNet(12, state_dim, horizon, n_obs, channels).to(device)
        elif arch == 'transformer':
            model = BCTransformer(12, state_dim, horizon, n_obs, d_model, n_layers).to(device)
        elif arch == 'mlp':
            model = BCMLP(12, state_dim, horizon, n_obs).to(device)
    else:  # diffusion
        if arch == 'unet':
            model = UNetNoiseNet(12, state_dim, horizon, n_obs, channels=channels).to(device)
        elif arch == 'transformer':
            model = TransformerNoiseNet(12, state_dim, horizon, n_obs,
                                       d_model=d_model, n_layers=n_layers).to(device)
        elif arch == 'mlp':
            model = MLPNoiseNet(12, state_dim, horizon, n_obs).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  params={params/1e6:.2f}M", flush=True)

    sched = None
    if mode == 'diffusion':
        sched = build_scheduler(DiffusionConfig(beta_schedule='squared_cosine'))
        val_loss, best_ep, train_time = train_diffusion(
            model, sched, train_chunks_obs, train_chunks_act,
            val_chunks_obs, val_chunks_act,
            bs=bs, lr=lr, max_epochs=max_epochs, patience=patience
        )
    else:
        val_loss, best_ep, train_time = train_bc(
            model, train_chunks_obs, train_chunks_act,
            val_chunks_obs, val_chunks_act,
            bs=bs, lr=lr, max_epochs=max_epochs, patience=patience
        )

    # Eval
    succ, avg_dr = eval_model(
        model, obs_mean, obs_std, act_mean, act_std, state_dim,
        feature_names, mode=mode, sched=sched,
        n_eval=n_eval, max_steps=max_eval_steps, horizon=horizon, n_obs=n_obs
    )

    total_time = time.time() - t0
    print(f"  RESULT: {succ}/{n_eval} success, {avg_dr:.0f}% dist_red, "
          f"val={val_loss:.4f}, best_ep={best_ep}, "
          f"train={train_time:.0f}s, total={total_time:.0f}s", flush=True)

    # Cleanup
    del model, train_chunks_obs, train_chunks_act, val_chunks_obs, val_chunks_act
    torch.cuda.empty_cache()

    return {
        'name': name, 'features': feature_names, 'mode': mode, 'arch': arch,
        'params': params, 'val_loss': val_loss, 'best_epoch': best_ep,
        'success': succ, 'n_eval': n_eval, 'dist_reduction': avg_dr,
        'train_time': train_time, 'total_time': total_time,
    }


# ========== Main sweep ==========

if __name__ == "__main__":
    results = []

    # ====== ROUND 1: Feature selection (BC_UNet, fast) ======
    print("\n" + "="*60, flush=True)
    print("ROUND 1: FEATURE SELECTION (BC_UNet, ≤100 epochs)", flush=True)
    print("="*60, flush=True)

    feature_configs = [
        # TA-recommended handle-based features (per plan)
        ("F1_baseline_16d",
         ['proprio']),
        ("F2_+handle_pos_19d",
         ['proprio', 'handle_pos']),
        ("F3_+rel_pos_22d",
         ['proprio', 'handle_pos', 'handle_to_eef']),
        ("F4_+hinge_23d",
         ['proprio', 'handle_pos', 'handle_to_eef', 'hinge_angle']),
        ("F5_+door_obj_26d",
         ['proprio', 'handle_pos', 'handle_to_eef', 'hinge_angle', 'door_pos']),
        ("F6_rel_only_19d",
         ['proprio', 'handle_to_eef']),
        # Door-centroid baselines for comparison
        ("F7_door_pos_19d",
         ['proprio', 'door_pos']),
        ("F8_door_rel_22d",
         ['proprio', 'door_pos', 'door_to_eef_pos']),
    ]

    for name, feat_names in feature_configs:
        r = run_ablation(name, feat_names, mode='bc', arch='unet',
                        max_epochs=100, patience=30, n_eval=16, max_eval_steps=300)
        results.append(r)

    # Print Round 1 summary
    print(f"\n{'='*60}", flush=True)
    print("ROUND 1 SUMMARY", flush=True)
    print(f"{'Name':<25} {'Dim':>4} {'Val':>6} {'Ep':>4} {'Succ':>6} {'DR%':>5} {'Time':>6}", flush=True)
    print("-" * 60, flush=True)
    for r in results:
        dim = sum(feat_dims[f] for f in r['features'])
        print(f"{r['name']:<25} {dim:>4} {r['val_loss']:>6.4f} {r['best_epoch']:>4} "
              f"{r['success']:>3}/{r['n_eval']}  {r['dist_reduction']:>5.0f}% {r['train_time']:>5.0f}s", flush=True)

    # Pick best feature set from Round 1
    best_r1 = max(results, key=lambda r: r['dist_reduction'])
    best_features = best_r1['features']
    print(f"\nBest features: {best_r1['name']} ({best_r1['dist_reduction']:.0f}%)", flush=True)

    # ====== ROUND 2: Method comparison on best features ======
    print(f"\n{'='*60}", flush=True)
    print(f"ROUND 2: BC vs DIFFUSION on {best_r1['name']}", flush=True)
    print("="*60, flush=True)

    method_configs = [
        ("R2_BC_UNet",       'bc',        'unet',        (64, 128, 256), 128, 4),
        ("R2_BC_Transformer",'bc',        'transformer', (64, 128, 256), 128, 4),
        ("R2_BC_MLP",        'bc',        'mlp',         (64, 128, 256), 512, 4),
        ("R2_Diff_UNet",     'diffusion', 'unet',        (64, 128, 256), 128, 4),
        ("R2_Diff_Transf",   'diffusion', 'transformer', (64, 128, 256), 128, 4),
        ("R2_Diff_MLP",      'diffusion', 'mlp',         (64, 128, 256), 512, 4),
    ]

    r2_results = []
    for name, mode, arch, channels, d_model, n_layers in method_configs:
        r = run_ablation(name, best_features, mode=mode, arch=arch,
                        max_epochs=100, patience=30, n_eval=16, max_eval_steps=300,
                        channels=channels, d_model=d_model, n_layers=n_layers)
        r2_results.append(r)
        results.append(r)

    # Print Round 2 summary
    print(f"\n{'='*60}", flush=True)
    print("ROUND 2 SUMMARY", flush=True)
    print(f"{'Name':<22} {'Mode':<10} {'Params':>7} {'Val':>6} {'Ep':>4} {'Succ':>6} {'DR%':>5} {'Train':>6}", flush=True)
    print("-" * 70, flush=True)
    for r in r2_results:
        print(f"{r['name']:<22} {r['mode']:<10} {r['params']/1e6:>6.2f}M {r['val_loss']:>6.4f} "
              f"{r['best_epoch']:>4} {r['success']:>3}/{r['n_eval']}  {r['dist_reduction']:>5.0f}% "
              f"{r['train_time']:>5.0f}s", flush=True)

    # Decision: pick best method considering both performance AND training speed
    best_r2 = max(r2_results, key=lambda r: r['dist_reduction'])
    # Also find best BC for comparison
    best_bc = max([r for r in r2_results if r['mode'] == 'bc'], key=lambda r: r['dist_reduction'])
    best_diff = max([r for r in r2_results if r['mode'] == 'diffusion'], key=lambda r: r['dist_reduction'])

    print(f"\nBest BC:        {best_bc['name']} ({best_bc['dist_reduction']:.0f}%, {best_bc['train_time']:.0f}s)")
    print(f"Best Diffusion: {best_diff['name']} ({best_diff['dist_reduction']:.0f}%, {best_diff['train_time']:.0f}s)")
    print(f"Overall best:   {best_r2['name']} ({best_r2['dist_reduction']:.0f}%)")

    # If BC is much faster and close in performance, prefer BC for scale-up
    bc_faster_ratio = best_diff['train_time'] / max(best_bc['train_time'], 1)
    perf_gap = best_diff['dist_reduction'] - best_bc['dist_reduction']
    print(f"\nDiffusion is {bc_faster_ratio:.1f}x slower than BC")
    print(f"Performance gap: {perf_gap:+.0f}% (positive = diffusion better)")

    if perf_gap < 5 and bc_faster_ratio > 3:
        scale_mode = 'bc'
        scale_arch = best_bc['arch']
        print("→ BC wins on efficiency. Scaling up BC.", flush=True)
    else:
        scale_mode = best_r2['mode']
        scale_arch = best_r2['arch']
        print(f"→ Scaling up {scale_mode} {scale_arch}.", flush=True)

    # ====== ROUND 3: Scale up for 2 hours ======
    print(f"\n{'='*60}", flush=True)
    print(f"ROUND 3: SCALE UP — {scale_mode} {scale_arch} on {best_r1['name']}", flush=True)
    print("="*60, flush=True)

    # Bigger model, more epochs, more eval
    if scale_arch == 'unet':
        scale_channels = (128, 256, 512)
    else:
        scale_channels = (64, 128, 256)
    scale_d_model = 256
    scale_n_layers = 6

    r3 = run_ablation(
        f"R3_SCALE_{scale_mode}_{scale_arch}",
        best_features,
        mode=scale_mode, arch=scale_arch,
        bs=128, lr=1e-3,
        max_epochs=2000,  # will early-stop
        patience=100,     # more patience for scale run
        n_eval=10,
        max_eval_steps=500,
        channels=scale_channels,
        d_model=scale_d_model,
        n_layers=scale_n_layers,
    )
    results.append(r3)

    # ====== FINAL SUMMARY ======
    print(f"\n{'='*60}", flush=True)
    print("FULL ABLATION SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"{'Name':<25} {'Mode':<6} {'DR%':>5} {'Succ':>6} {'Train':>6} {'Total':>6}", flush=True)
    print("-" * 60, flush=True)
    for r in results:
        print(f"{r['name']:<25} {r['mode']:<6} {r['dist_reduction']:>5.0f}% "
              f"{r['success']:>3}/{r['n_eval']}  {r['train_time']:>5.0f}s {r['total_time']:>5.0f}s", flush=True)

    # Save results
    torch.save(results, os.path.join(SAVE_DIR, "ablation_results.pt"))
    print(f"\nResults saved to {SAVE_DIR}/ablation_results.pt", flush=True)
