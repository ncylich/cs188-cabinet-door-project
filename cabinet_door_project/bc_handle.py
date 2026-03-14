"""BC policy with handle-relative oracle state.

Incorporates the key innovations from the best-performing baseline:
  - Actual handle site position (vs door body centroid)
  - Transformer with causal 16-step history
  - gripper_loss_weight=2.0 to counter MSE averaging of bimodal gripper
  - static_action_mask for near-constant action dims
  - Huber (smooth_l1) loss
  - Parallelized handle feature cache building

Feature set (44-dim):
  proprio (16) | door_pos (3) | door_quat (4) | eef_pos (3) | eef_quat (4)
  | door_to_eef_pos (3) | door_to_eef_quat (4) | door_dist (1)
  | handle_pos (3) | handle_to_eef_pos (3)

Usage:
    python bc_handle.py                     # train then eval 20 eps
    python bc_handle.py --eval_only         # eval last saved checkpoint
    python bc_handle.py --n_workers 6       # parallel cache (default 4)
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── paths ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

SAVE_DIR     = Path('/tmp/diffusion_policy_checkpoints')
HANDLE_CACHE = SAVE_DIR / 'handle_cache'
LEROBOT_ROOT = Path(
    '/home/noahcylich/cs188-cabinet-door-project/robocasa/datasets'
    '/v1.0/pretrain/atomic/OpenCabinet/20250819/lerobot'
)
CKPT_PATH = SAVE_DIR / 'bc_handle_best.pt'

# ── feature layout ─────────────────────────────────────────────────────────
# Concatenated in this order from preprocessed_all_states.pt + handle cache
FEAT_NAMES = [
    'proprio',              # 16
    'door_pos',             # 3
    'door_quat',            # 4
    'eef_pos',              # 3
    'eef_quat',             # 4
    'door_to_eef_pos',      # 3
    'door_to_eef_quat',     # 4
    'gripper_to_door_dist', # 1
    'handle_pos',           # 3  ← sim-extracted
    'handle_to_eef_pos',    # 3  ← eef_pos - handle_pos (world frame)
]
STATE_DIM = 44   # 16+3+4+3+4+3+4+1+3+3

# Feature index subsets for ablation experiments.
# Layout: proprio(0:16) door_pos(16:19) door_quat(19:23) eef_pos(23:26) eef_quat(26:30)
#         door_to_eef_pos(30:33) door_to_eef_quat(33:37) gripper_dist(37:38)
#         handle_pos(38:41) handle_to_eef_pos(41:44)
FEATURE_CONFIGS = {
    'full':        list(range(44)),                                      # 44-dim default
    'no_handle':   list(range(38)),                                      # 38-dim: drop handle_pos + handle_to_eef
    'handle_only': list(range(16)) + list(range(23, 30)) + list(range(37, 44)),  # 30-dim: no door centroid
}

# Action dim 11 = gripper (raw LeRobot ordering, mapped to env dim 6 at exec time)
# Action dim  3 = torso/reserve (static, std ≈ 0)
GRIPPER_DIM = 11
STATIC_STD_THRESH = 1e-5

from diffusion_policy.evaluation import (
    STATE_KEYS_ORDERED,
    dataset_action_to_env_action,
    get_handle_pos_from_env,
    compute_eef_pos_from_obs,
)
from diffusion_policy.scheduler import DDPMScheduler
from diffusion_policy.models.mlp import TimestepMLP


def create_env(split='pretrain', seed=0):
    """Headless env with no offscreen renderer for fast eval."""
    import robocasa  # noqa
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    if split == 'pretrain':
        obj_instance_split = 'pretrain'
        layout_ids, style_ids = -2, -2
    elif split == 'target':
        obj_instance_split = 'target'
        layout_ids, style_ids = None, None
    else:
        obj_instance_split = None
        layout_ids, style_ids = -3, -3

    return robosuite.make(
        env_name='OpenCabinet',
        robots='PandaOmron',
        controller_configs=load_composite_controller_config(robot='PandaOmron'),
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        layout_ids=layout_ids,
        style_ids=style_ids,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Parallel handle-cache building
# ═══════════════════════════════════════════════════════════════════════════

def _worker(args):
    """Spawn-safe worker: creates its own MuJoCo env, processes a batch of episodes."""
    ep_batch, lerobot_root, cache_dir, eef_pos_by_ep = args

    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    lerobot_root = Path(lerobot_root)
    cache_dir    = Path(cache_dir)

    import robocasa  # noqa
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    from diffusion_policy.evaluation import get_handle_site_names

    env = robosuite.make(
        env_name='OpenCabinet', robots='PandaOmron',
        controller_configs=load_composite_controller_config(robot='PandaOmron'),
        has_renderer=False, has_offscreen_renderer=False,
        ignore_done=True, use_object_obs=True, use_camera_obs=False,
        camera_depths=False, seed=0,
        obj_instance_split='pretrain', layout_ids=-2, style_ids=-2,
    )

    n_done = 0
    try:
        for ep_id, start, end in ep_batch:
            cache_path = cache_dir / f'episode_{int(ep_id):06d}.npy'
            if cache_path.exists():
                n_done += 1
                continue

            ep_dir = lerobot_root / 'extras' / f'episode_{int(ep_id):06d}'
            sim_states = np.load(ep_dir / 'states.npz')['states']
            with open(ep_dir / 'ep_meta.json') as f:
                ep_meta = json.load(f)
            with gzip.open(ep_dir / 'model.xml.gz', 'rb') as f:
                model_xml = f.read().decode('utf-8')

            if hasattr(env, 'set_ep_meta'):
                env.set_ep_meta(ep_meta)
            env.reset()
            env.reset_from_xml_string(env.edit_model_xml(model_xml))
            env.sim.reset()

            sites = get_handle_site_names(env)
            T = int(end) - int(start)
            eef_ep = eef_pos_by_ep[int(ep_id)]   # (T, 3)
            ep_handle = np.zeros((T, 3), dtype=np.float32)
            active = sites[0] if sites else None

            for i in range(min(T, len(sim_states))):
                env.sim.set_state_from_flattened(sim_states[i])
                env.sim.forward()
                if not sites:
                    continue
                eef = eef_ep[i]
                if len(sites) > 1:
                    active = min(sites, key=lambda s: np.linalg.norm(
                        env.sim.data.site_xpos[env.sim.model.site_name2id(s)] - eef))
                sid = env.sim.model.site_name2id(active)
                ep_handle[i] = env.sim.data.site_xpos[sid]

            np.save(cache_path, ep_handle)
            n_done += 1
    finally:
        env.close()

    return n_done


def build_handle_cache(eef_pos_all, ep_boundaries, n_workers=4):
    """Build per-timestep handle positions via parallel episode replay.

    Returns handle_pos_all: (N, 3) float32.
    """
    HANDLE_CACHE.mkdir(parents=True, exist_ok=True)

    # Pre-slice eef_pos per episode so workers don't carry the full array
    eef_np = eef_pos_all.numpy() if isinstance(eef_pos_all, torch.Tensor) else np.asarray(eef_pos_all)
    eef_by_ep = {int(eid): eef_np[int(s):int(e)] for eid, s, e in ep_boundaries}

    # Split episodes across workers
    ep_list = [(eid, s, e) for eid, s, e in ep_boundaries]
    batches = [ep_list[i::n_workers] for i in range(n_workers)]
    worker_args = [(b, str(LEROBOT_ROOT), str(HANDLE_CACHE), eef_by_ep) for b in batches]

    n_already = sum(1 for _, s, e in ep_list
                    if (HANDLE_CACHE / f'episode_{int(_):06d}.npy').exists())
    if n_already == len(ep_list):
        print(f"Handle cache complete ({n_already}/{len(ep_list)} episodes cached).")
    else:
        print(f"Building handle cache: {n_already}/{len(ep_list)} cached, "
              f"running {n_workers} workers in parallel...", flush=True)
        ctx = mp.get_context('spawn')
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_worker, worker_args)
        print(f"  Done. {sum(results)} episodes processed.", flush=True)

    # Assemble full array from cache files
    N = len(eef_np)
    handle_pos = np.zeros((N, 3), dtype=np.float32)
    for eid, start, end in ep_boundaries:
        p = HANDLE_CACHE / f'episode_{int(eid):06d}.npy'
        if p.exists():
            arr = np.load(p)
            handle_pos[int(start):int(end)] = arr[:int(end)-int(start)]
    return handle_pos


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_data(n_workers=4, feat_indices=None):
    data     = torch.load(SAVE_DIR / 'preprocessed_all_states.pt', weights_only=False)
    feats    = data['features']
    actions  = data['actions'].float()
    ep_bounds = data['ep_boundaries']

    handle_pos = build_handle_cache(feats['eef_pos'], ep_bounds, n_workers=n_workers)
    handle_pos_t     = torch.from_numpy(handle_pos)
    handle_to_eef_t  = feats['eef_pos'] - handle_pos_t   # world-frame vector

    obs_full = torch.cat(
        [feats[n] for n in FEAT_NAMES[:-2]] + [handle_pos_t, handle_to_eef_t],
        dim=-1,
    ).float()
    assert obs_full.shape[-1] == STATE_DIM, f"Got {obs_full.shape[-1]}, expected {STATE_DIM}"

    if feat_indices is not None and len(feat_indices) < STATE_DIM:
        obs = obs_full[:, feat_indices]
    else:
        obs = obs_full
    return obs, actions, ep_bounds


def train_val_split(obs, actions, ep_bounds, val_frac=0.10, seed=0):
    rng   = np.random.default_rng(seed)
    n_eps = len(ep_bounds)
    idx   = np.arange(n_eps)
    rng.shuffle(idx)
    n_val = max(1, int(round(n_eps * val_frac)))
    val_set = set(idx[:n_val].tolist())

    tr_i, va_i = [], []
    tr_ep_starts, va_ep_starts = [], []  # episode start positions within each split
    for i, (_, s, e) in enumerate(ep_bounds):
        if i in val_set:
            va_ep_starts.append(len(va_i))
            va_i.extend(range(int(s), int(e)))
        else:
            tr_ep_starts.append(len(tr_i))
            tr_i.extend(range(int(s), int(e)))

    return (obs[tr_i].float(), actions[tr_i].float(),
            obs[va_i].float(), actions[va_i].float(),
            tr_ep_starts, va_ep_starts)


# ═══════════════════════════════════════════════════════════════════════════
# Sequence dataset (per-step causal history windows)
# ═══════════════════════════════════════════════════════════════════════════

def build_seq_tensors(obs, actions, seq_len, ep_starts=None):
    """Return (seqs, masks, acts) respecting episode boundaries."""
    N, D = obs.shape
    seqs  = torch.zeros(N, seq_len, D)
    masks = torch.ones(N, seq_len, dtype=torch.bool)   # True = padding

    # Build per-frame episode start lookup (which episode start applies to frame i)
    if ep_starts:
        ep_starts_sorted = sorted(ep_starts)
        frame_ep_start = np.zeros(N, dtype=np.int64)
        for ep_s in ep_starts_sorted:
            frame_ep_start[ep_s:] = ep_s
    else:
        frame_ep_start = np.zeros(N, dtype=np.int64)

    for i in range(N):
        s = max(int(frame_ep_start[i]), i - seq_len + 1)
        hist = obs[s:i+1]
        L    = len(hist)
        seqs[i, -L:]  = hist
        masks[i, -L:] = False
    return seqs, masks, actions.clone()


# ═══════════════════════════════════════════════════════════════════════════
# Model  (matches Noah's TemporalBCTransformer exactly)
# ═══════════════════════════════════════════════════════════════════════════

class BCTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, seq_len=16,
                 d_model=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.seq_len   = seq_len
        self.proj      = nn.Linear(state_dim, d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder   = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm      = nn.LayerNorm(d_model)
        self.head      = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, action_dim),
        )
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, states, padding_mask):
        x = self.proj(states) + self.pos_emb[:, :states.shape[1]]
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        lengths = (~padding_mask).sum(1).clamp(min=1) - 1
        pooled  = x[torch.arange(x.shape[0], device=x.device), lengths]
        return self.head(self.norm(pooled))


class BCMLP(nn.Module):
    """Flat MLP — no temporal context. Same interface as BCTransformer."""
    def __init__(self, state_dim, action_dim, hidden=512, n_layers=3, dropout=0.1, **kwargs):
        super().__init__()
        dims = [state_dim] + [hidden] * n_layers + [action_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, states, padding_mask=None):
        # states: (B, seq_len, D) → use only the most recent obs
        x = states[:, -1] if states.dim() == 3 else states
        return self.net(x)


class BCTransformerBinaryGripper(nn.Module):
    """BCTransformer with a separate BCE classification head for the gripper dim."""
    def __init__(self, state_dim, action_dim, seq_len=16,
                 d_model=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        cont_dim = action_dim - 1   # continuous dims (all except gripper)
        self.seq_len = seq_len
        self.proj    = nn.Linear(state_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder    = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm       = nn.LayerNorm(d_model)
        self.head_cont  = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, cont_dim),
        )
        self.head_grip  = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),          # logit; apply sigmoid at inference
        )
        nn.init.normal_(self.pos_emb, std=0.02)

    def _pool(self, states, padding_mask):
        x = self.proj(states) + self.pos_emb[:, :states.shape[1]]
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        lengths = (~padding_mask).sum(1).clamp(min=1) - 1
        pooled  = x[torch.arange(x.shape[0], device=x.device), lengths]
        return self.norm(pooled)

    def forward(self, states, padding_mask):
        pooled = self._pool(states, padding_mask)
        return self.head_cont(pooled), self.head_grip(pooled)   # (B,11), (B,1)


class HandleContextEncoder(nn.Module):
    """BCTransformer encoder body without an action head — returns context vector."""
    def __init__(self, state_dim, seq_len=16, d_model=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.proj    = nn.Linear(state_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm    = nn.LayerNorm(d_model)
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, states, padding_mask):
        x = self.proj(states) + self.pos_emb[:, :states.shape[1]]
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        lengths = (~padding_mask).sum(1).clamp(min=1) - 1
        pooled  = x[torch.arange(x.shape[0], device=x.device), lengths]
        return self.norm(pooled)   # (B, d_model)


class MLPDenoiser(nn.Module):
    """3-layer MLP that predicts noise given (noisy_action, context, timestep)."""
    def __init__(self, action_dim=12, context_dim=256, hidden=512):
        super().__init__()
        self.time_embed = TimestepMLP(128, hidden)
        self.net = nn.Sequential(
            nn.Linear(action_dim + context_dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x, context, t):
        """x: (B,A) noisy action  context: (B,C)  t: (B,) → noise (B,A)"""
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, context, t_emb], dim=-1))


class TemporalDiffusionPolicy(nn.Module):
    """Diffusion policy conditioned on 16-step causal obs history.

    encode(states, mask) → context (B, d_model)
    forward(noisy_action, context, t) → predicted_noise   (DDPMScheduler interface)
    """
    def __init__(self, state_dim, action_dim=12, seq_len=16,
                 d_model=256, n_heads=8, n_layers=4, dropout=0.1, denoiser_hidden=512):
        super().__init__()
        self.action_dim = action_dim
        self.context_encoder = HandleContextEncoder(
            state_dim, seq_len, d_model, n_heads, n_layers, dropout)
        self.denoiser = MLPDenoiser(action_dim, d_model, denoiser_hidden)

    def encode(self, states, padding_mask):
        return self.context_encoder(states, padding_mask)

    def forward(self, x, context, t):
        """Denoiser interface — called by DDPMScheduler.denoise_ddim."""
        return self.denoiser(x, context, t)


def build_model(arch, state_dim, action_dim, seq_len, d_model, n_heads, n_layers, dropout,
                binary_gripper=False, denoiser_hidden=512, **_):
    if arch == 'diffusion':
        return TemporalDiffusionPolicy(state_dim, action_dim, seq_len, d_model,
                                       n_heads, n_layers, dropout, denoiser_hidden)
    if arch == 'mlp':
        return BCMLP(state_dim, action_dim, hidden=d_model, n_layers=n_layers, dropout=dropout)
    if binary_gripper:
        return BCTransformerBinaryGripper(state_dim, action_dim, seq_len,
                                          d_model, n_heads, n_layers, dropout)
    return BCTransformer(state_dim, action_dim, seq_len, d_model, n_heads, n_layers, dropout)


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train(tr_obs, tr_act, va_obs, va_act,
          tr_ep_starts=None, va_ep_starts=None,
          arch='transformer',
          seq_len=16, d_model=256, n_heads=8, n_layers=4, dropout=0.1,
          lr=3e-4, weight_decay=1e-4, grad_clip=1.0,
          max_epochs=150, patience=20, batch_size=128,
          gripper_weight=2.0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dim  = tr_obs.shape[-1]
    action_dim = tr_act.shape[-1]

    # Normalization
    obs_mean = tr_obs.mean(0);  obs_std = tr_obs.std(0).clamp(min=1e-3)
    act_mean = tr_act.mean(0);  act_std = tr_act.std(0).clamp(min=1e-3)

    # Static action mask: dims whose expert std is effectively 0
    static_mask = tr_act.std(0) < STATIC_STD_THRESH
    static_vals = act_mean.clone()

    tr_obs_n = (tr_obs - obs_mean) / obs_std
    va_obs_n = (va_obs - obs_mean) / obs_std
    tr_act_n = (tr_act - act_mean) / act_std
    va_act_n = (va_act - act_mean) / act_std

    print('Building sequence tensors...', flush=True)
    t0 = time.time()
    tr_seq, tr_msk, tr_a = build_seq_tensors(tr_obs_n, tr_act_n, seq_len, tr_ep_starts)
    va_seq, va_msk, va_a = build_seq_tensors(va_obs_n, va_act_n, seq_len, va_ep_starts)
    print(f'  train={len(tr_seq):,}  val={len(va_seq):,}  ({time.time()-t0:.1f}s)', flush=True)

    # Per-dim loss weights — upweight gripper dim
    w = torch.ones(action_dim, device=device)
    w[GRIPPER_DIM] = gripper_weight

    def loss_fn(pred, target):
        return (nn.functional.smooth_l1_loss(pred, target, reduction='none') * w).mean()

    tr_seq = tr_seq.to(device); tr_msk = tr_msk.to(device); tr_a = tr_a.to(device)
    va_seq = va_seq.to(device); va_msk = va_msk.to(device); va_a = va_a.to(device)

    model = build_model(arch, state_dim, action_dim, seq_len,
                        d_model, n_heads, n_layers, dropout).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    best_val = float('inf'); best_state = None; wait = 0
    N = len(tr_seq)

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        ep_loss = 0.0; nb = 0
        for b in range(0, N - batch_size + 1, batch_size):
            idx = perm[b:b+batch_size]
            loss = loss_fn(model(tr_seq[idx], tr_msk[idx]), tr_a[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(va_seq, va_msk), va_a).item()

        print(f'  Epoch {epoch+1:4d}/{max_epochs}  '
              f'train={ep_loss/nb:.5f}  val={vl:.5f}  '
              f'lr={sched.get_last_lr()[0]:.2e}', flush=True)

        if vl < best_val:
            best_val = vl; best_state = deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'\nEarly stopping (epoch {epoch+1}, best val={best_val:.5f})', flush=True)
                break

    model.load_state_dict(best_state)
    model.eval()
    print(f'Best val loss: {best_val:.5f}', flush=True)
    return model, obs_mean, obs_std, act_mean, act_std, static_mask, static_vals


# ═══════════════════════════════════════════════════════════════════════════
# State extraction at eval time
# ═══════════════════════════════════════════════════════════════════════════

def _quat_mul(q1, q2):
    x1,y1,z1,w1 = q1;  x2,y2,z2,w2 = q2
    return np.array([
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
        w1*w2-x1*x2-y1*y2-z1*z2,
    ], dtype=np.float32)


def extract_state(obs, env, active_handle_site):
    """Build 44-dim state vector from live obs + env sim state."""
    proprio = np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED], dtype=np.float32)

    door_pos  = obs['door_obj_pos'].flatten().astype(np.float32)
    door_quat = obs['door_obj_quat'].flatten().astype(np.float32)

    eef_pos  = compute_eef_pos_from_obs(obs)
    base_quat        = obs['robot0_base_quat'].flatten().astype(np.float32)
    base_to_eef_quat = obs['robot0_base_to_eef_quat'].flatten().astype(np.float32)
    eef_quat = _quat_mul(base_quat, base_to_eef_quat)

    door_to_eef_pos  = eef_pos - door_pos
    door_quat_conj   = np.array([-door_quat[0], -door_quat[1], -door_quat[2], door_quat[3]])
    door_to_eef_quat = _quat_mul(door_quat_conj, eef_quat)
    door_dist = np.array([np.linalg.norm(door_to_eef_pos)], dtype=np.float32)

    handle_pos, active_handle_site = get_handle_pos_from_env(env, active_handle_site, eef_pos)
    handle_to_eef_pos = eef_pos - handle_pos

    state = np.concatenate([
        proprio, door_pos, door_quat, eef_pos, eef_quat,
        door_to_eef_pos, door_to_eef_quat, door_dist,
        handle_pos, handle_to_eef_pos,
    ])
    return state.astype(np.float32), active_handle_site


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _eval_worker(args):
    """Spawn-safe eval worker: loads checkpoint from disk, creates own env, runs episode subset."""
    ckpt_path, ep_indices, seq_len, max_steps, split, base_seed = args

    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    arch = ckpt.get('arch', 'transformer')
    feat_indices = ckpt.get('feat_indices', None)
    mkw = ckpt['model_kwargs']
    model = build_model(arch, **mkw)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    obs_mean_np = ckpt['obs_mean'].numpy()
    obs_std_np  = ckpt['obs_std'].numpy()
    act_mean_np = ckpt['act_mean'].numpy()
    act_std_np  = ckpt['act_std'].numpy()
    static_np   = ckpt['static_mask'].numpy()
    svals_np    = ckpt['static_vals'].numpy()

    # Model's actual input dim (may differ from STATE_DIM with feat subsets)
    model_state_dim = mkw['state_dim']

    # Warm up model to trigger any JIT / lazy initialization before timed episodes
    _dummy_seq  = torch.zeros(1, seq_len, model_state_dim)
    _dummy_mask = torch.ones(1, seq_len, dtype=torch.bool)
    with torch.no_grad():
        model(_dummy_seq, _dummy_mask)

    env = create_env(split=split, seed=base_seed + ep_indices[0])

    # ONE slow reset per worker — picks this worker's kitchen, compiles MuJoCo model.
    # Subsequent episodes reuse the same compiled model via fast state-restore.
    obs = env.reset()
    init_flat = env.sim.get_state().flatten()   # [time, qpos, qvel]

    results = []

    # Pre-allocate seq/mask buffers — reused every step (no per-step alloc)
    seq_buf  = np.zeros((seq_len, model_state_dim), dtype=np.float32)
    mask_buf = np.ones(seq_len, dtype=bool)
    seq_t    = torch.from_numpy(seq_buf).unsqueeze(0)   # shares storage
    mask_t   = torch.from_numpy(mask_buf).unsqueeze(0)

    for i, ep_i in enumerate(ep_indices):
        if i > 0:
            # Fast reset: restore physics to episode-start state without recompiling XML.
            env.sim.set_state_from_flattened(init_flat)
            env.sim.forward()
            for robot in env.robots:
                ctrl = getattr(robot, 'composite_controller', None) or getattr(robot, 'controller', None)
                if ctrl is not None and hasattr(ctrl, 'reset'):
                    ctrl.reset()
            obs = env._get_observations()

        active_site = None
        history = deque(maxlen=seq_len)
        success = False
        last_state_full = None  # always 44-dim for hdist metric

        # Prime first state before the loop
        state_full, active_site = extract_state(obs, env, active_site)

        for step in range(max_steps):
            last_state_full = state_full
            # Slice to model's feature subset if needed
            state = state_full[feat_indices] if feat_indices is not None else state_full
            state_n = (state - obs_mean_np) / obs_std_np
            history.append(state_n)
            L = len(history)

            # Fill pre-allocated buffers in-place
            seq_buf[:] = 0.0
            seq_buf[-L:] = np.stack(list(history))
            mask_buf[:] = True
            mask_buf[-L:] = False

            with torch.no_grad():
                pred_n = model(seq_t, mask_t).numpy()[0]

            raw_act = pred_n * act_std_np + act_mean_np
            raw_act[static_np] = svals_np[static_np]
            raw_act = np.clip(raw_act, -1.0, 1.0)
            env_act = np.clip(dataset_action_to_env_action(raw_act), -1.0, 1.0)
            obs, _, _, _ = env.step(env_act)

            if env._check_success():
                success = True
                state_full, active_site = extract_state(obs, env, active_site)
                break

            state_full, active_site = extract_state(obs, env, active_site)

        # handle_to_eef_pos is last 3 dims of state_full (indices 41:44)
        hdist = np.linalg.norm(last_state_full[-3:]) if last_state_full is not None else 0.0
        results.append((ep_i, success, step + 1, hdist))

    env.close()
    return results


def evaluate(model, obs_mean, obs_std, act_mean, act_std,
             static_mask, static_vals,
             seq_len=16, n_eps=20, max_steps=500, split='pretrain', seed=0,
             n_workers=4, ckpt_path=None):

    if ckpt_path is None:
        ckpt_path = CKPT_PATH

    # Split episode indices across workers
    ep_indices = list(range(n_eps))
    batches = [ep_indices[i::n_workers] for i in range(n_workers)]
    batches = [b for b in batches if b]  # drop empty batches

    worker_args = [
        (str(ckpt_path), batch, seq_len, max_steps, split, seed)
        for batch in batches
    ]

    print(f'  Launching {len(batches)} eval workers for {n_eps} episodes...', flush=True)
    ctx = mp.get_context('spawn')
    with ctx.Pool(len(batches)) as pool:
        all_results = pool.map(_eval_worker, worker_args)

    # Flatten and sort by episode index
    flat = sorted([item for batch_res in all_results for item in batch_res],
                  key=lambda x: x[0])

    successes = []; lengths = []
    for ep_i, success, steps, hdist in flat:
        successes.append(success); lengths.append(steps)
        print(f'  Ep {ep_i+1:2d}/{n_eps}: {"OK" if success else "X "}  '
              f'steps={steps:3d}  handle_dist={hdist:.3f}m', flush=True)

    n_succ = sum(successes)
    print(f'\nResult: {n_succ}/{n_eps} ({100*n_succ/n_eps:.1f}%)  '
          f'avg_steps={np.mean(lengths):.1f}', flush=True)
    return n_succ, n_eps


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_only',  action='store_true')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--arch',        default='transformer', choices=['transformer', 'mlp'])
    parser.add_argument('--feat_subset', default='full', choices=['full', 'no_handle', 'handle_only'])
    parser.add_argument('--n_eps',          type=int,   default=20)
    parser.add_argument('--max_steps',      type=int,   default=500)
    parser.add_argument('--n_eval_workers', type=int,   default=4)
    parser.add_argument('--seq_len',    type=int,   default=16)
    parser.add_argument('--d_model',    type=int,   default=256)
    parser.add_argument('--n_heads',    type=int,   default=8)
    parser.add_argument('--n_layers',   type=int,   default=4)
    parser.add_argument('--dropout',    type=float, default=0.1)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--patience',   type=int,   default=20)
    parser.add_argument('--batch_size', type=int,   default=128)
    parser.add_argument('--gripper_weight', type=float, default=2.0)
    parser.add_argument('--n_workers',  type=int,   default=4)
    parser.add_argument('--val_frac',   type=float, default=0.10)
    parser.add_argument('--seed',       type=int,   default=0)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else CKPT_PATH
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat_indices = FEATURE_CONFIGS[args.feat_subset]
    if len(feat_indices) == STATE_DIM:
        feat_indices = None   # no slicing needed for full set

    if not args.eval_only:
        print('=== Loading data ===', flush=True)
        obs_all, actions, ep_bounds = load_data(n_workers=args.n_workers,
                                                feat_indices=feat_indices)
        print(f'State {obs_all.shape[-1]}-dim  Actions {actions.shape[-1]}-dim  '
              f'Frames {len(obs_all):,}  feat_subset={args.feat_subset}  arch={args.arch}',
              flush=True)

        tr_obs, tr_act, va_obs, va_act, tr_ep_starts, va_ep_starts = train_val_split(
            obs_all, actions, ep_bounds, val_frac=args.val_frac)
        print(f'Train {len(tr_obs):,}  Val {len(va_obs):,}', flush=True)

        print('\n=== Training ===', flush=True)
        model, obs_mean, obs_std, act_mean, act_std, static_mask, static_vals = train(
            tr_obs, tr_act, va_obs, va_act,
            tr_ep_starts=tr_ep_starts, va_ep_starts=va_ep_starts,
            arch=args.arch,
            seq_len=args.seq_len, d_model=args.d_model,
            n_heads=args.n_heads, n_layers=args.n_layers,
            dropout=args.dropout, lr=args.lr,
            max_epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size, gripper_weight=args.gripper_weight,
        )

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(
            model_state=model.state_dict(),
            arch=args.arch,
            feat_indices=feat_indices,
            model_kwargs=dict(state_dim=obs_all.shape[-1], action_dim=actions.shape[-1],
                              seq_len=args.seq_len, d_model=args.d_model,
                              n_heads=args.n_heads, n_layers=args.n_layers,
                              dropout=args.dropout),
            obs_mean=obs_mean, obs_std=obs_std,
            act_mean=act_mean, act_std=act_std,
            static_mask=static_mask, static_vals=static_vals,
        ), ckpt_path)
        print(f'Checkpoint saved → {ckpt_path}', flush=True)

    else:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        arch_loaded = ckpt.get('arch', 'transformer')
        feat_indices = ckpt.get('feat_indices', None)
        model = build_model(arch_loaded, **ckpt['model_kwargs'])
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        obs_mean    = ckpt['obs_mean']
        obs_std     = ckpt['obs_std']
        act_mean    = ckpt['act_mean']
        act_std     = ckpt['act_std']
        static_mask = ckpt['static_mask']
        static_vals = ckpt['static_vals']
        print(f'Loaded checkpoint from {ckpt_path}  arch={arch_loaded}  '
              f'feat_indices={feat_indices}', flush=True)

    model = model.to(device)
    print(f'\n=== Evaluating ({args.n_eps} episodes, {args.max_steps} steps) ===',
          flush=True)
    evaluate(
        model, obs_mean, obs_std, act_mean, act_std, static_mask, static_vals,
        seq_len=args.seq_len, n_eps=args.n_eps, max_steps=args.max_steps, seed=args.seed,
        n_workers=args.n_eval_workers, ckpt_path=ckpt_path,
    )


if __name__ == '__main__':
    main()
