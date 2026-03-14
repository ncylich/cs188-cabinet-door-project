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
from diffusion_policy.models.unet import UNetNoiseNet


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


class GripperMLP(nn.Module):
    """Independent binary gripper classifier (for split-gripper arch)."""
    def __init__(self, state_dim, hidden=128, n_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        dims = [state_dim] + [hidden] * n_layers + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, states, padding_mask=None):
        # Use only the most recent frame
        x = states[:, -1] if states.dim() == 3 else states
        return self.net(x)  # (B, 1) raw logit


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
                binary_gripper=False, denoiser_hidden=512, horizon=16, n_obs_steps=2, **_):
    if arch == 'unet':
        return UNetNoiseNet(action_dim=action_dim, state_dim=state_dim,
                            horizon=horizon, n_obs_steps=n_obs_steps,
                            channels=(256, 512, 1024))
    if arch == 'diffusion':
        return TemporalDiffusionPolicy(state_dim, action_dim, seq_len, d_model,
                                       n_heads, n_layers, dropout, denoiser_hidden)
    if arch == 'mlp':
        return BCMLP(state_dim, action_dim, hidden=d_model, n_layers=n_layers, dropout=dropout)
    if binary_gripper:
        return BCTransformerBinaryGripper(state_dim, action_dim, seq_len,
                                          d_model, n_heads, n_layers, dropout)
    # 'transformer' or 'split_gripper' arm portion — same BCTransformer body
    return BCTransformer(state_dim, action_dim, seq_len, d_model, n_heads, n_layers, dropout)


# ═══════════════════════════════════════════════════════════════════════════
# U-Net windowed dataset builder
# ═══════════════════════════════════════════════════════════════════════════

def build_unet_tensors(obs, actions, horizon, n_obs_steps, ep_starts=None):
    """Build (obs_context, action_horizon) pairs for U-Net diffusion training.

    Each sample: obs_context = last n_obs_steps states (flat), action_horizon = next horizon actions.
    Episode boundaries respected — no cross-episode windows.
    """
    N, D = obs.shape
    A = actions.shape[-1]
    obs_np  = obs.numpy()     if isinstance(obs,     torch.Tensor) else np.asarray(obs,     dtype=np.float32)
    act_np  = actions.numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions, dtype=np.float32)

    if ep_starts:
        ep_starts_s = sorted(ep_starts)
        ep_ends_s   = ep_starts_s[1:] + [N]
        frame_ep_start = np.zeros(N, dtype=np.int64)
        frame_ep_end   = np.full(N, N, dtype=np.int64)
        for s, e in zip(ep_starts_s, ep_ends_s):
            frame_ep_start[s:e] = s
            frame_ep_end[s:e]   = e
    else:
        frame_ep_start = np.zeros(N, dtype=np.int64)
        frame_ep_end   = np.full(N, N, dtype=np.int64)

    valid = [i for i in range(N)
             if (i - n_obs_steps + 1 >= frame_ep_start[i] and
                 i + horizon         <= frame_ep_end[i])]

    M = len(valid)
    obs_ctxs    = np.zeros((M, n_obs_steps * D), dtype=np.float32)
    act_horizons = np.zeros((M, horizon, A),      dtype=np.float32)
    for k, i in enumerate(valid):
        obs_ctxs[k]     = obs_np[i - n_obs_steps + 1:i + 1].reshape(-1)
        act_horizons[k] = act_np[i:i + horizon]
    return torch.from_numpy(obs_ctxs), torch.from_numpy(act_horizons)


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train(tr_obs, tr_act, va_obs, va_act,
          tr_ep_starts=None, va_ep_starts=None,
          arch='transformer',
          binary_gripper=False, bce_weight=2.0,
          ddpm_steps=100, denoiser_hidden=512,
          horizon=16, n_obs_steps=2, n_action_steps=8,
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

    # Raw gripper labels for binary head (before normalization)
    tr_raw_grip = tr_act[:, GRIPPER_DIM].clone()
    va_raw_grip = va_act[:, GRIPPER_DIM].clone()

    tr_obs_n = (tr_obs - obs_mean) / obs_std
    va_obs_n = (va_obs - obs_mean) / obs_std
    tr_act_n = (tr_act - act_mean) / act_std
    va_act_n = (va_act - act_mean) / act_std

    # ── U-Net diffusion: windowed action-horizon training ──────────────────
    if arch == 'unet':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Building U-Net windowed tensors...', flush=True)
        tr_ctx, tr_ah = build_unet_tensors(tr_obs_n, tr_act_n, horizon, n_obs_steps, tr_ep_starts)
        va_ctx, va_ah = build_unet_tensors(va_obs_n, va_act_n, horizon, n_obs_steps, va_ep_starts)
        print(f'  train={len(tr_ctx):,}  val={len(va_ctx):,}', flush=True)
        tr_ctx = tr_ctx.to(device); tr_ah = tr_ah.to(device)
        va_ctx = va_ctx.to(device); va_ah = va_ah.to(device)

        diff_scheduler = DDPMScheduler(num_train_steps=ddpm_steps, beta_schedule='squared_cosine')
        model = build_model(arch, state_dim, action_dim, seq_len, d_model, n_heads, n_layers,
                            dropout, horizon=horizon, n_obs_steps=n_obs_steps).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

        best_val = float('inf'); best_state = None; wait = 0
        M = len(tr_ctx)
        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(M, device=device)
            ep_loss = 0.0; nb = 0
            for b in range(0, M - batch_size + 1, batch_size):
                idx = perm[b:b+batch_size]
                ctx_b = tr_ctx[idx]; ah_b = tr_ah[idx]
                noise   = torch.randn_like(ah_b)
                t       = torch.randint(0, ddpm_steps, (len(idx),), device=device)
                noisy   = diff_scheduler.add_noise(ah_b, noise, t)
                loss    = nn.functional.mse_loss(model(noisy, ctx_b, t), noise)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                ep_loss += loss.item(); nb += 1
            sched.step()
            model.eval()
            with torch.no_grad():
                va_noise = torch.randn_like(va_ah)
                va_t     = torch.randint(0, ddpm_steps, (len(va_ctx),), device=device)
                va_noisy = diff_scheduler.add_noise(va_ah, va_noise, va_t)
                vl = nn.functional.mse_loss(model(va_noisy, va_ctx, va_t), va_noise).item()
            print(f'  Epoch {epoch+1:4d}/{max_epochs}  train={ep_loss/nb:.5f}  val={vl:.5f}  '
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

    print('Building sequence tensors...', flush=True)
    t0 = time.time()
    tr_seq, tr_msk, tr_a = build_seq_tensors(tr_obs_n, tr_act_n, seq_len, tr_ep_starts)
    va_seq, va_msk, va_a = build_seq_tensors(va_obs_n, va_act_n, seq_len, va_ep_starts)
    print(f'  train={len(tr_seq):,}  val={len(va_seq):,}  ({time.time()-t0:.1f}s)', flush=True)

    tr_seq = tr_seq.to(device); tr_msk = tr_msk.to(device); tr_a = tr_a.to(device)
    va_seq = va_seq.to(device); va_msk = va_msk.to(device); va_a = va_a.to(device)

    # Gripper labels for binary head: {-1,+1} → {0,1}
    tr_grip_lbl = ((tr_raw_grip + 1.0) / 2.0).to(device)
    va_grip_lbl = ((va_raw_grip + 1.0) / 2.0).to(device)

    # Per-dim loss weights (continuous dims only for non-binary, full for standard)
    if binary_gripper:
        w = torch.ones(action_dim - 1, device=device)   # 11 continuous dims
    else:
        w = torch.ones(action_dim, device=device)
        w[GRIPPER_DIM] = gripper_weight

    def loss_fn_standard(pred, target):
        return (nn.functional.smooth_l1_loss(pred, target, reduction='none') * w).mean()

    def loss_fn_binary(pred_cont, grip_logit, target, grip_label):
        huber = (nn.functional.smooth_l1_loss(
            pred_cont, target[:, :action_dim-1], reduction='none') * w).mean()
        bce = nn.functional.binary_cross_entropy_with_logits(
            grip_logit.squeeze(-1), grip_label)
        return huber + bce_weight * bce

    # Diffusion scheduler (only used when arch=='diffusion')
    diff_scheduler = None
    if arch == 'diffusion':
        diff_scheduler = DDPMScheduler(num_train_steps=ddpm_steps,
                                       beta_schedule='squared_cosine')

    model = build_model(arch, state_dim, action_dim, seq_len, d_model, n_heads, n_layers,
                        dropout, binary_gripper=binary_gripper,
                        denoiser_hidden=denoiser_hidden).to(device)
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
            seq_b, msk_b, act_b = tr_seq[idx], tr_msk[idx], tr_a[idx]

            if arch == 'diffusion':
                context = model.encode(seq_b, msk_b)
                noise   = torch.randn_like(act_b)
                t       = torch.randint(0, ddpm_steps, (len(idx),), device=device)
                noisy   = diff_scheduler.add_noise(act_b, noise, t)
                loss    = nn.functional.mse_loss(model(noisy, context, t), noise)
            elif binary_gripper:
                pred_cont, grip_logit = model(seq_b, msk_b)
                loss = loss_fn_binary(pred_cont, grip_logit, act_b, tr_grip_lbl[idx])
            else:
                loss = loss_fn_standard(model(seq_b, msk_b), act_b)

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            if arch == 'diffusion':
                _va_noise = torch.randn(len(va_a), action_dim, device=device)
                _va_t     = torch.randint(0, ddpm_steps, (len(va_a),), device=device)
                _va_ctx   = model.encode(va_seq, va_msk)
                _va_noisy = diff_scheduler.add_noise(va_a, _va_noise, _va_t)
                vl = nn.functional.mse_loss(
                    model(_va_noisy, _va_ctx, _va_t), _va_noise).item()
            elif binary_gripper:
                pred_cont_v, grip_logit_v = model(va_seq, va_msk)
                vl = loss_fn_binary(pred_cont_v, grip_logit_v,
                                    va_a, va_grip_lbl).item()
            else:
                vl = loss_fn_standard(model(va_seq, va_msk), va_a).item()

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
# Split-gripper training (two independent models, no shared gradient)
# ═══════════════════════════════════════════════════════════════════════════

def train_split_gripper(tr_obs, tr_act, va_obs, va_act,
                        tr_ep_starts=None, va_ep_starts=None,
                        seq_len=16, d_model=256, n_heads=8, n_layers=4, dropout=0.1,
                        lr=3e-4, weight_decay=1e-4, grad_clip=1.0,
                        max_epochs=150, patience=20, batch_size=128):
    """Train arm BCTransformer and GripperMLP as two completely independent models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dim  = tr_obs.shape[-1]
    action_dim = tr_act.shape[-1]

    obs_mean = tr_obs.mean(0); obs_std = tr_obs.std(0).clamp(min=1e-3)
    act_mean = tr_act.mean(0); act_std = tr_act.std(0).clamp(min=1e-3)
    static_mask = tr_act.std(0) < STATIC_STD_THRESH
    static_vals = act_mean.clone()

    tr_obs_n = (tr_obs - obs_mean) / obs_std
    va_obs_n = (va_obs - obs_mean) / obs_std
    tr_act_n = (tr_act - act_mean) / act_std
    va_act_n = (va_act - act_mean) / act_std

    # Raw gripper labels: {-1,+1} → {0,1}
    tr_grip = ((tr_act[:, GRIPPER_DIM] + 1.0) / 2.0)
    va_grip = ((va_act[:, GRIPPER_DIM] + 1.0) / 2.0)

    print('Building sequence tensors...', flush=True)
    tr_seq, tr_msk, tr_a = build_seq_tensors(tr_obs_n, tr_act_n, seq_len, tr_ep_starts)
    va_seq, va_msk, va_a = build_seq_tensors(va_obs_n, va_act_n, seq_len, va_ep_starts)
    print(f'  train={len(tr_seq):,}  val={len(va_seq):,}', flush=True)

    tr_seq = tr_seq.to(device); tr_msk = tr_msk.to(device)
    tr_a   = tr_a.to(device);   tr_grip = tr_grip.to(device)
    va_seq = va_seq.to(device); va_msk = va_msk.to(device)
    va_a   = va_a.to(device);   va_grip = va_grip.to(device)

    # Arm model: BCTransformer → (action_dim-1) continuous dims (no gripper)
    arm_model = BCTransformer(state_dim, action_dim - 1, seq_len, d_model,
                               n_heads, n_layers, dropout).to(device)
    # Gripper model: GripperMLP → 1-dim logit
    grip_model = GripperMLP(state_dim, hidden=128, dropout=dropout).to(device)

    # Per-dim Huber weights for arm (dims 0:GRIPPER_DIM + GRIPPER_DIM+1:end)
    arm_act_indices = list(range(GRIPPER_DIM)) + list(range(GRIPPER_DIM + 1, action_dim))
    w_arm = torch.ones(action_dim - 1, device=device)

    arm_opt  = torch.optim.AdamW(arm_model.parameters(),  lr=lr, weight_decay=weight_decay)
    grip_opt = torch.optim.AdamW(grip_model.parameters(), lr=lr, weight_decay=weight_decay)
    arm_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(arm_opt,  T_max=max_epochs)
    grip_sched = torch.optim.lr_scheduler.CosineAnnealingLR(grip_opt, T_max=max_epochs)

    # Target for arm: all action dims except gripper
    tr_arm_tgt = tr_a[:, arm_act_indices]
    va_arm_tgt = va_a[:, arm_act_indices]

    best_val = float('inf'); best_arm_state = None; best_grip_state = None; wait = 0
    N = len(tr_seq)

    for epoch in range(max_epochs):
        arm_model.train(); grip_model.train()
        perm = torch.randperm(N, device=device)
        ep_arm_loss = ep_grip_loss = 0.0; nb = 0
        for b in range(0, N - batch_size + 1, batch_size):
            idx = perm[b:b+batch_size]
            seq_b, msk_b = tr_seq[idx], tr_msk[idx]
            # Arm step
            arm_pred = arm_model(seq_b, msk_b)
            arm_loss = (nn.functional.smooth_l1_loss(arm_pred, tr_arm_tgt[idx],
                                                     reduction='none') * w_arm).mean()
            arm_opt.zero_grad(); arm_loss.backward()
            nn.utils.clip_grad_norm_(arm_model.parameters(), grad_clip)
            arm_opt.step()
            # Gripper step (completely independent)
            grip_logit = grip_model(seq_b, msk_b)
            grip_loss = nn.functional.binary_cross_entropy_with_logits(
                grip_logit.squeeze(-1), tr_grip[idx])
            grip_opt.zero_grad(); grip_loss.backward()
            nn.utils.clip_grad_norm_(grip_model.parameters(), grad_clip)
            grip_opt.step()
            ep_arm_loss += arm_loss.item(); ep_grip_loss += grip_loss.item(); nb += 1
        arm_sched.step(); grip_sched.step()

        arm_model.eval(); grip_model.eval()
        with torch.no_grad():
            va_arm_pred  = arm_model(va_seq, va_msk)
            va_grip_pred = grip_model(va_seq, va_msk)
            arm_vl  = (nn.functional.smooth_l1_loss(va_arm_pred, va_arm_tgt,
                                                    reduction='none') * w_arm).mean().item()
            grip_vl = nn.functional.binary_cross_entropy_with_logits(
                va_grip_pred.squeeze(-1), va_grip).item()
            vl = arm_vl + grip_vl
        print(f'  Epoch {epoch+1:4d}/{max_epochs}  arm={ep_arm_loss/nb:.5f}  '
              f'grip={ep_grip_loss/nb:.5f}  val={vl:.5f}', flush=True)
        if vl < best_val:
            best_val = vl
            best_arm_state  = deepcopy(arm_model.state_dict())
            best_grip_state = deepcopy(grip_model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'\nEarly stopping (epoch {epoch+1}, best val={best_val:.5f})', flush=True)
                break

    arm_model.load_state_dict(best_arm_state)
    grip_model.load_state_dict(best_grip_state)
    arm_model.eval(); grip_model.eval()
    print(f'Best val loss: {best_val:.5f}', flush=True)
    return (arm_model, grip_model, obs_mean, obs_std, act_mean, act_std,
            static_mask, static_vals, arm_act_indices)


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

def _any_door_open(env, th=0.90):
    """Return True if ANY cabinet door joint is ≥ th open (professor's recommendation).

    Falls back to env._check_success() if fixture/joints unavailable.
    """
    try:
        fxtr = getattr(env, 'fxtr', None)
        if fxtr is None:
            return env._check_success()
        joint_names = getattr(fxtr, 'door_joint_names', [])
        if not joint_names:
            return env._check_success()
        joint_state = fxtr.get_joint_state(env, joint_names)
        if not joint_state:
            return env._check_success()
        return any(v >= th for v in joint_state.values())
    except Exception:
        return env._check_success()


def _eval_worker(args):
    """Spawn-safe eval worker: loads checkpoint from disk, creates own env, runs episode subset."""
    ckpt_path, ep_indices, seq_len, max_steps, split, base_seed = args

    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    arch           = ckpt.get('arch', 'transformer')
    feat_indices   = ckpt.get('feat_indices', None)
    binary_gripper = ckpt.get('binary_gripper', False)
    ddpm_steps     = ckpt.get('ddpm_steps', 100)
    ddim_steps     = ckpt.get('ddim_steps', 10)
    n_obs_steps    = ckpt.get('n_obs_steps', 2)
    horizon        = ckpt.get('horizon', 16)
    n_action_steps = ckpt.get('n_action_steps', 8)
    mkw = ckpt['model_kwargs']
    obs_mean_np = ckpt['obs_mean'].cpu().numpy()
    obs_std_np  = ckpt['obs_std'].cpu().numpy()
    act_mean_np = ckpt['act_mean'].cpu().numpy()
    act_std_np  = ckpt['act_std'].cpu().numpy()
    static_np   = ckpt['static_mask'].numpy()
    svals_np    = ckpt['static_vals'].numpy()
    model_state_dim = mkw['state_dim']
    action_dim      = mkw['action_dim']

    # Load model(s)
    if arch == 'split_gripper':
        arm_model  = build_model('transformer', binary_gripper=False,
                                 **{**mkw, 'action_dim': action_dim - 1})
        grip_model = GripperMLP(model_state_dim, hidden=128,
                                dropout=mkw.get('dropout', 0.1))
        arm_model.load_state_dict(ckpt['model_state'])
        grip_model.load_state_dict(ckpt['grip_model_state'])
        arm_model.eval(); grip_model.eval()
        arm_act_indices = ckpt.get('arm_act_indices', list(range(GRIPPER_DIM)) +
                                   list(range(GRIPPER_DIM + 1, action_dim)))
        model = None  # not used for split_gripper
    else:
        model = build_model(arch, binary_gripper=binary_gripper, **mkw)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

    # Diffusion scheduler
    diff_scheduler = None
    if arch in ('diffusion', 'unet'):
        diff_scheduler = DDPMScheduler(num_train_steps=ddpm_steps,
                                       beta_schedule='squared_cosine')

    # Warm up model
    if arch == 'split_gripper':
        _dummy_seq  = torch.zeros(1, seq_len, model_state_dim)
        _dummy_mask = torch.ones(1, seq_len, dtype=torch.bool)
        with torch.no_grad():
            arm_model(_dummy_seq, _dummy_mask)
            grip_model(_dummy_seq, _dummy_mask)
    elif arch == 'unet':
        _dummy_ctx = torch.zeros(1, n_obs_steps * model_state_dim)
        _dummy_act = torch.zeros(1, horizon, action_dim)
        with torch.no_grad():
            model(_dummy_act, _dummy_ctx, torch.zeros(1, dtype=torch.long))
    else:
        _dummy_seq  = torch.zeros(1, seq_len, model_state_dim)
        _dummy_mask = torch.ones(1, seq_len, dtype=torch.bool)
        with torch.no_grad():
            if arch == 'diffusion':
                _ctx = model.encode(_dummy_seq, _dummy_mask)
                model(torch.zeros(1, action_dim), _ctx, torch.zeros(1, dtype=torch.long))
            else:
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

        active_site    = None
        history        = deque(maxlen=seq_len)       # used by transformer/mlp/diffusion
        obs_deque      = deque(maxlen=n_obs_steps)   # used by unet
        action_queue   = deque()                     # used by unet
        success        = False
        last_state_full = None

        state_full, active_site = extract_state(obs, env, active_site)

        for step in range(max_steps):
            last_state_full = state_full
            state = state_full[feat_indices] if feat_indices is not None else state_full
            state_n = (state - obs_mean_np) / obs_std_np

            # ── U-Net: receding-horizon action chunking ─────────────────────
            if arch == 'unet':
                obs_deque.append(state_n)
                if len(action_queue) == 0:
                    while len(obs_deque) < n_obs_steps:
                        obs_deque.appendleft(obs_deque[0])
                    obs_ctx = torch.from_numpy(
                        np.concatenate(list(obs_deque))).unsqueeze(0)  # (1, n_obs*D)
                    x_T = torch.randn(1, horizon, action_dim)
                    with torch.no_grad():
                        pred_h = diff_scheduler.denoise_ddim(
                            model, x_T, obs_ctx,
                            num_inference_steps=ddim_steps)           # (1,H,A)
                    pred_np = pred_h[0].numpy()                       # (H,A)
                    for h in range(min(n_action_steps, horizon)):
                        raw_h = pred_np[h] * act_std_np + act_mean_np
                        raw_h[static_np] = svals_np[static_np]
                        action_queue.append(np.clip(raw_h, -1.0, 1.0))
                raw_act = action_queue.popleft()

            # ── Split-gripper: arm + independent gripper ────────────────────
            elif arch == 'split_gripper':
                history.append(state_n)
                L = len(history)
                seq_buf[:] = 0.0; seq_buf[-L:] = np.stack(list(history))
                mask_buf[:] = True; mask_buf[-L:] = False
                with torch.no_grad():
                    arm_pred_n  = arm_model(seq_t, mask_t).numpy()[0]   # (action_dim-1,)
                    grip_logit  = grip_model(seq_t, mask_t)
                    grip_binary = 1.0 if grip_logit.sigmoid().item() > 0.5 else 0.0
                raw_act = np.empty(action_dim, dtype=np.float32)
                arm_raw = arm_pred_n * act_std_np[arm_act_indices] + act_mean_np[arm_act_indices]
                arm_raw[static_np[arm_act_indices]] = svals_np[arm_act_indices][
                    static_np[arm_act_indices]]
                raw_act[arm_act_indices]  = np.clip(arm_raw, -1.0, 1.0)
                raw_act[GRIPPER_DIM]      = np.clip(
                    grip_binary * act_std_np[GRIPPER_DIM] + act_mean_np[GRIPPER_DIM], -1.0, 1.0)

            # ── Standard transformer / MLP / old diffusion ──────────────────
            else:
                history.append(state_n)
                L = len(history)
                seq_buf[:] = 0.0; seq_buf[-L:] = np.stack(list(history))
                mask_buf[:] = True; mask_buf[-L:] = False
                with torch.no_grad():
                    if arch == 'diffusion':
                        _ctx   = model.encode(seq_t, mask_t)
                        _x_T   = torch.randn(1, action_dim)
                        pred_n = diff_scheduler.denoise_ddim(
                            model, _x_T, _ctx,
                            num_inference_steps=ddim_steps).numpy()[0]
                    elif binary_gripper:
                        pred_cont, grip_logit = model(seq_t, mask_t)
                        pred_cont_np = pred_cont.numpy()[0]
                        grip_prob    = float(grip_logit.sigmoid().numpy()[0, 0])
                        pred_n       = np.empty(action_dim, dtype=np.float32)
                        pred_n[:action_dim-1] = pred_cont_np
                        pred_n[action_dim-1]  = grip_prob
                    else:
                        pred_n = model(seq_t, mask_t).numpy()[0]
                if binary_gripper:
                    raw_act = np.empty(action_dim, dtype=np.float32)
                    raw_act[:action_dim-1] = (pred_n[:action_dim-1]
                                              * act_std_np[:action_dim-1]
                                              + act_mean_np[:action_dim-1])
                    raw_act[action_dim-1]  = pred_n[action_dim-1]
                    raw_act[:action_dim-1][static_np[:action_dim-1]] = \
                        svals_np[:action_dim-1][static_np[:action_dim-1]]
                    raw_act[:action_dim-1] = np.clip(raw_act[:action_dim-1], -1.0, 1.0)
                else:
                    raw_act = pred_n * act_std_np + act_mean_np
                    raw_act[static_np] = svals_np[static_np]
                    raw_act = np.clip(raw_act, -1.0, 1.0)

            env_act = np.clip(dataset_action_to_env_action(raw_act), -1.0, 1.0)
            obs, _, _, _ = env.step(env_act)

            if _any_door_open(env):
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
    parser.add_argument('--arch',        default='transformer',
                        choices=['transformer', 'mlp', 'diffusion', 'unet', 'split_gripper'])
    parser.add_argument('--feat_subset', default='full',
                        choices=['full', 'no_handle', 'handle_only'])
    parser.add_argument('--binary_gripper', action='store_true',
                        help='Separate BCE classification head for gripper dim')
    parser.add_argument('--bce_weight',  type=float, default=2.0,
                        help='Weight on BCE gripper loss (only with --binary_gripper)')
    parser.add_argument('--ddpm_steps',  type=int,   default=100,
                        help='DDPM training timesteps T (only with --arch diffusion)')
    parser.add_argument('--ddim_steps',  type=int,   default=10,
                        help='DDIM inference steps (only with --arch diffusion)')
    parser.add_argument('--denoiser_hidden', type=int, default=512,
                        help='Hidden dim for MLP denoiser (only with --arch diffusion)')
    parser.add_argument('--horizon',      type=int, default=16,
                        help='Action prediction horizon (unet only)')
    parser.add_argument('--n_obs_steps',  type=int, default=2,
                        help='Number of obs frames to concat as context (unet only)')
    parser.add_argument('--n_action_steps', type=int, default=8,
                        help='Actions to execute per replan (unet only)')
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

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.arch == 'unet':
        ckpt_path = SAVE_DIR / 'bc_unet_best.pt'
    elif args.arch == 'split_gripper':
        ckpt_path = SAVE_DIR / 'bc_split_grip_best.pt'
    else:
        ckpt_path = CKPT_PATH
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
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        if args.arch == 'split_gripper':
            (arm_model, grip_model, obs_mean, obs_std,
             act_mean, act_std, static_mask, static_vals,
             arm_act_indices) = train_split_gripper(
                tr_obs, tr_act, va_obs, va_act,
                tr_ep_starts=tr_ep_starts, va_ep_starts=va_ep_starts,
                seq_len=args.seq_len, d_model=args.d_model,
                n_heads=args.n_heads, n_layers=args.n_layers,
                dropout=args.dropout, lr=args.lr,
                max_epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size,
            )
            torch.save(dict(
                model_state=arm_model.state_dict(),
                grip_model_state=grip_model.state_dict(),
                arch='split_gripper',
                feat_indices=feat_indices,
                binary_gripper=False,
                arm_act_indices=arm_act_indices,
                n_obs_steps=args.n_obs_steps,
                horizon=args.horizon,
                n_action_steps=args.n_action_steps,
                ddpm_steps=args.ddpm_steps,
                ddim_steps=args.ddim_steps,
                model_kwargs=dict(state_dim=obs_all.shape[-1],
                                  action_dim=actions.shape[-1],
                                  seq_len=args.seq_len, d_model=args.d_model,
                                  n_heads=args.n_heads, n_layers=args.n_layers,
                                  dropout=args.dropout,
                                  denoiser_hidden=args.denoiser_hidden),
                obs_mean=obs_mean, obs_std=obs_std,
                act_mean=act_mean, act_std=act_std,
                static_mask=static_mask, static_vals=static_vals,
            ), ckpt_path)
            model = arm_model  # for evaluate() signature
        else:
            model, obs_mean, obs_std, act_mean, act_std, static_mask, static_vals = train(
                tr_obs, tr_act, va_obs, va_act,
                tr_ep_starts=tr_ep_starts, va_ep_starts=va_ep_starts,
                arch=args.arch,
                binary_gripper=args.binary_gripper, bce_weight=args.bce_weight,
                ddpm_steps=args.ddpm_steps, denoiser_hidden=args.denoiser_hidden,
                horizon=args.horizon, n_obs_steps=args.n_obs_steps,
                n_action_steps=args.n_action_steps,
                seq_len=args.seq_len, d_model=args.d_model,
                n_heads=args.n_heads, n_layers=args.n_layers,
                dropout=args.dropout, lr=args.lr,
                max_epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size, gripper_weight=args.gripper_weight,
            )
            torch.save(dict(
                model_state=model.state_dict(),
                arch=args.arch,
                feat_indices=feat_indices,
                binary_gripper=args.binary_gripper,
                ddpm_steps=args.ddpm_steps,
                ddim_steps=args.ddim_steps,
                n_obs_steps=args.n_obs_steps,
                horizon=args.horizon,
                n_action_steps=args.n_action_steps,
                model_kwargs=dict(state_dim=obs_all.shape[-1],
                                  action_dim=actions.shape[-1],
                                  seq_len=args.seq_len, d_model=args.d_model,
                                  n_heads=args.n_heads, n_layers=args.n_layers,
                                  dropout=args.dropout,
                                  denoiser_hidden=args.denoiser_hidden,
                                  horizon=args.horizon,
                                  n_obs_steps=args.n_obs_steps),
                obs_mean=obs_mean, obs_std=obs_std,
                act_mean=act_mean, act_std=act_std,
                static_mask=static_mask, static_vals=static_vals,
            ), ckpt_path)
        print(f'Checkpoint saved → {ckpt_path}', flush=True)

    else:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        arch_loaded    = ckpt.get('arch', 'transformer')
        feat_indices   = ckpt.get('feat_indices', None)
        bg_loaded      = ckpt.get('binary_gripper', False)
        model = build_model(arch_loaded, binary_gripper=bg_loaded, **ckpt['model_kwargs'])
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
