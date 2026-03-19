"""Generate website media assets: GIFs and teaser MP4.

Trains three BC UNet model variants and renders rollout videos:
  - proprio_failure     → assets/proprio-failure.gif
  - door_near_miss      → assets/door-centroid-near-miss.gif
  - final_success       → assets/final-success.gif
  - teaser_mp4          → assets/final-demo.mp4 (all 3 clips concatenated)

Usage (run from cabinet_door_project/):
    python generate_media.py --mode proprio_failure
    python generate_media.py --mode door_near_miss
    python generate_media.py --mode final_success
    python generate_media.py --mode teaser_mp4
"""
import os
import sys
import argparse
import logging
import time
from copy import deepcopy
from collections import deque

# Must set osmesa before any GL import
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import torch
import torch.nn as nn
import imageio

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
PERM_DIR    = os.path.join(os.path.dirname(__file__), "checkpoints")
SAVE_DIR    = "/tmp/diffusion_policy_checkpoints"
ASSETS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
PT_PATH     = os.path.join(SAVE_DIR, "preprocessed_all_states.pt")

HORIZON      = 16
N_OBS        = 2
N_ACT_STEPS  = 8
RENDER_H     = 320
RENDER_W     = 480
# h264 requires dimensions divisible by 2
GIF_W, GIF_H = 256, 172
GIF_FPS      = 20
MP4_FPS      = 20

# ── Model ────────────────────────────────────────────────────────────────────

class BCUNet(nn.Module):
    def __init__(self, action_dim, state_dim, horizon, n_obs_steps, channels=(64, 128, 256)):
        super().__init__()
        self.horizon    = horizon
        self.action_dim = action_dim
        obs_in          = state_dim * n_obs_steps
        self.obs_proj   = nn.Linear(obs_in, channels[0])
        self.decoder    = nn.Sequential(
            nn.Linear(channels[0], channels[1]), nn.ReLU(),
            nn.Linear(channels[1], channels[2]), nn.ReLU(),
            nn.Linear(channels[2], horizon * action_dim),
        )

    def forward(self, obs):
        x = obs.reshape(obs.shape[0], -1)
        x = torch.relu(self.obs_proj(x))
        return self.decoder(x).reshape(-1, self.horizon, self.action_dim)


# ── Data loading ─────────────────────────────────────────────────────────────

def _resolve_file(filename):
    """Return path to filename, preferring SAVE_DIR but falling back to PERM_DIR."""
    tmp_path  = os.path.join(SAVE_DIR, filename)
    perm_path = os.path.join(PERM_DIR, filename)
    if os.path.exists(tmp_path):
        return tmp_path
    if os.path.exists(perm_path):
        return perm_path
    return tmp_path  # return tmp so error messages point there


def _ensure_file_in_tmp(filename):
    """Copy file from PERM_DIR to SAVE_DIR if missing from SAVE_DIR."""
    tmp_path  = os.path.join(SAVE_DIR, filename)
    perm_path = os.path.join(PERM_DIR, filename)
    if not os.path.exists(tmp_path) and os.path.exists(perm_path):
        import shutil
        os.makedirs(SAVE_DIR, exist_ok=True)
        if os.path.isdir(perm_path):
            shutil.copytree(perm_path, tmp_path)
        else:
            shutil.copy2(perm_path, tmp_path)
        logger.info(f"Restored {filename} from permanent storage → {tmp_path}")


def load_data():
    """Load full preprocessed data; restores from permanent storage if /tmp was cleared."""
    for f in ["preprocessed_all_states.pt", "handle_cache", "hinge_cache",
              "door_positions.npz", "door_quats.npz"]:
        _ensure_file_in_tmp(f)

    pt_path = _resolve_file("preprocessed_all_states.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"{pt_path} not found. Run: python preprocess_parallel.py"
        )
    data = torch.load(pt_path, weights_only=False)
    if "handle_pos" not in data["features"]:
        raise FileNotFoundError(
            "handle_pos missing from preprocessed data. "
            "Run: python preprocess_parallel.py"
        )
    return data


def load_data_proprio_only():
    """Load just proprio+actions from parquet — no sim replay needed."""
    from diffusion_policy.data import get_dataset_path, load_episodes
    import torch

    episodes = load_episodes(get_dataset_path())
    episodes.sort(key=lambda e: e["episode_index"])

    all_states  = []
    all_actions = []
    ep_bounds   = []
    total = 0
    for ep in episodes:
        T = len(ep["states"])
        ep_bounds.append((ep["episode_index"], total, total + T))
        all_states.append(ep["states"])
        all_actions.append(ep["actions"])
        total += T

    states_t  = torch.from_numpy(np.concatenate(all_states,  axis=0)).float()
    actions_t = torch.from_numpy(np.concatenate(all_actions, axis=0)).float()
    ep_bounds = np.array(ep_bounds)

    return {
        "features":      {"proprio": states_t},
        "actions":       actions_t,
        "ep_boundaries": ep_bounds,
    }


# ── Training ─────────────────────────────────────────────────────────────────

def build_chunked(obs_t, act_t, horizon, n_obs):
    obs_list, act_list = [], []
    N = len(obs_t)
    for j in range(max(0, N - horizon - n_obs + 1)):
        obs_list.append(obs_t[j : j + n_obs])
        act_list.append(act_t[j + n_obs - 1 : j + n_obs - 1 + horizon])
    return torch.stack(obs_list), torch.stack(act_list)


def train_model(data, feature_names, ckpt_path):
    """Train BC UNet on selected features; saves checkpoint. Returns stats dicts."""
    from diffusion_policy.training import get_cosine_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features   = data["features"]
    actions    = data["actions"]
    ep_bounds  = data["ep_boundaries"]

    # Episode-level train/val split (15% val)
    rng   = np.random.RandomState(42)
    n_eps = len(ep_bounds)
    perm  = rng.permutation(n_eps)
    val_eps = set(perm[: max(1, int(n_eps * 0.15))])

    obs_all   = torch.cat([features[n] for n in feature_names], dim=-1)
    state_dim = obs_all.shape[-1]
    logger.info(f"Training on features {feature_names} → {state_dim}-dim")

    train_idxs, val_idxs = [], []
    for i, (_, start, end) in enumerate(ep_bounds):
        idxs = list(range(int(start), int(end)))
        (val_idxs if i in val_eps else train_idxs).extend(idxs)

    tr_obs_flat = obs_all[train_idxs]
    tr_act_flat = actions[train_idxs]
    va_obs_flat = obs_all[val_idxs]
    va_act_flat = actions[val_idxs]

    obs_mean = tr_obs_flat.mean(0); obs_std = tr_obs_flat.std(0).clamp(min=1e-6)
    act_mean = tr_act_flat.mean(0); act_std = tr_act_flat.std(0).clamp(min=1e-6)

    tr_obs = ((tr_obs_flat - obs_mean) / obs_std).to(device)
    tr_act = ((tr_act_flat - act_mean) / act_std).to(device)
    va_obs = ((va_obs_flat - obs_mean) / obs_std).to(device)
    va_act = ((va_act_flat - act_mean) / act_std).to(device)

    tr_obs_c, tr_act_c = build_chunked(tr_obs, tr_act, HORIZON, N_OBS)
    va_obs_c, va_act_c = build_chunked(va_obs, va_act, HORIZON, N_OBS)
    logger.info(f"  Train chunks: {len(tr_obs_c)}, Val chunks: {len(va_obs_c)}")

    model = BCUNet(action_dim=12, state_dim=state_dim, horizon=HORIZON,
                   n_obs_steps=N_OBS).to(device)
    bs, lr, max_ep, patience = 128, 1e-3, 100, 30
    ns  = len(tr_obs_c)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    nbpe = max(1, ns // bs)
    lr_s = get_cosine_schedule_with_warmup(opt, min(10, max_ep // 10), max_ep * nbpe)

    best_val, best_state, best_ep, wait = float("inf"), None, 0, 0
    t0 = time.time()

    for epoch in range(max_ep):
        model.train()
        perm2 = torch.randperm(ns, device=device)
        el, nb = 0.0, 0
        for b in range(0, ns - bs + 1, bs):
            idx = perm2[b : b + bs]
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                loss = nn.functional.mse_loss(model(tr_obs_c[idx]), tr_act_c[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); lr_s.step()
            el += loss.item(); nb += 1

        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            vl = nn.functional.mse_loss(model(va_obs_c), va_act_c).item()

        if vl < best_val:
            best_val, best_state, best_ep, wait = vl, deepcopy(model.state_dict()), epoch + 1, 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    logger.info(f"  Done: best_ep={best_ep} val={best_val:.4f} in {time.time()-t0:.0f}s")

    ckpt = {
        "model":         deepcopy(model).cpu(),
        "obs_mean":      obs_mean,
        "obs_std":       obs_std,
        "act_mean":      act_mean,
        "act_std":       act_std,
        "state_dim":     state_dim,
        "feature_names": feature_names,
    }
    torch.save(ckpt, ckpt_path)
    logger.info(f"  Saved checkpoint → {ckpt_path}")
    return ckpt


# ── Env creation with cameras ────────────────────────────────────────────────

def make_render_env(seed):
    import robocasa  # noqa: F401
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    return robosuite.make(
        env_name="OpenCabinet",
        robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="robot0_agentview_center",
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,   # we render manually below
        camera_depths=False,
        control_freq=20,
        seed=seed,
        obj_instance_split="pretrain",
        layout_ids=-2,
        style_ids=-2,
    )


# ── Observation builders per feature set ────────────────────────────────────

from diffusion_policy.evaluation import (
    STATE_KEYS_ORDERED,
    dataset_action_to_env_action,
    get_handle_pos_from_env,
    check_one_door_success,
)


def build_obs_proprio(obs, env, ctx):
    return np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED]).astype(np.float32)


def build_obs_door_pos(obs, env, ctx):
    proprio = np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED]).astype(np.float32)
    door_pos = obs["door_obj_pos"].flatten().astype(np.float32)
    return np.concatenate([proprio, door_pos])


def build_obs_handle(obs, env, ctx):
    proprio  = np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED]).astype(np.float32)
    eef_pos  = obs.get("robot0_eef_pos", np.zeros(3)).flatten().astype(np.float32)
    hp, ctx["active_site"] = get_handle_pos_from_env(env, ctx.get("active_site"), eef_pos)
    h2e = (eef_pos - hp).astype(np.float32)
    return np.concatenate([proprio, hp, h2e])


OBS_BUILDERS = {
    "proprio":   build_obs_proprio,
    "door_pos":  build_obs_door_pos,
    "handle":    build_obs_handle,
}


# ── Parallel seed scanner (no rendering) ─────────────────────────────────────

def _fast_scan_worker(args):
    """Spawn-safe: evaluate one seed without rendering; return (seed, success, dr)."""
    seed, ckpt_path, feature_type, max_steps = args

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    import sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))

    import numpy as _np
    import torch as _torch
    from collections import deque as _deque

    from diffusion_policy.evaluation import (
        STATE_KEYS_ORDERED as _SK,
        dataset_action_to_env_action as _d2e,
        get_handle_pos_from_env as _ghp,
        check_one_door_success as _chk,
    )

    def _build_obs(obs, env, ctx, ftype):
        proprio = _np.concatenate([obs[k].flatten() for k in _SK]).astype(_np.float32)
        if ftype == "proprio":
            return proprio
        elif ftype == "door_pos":
            return _np.concatenate([proprio, obs["door_obj_pos"].flatten().astype(_np.float32)])
        else:  # handle
            eef = obs.get("robot0_eef_pos", _np.zeros(3)).flatten().astype(_np.float32)
            hp, ctx["active_site"] = _ghp(env, ctx.get("active_site"), eef)
            return _np.concatenate([proprio, hp, (eef - hp).astype(_np.float32)])

    import robocasa  # noqa
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    env = robosuite.make(
        env_name="OpenCabinet", robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        has_renderer=False, has_offscreen_renderer=False,
        ignore_done=True, use_object_obs=True, use_camera_obs=False,
        camera_depths=False, control_freq=20, seed=seed,
        obj_instance_split="pretrain", layout_ids=-2, style_ids=-2,
    )

    ckpt = _torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model = ckpt["model"].eval()
    obs_mean = ckpt["obs_mean"].float()
    obs_std  = ckpt["obs_std"].float()
    act_mean = ckpt["act_mean"].float()
    act_std  = ckpt["act_std"].float()

    obs = env.reset()
    ctx = {}
    aug = _build_obs(obs, env, ctx, feature_type)
    oh = _deque([aug] * N_OBS, maxlen=N_OBS)
    aq = _deque()

    eef0 = obs.get("robot0_eef_pos", _np.zeros(3)).flatten().astype(_np.float32)
    hp0, _ = _ghp(env, None, eef0)
    init_dist = float(_np.linalg.norm(eef0 - hp0))
    min_dist = init_dist
    success = False

    for step in range(max_steps):
        if not aq:
            oc = _torch.from_numpy(_np.stack(list(oh))).float().unsqueeze(0)
            oc = (oc - obs_mean) / obs_std
            with _torch.no_grad():
                acts = model(oc).reshape(1, HORIZON, 12) * act_std + act_mean
            for i in range(min(N_ACT_STEPS, HORIZON)):
                aq.append(acts[0, i].numpy())
        env_act = _d2e(aq.popleft())
        env_act = _np.clip(env_act, -1.0, 1.0)
        obs, _, _, _ = env.step(env_act)
        aug = _build_obs(obs, env, ctx, feature_type)
        oh.append(aug)
        eef = obs.get("robot0_eef_pos", _np.zeros(3)).flatten().astype(_np.float32)
        hp, ctx["active_site"] = _ghp(env, ctx.get("active_site"), eef)
        min_dist = min(min_dist, float(_np.linalg.norm(eef - hp)))
        if _chk(env):
            success = True
            break

    env.close()
    dr = (1 - min_dist / max(init_dist, 1e-6)) * 100
    return seed, success, dr


def scan_seeds_parallel(ckpt_path, feature_type, seeds, max_steps=400, n_workers=8):
    """Evaluate multiple seeds in parallel (no rendering). Returns list of (seed, success, dr)."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    args = [(s, ckpt_path, feature_type, max_steps) for s in seeds]
    with ctx.Pool(min(n_workers, len(seeds))) as pool:
        results = pool.map(_fast_scan_worker, args)
    return results  # [(seed, success, dr), ...]


# ── Rollout with frame capture ───────────────────────────────────────────────

def run_rollout(ckpt, obs_builder, seed, max_steps=500, max_frames=None,
                post_success_steps=0, render_hw=None):
    """Run one episode; return (frames, success, dr_pct).

    post_success_steps: keep executing the policy after success so the door
    continues to swing open visibly (0 = stop immediately as before).
    render_hw: (height, width) for frame capture; defaults to (RENDER_H, RENDER_W).
    """
    rh, rw = render_hw if render_hw else (RENDER_H, RENDER_W)
    model = ckpt["model"].eval()
    obs_mean = ckpt["obs_mean"].float()
    obs_std  = ckpt["obs_std"].float()
    act_mean = ckpt["act_mean"].float()
    act_std  = ckpt["act_std"].float()

    env = make_render_env(seed)
    obs = env.reset()

    ctx = {}  # mutable state for obs_builder (e.g. active_site)
    aug = obs_builder(obs, env, ctx)
    oh  = deque([aug] * N_OBS, maxlen=N_OBS)
    aq  = deque()

    eef0 = obs.get("robot0_eef_pos", np.zeros(3)).flatten().astype(np.float32)
    hp0, _ = get_handle_pos_from_env(env, None, eef0)
    init_dist = float(np.linalg.norm(eef0 - hp0))
    min_dist  = init_dist

    frames  = []
    success = False

    for step in range(max_steps):
        if not aq:
            oc   = torch.from_numpy(np.stack(list(oh))).float().unsqueeze(0)
            oc   = (oc - obs_mean) / obs_std
            with torch.no_grad():
                acts = model(oc).reshape(1, HORIZON, 12) * act_std + act_mean
            for i in range(min(N_ACT_STEPS, HORIZON)):
                aq.append(acts[0, i].numpy())

        env_act = dataset_action_to_env_action(aq.popleft())
        env_act = np.clip(env_act, -1.0, 1.0)
        obs, _, _, _ = env.step(env_act)

        aug = obs_builder(obs, env, ctx)
        oh.append(aug)

        # Capture frame
        frame = env.sim.render(height=rh, width=rw,
                               camera_name="robot0_agentview_center")[::-1]
        frames.append(frame.copy())
        if max_frames and len(frames) >= max_frames:
            break

        # Track distance
        eef_cur = obs.get("robot0_eef_pos", np.zeros(3)).flatten().astype(np.float32)
        hp_cur, ctx["active_site"] = get_handle_pos_from_env(
            env, ctx.get("active_site"), eef_cur)
        min_dist = min(min_dist, float(np.linalg.norm(eef_cur - hp_cur)))

        if check_one_door_success(env):
            success = True
            # Keep running the policy so the door continues to swing open
            extra = post_success_steps if post_success_steps > 0 else 30
            for _ in range(extra):
                if not aq:
                    oc = torch.from_numpy(np.stack(list(oh))).float().unsqueeze(0)
                    oc = (oc - obs_mean) / obs_std
                    with torch.no_grad():
                        acts = model(oc).reshape(1, HORIZON, 12) * act_std + act_mean
                    for i in range(min(N_ACT_STEPS, HORIZON)):
                        aq.append(acts[0, i].numpy())
                env_act = dataset_action_to_env_action(aq.popleft())
                env_act = np.clip(env_act, -1.0, 1.0)
                obs, _, _, _ = env.step(env_act)
                aug = obs_builder(obs, env, ctx)
                oh.append(aug)
                frame = env.sim.render(height=rh, width=rw,
                                       camera_name="robot0_agentview_center")[::-1]
                frames.append(frame.copy())
            break

    env.close()
    dr = (1 - min_dist / max(init_dist, 1e-6)) * 100
    logger.info(f"  seed={seed} success={success} DR={dr:.0f}% frames={len(frames)}")
    return frames, success, dr


# ── GIF / MP4 writers ────────────────────────────────────────────────────────

def save_gif(frames, path, fps=GIF_FPS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"Writing GIF ({len(frames)} frames) → {path}")
    with imageio.get_writer(path, mode="I", fps=fps, loop=0) as w:
        for f in frames:
            w.append_data(f)
    size_kb = os.path.getsize(path) // 1024
    logger.info(f"  Saved {size_kb} KB")


def save_mp4(frames, path, fps=MP4_FPS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"Writing MP4 ({len(frames)} frames) → {path}")
    with imageio.get_writer(path, fps=fps, codec="libx264",
                             quality=8, macro_block_size=None) as w:
        for f in frames:
            w.append_data(f)
    size_kb = os.path.getsize(path) // 1024
    logger.info(f"  Saved {size_kb} KB")


# ── Mode implementations ──────────────────────────────────────────────────────

def mode_proprio_failure(data):
    """Train proprio-only model; find a visually interesting failure; save GIF."""
    _ensure_file_in_tmp("media_proprio_only.pt")
    ckpt_path = os.path.join(SAVE_DIR, "media_proprio_only.pt")
    if os.path.exists(ckpt_path):
        logger.info(f"Loading cached checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=False)
    else:
        ckpt = train_model(data, feature_names=["proprio"], ckpt_path=ckpt_path)

    out_path = os.path.join(ASSETS_DIR, "proprio-failure.gif")

    # Phase 14 data: seeds 17,19,25,28,35,78,84,90 had DR < 25%
    # Try seeds known to fail clearly; fall back to any low-DR episode
    candidate_seeds = [18, 20, 26, 85, 91, 36, 0, 1, 5]
    best_frames, best_dr = None, 999.0

    for seed in candidate_seeds:
        logger.info(f"Trying proprio-failure seed={seed}...")
        frames, success, dr = run_rollout(ckpt, build_obs_proprio, seed,
                                          max_steps=300, max_frames=200)
        if success:
            logger.info(f"  Skipping (succeeded unexpectedly at seed={seed})")
            continue
        # Want a failure with some movement but clearly lost (DR < 30%)
        if dr < 30.0:
            logger.info(f"  Good failure seed: DR={dr:.0f}%")
            best_frames, best_dr = frames, dr
            break
        # Also accept medium DR if we don't find a clear wandering failure
        if best_frames is None or dr < best_dr:
            best_frames, best_dr = frames, dr

    if best_frames is None:
        raise RuntimeError("Could not find any failure rollout for proprio-only model")

    save_gif(best_frames, out_path)
    logger.info(f"\n==> proprio-failure.gif saved (DR={best_dr:.0f}%)")
    return best_frames


def mode_door_near_miss(data):
    """Train proprio+door_pos model; find a near-miss; save GIF."""
    _ensure_file_in_tmp("media_door_pos.pt")
    ckpt_path = os.path.join(SAVE_DIR, "media_door_pos.pt")
    if os.path.exists(ckpt_path):
        logger.info(f"Loading cached checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=False)
    else:
        ckpt = train_model(data, feature_names=["proprio", "door_pos"], ckpt_path=ckpt_path)

    out_path = os.path.join(ASSETS_DIR, "door-centroid-near-miss.gif")

    # Parallel-scan seeds 0-29 to find DR 70-80% (clearly approaches but fails)
    tmp_ckpt_path = os.path.join(SAVE_DIR, "_scan_door_pos_tmp.pt")
    torch.save(ckpt, tmp_ckpt_path)
    logger.info("Parallel-scanning 30 seeds for door_pos model...")
    scan_results = scan_seeds_parallel(tmp_ckpt_path, "door_pos", list(range(30)),
                                       max_steps=400, n_workers=8)
    os.remove(tmp_ckpt_path)

    for seed, success, dr in scan_results:
        logger.info(f"  seed={seed} success={success} DR={dr:.0f}%")

    # Pick best seed in 70-80% range (not a success)
    candidates = [(s, dr) for s, succ, dr in scan_results if not succ and 70.0 <= dr <= 80.0]
    if not candidates:
        # Widen to 60-85%
        candidates = [(s, dr) for s, succ, dr in scan_results if not succ and 60.0 <= dr <= 85.0]
    if not candidates:
        # Fall back to highest DR non-success
        candidates = [(s, dr) for s, succ, dr in scan_results if not succ]

    best_seed, best_dr = max(candidates, key=lambda x: x[1])
    logger.info(f"Selected seed={best_seed} DR={best_dr:.0f}% — rendering...")
    best_frames, _, _ = run_rollout(ckpt, build_obs_door_pos, best_seed,
                                    max_steps=400, max_frames=240)
    found_near_miss = True

    if best_frames is None:
        raise RuntimeError("Could not find any rollout for door-pos model")

    save_gif(best_frames, out_path)
    logger.info(f"\n==> door-centroid-near-miss.gif saved (DR={best_dr:.0f}%)")
    return best_frames


def mode_final_success(data):
    """Train best F3 model; find a clean success; save GIF."""
    _ensure_file_in_tmp("best_f3_bc_unet.pt")
    ckpt_path = os.path.join(SAVE_DIR, "best_f3_bc_unet.pt")
    if os.path.exists(ckpt_path):
        logger.info(f"Loading cached checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=False)
    else:
        ckpt = train_model(data, feature_names=["proprio", "handle_pos", "handle_to_eef"],
                           ckpt_path=ckpt_path)

    out_path = os.path.join(ASSETS_DIR, "final-success.gif")

    # Phase 14: seeds 3,5,7,10,15,17,23... succeeded (ep4,6,8,11,16,18,24)
    priority_seeds = [3, 5, 7, 10, 0, 1, 2, 4, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    for seed in priority_seeds:
        logger.info(f"Trying final-success seed={seed}...")
        frames, success, dr = run_rollout(ckpt, build_obs_handle, seed,
                                          max_steps=500, post_success_steps=150)
        if success:
            logger.info(f"  SUCCESS at seed={seed}, DR={dr:.0f}%")
            save_gif(frames, out_path)
            logger.info(f"\n==> final-success.gif saved")
            return frames
        logger.info(f"  Failed (DR={dr:.0f}%)")

    raise RuntimeError("No success found across 20 seeds for F3 model")


def _render_clip_1080p(ckpt, obs_builder, seed, max_steps, max_frames=None,
                       post_success_steps=0):
    """Render one clip at 1080p (1080h x 1620w, maintaining 2:3 aspect ratio)."""
    # 1080 * (480/320) = 1620; both divisible by 2
    return run_rollout(ckpt, obs_builder, seed, max_steps=max_steps,
                       max_frames=max_frames, post_success_steps=post_success_steps,
                       render_hw=(1080, 1620))


def _clip_render_worker(args):
    """Spawn-safe: render one 1080p clip and save to cache_path as .npy.

    args: (clip_name, ckpt_path, feature_type, seed,
           max_steps, max_frames, post_success_steps, cache_path)
    """
    clip_name, ckpt_path, feature_type, seed, max_steps, max_frames, post_suc, cache_path = args

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    import sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))

    import logging as _log
    _log.basicConfig(level=_log.INFO, stream=_sys.stdout, force=True)
    _logger = _log.getLogger(clip_name)

    import numpy as _np
    import torch as _torch
    from collections import deque as _deque

    from diffusion_policy.evaluation import (
        STATE_KEYS_ORDERED as _SK,
        dataset_action_to_env_action as _d2e,
        get_handle_pos_from_env as _ghp,
        check_one_door_success as _chk,
    )

    import robocasa  # noqa
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    _logger.info(f"[{clip_name}] starting render seed={seed} at 1080p")

    env = robosuite.make(
        env_name="OpenCabinet", robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        has_renderer=False, has_offscreen_renderer=True,
        render_camera="robot0_agentview_center",
        ignore_done=True, use_object_obs=True, use_camera_obs=False,
        camera_depths=False, control_freq=20, seed=seed,
        obj_instance_split="pretrain", layout_ids=-2, style_ids=-2,
    )

    ckpt = _torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model = ckpt["model"].eval()
    obs_mean = ckpt["obs_mean"].float()
    obs_std  = ckpt["obs_std"].float()
    act_mean = ckpt["act_mean"].float()
    act_std  = ckpt["act_std"].float()

    def _build_obs(obs, ctx):
        proprio = _np.concatenate([obs[k].flatten() for k in _SK]).astype(_np.float32)
        if feature_type == "proprio":
            return proprio
        elif feature_type == "door_pos":
            return _np.concatenate([proprio, obs["door_obj_pos"].flatten().astype(_np.float32)])
        else:  # handle
            eef = obs.get("robot0_eef_pos", _np.zeros(3)).flatten().astype(_np.float32)
            hp, ctx["active_site"] = _ghp(env, ctx.get("active_site"), eef)
            return _np.concatenate([proprio, hp, (eef - hp).astype(_np.float32)])

    obs = env.reset()
    ctx = {}
    aug = _build_obs(obs, ctx)
    oh  = _deque([aug] * N_OBS, maxlen=N_OBS)
    aq  = _deque()

    frames  = []
    success = False

    for step in range(max_steps):
        if not aq:
            oc = _torch.from_numpy(_np.stack(list(oh))).float().unsqueeze(0)
            oc = (oc - obs_mean) / obs_std
            with _torch.no_grad():
                acts = model(oc).reshape(1, HORIZON, 12) * act_std + act_mean
            for i in range(min(N_ACT_STEPS, HORIZON)):
                aq.append(acts[0, i].numpy())

        env_act = _d2e(aq.popleft())
        env_act = _np.clip(env_act, -1.0, 1.0)
        obs, _, _, _ = env.step(env_act)
        aug = _build_obs(obs, ctx)
        oh.append(aug)

        frame = env.sim.render(height=1080, width=1620,
                               camera_name="robot0_agentview_center")[::-1]
        frames.append(frame.copy())
        if max_frames and len(frames) >= max_frames:
            break

        if _chk(env):
            success = True
            for _ in range(post_suc):
                if not aq:
                    oc = _torch.from_numpy(_np.stack(list(oh))).float().unsqueeze(0)
                    oc = (oc - obs_mean) / obs_std
                    with _torch.no_grad():
                        acts = model(oc).reshape(1, HORIZON, 12) * act_std + act_mean
                    for i in range(min(N_ACT_STEPS, HORIZON)):
                        aq.append(acts[0, i].numpy())
                env_act = _d2e(aq.popleft())
                env_act = _np.clip(env_act, -1.0, 1.0)
                obs, _, _, _ = env.step(env_act)
                aug = _build_obs(obs, ctx)
                oh.append(aug)
                frame = env.sim.render(height=1080, width=1620,
                                       camera_name="robot0_agentview_center")[::-1]
                frames.append(frame.copy())
            break

    env.close()
    _np.save(cache_path, _np.array(frames, dtype=_np.uint8))
    _logger.info(f"[{clip_name}] done: {len(frames)} frames → {cache_path} (success={success})")
    return len(frames)


def mode_teaser_mp4(data):
    """Render all three clips in parallel at 1080p (cached), then compose MP4.

    Delete a clip's .npy cache to force re-render of just that clip.
    """
    out_path = os.path.join(ASSETS_DIR, "final-demo.mp4")

    for fname in ["media_proprio_only.pt", "media_door_pos.pt", "best_f3_bc_unet.pt"]:
        _ensure_file_in_tmp(fname)
    for ckpt_file in [
        os.path.join(SAVE_DIR, "media_proprio_only.pt"),
        os.path.join(SAVE_DIR, "media_door_pos.pt"),
        os.path.join(SAVE_DIR, "best_f3_bc_unet.pt"),
    ]:
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(
                f"Missing checkpoint {ckpt_file}. Run the individual modes first."
            )

    clips = [
        # (clip_name,  ckpt_file,            feature_type, seed, max_steps, max_frames, post_suc, cache)
        ("fail",  "media_proprio_only.pt", "proprio",  91,  300, 200, 0,   "clip_fail_1080p.npy"),
        ("near",  "media_door_pos.pt",     "door_pos", 15,  400, 240, 0,   "clip_near_1080p.npy"),
        ("suc",   "best_f3_bc_unet.pt",    "handle",   10,  500, None, 150, "clip_suc_1080p.npy"),
    ]

    to_render = []
    for name, ckpt_file, ftype, seed, ms, mf, ps, cache_name in clips:
        cache_path = os.path.join(SAVE_DIR, cache_name)
        if os.path.exists(cache_path):
            logger.info(f"  [{name}] cache found — skipping render: {cache_path}")
        else:
            to_render.append((
                name,
                os.path.join(SAVE_DIR, ckpt_file),
                ftype, seed, ms, mf, ps,
                cache_path,
            ))

    if to_render:
        logger.info(f"=== Rendering {len(to_render)} clip(s) in parallel at 1080p ===")
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(to_render)) as pool:
            pool.map(_clip_render_worker, to_render)
    else:
        logger.info("All clips cached — skipping rendering.")

    # ── Compose ───────────────────────────────────────────────────────────────
    logger.info("Loading clips and composing MP4...")
    fail_frames = list(np.load(os.path.join(SAVE_DIR, "clip_fail_1080p.npy")))
    near_frames = list(np.load(os.path.join(SAVE_DIR, "clip_near_1080p.npy")))
    suc_frames  = list(np.load(os.path.join(SAVE_DIR, "clip_suc_1080p.npy")))

    h, w = fail_frames[0].shape[:2]
    pause = [np.zeros((h, w, 3), dtype=np.uint8)] * 10

    all_frames = (
        fail_frames[:200] + pause +
        near_frames[:200] + pause +
        suc_frames[:212]
    )

    save_mp4(all_frames, out_path)
    logger.info(f"\n==> final-demo.mp4 saved at {w}x{h}, {len(all_frames)} frames")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["proprio_failure", "door_near_miss",
                                 "final_success", "teaser_mp4"],
                        help="Which asset to generate")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain even if checkpoint exists")
    args = parser.parse_args()

    if args.retrain:
        for f in [
            os.path.join(SAVE_DIR, "media_proprio_only.pt"),
            os.path.join(SAVE_DIR, "media_door_pos.pt"),
            os.path.join(SAVE_DIR, "best_f3_bc_unet.pt"),
        ]:
            if os.path.exists(f):
                os.remove(f)
                logger.info(f"Removed {f}")

    os.makedirs(ASSETS_DIR, exist_ok=True)
    logger.info(f"Assets dir: {ASSETS_DIR}")

    if args.mode == "proprio_failure":
        logger.info("Loading proprio from parquet (no preprocessing needed)...")
        data = load_data_proprio_only()
        mode_proprio_failure(data)
    elif args.mode in ("door_near_miss", "final_success", "teaser_mp4"):
        logger.info("Loading full preprocessed dataset...")
        data = load_data()
        if args.mode == "door_near_miss":
            mode_door_near_miss(data)
        elif args.mode == "final_success":
            mode_final_success(data)
        elif args.mode == "teaser_mp4":
            mode_teaser_mp4(data)


if __name__ == "__main__":
    main()
