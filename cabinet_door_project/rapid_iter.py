"""Rapid iteration: BC transformer (no diffusion) + aggressive diffusion variants.
Each config should finish train+eval in < 5 min."""
import torch
import logging
import time
import os
import sys
import numpy as np
import torch.nn as nn
from collections import deque

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

from diffusion_policy.config import DiffusionConfig
from diffusion_policy.data import get_dataset_path, load_episodes, load_stats, Normalizer
from diffusion_policy.models.unet import UNetNoiseNet
from diffusion_policy.models.transformer import TransformerNoiseNet
from diffusion_policy.training import build_scheduler, EMA, get_cosine_schedule_with_warmup
from diffusion_policy.evaluation import create_env, extract_state, dataset_action_to_env_action

device = torch.device("cuda")
N_EVAL = 3
MAX_EVAL_STEPS = 200

ds = get_dataset_path()
episodes = load_episodes(ds)

preproc = torch.load("/tmp/diffusion_policy_checkpoints/preprocessed_19dim.pt", weights_only=False)
obs_mean = preproc["obs_mean"]
obs_std = preproc["obs_std"]
act_mean = preproc["act_mean"]
act_std = preproc["act_std"]
sn = Normalizer(obs_mean, obs_std)
an = Normalizer(act_mean, act_std)

dp_data = np.load("/tmp/diffusion_policy_checkpoints/door_positions.npz")
door_positions = {int(k): v.astype(np.float32) for k, v in dp_data.items()}


def build_dataset(horizon, n_obs):
    obs_list, act_list = [], []
    for ep in episodes:
        eid = ep["episode_index"]
        dp = door_positions[eid]
        aug = np.concatenate(
            [ep["states"], np.tile(dp, (len(ep["states"]), 1))], axis=1
        ).astype(np.float32)
        actions = ep["actions"].astype(np.float32)
        for j in range(max(0, len(ep["states"]) - horizon - n_obs + 1)):
            obs_list.append(sn.normalize(torch.from_numpy(aug[j : j + n_obs])))
            act_list.append(
                an.normalize(
                    torch.from_numpy(actions[j + n_obs - 1 : j + n_obs - 1 + horizon])
                )
            )
    return torch.stack(obs_list).to(device), torch.stack(act_list).to(device)


# ============================================================
# BC Transformer: no diffusion, direct action prediction
# ============================================================
class BCTransformer(nn.Module):
    """Simple behavioral cloning transformer. Predicts actions directly."""
    def __init__(self, action_dim=12, state_dim=19, horizon=16, n_obs_steps=2,
                 n_layers=4, n_heads=4, d_model=128):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.d_model = d_model

        self.state_proj = nn.Linear(state_dim, d_model)
        # Learnable action queries (one per horizon step)
        self.action_queries = nn.Parameter(torch.randn(horizon, d_model) * 0.02)
        self.pos_embed = nn.Embedding(n_obs_steps + horizon, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, action_dim)

    def forward(self, obs_context):
        batch = obs_context.shape[0]
        if obs_context.dim() == 2:
            obs_context = obs_context.reshape(batch, self.n_obs_steps, -1)

        state_tokens = self.state_proj(obs_context)
        # Expand action queries for the batch
        aq = self.action_queries.unsqueeze(0).expand(batch, -1, -1)
        seq = torch.cat([state_tokens, aq], dim=1)
        seq_len = seq.shape[1]
        positions = torch.arange(seq_len, device=seq.device)
        seq = seq + self.pos_embed(positions).unsqueeze(0)

        out = self.transformer(seq)
        action_out = out[:, self.n_obs_steps:]
        return self.output_proj(action_out)


@torch.no_grad()
def eval_bc(model, horizon, n_obs, n_action_steps=8):
    """Evaluate a BC model (no diffusion)."""
    model.eval()
    env = create_env(split="pretrain", seed=0)
    successes = 0
    dist_reductions = []

    for ep in range(N_EVAL):
        obs = env.reset()
        state = extract_state(obs)
        dp = obs["door_obj_pos"].flatten().astype(np.float32)
        aug = np.concatenate([state, dp])
        oh = deque([aug] * n_obs, maxlen=n_obs)
        aq = deque()
        success = False
        d2e = obs.get("door_obj_to_robot0_eef_pos", np.zeros(3))
        init_dist = np.linalg.norm(d2e)
        min_dist = init_dist

        for step in range(MAX_EVAL_STEPS):
            if not aq:
                oc = sn.normalize(
                    torch.from_numpy(np.stack(list(oh))).float().unsqueeze(0).to(device)
                )
                acts = model(oc)  # Direct prediction, no denoising!
                acts = an.denormalize(acts.reshape(1, horizon, 12))
                for i in range(min(n_action_steps, horizon)):
                    aq.append(acts[0, i].cpu().numpy())

            env_act = dataset_action_to_env_action(aq.popleft())
            env_act = np.clip(env_act, -1.0, 1.0)
            obs, reward, done, info = env.step(env_act)
            state = extract_state(obs)
            dp = obs["door_obj_pos"].flatten().astype(np.float32)
            oh.append(np.concatenate([state, dp]))
            d2e = obs.get("door_obj_to_robot0_eef_pos", np.zeros(3))
            min_dist = min(min_dist, np.linalg.norm(d2e))
            if env._check_success():
                success = True
                break

        if success:
            successes += 1
        dr = (1 - min_dist / max(init_dist, 1e-6)) * 100
        dist_reductions.append(dr)
        s = "OK" if success else "X"
        print(f"    Ep{ep+1}: {s} d={init_dist:.2f}->{min_dist:.2f} ({dr:.0f}%)", flush=True)

    env.close()
    return successes, np.mean(dist_reductions)


@torch.no_grad()
def eval_diffusion(model, sched, horizon, n_obs, n_action_steps=8, ddim_steps=16):
    """Evaluate a diffusion model."""
    model.eval()
    env = create_env(split="pretrain", seed=0)
    successes = 0
    dist_reductions = []

    for ep in range(N_EVAL):
        obs = env.reset()
        state = extract_state(obs)
        dp = obs["door_obj_pos"].flatten().astype(np.float32)
        aug = np.concatenate([state, dp])
        oh = deque([aug] * n_obs, maxlen=n_obs)
        aq = deque()
        success = False
        d2e = obs.get("door_obj_to_robot0_eef_pos", np.zeros(3))
        init_dist = np.linalg.norm(d2e)
        min_dist = init_dist

        for step in range(MAX_EVAL_STEPS):
            if not aq:
                oc = sn.normalize(
                    torch.from_numpy(np.stack(list(oh))).float().unsqueeze(0).to(device)
                )
                xT = torch.randn(1, horizon, 12, device=device)
                den = sched.denoise_ddim(model, xT, oc, num_inference_steps=ddim_steps)
                acts = an.denormalize(den.reshape(1, horizon, 12))
                for i in range(min(n_action_steps, horizon)):
                    aq.append(acts[0, i].cpu().numpy())

            env_act = dataset_action_to_env_action(aq.popleft())
            env_act = np.clip(env_act, -1.0, 1.0)
            obs, reward, done, info = env.step(env_act)
            state = extract_state(obs)
            dp = obs["door_obj_pos"].flatten().astype(np.float32)
            oh.append(np.concatenate([state, dp]))
            d2e = obs.get("door_obj_to_robot0_eef_pos", np.zeros(3))
            min_dist = min(min_dist, np.linalg.norm(d2e))
            if env._check_success():
                success = True
                break

        if success:
            successes += 1
        dr = (1 - min_dist / max(init_dist, 1e-6)) * 100
        dist_reductions.append(dr)
        s = "OK" if success else "X"
        print(f"    Ep{ep+1}: {s} d={init_dist:.2f}->{min_dist:.2f} ({dr:.0f}%)", flush=True)

    env.close()
    return successes, np.mean(dist_reductions)


# ============================================================
# Training functions
# ============================================================

def train_bc(name, horizon=16, n_obs=2, epochs=10, bs=128, lr=1e-3,
             n_layers=4, n_heads=4, d_model=128, n_action_steps=8):
    """Train BC transformer (no diffusion)."""
    all_obs, all_act = build_dataset(horizon, n_obs)
    ns = all_obs.shape[0]

    print(f"\n{'='*60}", flush=True)
    print(f"BC: {name} (h={horizon}, obs={n_obs}, d={d_model}, L={n_layers}, ep={epochs})", flush=True)
    print(f"{'='*60}", flush=True)

    model = BCTransformer(
        action_dim=12, state_dim=19, horizon=horizon, n_obs_steps=n_obs,
        n_layers=n_layers, n_heads=n_heads, d_model=d_model,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params/1e6:.2f}M, BS={bs}, LR={lr}, epochs={epochs}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    nbpe = max(1, ns // bs)
    lr_s = get_cosine_schedule_with_warmup(opt, max(1, nbpe), epochs * nbpe)
    best = float("inf")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        el = 0.0
        nb = 0
        perm = torch.randperm(ns, device=device)
        for bs_start in range(0, ns - bs + 1, bs):
            idx = perm[bs_start : bs_start + bs]
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                pred = model(all_obs[idx])
                loss = nn.functional.mse_loss(pred, all_act[idx].reshape(pred.shape))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lr_s.step()
            el += loss.item()
            nb += 1
        avg = el / max(nb, 1)
        if avg < best:
            best = avg
        if epoch == 0 or (epoch + 1) % max(1, epochs // 4) == 0 or epoch == epochs - 1:
            print(f"  Ep {epoch+1}/{epochs}  loss={avg:.6f}  t={time.time()-t0:.0f}s", flush=True)

    train_time = time.time() - t0
    print(f"  Train done: best_loss={best:.6f} in {train_time:.0f}s", flush=True)

    succ, avg_dr = eval_bc(model, horizon, n_obs, n_action_steps)
    eval_time = time.time() - t0 - train_time
    print(f"  Result: {succ}/{N_EVAL} success, avg_dist_reduction={avg_dr:.0f}%", flush=True)
    print(f"  Time: train={train_time:.0f}s, eval={eval_time:.0f}s", flush=True)

    del model, opt, all_obs, all_act
    torch.cuda.empty_cache()
    return name, best, succ, avg_dr, train_time


def train_diffusion(name, backbone="unet", horizon=16, n_obs=2, epochs=50, bs=128, lr=1e-3,
                    channels=(64, 128, 256), n_action_steps=8, ddim_steps=8,
                    n_train_steps=100):
    """Train diffusion model with aggressive settings."""
    all_obs, all_act = build_dataset(horizon, n_obs)
    ns = all_obs.shape[0]

    print(f"\n{'='*60}", flush=True)
    print(f"DIFF: {name} (bb={backbone}, h={horizon}, obs={n_obs}, T={n_train_steps})", flush=True)
    print(f"{'='*60}", flush=True)

    if backbone == "transformer":
        model = TransformerNoiseNet(
            action_dim=12, state_dim=19, horizon=horizon, n_obs_steps=n_obs,
            n_layers=4, n_heads=4, d_model=128,
        ).to(device)
    else:
        model = UNetNoiseNet(
            action_dim=12, state_dim=19, horizon=horizon, n_obs_steps=n_obs,
            channels=channels,
        ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params/1e6:.2f}M, BS={bs}, LR={lr}, epochs={epochs}, T={n_train_steps}", flush=True)

    cfg = DiffusionConfig(beta_schedule="squared_cosine", num_diffusion_steps=n_train_steps)
    sched = build_scheduler(cfg)
    ema = EMA(model, decay=0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    nbpe = max(1, ns // bs)
    lr_s = get_cosine_schedule_with_warmup(opt, max(1, nbpe), epochs * nbpe)
    best = float("inf")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        el = 0.0
        nb = 0
        perm = torch.randperm(ns, device=device)
        for bs_start in range(0, ns - bs + 1, bs):
            idx = perm[bs_start : bs_start + bs]
            noise = torch.randn_like(all_act[idx])
            tt = torch.randint(0, sched.num_train_steps, (bs,), device=device)
            na = sched.add_noise(all_act[idx], noise, tt)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                pred = model(na, all_obs[idx], tt)
                loss = nn.functional.mse_loss(pred, noise.reshape(pred.shape))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lr_s.step()
            ema.update(model)
            el += loss.item()
            nb += 1
        avg = el / max(nb, 1)
        if avg < best:
            best = avg
        if epoch == 0 or (epoch + 1) % max(1, epochs // 4) == 0 or epoch == epochs - 1:
            print(f"  Ep {epoch+1}/{epochs}  loss={avg:.6f}  t={time.time()-t0:.0f}s", flush=True)

    train_time = time.time() - t0
    print(f"  Train done: best_loss={best:.6f} in {train_time:.0f}s", flush=True)

    ema.apply(model)
    succ, avg_dr = eval_diffusion(model, sched, horizon, n_obs, n_action_steps, ddim_steps)
    eval_time = time.time() - t0 - train_time
    print(f"  Result: {succ}/{N_EVAL} success, avg_dist_reduction={avg_dr:.0f}%", flush=True)
    print(f"  Time: train={train_time:.0f}s, eval={eval_time:.0f}s", flush=True)

    del model, opt, ema, all_obs, all_act
    torch.cuda.empty_cache()
    return name, best, succ, avg_dr, train_time


# ============================================================
# ROUND 1: BC Transformer sweep (no diffusion!)
# ============================================================
results = []

print("\n" + "#" * 60, flush=True)
print("# ROUND 1: BC Transformer (NO diffusion) — like friend's approach", flush=True)
print("#" * 60, flush=True)

bc_configs = [
    # (name, horizon, n_obs, epochs, bs, lr, n_layers, n_heads, d_model, n_action_steps)
    ("bc_tiny_2ep",    16, 2, 2,   128, 1e-3, 2, 4, 64,  8),  # Friend's ~2 epoch approach
    ("bc_tiny_10ep",   16, 2, 10,  128, 1e-3, 2, 4, 64,  8),  # Slightly more training
    ("bc_small_10ep",  16, 2, 10,  128, 1e-3, 4, 4, 128, 8),  # Bigger model
    ("bc_small_50ep",  16, 2, 50,  128, 1e-3, 4, 4, 128, 8),  # More epochs
    ("bc_med_10ep",    16, 2, 10,  128, 1e-3, 6, 4, 256, 8),  # Medium model
]

for name, h, n_obs, ep, bs, lr, nl, nh, dm, nas in bc_configs:
    r = train_bc(name, h, n_obs, ep, bs, lr, nl, nh, dm, nas)
    results.append(r)

# ============================================================
# ROUND 2: Aggressive diffusion (fewer timesteps, small model)
# ============================================================
print("\n" + "#" * 60, flush=True)
print("# ROUND 2: Aggressive diffusion (few timesteps, small model)", flush=True)
print("#" * 60, flush=True)

diff_configs = [
    # (name, backbone, horizon, n_obs, epochs, bs, lr, channels, n_action_steps, ddim_steps, n_train_steps)
    ("diff_T20_50ep",  "unet", 16, 2, 50, 128, 1e-3, (64, 128, 256), 8, 4, 20),  # Only 20 diffusion steps
    ("diff_T50_50ep",  "unet", 16, 2, 50, 128, 1e-3, (64, 128, 256), 8, 8, 50),  # 50 diffusion steps
    ("diff_xfmr_50ep", "transformer", 16, 2, 50, 128, 1e-3, None, 8, 8, 100),     # Transformer backbone
]

for name, bb, h, n_obs, ep, bs, lr, ch, nas, ddim, nts in diff_configs:
    r = train_diffusion(name, bb, h, n_obs, ep, bs, lr, ch or (64, 128, 256), nas, ddim, nts)
    results.append(r)

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*70}", flush=True)
print("RAPID ITERATION SUMMARY", flush=True)
print(f"{'='*70}", flush=True)
print(f"{'Name':<20} {'Loss':>8} {'Success':>8} {'DistRed':>8} {'TrainT':>8}", flush=True)
print("-" * 56, flush=True)
for name, loss, succ, dr, tt in results:
    print(f"{name:<20} {loss:>8.4f} {succ:>5}/{N_EVAL}  {dr:>6.0f}% {tt:>6.0f}s", flush=True)
print(flush=True)
