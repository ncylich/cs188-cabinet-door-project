"""Fast ablation sweep: BS=128, LR=1e-3, 100 epochs, eval 5 rollouts each."""
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
from diffusion_policy.models.mlp import MLPNoiseNet
from diffusion_policy.training import build_scheduler, EMA, get_cosine_schedule_with_warmup
from diffusion_policy.evaluation import create_env, extract_state, dataset_action_to_env_action

device = torch.device("cuda")
BS = 128
LR = 1e-3
EPOCHS = 100
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


@torch.no_grad()
def eval_checkpoint(model, sched, horizon, n_obs, n_action_steps=8):
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
                den = sched.denoise_ddim(model, xT, oc, num_inference_steps=16)
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


def run_config(name, backbone, horizon, n_obs, channels=(256, 512, 1024)):
    all_obs, all_act = build_dataset(horizon, n_obs)
    ns = all_obs.shape[0]
    n_action_steps = min(8, horizon)

    print(f"\n{'='*60}", flush=True)
    print(f"{name} (bb={backbone}, h={horizon}, obs={n_obs}, n={ns})", flush=True)
    print(f"{'='*60}", flush=True)

    if backbone == "mlp":
        model = MLPNoiseNet(
            action_dim=12, state_dim=19, horizon=horizon, n_obs_steps=n_obs,
            hidden_dim=512, n_layers=4,
        ).to(device)
    else:
        model = UNetNoiseNet(
            action_dim=12, state_dim=19, horizon=horizon, n_obs_steps=n_obs,
            channels=channels,
        ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params/1e6:.1f}M, BS={BS}, LR={LR}, epochs={EPOCHS}", flush=True)

    sched = build_scheduler(DiffusionConfig(beta_schedule="squared_cosine"))
    ema = EMA(model, decay=0.9999)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    nbpe = max(1, ns // BS)
    lr_s = get_cosine_schedule_with_warmup(opt, 10, EPOCHS * nbpe)
    best = float("inf")
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        el = 0.0
        nb = 0
        perm = torch.randperm(ns, device=device)
        for bs_start in range(0, ns - BS + 1, BS):
            idx = perm[bs_start : bs_start + BS]
            noise = torch.randn_like(all_act[idx])
            tt = torch.randint(0, sched.num_train_steps, (BS,), device=device)
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
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  Ep {epoch+1}/{EPOCHS}  loss={avg:.6f}  t={time.time()-t0:.0f}s", flush=True)

    train_time = time.time() - t0
    print(f"  Train done: best_loss={best:.6f} in {train_time:.0f}s", flush=True)

    # Apply EMA and eval
    ema.apply(model)
    succ, avg_dr = eval_checkpoint(model, sched, horizon, n_obs, n_action_steps)
    eval_time = time.time() - t0 - train_time
    print(f"  Result: {succ}/{N_EVAL} success, avg_dist_reduction={avg_dr:.0f}%", flush=True)
    print(f"  Time: train={train_time:.0f}s, eval={eval_time:.0f}s, total={time.time()-t0:.0f}s", flush=True)

    del model, opt, ema, all_obs, all_act
    torch.cuda.empty_cache()
    return name, best, succ, avg_dr


# ========== RUN ALL CONFIGS ==========
results = []

configs = [
    # Round 1: model size sweep (pick best, then ablate other dims)
    ("med_unet", "unet", 16, 2, (128, 256, 512)),     # 8M params, fastest
    ("big_unet", "unet", 16, 2, (256, 512, 1024)),     # 30M params
    ("small_unet", "unet", 16, 2, (64, 128, 256)),     # 2.3M params
]

for name, backbone, horizon, n_obs, channels in configs:
    r = run_config(name, backbone, horizon, n_obs, channels or (256, 512, 1024))
    results.append(r)

# ========== SUMMARY ==========
print(f"\n{'='*60}", flush=True)
print("FAST ABLATION SUMMARY (BS=128, LR=1e-3, 100 epochs)", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Name':<18} {'Loss':>8} {'Success':>8} {'DistRed':>8}", flush=True)
print("-" * 46, flush=True)
for name, loss, succ, dr in results:
    print(f"{name:<18} {loss:>8.4f} {succ:>5}/{N_EVAL}  {dr:>6.0f}%", flush=True)
print(flush=True)
