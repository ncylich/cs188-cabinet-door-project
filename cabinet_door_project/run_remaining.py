"""Run remaining ablation training + evaluate ALL ablation checkpoints sequentially."""
import torch
import logging
import time
import os
import sys
import numpy as np
import torch.nn as nn

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

from diffusion_policy.config import DiffusionConfig
from diffusion_policy.data import get_dataset_path, load_episodes, load_stats, Normalizer
from diffusion_policy.models.unet import UNetNoiseNet
from diffusion_policy.models.mlp import MLPNoiseNet
from diffusion_policy.training import build_scheduler, EMA, get_cosine_schedule_with_warmup

device = torch.device("cuda")
BS = 2048
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


def train_one(name, backbone, horizon, n_obs, epochs):
    ckpt_path = f"/tmp/diffusion_policy_checkpoints/{name}/best.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Skip if trained past 80% of target epochs
        if ckpt.get("epoch", 0) >= epochs * 0.8:
            print(f"SKIP {name}: already trained (epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f})", flush=True)
            return

    all_obs, all_act = build_dataset(horizon, n_obs)
    ns = all_obs.shape[0]
    print(
        f"=== TRAIN {name} (bb={backbone}, h={horizon}, obs={n_obs}, ep={epochs}, n={ns}) ===",
        flush=True,
    )

    if backbone == "mlp":
        model = MLPNoiseNet(
            action_dim=12, state_dim=19, horizon=horizon, n_obs_steps=n_obs,
            hidden_dim=512, n_layers=4,
        ).to(device)
    else:
        model = UNetNoiseNet(
            action_dim=12, state_dim=19, horizon=horizon, n_obs_steps=n_obs,
        ).to(device)

    sched = build_scheduler(DiffusionConfig(beta_schedule="squared_cosine"))
    ema = EMA(model, decay=0.9999)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    nbpe = max(1, ns // BS)
    lr_s = get_cosine_schedule_with_warmup(opt, 500, epochs * nbpe)
    ckpt_dir = f"/tmp/diffusion_policy_checkpoints/{name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    best = float("inf")
    t0 = time.time()

    for epoch in range(epochs):
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
            opt.step()
            lr_s.step()
            ema.update(model)
            el += loss.item()
            nb += 1
        avg = el / max(nb, 1)
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(
                f"  Ep {epoch+1}/{epochs}  loss={avg:.6f}  t={time.time()-t0:.0f}s",
                flush=True,
            )
        if avg < best:
            best = avg
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "state_dim": 19,
                    "obs_mean": obs_mean,
                    "obs_std": obs_std,
                    "act_mean": act_mean,
                    "act_std": act_std,
                    "horizon": horizon,
                    "n_obs_steps": n_obs,
                    "backbone": backbone,
                    "epoch": epoch,
                    "loss": avg,
                },
                os.path.join(ckpt_dir, "best.pt"),
            )

    print(f"{name}: best={best:.6f} in {time.time()-t0:.0f}s", flush=True)
    del model, opt, ema, all_obs, all_act
    torch.cuda.empty_cache()


# ========== PHASE 1: Train remaining ablations ==========
print("\n" + "=" * 60, flush=True)
print("PHASE 1: Training remaining ablations", flush=True)
print("=" * 60, flush=True)

ablations = [
    ("A2_mlp", "mlp", 16, 2, 3000),
    ("A3_h8", "unet", 8, 2, 3000),
    ("A4_h32", "unet", 32, 2, 3000),
    ("A5_obs1", "unet", 16, 1, 3000),
    ("A6_5k", "unet", 16, 2, 5000),
]

for name, backbone, horizon, n_obs, epochs in ablations:
    train_one(name, backbone, horizon, n_obs, epochs)

print("\nALL ABLATION TRAINING DONE!", flush=True)

# ========== PHASE 2: Evaluate all checkpoints ==========
print("\n" + "=" * 60, flush=True)
print("PHASE 2: Evaluating all checkpoints (10 rollouts each)", flush=True)
print("=" * 60, flush=True)

# Clean up GPU memory before eval
torch.cuda.empty_cache()

from eval_oracle import run_eval

eval_configs = [
    ("A2_mlp", "/tmp/diffusion_policy_checkpoints/A2_mlp/best.pt"),
    ("A3_h8", "/tmp/diffusion_policy_checkpoints/A3_h8/best.pt"),
    ("A4_h32", "/tmp/diffusion_policy_checkpoints/A4_h32/best.pt"),
    ("A5_obs1", "/tmp/diffusion_policy_checkpoints/A5_obs1/best.pt"),
    ("A6_5k", "/tmp/diffusion_policy_checkpoints/A6_5k/best.pt"),
]

results = {}
for name, ckpt_path in eval_configs:
    if not os.path.exists(ckpt_path):
        print(f"SKIP EVAL {name}: no checkpoint", flush=True)
        continue
    print(f"\n=== EVAL {name} ===", flush=True)
    try:
        r = run_eval(ckpt_path, num_rollouts=10, max_steps=500, split="pretrain", seed=0)
        results[name] = r
        print(f"{name}: {r['success_rate']*100:.0f}% success ({sum(r['successes'])}/10)", flush=True)
    except Exception as e:
        print(f"{name}: EVAL FAILED: {e}", flush=True)
        results[name] = {"success_rate": -1, "error": str(e)}

# ========== PHASE 3: Summary ==========
print("\n" + "=" * 60, flush=True)
print("ABLATION RESULTS SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"{'Name':<12} {'Success':>8} {'Avg Dist Red':>14}", flush=True)
print("-" * 40, flush=True)
for name, r in results.items():
    if r.get("success_rate", -1) >= 0:
        sr = r["success_rate"]
        avg_red = np.mean([d["reduction"] for d in r["distances"]])
        print(f"{name:<12} {sr*100:>7.0f}% {avg_red:>13.0f}%", flush=True)
    else:
        print(f"{name:<12} {'FAILED':>8}", flush=True)

print("\nALL DONE!", flush=True)
