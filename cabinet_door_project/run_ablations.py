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


ablations = [
    ("A2_mlp", "mlp", 16, 2, 3000),
    ("A3_h8", "unet", 8, 2, 3000),
    ("A4_h32", "unet", 32, 2, 3000),
    ("A5_obs1", "unet", 16, 1, 3000),
    ("A6_5k", "unet", 16, 2, 5000),
]

for name, backbone, horizon, n_obs, epochs in ablations:
    all_obs, all_act = build_dataset(horizon, n_obs)
    ns = all_obs.shape[0]
    print(
        f"=== {name} (bb={backbone}, h={horizon}, obs={n_obs}, ep={epochs}, n={ns}) ===",
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

print("ALL ABLATIONS DONE!", flush=True)
