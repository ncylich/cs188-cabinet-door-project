"""GMM action head experiment: model multi-modal action distributions.

MSE loss averages modes → produces "in-between" actions that fail at contacts.
GMM loss fits a mixture of Gaussians → can capture multiple valid strategies.
"""
import torch, logging, time, os, sys
import numpy as np
import torch.nn as nn
from copy import deepcopy
from collections import deque
import math

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

from diffusion_policy.data import Normalizer
from diffusion_policy.evaluation import create_env, extract_state, dataset_action_to_env_action, STATE_KEYS_ORDERED
from diffusion_policy.training import get_cosine_schedule_with_warmup

device = torch.device('cuda')
SAVE_DIR = '/tmp/diffusion_policy_checkpoints'
data = torch.load(os.path.join(SAVE_DIR, 'preprocessed_all_states.pt'), weights_only=False)
features = data['features']
actions = data['actions']
ep_bounds = data['ep_boundaries']

feature_names = ['proprio', 'door_pos', 'door_to_eef_pos', 'gripper_to_door_dist']
obs_all = torch.cat([features[n] for n in feature_names], dim=-1)
state_dim = obs_all.shape[-1]

rng = np.random.RandomState(42)
n_eps = len(ep_bounds)
perm = rng.permutation(n_eps)
n_val = max(1, int(n_eps * 0.15))
val_eps = set(perm[:n_val])
train_idxs, val_idxs = [], []
for i, (eid, start, end) in enumerate(ep_bounds):
    idxs = list(range(start, end))
    (val_idxs if i in val_eps else train_idxs).extend(idxs)

train_obs, train_act = obs_all[train_idxs], actions[train_idxs]
val_obs, val_act = obs_all[val_idxs], actions[val_idxs]
obs_mean = train_obs.mean(0); obs_std = train_obs.std(0).clamp(min=1e-6)
act_mean = train_act.mean(0); act_std = train_act.std(0).clamp(min=1e-6)
train_obs = (train_obs - obs_mean) / obs_std; val_obs = (val_obs - obs_mean) / obs_std
train_act = (train_act - act_mean) / act_std; val_act = (val_act - act_mean) / act_std

H, NOBS = 16, 2
def chunk(obs, act):
    ol, al = [], []
    for j in range(max(0, len(obs) - H - NOBS + 1)):
        ol.append(obs[j:j+NOBS]); al.append(act[j+NOBS-1:j+NOBS-1+H])
    return torch.stack(ol).to(device), torch.stack(al).to(device)
train_co, train_ca = chunk(train_obs, train_act)
val_co, val_ca = chunk(val_obs, val_act)
ns, nv = len(train_co), len(val_co)


class BCMLPBaseline(nn.Module):
    """Standard BC MLP with MSE loss."""
    def __init__(self, action_dim, state_dim, horizon, n_obs_steps, hidden_dim=128, n_layers=3, dropout=0.3):
        super().__init__()
        self.horizon = horizon; self.action_dim = action_dim
        in_dim = state_dim * n_obs_steps
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, horizon * action_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, obs):
        return self.net(obs.reshape(obs.shape[0], -1)).reshape(-1, self.horizon, self.action_dim)


class BCMLPGMM(nn.Module):
    """BC MLP with GMM action head."""
    def __init__(self, action_dim, state_dim, horizon, n_obs_steps, hidden_dim=128, n_layers=3,
                 dropout=0.3, n_modes=5, min_std=0.01):
        super().__init__()
        self.horizon = horizon; self.action_dim = action_dim
        self.n_modes = n_modes; self.min_std = min_std
        in_dim = state_dim * n_obs_steps
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        self.backbone = nn.Sequential(*layers)

        flat_act = horizon * action_dim
        # Per mode: mean + log_std + logit_weight
        self.mean_head = nn.Linear(hidden_dim, n_modes * flat_act)
        self.logstd_head = nn.Linear(hidden_dim, n_modes * flat_act)
        self.logit_head = nn.Linear(hidden_dim, n_modes)

    def forward(self, obs):
        """Returns most likely mode's mean for inference."""
        h = self.backbone(obs.reshape(obs.shape[0], -1))
        means = self.mean_head(h).reshape(-1, self.n_modes, self.horizon, self.action_dim)
        logits = self.logit_head(h)  # (B, n_modes)
        # Return highest-weight mode
        best_mode = logits.argmax(dim=-1)  # (B,)
        return means[torch.arange(len(means)), best_mode]  # (B, H, action_dim)

    def gmm_loss(self, obs, target):
        """Negative log-likelihood under GMM."""
        B = obs.shape[0]
        h = self.backbone(obs.reshape(B, -1))
        flat_act = self.horizon * self.action_dim

        means = self.mean_head(h).reshape(B, self.n_modes, flat_act)
        log_stds = self.logstd_head(h).reshape(B, self.n_modes, flat_act)
        log_stds = log_stds.clamp(-5, 2)  # stability
        stds = log_stds.exp().clamp(min=self.min_std)
        logits = self.logit_head(h)  # (B, n_modes)
        log_weights = torch.log_softmax(logits, dim=-1)  # (B, n_modes)

        target_flat = target.reshape(B, 1, flat_act).expand_as(means)

        # Log probability under each Gaussian
        log_probs = -0.5 * (((target_flat - means) / stds) ** 2 + 2 * log_stds + math.log(2 * math.pi))
        log_probs = log_probs.sum(dim=-1)  # (B, n_modes) — sum over action dims

        # Log-sum-exp over modes
        log_mixture = torch.logsumexp(log_weights + log_probs, dim=-1)  # (B,)

        return -log_mixture.mean()


class BCMLPBetterLoss(nn.Module):
    """BC MLP with L1 + Huber loss (more robust to outliers than MSE)."""
    def __init__(self, action_dim, state_dim, horizon, n_obs_steps, hidden_dim=128, n_layers=3, dropout=0.3):
        super().__init__()
        self.horizon = horizon; self.action_dim = action_dim
        in_dim = state_dim * n_obs_steps
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, horizon * action_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, obs):
        return self.net(obs.reshape(obs.shape[0], -1)).reshape(-1, self.horizon, self.action_dim)


def train_and_eval(name, model, loss_fn, BS=128, LR=1e-3, max_epochs=200, patience=30):
    """Generic train + eval loop."""
    ns_local = len(train_co)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    nbpe = max(1, ns_local // BS)
    lr_s = get_cosine_schedule_with_warmup(opt, 10, max_epochs * nbpe)
    best_val = float('inf'); best_state = None; best_ep = 0; wait = 0; t0 = time.time()
    params = sum(p.numel() for p in model.parameters())
    print(f'\n=== {name} ({params/1e6:.2f}M params) ===', flush=True)

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(ns_local, device=device)
        el, nb = 0.0, 0
        for b in range(0, ns_local - BS + 1, BS):
            idx = perm[b:b+BS]
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                loss = loss_fn(model, train_co[idx], train_ca[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); lr_s.step()
            el += loss.item(); nb += 1

        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            vl = loss_fn(model, val_co, val_ca).item()

        if vl < best_val:
            best_val = vl; best_state = deepcopy(model.state_dict()); best_ep = epoch+1; wait = 0
        else:
            wait += 1
            if wait >= patience: break

        if (epoch+1) % 25 == 0 or epoch == 0:
            print(f'  Ep {epoch+1} train={el/max(nb,1):.4f} val={vl:.4f} best_ep={best_ep}', flush=True)

    model.load_state_dict(best_state)
    print(f'  Done: best_ep={best_ep} val={best_val:.4f} in {time.time()-t0:.0f}s', flush=True)

    # Eval
    model.eval()
    sn = Normalizer(obs_mean.cpu().numpy(), obs_std.cpu().numpy())
    an = Normalizer(act_mean.cpu().numpy(), act_std.cpu().numpy())

    env = create_env(split='pretrain', seed=0)
    succ, drs = 0, []
    for ep_i in range(5):
        obs = env.reset()
        parts = [np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED]),
                 obs['door_obj_pos'].flatten(), obs['door_obj_to_robot0_eef_pos'].flatten(),
                 np.array([np.linalg.norm(obs['door_obj_to_robot0_eef_pos'])])]
        aug = np.concatenate(parts).astype(np.float32)
        oh = deque([aug]*NOBS, maxlen=NOBS); aq = deque()
        init_dist = np.linalg.norm(obs['door_obj_to_robot0_eef_pos']); min_dist = init_dist; success = False

        for step in range(500):
            if not aq:
                with torch.no_grad():
                    oc = sn.normalize(torch.from_numpy(np.stack(list(oh))).float().unsqueeze(0).to(device))
                    acts = an.denormalize(model(oc).reshape(1, H, 12))
                for i in range(8): aq.append(acts[0,i].cpu().numpy())
            env_act = dataset_action_to_env_action(aq.popleft())
            env_act = np.clip(env_act, -1.0, 1.0)
            obs, _, _, _ = env.step(env_act)
            parts = [np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED]),
                     obs['door_obj_pos'].flatten(), obs['door_obj_to_robot0_eef_pos'].flatten(),
                     np.array([np.linalg.norm(obs['door_obj_to_robot0_eef_pos'])])]
            oh.append(np.concatenate(parts).astype(np.float32))
            min_dist = min(min_dist, np.linalg.norm(obs['door_obj_to_robot0_eef_pos']))
            if env._check_success(): success = True; break

        if success: succ += 1
        dr = (1 - min_dist / max(init_dist, 1e-6)) * 100
        drs.append(dr)
        print(f'    Ep{ep_i+1}: {"OK" if success else "X"} d={init_dist:.2f}->{min_dist:.2f} ({dr:.0f}%)', flush=True)

    env.close()
    avg_dr = np.mean(drs)
    print(f'  RESULT: {succ}/5 success, {avg_dr:.0f}% dist_red', flush=True)
    del opt; torch.cuda.empty_cache()
    return name, best_ep, best_val, succ, avg_dr


results = []

# 1. Baseline MSE
def mse_loss(model, obs, act):
    return nn.functional.mse_loss(model(obs), act)
m = BCMLPBaseline(12, state_dim, H, NOBS, 128, 3, 0.3).to(device)
results.append(train_and_eval('mse_baseline', m, mse_loss)); del m

# 2. GMM 5 modes
def gmm_loss(model, obs, act):
    return model.gmm_loss(obs, act)
m = BCMLPGMM(12, state_dim, H, NOBS, 128, 3, 0.3, n_modes=5).to(device)
results.append(train_and_eval('gmm_5modes', m, gmm_loss)); del m

# 3. GMM 10 modes
m = BCMLPGMM(12, state_dim, H, NOBS, 128, 3, 0.3, n_modes=10).to(device)
results.append(train_and_eval('gmm_10modes', m, gmm_loss)); del m

# 4. Huber loss (robust to outliers)
def huber_loss(model, obs, act):
    return nn.functional.smooth_l1_loss(model(obs), act)
m = BCMLPBetterLoss(12, state_dim, H, NOBS, 128, 3, 0.3).to(device)
results.append(train_and_eval('huber_loss', m, huber_loss)); del m

# 5. Weighted loss: higher weight on gripper actions
def weighted_mse_loss(model, obs, act):
    pred = model(obs)
    # Weight gripper and base_mode actions higher (dims 6, 11 in 12-dim action)
    weights = torch.ones(12, device=device)
    weights[6] = 5.0   # gripper
    weights[11] = 3.0  # base_mode
    return (weights * (pred - act) ** 2).mean()
m = BCMLPBaseline(12, state_dim, H, NOBS, 128, 3, 0.3).to(device)
results.append(train_and_eval('weighted_mse', m, weighted_mse_loss)); del m

print('\n=== GMM/LOSS EXPERIMENT SUMMARY ===', flush=True)
for name, ep, vl, succ, dr in results:
    print(f'  {name:<20} best_ep={ep:>3} val={vl:.4f} succ={succ}/5 dr={dr:.0f}%', flush=True)
