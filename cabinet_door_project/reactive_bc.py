"""
Reactive BC experiment: compare different action horizons and gripper weighting.
Key hypothesis: smaller horizon = more reactive = better grasping.
Also test heavily-weighted gripper loss to improve grasp timing.
"""
import torch, time, os, sys, numpy as np
import torch.nn as nn
from copy import deepcopy
from collections import deque

os.chdir('/home/noahcylich/cs188-cabinet-door-project/cabinet_door_project')
sys.path.insert(0, '.')
from diffusion_policy.data import Normalizer
from diffusion_policy.evaluation import create_env, dataset_action_to_env_action, STATE_KEYS_ORDERED
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

train_obs = obs_all[train_idxs]; val_obs = obs_all[val_idxs]
train_act = actions[train_idxs]; val_act = actions[val_idxs]
obs_mean = train_obs.mean(0); obs_std = train_obs.std(0).clamp(min=1e-6)
act_mean = train_act.mean(0); act_std = train_act.std(0).clamp(min=1e-6)
train_obs_n = (train_obs - obs_mean) / obs_std; val_obs_n = (val_obs - obs_mean) / obs_std
train_act_n = (train_act - act_mean) / act_std; val_act_n = (val_act - act_mean) / act_std


class BCMLPDropout(nn.Module):
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


def build_chunks(horizon, nobs):
    def chunk(obs, act):
        ol, al = [], []
        for j in range(max(0, len(obs) - horizon - nobs + 1)):
            ol.append(obs[j:j+nobs]); al.append(act[j+nobs-1:j+nobs-1+horizon])
        return torch.stack(ol).to(device), torch.stack(al).to(device)
    return chunk(train_obs_n, train_act_n), chunk(val_obs_n, val_act_n)


def train_and_eval(name, model, train_co, train_ca, val_co, val_ca,
                   horizon, n_action_steps, gripper_weight=1.0,
                   BS=128, LR=1e-3, max_epochs=200, patience=30):
    """Train with optional gripper weighting + eval."""
    ns = len(train_co)
    params = sum(p.numel() for p in model.parameters())
    print(f'\n=== {name} (H={horizon}, exec={n_action_steps}, grip_w={gripper_weight}) ===', flush=True)
    print(f'  {params/1e6:.3f}M params, ns={ns}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    nbpe = max(1, ns // BS)
    lr_s = get_cosine_schedule_with_warmup(opt, 10, max_epochs * nbpe)
    best_val = float('inf'); best_state = None; best_ep = 0; wait = 0; t0 = time.time()

    # Loss weight vector: dim 11 = gripper
    act_weights = torch.ones(12, device=device)
    act_weights[11] = gripper_weight

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(ns, device=device)
        el, nb = 0.0, 0
        for b in range(0, ns - BS + 1, BS):
            idx = perm[b:b+BS]
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                pred = model(train_co[idx])
                err = (pred - train_ca[idx]) ** 2  # (B, H, 12)
                loss = (err * act_weights).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); lr_s.step()
            el += loss.item(); nb += 1

        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            vp = model(val_co)
            vl = nn.functional.mse_loss(vp, val_ca).item()  # unweighted val for comparison

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
    NOBS = train_co.shape[1]

    env = create_env(split='pretrain', seed=0)
    succ, drs = 0, []
    for ep_i in range(8):
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
                    acts = an.denormalize(model(oc).reshape(1, horizon, 12))
                for i in range(min(n_action_steps, horizon)):
                    aq.append(acts[0,i].cpu().numpy())
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
    print(f'  RESULT: {succ}/8 success, {avg_dr:.0f}% dist_red', flush=True)
    del opt; torch.cuda.empty_cache()
    return name, best_ep, best_val, succ, avg_dr


results = []

# ========== Exp 1: Baseline H=16, exec=8 (our best so far) ==========
(tc16, tca16), (vc16, vca16) = build_chunks(16, 2)
m = BCMLPDropout(12, state_dim, 16, 2, 128, 3, 0.3).to(device)
r = train_and_eval('baseline_H16_e8', m, tc16, tca16, vc16, vca16, 16, 8, gripper_weight=1.0)
results.append(r); del m; torch.cuda.empty_cache()

# ========== Exp 2: Reactive H=4, exec=2 ==========
(tc4, tca4), (vc4, vca4) = build_chunks(4, 2)
m = BCMLPDropout(12, state_dim, 4, 2, 128, 3, 0.3).to(device)
r = train_and_eval('reactive_H4_e2', m, tc4, tca4, vc4, vca4, 4, 2, gripper_weight=1.0)
results.append(r); del m; torch.cuda.empty_cache()

# ========== Exp 3: Pure reactive H=1, exec=1 ==========
(tc1, tca1), (vc1, vca1) = build_chunks(1, 2)
m = BCMLPDropout(12, state_dim, 1, 2, 128, 3, 0.3).to(device)
r = train_and_eval('pure_reactive_H1', m, tc1, tca1, vc1, vca1, 1, 1, gripper_weight=1.0)
results.append(r); del m; torch.cuda.empty_cache()

# ========== Exp 4: H=16 with heavy gripper weight ==========
m = BCMLPDropout(12, state_dim, 16, 2, 128, 3, 0.3).to(device)
r = train_and_eval('H16_grip_w50', m, tc16, tca16, vc16, vca16, 16, 8, gripper_weight=50.0)
results.append(r); del m; torch.cuda.empty_cache()

# ========== Exp 5: H=4 with heavy gripper weight ==========
m = BCMLPDropout(12, state_dim, 4, 2, 128, 3, 0.3).to(device)
r = train_and_eval('H4_grip_w50', m, tc4, tca4, vc4, vca4, 4, 2, gripper_weight=50.0)
results.append(r); del m; torch.cuda.empty_cache()

# ========== Exp 6: More obs context (NOBS=4) ==========
(tc16_n4, tca16_n4), (vc16_n4, vca16_n4) = build_chunks(16, 4)
m = BCMLPDropout(12, state_dim, 16, 4, 128, 3, 0.3).to(device)
r = train_and_eval('H16_nobs4', m, tc16_n4, tca16_n4, vc16_n4, vca16_n4, 16, 8, gripper_weight=1.0)
results.append(r); del m; torch.cuda.empty_cache()

print('\n=== REACTIVE BC SUMMARY ===', flush=True)
for name, ep, vl, succ, dr in results:
    print(f'  {name:<20} best_ep={ep:>3} val={vl:.4f} succ={succ}/8 dr={dr:.0f}%', flush=True)
