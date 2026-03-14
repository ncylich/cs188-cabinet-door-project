"""
Orientation-aware features experiment.
Hypothesis: door_to_eef_quat (4-dim) gives the model the approach angle info
needed for correct grasp, currently missing from our 23-dim feature set.
Also tests global EEF position as an alternative representation.
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

# Available feature groups
print("Available features:", list(features.keys()), flush=True)
for k, v in features.items():
    print(f"  {k}: {v.shape[-1]} dims", flush=True)

rng = np.random.RandomState(42)
n_eps = len(ep_bounds)
perm = rng.permutation(n_eps)
n_val = max(1, int(n_eps * 0.15))
val_eps = set(perm[:n_val])
train_idxs, val_idxs = [], []
for i, (eid, start, end) in enumerate(ep_bounds):
    idxs = list(range(start, end))
    (val_idxs if i in val_eps else train_idxs).extend(idxs)


def make_feature_set(feat_names):
    obs = torch.cat([features[n] for n in feat_names], dim=-1)
    state_dim = obs.shape[-1]
    train_obs = obs[train_idxs]; val_obs_raw = obs[val_idxs]
    train_act = actions[train_idxs]; val_act = actions[val_idxs]
    obs_mean = train_obs.mean(0); obs_std = train_obs.std(0).clamp(min=1e-6)
    act_mean = train_act.mean(0); act_std = train_act.std(0).clamp(min=1e-6)
    train_obs_n = (train_obs - obs_mean) / obs_std
    val_obs_n = (val_obs_raw - obs_mean) / obs_std
    train_act_n = (train_act - act_mean) / act_std
    val_act_n = (val_act - act_mean) / act_std
    return state_dim, train_obs_n, train_act_n, val_obs_n, val_act_n, obs_mean, obs_std, act_mean, act_std


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


H, NOBS = 16, 2

def build_chunks(train_obs_n, train_act_n, val_obs_n, val_act_n):
    def chunk(obs, act):
        ol, al = [], []
        for j in range(max(0, len(obs) - H - NOBS + 1)):
            ol.append(obs[j:j+NOBS]); al.append(act[j+NOBS-1:j+NOBS-1+H])
        return torch.stack(ol).to(device), torch.stack(al).to(device)
    return chunk(train_obs_n, train_act_n), chunk(val_obs_n, val_act_n)


def train_and_eval(name, model, feat_names, train_co, train_ca, val_co, val_ca,
                   obs_mean, obs_std, act_mean, act_std,
                   BS=128, LR=1e-3, max_epochs=200, patience=30):
    ns = len(train_co)
    params = sum(p.numel() for p in model.parameters())
    state_dim = train_co.shape[-1]
    print(f'\n=== {name} ({state_dim}-dim) ===', flush=True)
    print(f'  Features: {feat_names}', flush=True)
    print(f'  {params/1e6:.3f}M params', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    nbpe = max(1, ns // BS)
    lr_s = get_cosine_schedule_with_warmup(opt, 10, max_epochs * nbpe)
    best_val = float('inf'); best_state = None; best_ep = 0; wait = 0; t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(ns, device=device)
        el, nb = 0.0, 0
        for b in range(0, ns - BS + 1, BS):
            idx = perm[b:b+BS]
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                loss = nn.functional.mse_loss(model(train_co[idx]), train_ca[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); lr_s.step()
            el += loss.item(); nb += 1

        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            vl = nn.functional.mse_loss(model(val_co), val_ca).item()

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
    for ep_i in range(8):
        obs = env.reset()
        # Build obs from feature names
        parts = []
        for fn in feat_names:
            if fn == 'proprio':
                parts.append(np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED]))
            elif fn == 'door_pos':
                parts.append(obs['door_obj_pos'].flatten())
            elif fn == 'door_quat':
                parts.append(obs['door_obj_quat'].flatten())
            elif fn == 'eef_pos':
                # Compute global EEF pos from base + relative
                base_pos = np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED])[0:3]
                base_quat = np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED])[3:7]
                base_to_eef = np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED])[7:10]
                # Simple approximation: eef_pos ≈ obs directly from env
                # Use robot0_eef_pos if available
                if 'robot0_eef_pos' in obs:
                    parts.append(obs['robot0_eef_pos'].flatten())
                else:
                    parts.append(np.zeros(3))
            elif fn == 'eef_quat':
                if 'robot0_eef_quat' in obs:
                    parts.append(obs['robot0_eef_quat'].flatten())
                else:
                    parts.append(np.zeros(4))
            elif fn == 'door_to_eef_pos':
                parts.append(obs['door_obj_to_robot0_eef_pos'].flatten())
            elif fn == 'door_to_eef_quat':
                parts.append(obs['door_obj_to_robot0_eef_quat'].flatten())
            elif fn == 'gripper_to_door_dist':
                parts.append(np.array([np.linalg.norm(obs['door_obj_to_robot0_eef_pos'])]))
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
            # Update obs parts
            parts = []
            for fn in feat_names:
                if fn == 'proprio':
                    parts.append(np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED]))
                elif fn == 'door_pos':
                    parts.append(obs['door_obj_pos'].flatten())
                elif fn == 'door_quat':
                    parts.append(obs['door_obj_quat'].flatten())
                elif fn == 'eef_pos':
                    if 'robot0_eef_pos' in obs: parts.append(obs['robot0_eef_pos'].flatten())
                    else: parts.append(np.zeros(3))
                elif fn == 'eef_quat':
                    if 'robot0_eef_quat' in obs: parts.append(obs['robot0_eef_quat'].flatten())
                    else: parts.append(np.zeros(4))
                elif fn == 'door_to_eef_pos':
                    parts.append(obs['door_obj_to_robot0_eef_pos'].flatten())
                elif fn == 'door_to_eef_quat':
                    parts.append(obs['door_obj_to_robot0_eef_quat'].flatten())
                elif fn == 'gripper_to_door_dist':
                    parts.append(np.array([np.linalg.norm(obs['door_obj_to_robot0_eef_pos'])]))
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

# ========== Exp 1: Current best (23-dim) ==========
fn1 = ['proprio', 'door_pos', 'door_to_eef_pos', 'gripper_to_door_dist']
sd1, to1, ta1, vo1, va1, om1, os1, am1, as1 = make_feature_set(fn1)
(tc1, tca1), (vc1, vca1) = build_chunks(to1, ta1, vo1, va1)
m = BCMLPDropout(12, sd1, H, NOBS, 128, 3, 0.3).to(device)
r = train_and_eval('baseline_23dim', m, fn1, tc1, tca1, vc1, vca1, om1, os1, am1, as1)
results.append(r); del m; torch.cuda.empty_cache()

# ========== Exp 2: +door_to_eef_quat (27-dim) ==========
fn2 = ['proprio', 'door_pos', 'door_to_eef_pos', 'door_to_eef_quat', 'gripper_to_door_dist']
sd2, to2, ta2, vo2, va2, om2, os2, am2, as2 = make_feature_set(fn2)
(tc2, tca2), (vc2, vca2) = build_chunks(to2, ta2, vo2, va2)
m = BCMLPDropout(12, sd2, H, NOBS, 128, 3, 0.3).to(device)
r = train_and_eval('with_eef_quat_27dim', m, fn2, tc2, tca2, vc2, vca2, om2, os2, am2, as2)
results.append(r); del m; torch.cuda.empty_cache()

# ========== Exp 3: +eef_pos (26-dim) ==========
fn3 = ['proprio', 'door_pos', 'eef_pos', 'door_to_eef_pos', 'gripper_to_door_dist']
sd3, to3, ta3, vo3, va3, om3, os3, am3, as3 = make_feature_set(fn3)
(tc3, tca3), (vc3, vca3) = build_chunks(to3, ta3, vo3, va3)
m = BCMLPDropout(12, sd3, H, NOBS, 128, 3, 0.3).to(device)
r = train_and_eval('with_eef_pos_26dim', m, fn3, tc3, tca3, vc3, vca3, om3, os3, am3, as3)
results.append(r); del m; torch.cuda.empty_cache()

# ========== Exp 4: All oracle features (38-dim) ==========
fn4 = ['proprio', 'door_pos', 'door_quat', 'eef_pos', 'eef_quat', 'door_to_eef_pos', 'door_to_eef_quat', 'gripper_to_door_dist']
sd4, to4, ta4, vo4, va4, om4, os4, am4, as4 = make_feature_set(fn4)
(tc4, tca4), (vc4, vca4) = build_chunks(to4, ta4, vo4, va4)
m = BCMLPDropout(12, sd4, H, NOBS, 128, 3, 0.3).to(device)
r = train_and_eval('all_oracle_38dim', m, fn4, tc4, tca4, vc4, vca4, om4, os4, am4, as4)
results.append(r); del m; torch.cuda.empty_cache()

# ========== Exp 5: Best features + bigger model ==========
fn5 = fn2  # start with 27-dim
sd5, to5, ta5, vo5, va5, om5, os5, am5, as5 = make_feature_set(fn5)
(tc5, tca5), (vc5, vca5) = build_chunks(to5, ta5, vo5, va5)
m = BCMLPDropout(12, sd5, H, NOBS, 256, 4, 0.3).to(device)
r = train_and_eval('27dim_bigger', m, fn5, tc5, tca5, vc5, vca5, om5, os5, am5, as5)
results.append(r); del m; torch.cuda.empty_cache()

print('\n=== ORIENTATION FEATURES SUMMARY ===', flush=True)
for name, ep, vl, succ, dr in results:
    print(f'  {name:<25} best_ep={ep:>3} val={vl:.4f} succ={succ}/8 dr={dr:.0f}%', flush=True)
