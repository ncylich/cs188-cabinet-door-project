"""
Hybrid rule-based gripper experiment.
BC policy predicts arm+base actions, but gripper is rule-based:
  - if d(EEF, door) < threshold: CLOSE gripper (grasp attempt)
  - else: OPEN gripper (approach)

Tests whether the failure is due to wrong gripper timing vs wrong approach.
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

H, NOBS = 16, 2
def chunk(obs, act):
    ol, al = [], []
    for j in range(max(0, len(obs) - H - NOBS + 1)):
        ol.append(obs[j:j+NOBS]); al.append(act[j+NOBS-1:j+NOBS-1+H])
    return torch.stack(ol).to(device), torch.stack(al).to(device)
train_co, train_ca = chunk(train_obs_n, train_act_n)
val_co, val_ca = chunk(val_obs_n, val_act_n)
ns = len(train_co)


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


# Train model once, reuse for multiple eval configs
print("Training BC model...", flush=True)
model = BCMLPDropout(12, state_dim, H, NOBS, 128, 3, 0.3).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
BS = 128
lr_s = get_cosine_schedule_with_warmup(opt, 10, 200 * max(1, ns // BS))
best_val = float('inf'); best_state = None; wait = 0; t0 = time.time()

for epoch in range(200):
    model.train()
    perm = torch.randperm(ns, device=device)
    for b in range(0, ns - BS + 1, BS):
        idx = perm[b:b+BS]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = nn.functional.mse_loss(model(train_co[idx]), train_ca[idx])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); lr_s.step()
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        vl = nn.functional.mse_loss(model(val_co), val_ca).item()
    if vl < best_val: best_val = vl; best_state = deepcopy(model.state_dict()); wait = 0
    else:
        wait += 1
        if wait >= 30: break
model.load_state_dict(best_state)
model.eval()
print(f"Trained: val={best_val:.4f}", flush=True)

sn = Normalizer(obs_mean.cpu().numpy(), obs_std.cpu().numpy())
an = Normalizer(act_mean.cpu().numpy(), act_std.cpu().numpy())

# gripper +1 in raw dataset action → env_action[6] = +1.0 (CLOSE)
# gripper_raw = +1 → binarized ≥ 0.5 → env close
# We need to inject close (raw dataset dim 11 = +1) or open (-1)
# act_mean[11] ≈ -0.64, act_std[11] ≈ 0.77
# To inject +1 raw: normalized = (+1 - (-0.64)) / 0.77 = 2.13 (already in dataset space when normalized)
# Actually we override the DENORMALIZED raw action (before env conversion)
# dataset_action_to_env_action(raw) uses raw[11] directly
# So we set raw[11] = 2.0 (> 0.5 threshold) for close, or raw[11] = -2.0 for open


def eval_with_gripper_policy(gripper_policy, n_eps=12, max_steps=500, n_act=8, threshold=0.12):
    """
    gripper_policy: 'model' | 'always_close' | 'rule_close_at_dist' | 'rule_close_late'
    """
    env = create_env(split='pretrain', seed=0)
    succ, drs = 0, []
    close_count = 0

    for ep_i in range(n_eps):
        obs = env.reset()
        parts = [np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED]),
                 obs['door_obj_pos'].flatten(), obs['door_obj_to_robot0_eef_pos'].flatten(),
                 np.array([np.linalg.norm(obs['door_obj_to_robot0_eef_pos'])])]
        aug = np.concatenate(parts).astype(np.float32)
        oh = deque([aug]*NOBS, maxlen=NOBS); aq = deque()
        init_dist = np.linalg.norm(obs['door_obj_to_robot0_eef_pos']); min_dist = init_dist; success = False
        close_steps = 0

        for step in range(max_steps):
            if not aq:
                with torch.no_grad():
                    oc = sn.normalize(torch.from_numpy(np.stack(list(oh))).float().unsqueeze(0).to(device))
                    acts = an.denormalize(model(oc).reshape(1, H, 12))
                for i in range(min(n_act, H)):
                    aq.append(acts[0,i].cpu().numpy())

            raw_act = aq.popleft().copy()
            d2e = np.linalg.norm(obs['door_obj_to_robot0_eef_pos'])

            # Apply gripper policy override
            if gripper_policy == 'always_close':
                raw_act[11] = 2.0  # force close
                close_steps += 1
            elif gripper_policy == 'rule_close_at_dist':
                if d2e < threshold:
                    raw_act[11] = 2.0  # close
                    close_steps += 1
                else:
                    raw_act[11] = -2.0  # open
            elif gripper_policy == 'rule_close_late':
                # Close only when very close
                if d2e < 0.05:
                    raw_act[11] = 2.0
                    close_steps += 1
                else:
                    raw_act[11] = -2.0
            # else: model controls gripper

            env_act = dataset_action_to_env_action(raw_act)
            env_act = np.clip(env_act, -1.0, 1.0)
            obs, _, _, _ = env.step(env_act)
            parts = [np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED]),
                     obs['door_obj_pos'].flatten(), obs['door_obj_to_robot0_eef_pos'].flatten(),
                     np.array([np.linalg.norm(obs['door_obj_to_robot0_eef_pos'])])]
            oh.append(np.concatenate(parts).astype(np.float32))
            min_dist = min(min_dist, np.linalg.norm(obs['door_obj_to_robot0_eef_pos']))
            if env._check_success(): success = True; break

        if success: succ += 1; close_count += close_steps
        dr = (1 - min_dist / max(init_dist, 1e-6)) * 100
        drs.append(dr)
        print(f'    Ep{ep_i+1}: {"OK" if success else "X"} d={init_dist:.2f}->{min_dist:.2f} ({dr:.0f}%) close={close_steps}', flush=True)

    env.close()
    avg_dr = np.mean(drs)
    print(f'  RESULT: {succ}/{n_eps} success, {avg_dr:.0f}% dist_red', flush=True)
    return succ, avg_dr


results = []
policies = [
    ('model_gripper', 'model', 0.12),
    ('always_close', 'always_close', 0.12),
    ('rule_close_d<0.12', 'rule_close_at_dist', 0.12),
    ('rule_close_d<0.20', 'rule_close_at_dist', 0.20),
    ('rule_close_d<0.05', 'rule_close_late', 0.05),
]

for name, policy, thresh in policies:
    print(f'\n=== {name} ===', flush=True)
    succ, dr = eval_with_gripper_policy(policy, n_eps=12, threshold=thresh)
    results.append((name, succ, dr))

print('\n=== HYBRID GRIPPER SUMMARY ===', flush=True)
for name, succ, dr in results:
    print(f'  {name:<22} succ={succ:>2}/12  dr={dr:.0f}%', flush=True)
