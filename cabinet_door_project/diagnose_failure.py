"""Diagnose exactly why the robot fails: track gripper, EEF, handle proximity."""
import torch, os, sys, numpy as np
from collections import deque

os.chdir('/home/noahcylich/cs188-cabinet-door-project/cabinet_door_project')
sys.path.insert(0, '.')

from diffusion_policy.data import Normalizer
from diffusion_policy.evaluation import create_env, dataset_action_to_env_action, STATE_KEYS_ORDERED
from diffusion_policy.training import get_cosine_schedule_with_warmup
import torch.nn as nn

device = torch.device('cuda')
SAVE_DIR = '/tmp/diffusion_policy_checkpoints'

# Load preprocessing data
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

H, NOBS = 16, 2

# Quick train of best model
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

model = BCMLPDropout(12, state_dim, H, NOBS, 128, 3, 0.3).to(device)
train_obs_n = (obs_all[train_idxs] - obs_mean) / obs_std
val_obs_n = (obs_all[val_idxs] - obs_mean) / obs_std
train_act_n = (train_act - act_mean) / act_std
val_act_n = (val_act - act_mean) / act_std

def chunk(obs, act):
    ol, al = [], []
    for j in range(max(0, len(obs) - H - NOBS + 1)):
        ol.append(obs[j:j+NOBS]); al.append(act[j+NOBS-1:j+NOBS-1+H])
    return torch.stack(ol).to(device), torch.stack(al).to(device)
train_co, train_ca = chunk(train_obs_n, train_act_n)
val_co, val_ca = chunk(val_obs_n, val_act_n)
ns = len(train_co)

from copy import deepcopy
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
lr_s = get_cosine_schedule_with_warmup(opt, 10, 200 * max(1, ns//128))
best_val = float('inf'); best_state = None; wait = 0
BS = 128

print("Quick-training diagnostic model...", flush=True)
for epoch in range(200):
    model.train()
    perm = torch.randperm(ns, device=device)
    for b in range(0, ns - BS + 1, BS):
        idx = perm[b:b+BS]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = model(train_co[idx])
            loss = nn.functional.mse_loss(pred, train_ca[idx])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); lr_s.step()
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        vl = nn.functional.mse_loss(model(val_co), val_ca).item()
    if vl < best_val:
        best_val = vl; best_state = deepcopy(model.state_dict()); wait = 0
    else:
        wait += 1
        if wait >= 30: break
model.load_state_dict(best_state)
model.eval()
print(f"Trained, val={best_val:.4f}", flush=True)

sn = Normalizer(obs_mean.cpu().numpy(), obs_std.cpu().numpy())
an = Normalizer(act_mean.cpu().numpy(), act_std.cpu().numpy())

env = create_env(split='pretrain', seed=0)

print("\n=== DIAGNOSTIC ROLLOUT ===", flush=True)
print("Tracking: step, d2door, gripper_qpos, predicted_gripper_action", flush=True)

for ep_i in range(2):
    print(f"\n--- Episode {ep_i+1} ---", flush=True)
    obs = env.reset()

    # What obs keys are available?
    if ep_i == 0:
        obs_keys = [k for k in obs.keys() if 'door' in k.lower() or 'gripper' in k.lower() or 'handle' in k.lower()]
        print(f"Door/gripper obs keys: {obs_keys}", flush=True)
        for k in obs_keys:
            v = obs[k]
            print(f"  {k}: shape={np.asarray(v).shape}, val={np.asarray(v).flatten()[:5]}", flush=True)

    parts = [np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED]),
             obs['door_obj_pos'].flatten(), obs['door_obj_to_robot0_eef_pos'].flatten(),
             np.array([np.linalg.norm(obs['door_obj_to_robot0_eef_pos'])])]
    aug = np.concatenate(parts).astype(np.float32)
    oh = deque([aug]*NOBS, maxlen=NOBS); aq = deque()
    init_dist = np.linalg.norm(obs['door_obj_to_robot0_eef_pos'])
    min_dist = init_dist

    gripper_history = []
    dist_history = []

    for step in range(500):
        if not aq:
            with torch.no_grad():
                oc = sn.normalize(torch.from_numpy(np.stack(list(oh))).float().unsqueeze(0).to(device))
                acts = an.denormalize(model(oc).reshape(1, H, 12))
            for i in range(8):
                aq.append(acts[0,i].cpu().numpy())

        next_act = aq[0]  # peek at next action
        env_act = dataset_action_to_env_action(aq.popleft())
        env_act = np.clip(env_act, -1.0, 1.0)
        obs, _, _, _ = env.step(env_act)

        d2e = np.linalg.norm(obs['door_obj_to_robot0_eef_pos'])
        min_dist = min(min_dist, d2e)
        gripper_pos = np.asarray(obs.get('robot0_gripper_qpos', [0, 0])).flatten()
        pred_gripper_raw = next_act[6] if len(next_act) > 6 else 0

        gripper_history.append(gripper_pos[0] if len(gripper_pos) > 0 else 0)
        dist_history.append(d2e)

        if step % 20 == 0 or d2e < 0.15:
            print(f"  s={step:3d} d2door={d2e:.3f} gripper_open={gripper_pos[0]:.3f} pred_grip={pred_gripper_raw:.3f}", flush=True)

        parts = [np.concatenate([obs[k].flatten() for k in STATE_KEYS_ORDERED]),
                 obs['door_obj_pos'].flatten(), obs['door_obj_to_robot0_eef_pos'].flatten(),
                 np.array([np.linalg.norm(obs['door_obj_to_robot0_eef_pos'])])]
        oh.append(np.concatenate(parts).astype(np.float32))

        if env._check_success():
            print(f"  SUCCESS at step {step}!", flush=True)
            break

    print(f"\n  Summary: d={init_dist:.3f}->{min_dist:.3f}", flush=True)
    print(f"  Min dist at steps: {[i for i, d in enumerate(dist_history) if d < 0.15][:10]}", flush=True)
    print(f"  Gripper when close (d<0.15): {[g for d, g in zip(dist_history, gripper_history) if d < 0.15][:10]}", flush=True)

    # Also check what expert does at grasping
    print(f"\n  Expert data analysis:", flush=True)
    # Check gripper_qpos in training data
    # dim 14-15 of proprio = gripper_qpos
    grippers = obs_all[train_idxs, 14].cpu().numpy()
    print(f"  Expert gripper_qpos[0] stats: mean={grippers.mean():.3f}, std={grippers.std():.3f}, min={grippers.min():.3f}, max={grippers.max():.3f}", flush=True)

    # Action dim 6 = gripper action in dataset
    grip_acts = actions[train_idxs, 6].cpu().numpy()
    print(f"  Expert gripper_action stats: mean={grip_acts.mean():.3f}, std={grip_acts.std():.3f}, min={grip_acts.min():.3f}, max={grip_acts.max():.3f}", flush=True)
    print(f"  Expert gripper_action close (<0): {(grip_acts < 0).mean()*100:.1f}% of steps", flush=True)
    print(f"  Expert gripper_action open (>=0): {(grip_acts >= 0).mean()*100:.1f}% of steps", flush=True)

env.close()
print("\nDiagnosis complete.", flush=True)
