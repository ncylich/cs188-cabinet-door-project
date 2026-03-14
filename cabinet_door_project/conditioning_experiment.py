"""Compare conditioning vs input for door features, plus auxiliary output."""
import torch, logging, time, os, sys
import numpy as np
import torch.nn as nn
from copy import deepcopy
from collections import deque

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

# Split features: proprio (16) vs door features (7: door_pos + door_to_eef_pos + dist)
proprio_all = features['proprio']  # 16-dim
door_feat_all = torch.cat([features['door_pos'], features['door_to_eef_pos'],
                           features['gripper_to_door_dist']], dim=-1)  # 7-dim
obs_all = torch.cat([proprio_all, door_feat_all], dim=-1)  # 23-dim
state_dim = obs_all.shape[-1]
proprio_dim = 16
door_dim = 7

# Train/val split by episode
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


# Model 1: Baseline BC MLP (input mode, with dropout=0.3)
class BCMLPInput(nn.Module):
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
        x = obs.reshape(obs.shape[0], -1)
        return self.net(x).reshape(-1, self.horizon, self.action_dim)


# Model 2: FiLM conditioning - door features modulate proprio processing
class BCMLPConditioned(nn.Module):
    def __init__(self, action_dim, proprio_dim, door_dim, horizon, n_obs_steps, hidden_dim=128, n_layers=3, dropout=0.3):
        super().__init__()
        self.horizon = horizon; self.action_dim = action_dim
        self.proprio_dim = proprio_dim; self.door_dim = door_dim

        # Door encoder produces FiLM parameters (scale + shift per hidden dim)
        self.door_encoder = nn.Sequential(
            nn.Linear(door_dim * n_obs_steps, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2 * n_layers),  # scale + shift for each layer
        )

        # Proprio processing layers
        self.input_proj = nn.Linear(proprio_dim * n_obs_steps, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
        self.output = nn.Linear(hidden_dim, horizon * action_dim)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

    def forward(self, obs):
        B = obs.shape[0]
        # Split into proprio and door features
        proprio = obs[..., :self.proprio_dim].reshape(B, -1)
        door = obs[..., self.proprio_dim:].reshape(B, -1)

        # Get FiLM params from door features
        film_params = self.door_encoder(door)
        film_params = film_params.reshape(B, self.n_layers, 2, self.hidden_dim)

        # Process proprio with FiLM conditioning
        x = torch.relu(self.input_proj(proprio))
        # Apply first FiLM
        gamma, beta = film_params[:, 0, 0], film_params[:, 0, 1]
        x = gamma * x + beta

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < self.n_layers:
                gamma, beta = film_params[:, i+1, 0], film_params[:, i+1, 1]
                x = gamma * x + beta

        return self.output(x).reshape(-1, self.horizon, self.action_dim)


# Model 3: Auxiliary output - predict actions + next relative position
class BCMLPAuxiliary(nn.Module):
    def __init__(self, action_dim, state_dim, horizon, n_obs_steps, hidden_dim=128, n_layers=3, dropout=0.3, aux_dim=3):
        super().__init__()
        self.horizon = horizon; self.action_dim = action_dim; self.aux_dim = aux_dim
        in_dim = state_dim * n_obs_steps
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(hidden_dim, horizon * action_dim)
        self.aux_head = nn.Linear(hidden_dim, horizon * aux_dim)  # predict door_to_eef_pos changes

    def forward(self, obs, return_aux=False):
        x = self.backbone(obs.reshape(obs.shape[0], -1))
        actions = self.action_head(x).reshape(-1, self.horizon, self.action_dim)
        if return_aux:
            aux = self.aux_head(x).reshape(-1, self.horizon, self.aux_dim)
            return actions, aux
        return actions


# Prepare auxiliary targets: door_to_eef_pos at future timesteps
# We need the actual door_to_eef_pos values for the action horizons
door_to_eef = torch.cat([features['door_to_eef_pos']], dim=-1)  # (N, 3)
d2e_mean = door_to_eef[train_idxs].mean(0); d2e_std = door_to_eef[train_idxs].std(0).clamp(min=1e-6)
door_to_eef_norm = (door_to_eef - d2e_mean) / d2e_std
def chunk_aux(obs, act, aux):
    ol, al, xl = [], [], []
    for j in range(max(0, len(obs) - H - NOBS + 1)):
        ol.append(obs[j:j+NOBS])
        al.append(act[j+NOBS-1:j+NOBS-1+H])
        xl.append(aux[j+NOBS-1:j+NOBS-1+H])
    return torch.stack(ol).to(device), torch.stack(al).to(device), torch.stack(xl).to(device)

train_aux_obs, train_aux_act, train_aux_tgt = chunk_aux(
    (obs_all[train_idxs] - obs_mean) / obs_std,
    (actions[train_idxs] - act_mean) / act_std,
    door_to_eef_norm[train_idxs])
val_aux_obs, val_aux_act, val_aux_tgt = chunk_aux(
    (obs_all[val_idxs] - obs_mean) / obs_std,
    (actions[val_idxs] - act_mean) / act_std,
    door_to_eef_norm[val_idxs])


def train_and_eval(name, model, train_o, train_a, val_o, val_a,
                   aux_train_tgt=None, aux_val_tgt=None, aux_weight=0.5):
    """Train with early stopping, then eval 5 episodes."""
    ns_local = len(train_o)
    BS, LR = 128, 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    nbpe = max(1, ns_local // BS)
    lr_s = get_cosine_schedule_with_warmup(opt, 10, 200 * nbpe)
    best_val = float('inf'); best_state = None; best_ep = 0; wait = 0; t0 = time.time()
    has_aux = aux_train_tgt is not None

    params = sum(p.numel() for p in model.parameters())
    print(f'\n=== {name} ({params/1e6:.2f}M params) ===', flush=True)

    for epoch in range(200):
        model.train()
        perm = torch.randperm(ns_local, device=device)
        el, nb = 0.0, 0
        for b in range(0, ns_local - BS + 1, BS):
            idx = perm[b:b+BS]
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                if has_aux:
                    pred_act, pred_aux = model(train_o[idx], return_aux=True)
                    loss_act = nn.functional.mse_loss(pred_act, train_a[idx])
                    loss_aux = nn.functional.mse_loss(pred_aux, aux_train_tgt[idx])
                    loss = loss_act + aux_weight * loss_aux
                else:
                    pred = model(train_o[idx])
                    loss = nn.functional.mse_loss(pred, train_a[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); lr_s.step()
            el += loss.item(); nb += 1

        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            if has_aux:
                vp, vaux = model(val_o, return_aux=True)
                vl = nn.functional.mse_loss(vp, val_a).item()
            else:
                vp = model(val_o); vl = nn.functional.mse_loss(vp, val_a).item()

        if vl < best_val:
            best_val = vl; best_state = deepcopy(model.state_dict()); best_ep = epoch+1; wait = 0
        else:
            wait += 1
            if wait >= 30: break

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

# Config 1: Baseline (input mode, drop=0.3) - same as dropout sweep winner
m1 = BCMLPInput(12, state_dim, H, NOBS, 128, 3, 0.3).to(device)
r = train_and_eval('input_baseline', m1, train_co, train_ca, val_co, val_ca)
results.append(r); del m1

# Config 2: FiLM conditioning
m2 = BCMLPConditioned(12, proprio_dim, door_dim, H, NOBS, 128, 3, 0.3).to(device)
r = train_and_eval('film_conditioned', m2, train_co, train_ca, val_co, val_ca)
results.append(r); del m2

# Config 3: Auxiliary output (predict door_to_eef_pos)
m3 = BCMLPAuxiliary(12, state_dim, H, NOBS, 128, 3, 0.3, aux_dim=3).to(device)
r = train_and_eval('aux_output_0.5', m3, train_aux_obs, train_aux_act, val_aux_obs, val_aux_act,
                   train_aux_tgt, val_aux_tgt, aux_weight=0.5)
results.append(r); del m3

# Config 4: Auxiliary with lower weight
m4 = BCMLPAuxiliary(12, state_dim, H, NOBS, 128, 3, 0.3, aux_dim=3).to(device)
r = train_and_eval('aux_output_0.1', m4, train_aux_obs, train_aux_act, val_aux_obs, val_aux_act,
                   train_aux_tgt, val_aux_tgt, aux_weight=0.1)
results.append(r); del m4

# Config 5: FiLM + auxiliary combined
class BCMLPCondAux(nn.Module):
    def __init__(self, action_dim, proprio_dim, door_dim, horizon, n_obs_steps, hidden_dim=128, n_layers=3, dropout=0.3, aux_dim=3):
        super().__init__()
        self.horizon = horizon; self.action_dim = action_dim; self.aux_dim = aux_dim
        self.proprio_dim = proprio_dim; self.door_dim = door_dim
        self.door_encoder = nn.Sequential(
            nn.Linear(door_dim * n_obs_steps, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2 * n_layers),
        )
        self.input_proj = nn.Linear(proprio_dim * n_obs_steps, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)))
        self.action_head = nn.Linear(hidden_dim, horizon * action_dim)
        self.aux_head = nn.Linear(hidden_dim, horizon * aux_dim)
        self.n_layers = n_layers; self.hidden_dim = hidden_dim

    def forward(self, obs, return_aux=False):
        B = obs.shape[0]
        proprio = obs[..., :self.proprio_dim].reshape(B, -1)
        door = obs[..., self.proprio_dim:].reshape(B, -1)
        film_params = self.door_encoder(door).reshape(B, self.n_layers, 2, self.hidden_dim)
        x = torch.relu(self.input_proj(proprio))
        gamma, beta = film_params[:, 0, 0], film_params[:, 0, 1]
        x = gamma * x + beta
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < self.n_layers:
                gamma, beta = film_params[:, i+1, 0], film_params[:, i+1, 1]
                x = gamma * x + beta
        actions = self.action_head(x).reshape(-1, self.horizon, self.action_dim)
        if return_aux:
            aux = self.aux_head(x).reshape(-1, self.horizon, self.aux_dim)
            return actions, aux
        return actions

m5 = BCMLPCondAux(12, proprio_dim, door_dim, H, NOBS, 128, 3, 0.3, 3).to(device)
r = train_and_eval('film+aux_0.5', m5, train_aux_obs, train_aux_act, val_aux_obs, val_aux_act,
                   train_aux_tgt, val_aux_tgt, aux_weight=0.5)
results.append(r); del m5

print('\n=== CONDITIONING EXPERIMENT SUMMARY ===', flush=True)
for name, ep, vl, succ, dr in results:
    print(f'  {name:<20} best_ep={ep:>3} val={vl:.4f} succ={succ}/5 dr={dr:.0f}%', flush=True)
