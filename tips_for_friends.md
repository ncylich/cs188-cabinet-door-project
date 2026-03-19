# Tips for OpenCabinet — What Actually Worked

Lessons learned after ~60 experiments. Read this before you waste days on things that don't matter.

---

## 1. Fix the Gripper Immediately

The action space uses raw signed values `{-1 = open, +1 = close}` — **not probabilities**.

When you go from dataset actions → env actions, you need to binarize the gripper dim. The natural threshold is `0.0` (midpoint of −1/+1), **not** `0.5`. If you use 0.5, your model will train fine but the gripper will never close during eval because training data is ~82% open (mean ≈ −0.643), so any near-zero prediction binarizes to open. You'll get 0% success with a robot that's clearly approaching the handle — very confusing.

Same fix applies to the `base_mode` dim if you're using it.

Also double-check the dimension ordering: dataset format and env format are different.
```
Dataset: [base_motion(3), torso(1), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]
Env:     [eef_pos(3), eef_rot(3), gripper(1), base_motion(3), torso(1), base_mode(1)]
```

---

## 2. Use the Handle Site, Not the Door Centroid

This is the single most impactful thing we found. The door's body position in the sim is the **geometric center of the door panel** — not where you grip it. If you augment your state with the door body position, your policy will approach the middle of the door and stall out with 0 successes even at 40–55% distance reduction.

You need the **actual handle site position** — extracted by replaying each training episode in simulation and recording the named MuJoCo site position per timestep.

The two features that matter:
- `handle_pos` (3d): absolute world position of handle site
- `handle_to_eef` (3d): `eef_pos − handle_pos` — the live error signal

`handle_to_eef` is the more important of the two. It tells the policy exactly how far and in which direction to move at every step. With just `proprio(16) + handle_to_eef(3)` we got 31% success; adding `handle_pos` bumps it to 38–44%.

**Do not bother with:** door centroid features, door quaternion, orientation quaternions in general (hard for shallow MLPs), global EEF position (redundant, actively hurts), hinge angle (increases DR but hurts success in our tests).

---

## 3. Use the Relaxed Success Criterion

The default `env._check_success()` in robocasa requires **all** door joints to be open. In practice, you only need to open one. Replace it with a check like:

```
any hinge joint qpos > 0.3 rad (~17 degrees)
```

This didn't magically fix our results, but it's the correct criterion — you shouldn't be penalized for partially opening a double-cabinet when the task says "open a cabinet door."

---

## 4. BC Beats Diffusion at This Data Scale

We built full DDPM/DDIM diffusion and ran exhaustive comparisons. On 107 demos, **plain behavior cloning consistently matches or beats diffusion** at a fraction of the training time. Diffusion's noise prediction acts as a regularizer (train ≈ val) but doesn't help task success.

The one exception: a Diffusion Transformer is competitive — Transformer attention + iterative refinement is synergistic. But it trains 14–20× slower for similar performance.

For rapid iteration, use BC.

---

## 5. Use a 1D U-Net Backbone (Not MLP or Transformer) for BC

Among MLP, Transformer, and 1D Conv U-Net as BC backbones:

- **BC_UNet consistently wins** — the skip connections produce temporally smooth action sequences, which matters for the sustained pulling motion needed to open a door
- BC_MLP is nearly as good and faster to iterate with
- BC_Transformer underperforms within BC (though it's the best diffusion backbone)

For the final model: 1D U-Net, small default channels, action chunking H=16 execute 8.

---

## 6. Train for Almost No Epochs

With 107 demos, BC models peak at **epoch 2–4**. Training longer actively degrades eval performance — the model memorizes training trajectories and stops generalizing.

You need a held-out validation set (15% episode-level split, no leakage) and early stopping. Without it, you'll think more training is helping because loss keeps dropping.

Rule of thumb: if your BC model is still improving past epoch 10, something is wrong with your validation setup.

Optimal hyperparameters: `BS=128, LR=1e-3, max 100 epochs, patience=20–30`. Best epoch will almost always be ≤ 5.

---

## 7. Bigger Models Always Hurt

We tested this many times across many architectures. Every scale-up made things worse:

- 10× bigger BC_MLP: 38% DR vs 51% for tiny model
- 3× wider U-Net: best_ep=2, 20% success vs 31% for small model
- The bigger model just overfits faster

The bottleneck is data (107 demos), not capacity. The fix is more data, not a bigger model.

---

## 8. Action Chunking Is Essential, Don't Go Below H=16

We tested H=1 (fully reactive) through H=16 (chunked):

| Horizon | Execute Steps | Dist Reduction |
|---------|--------------|---------------|
| H=16 | 8 | 54% |
| H=4 | 2 | 47% |
| H=1 | 1 | ~22% |

Reactive BC is terrible for this task — it produces jittery, oscillating trajectories that can't apply consistent force to open a door. The policy needs to commit to a motion plan across at least 8 steps.

---

## 9. Fix Episode Boundary Contamination

When you build causal observation sequences for training (e.g., seq_len=16 history), the first 16 timesteps of each episode look back into the previous episode. That's a different kitchen, different robot state — the context is meaningless.

Mask or skip those frames. It affects ~4% of your training data but it's the kind of subtle noise that degrades val loss and can prevent you from crossing the success threshold.

---

## 10. Eval Variance Is High — Use N ≥ 50

At true success rates of 5–10%, a 20-episode eval has a 30–40% chance of returning 0 for a working policy (P(0 successes | p=0.05, N=20) = 36%). Don't conclude your checkpoint is broken from a single 20-episode run.

Also: if you use parallel workers for eval speed, be aware that each worker reuses the same kitchen for all its episodes. With 4 workers × 5 episodes, you're only testing 4 distinct kitchen layouts. Budget for N=50–100 and separate `env.reset()` calls per episode for a real estimate.

---

## Summary

The recipe that got us to **44% success** on 100 random kitchens:

1. Correct action remapping + gripper threshold = 0.0
2. Oracle handle site features: `proprio(16) + handle_pos(3) + handle_to_eef(3)` = 22 dims
3. BC (no diffusion), 1D U-Net backbone
4. Action chunking H=16, execute 8
5. Episode-level val split + early stopping → best epoch ≈ 4
6. 21 seconds of training on a GPU
