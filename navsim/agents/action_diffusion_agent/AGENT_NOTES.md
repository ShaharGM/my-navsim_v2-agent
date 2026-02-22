# ActionDiffusionAgent — Implementation Notes

Complete reference for the `action_diffusion_agent` package.  
Intended audience: future sessions with an AI assistant, or any developer picking up this codebase.

---

## 1. Overview

`ActionDiffusionAgent` is a camera-based autonomous driving agent for the **NAVSIM** evaluation framework. It follows a diffusion-policy architecture:

```
front panoramic image (l0 + f0 + r0)
    │
    ▼
PerceptionBackbone  (resnet50 / any timm / VoV V-99-eSE)
    │  (B, 3, H, W) → (B, N_img, C)
    ▼
img_proj  (Linear: C → D)       status_encoder  (Linear+LN: 8 → D)
    │                                    │
    └──────────────┬─────────────────────┘
                   │  concat
                   ▼
        context KV  (B, N_img+1, D)       ← positional bias added
                   │
         ┌─────────┴───────────┐
         │  TRAINING           │  INFERENCE
         │                     │
         │  add noise          │  sample N Gaussian noises
         │  predict ε (MSE)    │  DDPM reverse chain (T steps)
         │                     │  → N denoised action sequences
         └─────────────────────┘
                   │
     action_space.action_to_traj()   ← UnicycleAccelCurvature
                   │
        dense trajectory (40 × 3)
                   │
         ↓ downsample ×5
        sparse trajectory (8 × 3)   ← returned to NAVSIM evaluator
```

The agent operates in a **unicycle action space** — all diffusion is done on
(acceleration, curvature) pairs, not directly on positions.  
This gives the head a physically grounded, lower-dimensional signal to learn.

---

## 2. File Structure

```
action_diffusion_agent/
├── __init__.py               public API exports
├── ad_config.py              ActionDiffusionConfig dataclass (all hyperparams)
├── ad_backbone.py            PerceptionBackbone (VoV + timm, → token sequence)
├── ad_diffusion_head.py      DDPM noise schedule + DenoisingTransformer + physics
├── ad_model.py               ActionDiffusionModel (wires backbone→context→head)
├── ad_features.py            Feature/target builders (NAVSIM AbstractFeatureBuilder)
├── ad_agent.py               ActionDiffusionAgent (NAVSIM AbstractAgent)
├── ad_callback.py            TensorBoard visualisation callback
└── AGENT_NOTES.md            ← this file
```

The Hydra config lives outside the package:
```
navsim/planning/script/config/common/agent/action_diffusion_agent.yaml
```

---

## 3. Module-by-Module Reference

### 3.1 `ad_config.py` — `ActionDiffusionConfig`

A plain `@dataclass` (no inheritance). All fields have defaults.

| Field | Default | Meaning |
|---|---|---|
| `trajectory_sampling` | 4 s @ 0.5 s | Output waypoint spec (→ NAVSIM evaluator) |
| `camera_width/height` | 2048 × 512 | Stitched panoramic image resolution |
| `seq_len` | 1 | Historical camera frames to load |
| `use_back_view` | False | Include rear panorama (l2+b0+r2) |
| `backbone_type` | `"timm"` | `"timm"` or `"vov"` |
| `timm_model_name` | `"resnet50"` | Any `timm.create_model` identifier |
| `timm_pretrained` | True | Load ImageNet weights via timm hub |
| `vov_ckpt` | `""` | Path to VoV checkpoint (only for `backbone_type="vov"`) |
| `freeze_backbone` | True | Freeze backbone during training |
| `img_vert_anchors` | 16 | Spatial pooling height |
| `img_horz_anchors` | 64 | Spatial pooling width |
| `model_dim` | 256 | D — transformer/projection dimension |
| `ffn_dim` | 1024 | Feed-forward inner dim in denoising transformer |
| `num_heads` | 8 | Attention heads |
| `ego_status_dim` | 8 | Length of status vector: cmd(4)+vx+vy+ax+ay |
| `num_diffusion_layers` | 4 | TransformerDecoder layers in denoiser |
| `num_diffusion_timesteps` | 100 | DDPM training/inference T |
| `num_inference_proposals` | 64 | N parallel trajectories generated at inference |
| `internal_dt` | 0.1 | Physics integration step [s] |
| `internal_horizon` | 40 | 4 s / 0.1 s — dense action sequence length |
| `action_dim` | 2 | (acceleration, curvature) |
| `accel_mean/std` | 0.057 / 1.12 | Z-score stats estimated on mini split |
| `curv_mean/std` | 0.002 / 0.023 | Z-score stats estimated on mini split |
| `pretrained_ckpt` | `""` | Checkpoint to load backbone+proj from (skips head) |
| `weight_decay` | 0.0 | Adam weight decay |
| `dp_loss_weight` | 10.0 | Multiplier on the MSE diffusion loss |

**Key distinction between two checkpoint fields:**
- `config.pretrained_ckpt` — loads only backbone + projection weights (strips diffusion head). Used to warm-start the visual encoder from a prior run.
- `agent.checkpoint_path` — loads the **complete** Lightning state dict for inference or training resume.

---

### 3.2 `ad_backbone.py` — `PerceptionBackbone`

```
Input:  (B, 3, H, W)
Output: (B, N, C)   where N = img_vert_anchors × img_horz_anchors
```

**timm path:**
```python
self._encoder = timm.create_model(name, pretrained=True, features_only=True, out_indices=(-1,))
feat = self._encoder(image)[-1]   # always a list; take last
```
Channel count is probed once at init with a zero-tensor dry run on CPU.

**VoV path:**
```python
self._encoder = VoVNet("V-99-eSE", out_features=["stage4","stage5"], ...)
feat = self._encoder(image)[-1]   # also a plain list; 1024 channels
```

Both are followed by `AdaptiveAvgPool2d((vert, horz))` → flatten to `(B, N, C)`.

Properties: `backbone.out_channels`, `backbone.num_tokens`

---

### 3.3 `ad_diffusion_head.py` — `ActionDiffusionHead`

#### Building blocks
| Class | Purpose |
|---|---|
| `_FourierEncoder` | Log-spaced sin/cos encoding of scalars. dim=20, max_freq=100 |
| `_RMSNorm` | Root-mean-square norm (no bias) |
| `_SinusoidalPosEmb` | Integer timestep → (B, D) sinusoidal embedding |
| `_build_mlp` | RMSNorm + SiLU depth-5 MLP |
| `_PerWaypointEncoder` | (B,T,A) + timestep → (B,T,D); per-dim Fourier + MLP + LN |
| `_DenoisingTransformer` | Pre-LN TransformerDecoder; action queries cross-attend context KV |

#### `_DenoisingTransformer`
- Input:  `noise (B,T,A)`, `timesteps (B,)`, `context (B,L,D)`
- Time token — sinusoidal embedding appended to the front of the context KV: `(B, L+1, D)`
- Learned positional biases: `action_pos (1,T,D)` on query side, `cond_pos (1,L+1,D)` on KV side
- Architecture: Pre-LN multi-head cross-attention → Pre-LN FFN (GeLU), N layers
- Output: predicted noise `(B, T, A)`

#### `ActionDiffusionHead.compute_loss()` — training
```python
# 1. Convert GT dense trajectory → normalised (accel, curvature) actions
gt_actions = action_space.traj_to_action(gt_xyz, gt_rot, hist_xyz, hist_rot, t0_states)

# 2. DDPM forward process
noise = torch.randn_like(gt_actions)
timesteps = randint(0, T, (B,))
noisy_actions = noise_scheduler.add_noise(gt_actions, noise, timesteps)

# 3. Predict noise; MSE loss
pred_noise = denoiser(noisy_actions, timesteps, context)
return F.mse_loss(pred_noise, noise)
```

#### `ActionDiffusionHead.forward()` — inference
```python
# 1. Start from pure Gaussian noise (B*N proposals)
noise = randn(B*N, 40, 2)

# 2. Full DDPM reverse chain
scheduler.set_timesteps(T, device)
for t in scheduler.timesteps:
    pred_eps = denoiser(noise, t.expand(B*N), ctx)
    noise = scheduler.step(pred_eps, t, noise).prev_sample

# 3. action_to_traj → dense 40-step trajectory
# 4. Downsample to 8 sparse steps (indices 4, 9, 14, ..., 39)
return {"dp_pred": (B,N,8,3), "pred_actions": (B,N,40,2)}
```

#### History construction (`_build_history`)
`hist_status_feature` is a flat `(B, N_hist×7)` tensor.  
Layout per historical frame: `[vx, vy, ax, ay, px, py, heading]`.  
The last 3 values (px, py, heading) are extracted for each frame, and a zero current-frame placeholder is appended → `(B, N_hist+1, 3)` → converted to xyz/rotation tensors for the unicycle model.

Default `seq_len=1` gives 3 history frames + 1 current = `N_hist=3`, so `hist_status_feature` is `(B, 21)`.

---

### 3.4 `ad_model.py` — `ActionDiffusionModel`

Wires everything together:

```
context_len = img_vert_anchors × img_horz_anchors × (2 if use_back_view else 1)  +  1
           = 16×64 × 1  +  1  =  1025  (default)
```

**Context building (`_build_context`):**
1. `backbone(img_front)` → `(B, N, C)` → `img_proj` → `(B, N, D)`
2. Optional: same for rear view → `cat` along dim 1 → `(B, 2N, D)`  
3. `status_encoder(status)` → `(B, 1, D)` → unsqueeze
4. `cat([img_tokens, status_tok], dim=1)` → `(B, L, D)`
5. `+ pos_embedding.weight` → learnable positional bias

**Freeze / unfreeze:**
- `freeze_backbone=True` → `backbone` frozen at init; `train()` override forces them back to `.eval()` to prevent BatchNorm drift
- `pretrained_ckpt` → loads backbone+proj (strips `"agent.model."` / `"model."` prefixes, skips `"diffusion_head"` keys), then freezes

**Forward:**
- Training: returns `{context_kv, trajectory=zeros}` — loss computed externally
- Inference: runs `diffusion_head(context, features)`, picks one random proposal → `{context_kv, dp_pred, pred_actions, trajectory}`

---

### 3.5 `ad_features.py` — Feature & Target Builders

#### `ActionDiffusionFeatureBuilder.compute_features()`
Produces:
```
camera_feature:       (seq_len, 3, H, W)  — front panorama tensor
camera_feature_back:  (seq_len, 3, H, W)  — rear panorama (if use_back_view)
status_feature:       (8,)                — [cmd(4), vx, vy, ax, ay]
hist_status_feature:  (N_hist × 7,)       — flat [vx,vy,ax,ay,px,py,h] per frame
```

Camera stitching (`_stitch_trio`):
- Left:   `img[28:-28, 416:-416]` — removes outermost columns and top/bottom strip
- Center: `img[28:-28]`
- Right:  `img[28:-28, 416:-416]`
- Concatenate horizontally → `cv2.resize` to `(camera_width, camera_height)`
- `torchvision.transforms.ToTensor()` → normalised float32 `(3, H, W)`
- Falls back to black frame (zeros) if any image is None / empty

#### `ActionDiffusionTargetBuilder.compute_targets()`
Produces:
```
trajectory:         (8, 3)   — sparse 0.5 s GT [x, y, h] ego-relative
interpolated_traj:  (40, 3)  — dense 0.1 s GT [x, y, h] ego-relative
```
Uses `transform_trajectory + get_trajectory_as_array` (NAVSIM PDM utils) to upsample the 8-step trajectory to 40 dense steps with correct bicubic interpolation, then converts to ego-relative poses via `absolute_to_relative_poses`.

---

### 3.6 `ad_agent.py` — `ActionDiffusionAgent`

Subclasses `AbstractAgent` (which is itself `nn.Module + ABC`).

**Key methods:**

`get_sensor_config()` — requests `[0,1,2,3]` frames for f0/l0/r0; empty list for l1/r1/lidar; back cameras only when `use_back_view=True`.

`compute_loss(features, targets, predictions)`:
```python
context_kv = predictions["context_kv"]            # built by model.forward()
gt_dense   = targets["interpolated_traj"]          # (B, 40, 3)
loss = diffusion_head.compute_loss(context_kv, gt_dense, features) * dp_loss_weight
```

`get_optimizers()` — Adam with two param groups when `freeze_backbone=False`:
- Backbone + img_proj: `lr × 0.1`
- Everything else: `lr` (default 1e-4)

Schedulers: `None` | `"cosine"` (step) | `"step"` (epoch) | `"cycle"` (step, OneCycleLR) | `"plateau"` (epoch, monitors `"val/loss_epoch"`)

`initialize()` — loads full Lightning checkpoint for inference; strips `"agent."` prefix from state_dict keys.

---

### 3.7 `ad_callback.py` — `ActionDiffusionCallback`

Fires at `on_validation_epoch_end`:
1. Grabs first batch from val dataloader
2. Runs `agent.forward(features)` in eval+no_grad
3. Extracts `dp_pred (B,N,8,3)` and GT `trajectory (B,8,3)`
4. Plots `num_samples` (default 4) side-by-side subplots:
   - Red dashed semi-transparent: all N proposals
   - Green solid thick: ground truth
   - Black triangle at ego origin
   - Axes: y = lateral (horizontal), x = forward (vertical, up = forward)
5. Logs to TensorBoard as `"Val/Trajectories"` at the current epoch

---

## 4. How to Use

### 4.1 Training

```bash
# From navsim_workspace/navsim — minimal example
python navsim/planning/script/run_training.py \
  agent=action_diffusion_agent \
  train_test_split=navtrain

# Override backbone
python navsim/planning/script/run_training.py \
  agent=action_diffusion_agent \
  agent.config.backbone_type=timm \
  agent.config.timm_model_name=convnext_small \
  agent.lr=3e-4

# Full end-to-end fine-tuning (no backbone freeze)
python navsim/planning/script/run_training.py \
  agent=action_diffusion_agent \
  agent.config.freeze_backbone=false \
  agent.lr=5e-5

# VoV backbone from GTRS/DiffusionDrive
python navsim/planning/script/run_training.py \
  agent=action_diffusion_agent \
  agent.config.backbone_type=vov \
  agent.config.vov_ckpt=/path/to/dd3d_det_final.pth \
  agent.config.freeze_backbone=true

# With cosine LR schedule
python navsim/planning/script/run_training.py \
  agent=action_diffusion_agent \
  agent.scheduler=cosine \
  agent.total_train_steps=50000
```

### 4.2 Inference / Evaluation

```bash
python navsim/planning/script/run_pdm_score.py \
  agent=action_diffusion_agent \
  agent.checkpoint_path=/path/to/epoch_042-val_loss_0.1234.ckpt \
  train_test_split=navtest
```

### 4.3 Warm-starting the backbone from a prior run

If you have a prior checkpoint from which you only want the visual encoder:
```yaml
# in action_diffusion_agent.yaml or on CLI
agent.config.pretrained_ckpt: /path/to/prior_checkpoint.ckpt
agent.config.freeze_backbone: true   # optional — freeze the warm-started encoder
```
Keys matching `"diffusion_head"` are always skipped so the head trains from scratch.

### 4.4 Reducing inference cost

```yaml
agent.config.num_inference_proposals: 16   # was 64
agent.config.num_diffusion_timesteps: 50   # was 100 (also affects training)
```

---

## 5. NAVSIM Framework Conventions

The agent must satisfy the `AbstractAgent` interface:

| Method | What it does |
|---|---|
| `name()` | Returns string identifier |
| `initialize()` | Called before inference; load checkpoint |
| `get_sensor_config()` | Which cameras / lidar / how many history frames |
| `get_feature_builders()` | List of `AbstractFeatureBuilder` |
| `get_target_builders()` | List of `AbstractTargetBuilder` |
| `forward(features)` | Returns a dict; must include `"trajectory"` key of shape `(B,8,3)` |
| `compute_loss(features, targets, predictions)` | Returns scalar tensor |
| `get_optimizers()` | Returns optimizer or `{optimizer, lr_scheduler}` dict |
| `get_training_callbacks()` | List of `pl.Callback` |

The Lightning loop (`AgentLightningModule`) calls these in order:
```
forward(features) → compute_loss(features, targets, predictions)
```

The `"trajectory"` output is evaluated by the NAVSIM PDM score (not by a loss). During training it is safe to return a zero placeholder.

---

## 6. Tensor Shapes — Quick Reference

```
# Feature tensors
camera_feature:       (B, seq_len, 3, H, W)   H=512, W=2048
status_feature:       (B, 8)                  [cmd(4), vx, vy, ax, ay]
hist_status_feature:  (B, N_hist*7)           N_hist = seq_len-1+3 = 3 typically

# Internal context
backbone tokens:      (B, N_img, C)           N_img = 16×64 = 1024
img_proj output:      (B, N_img, D)           D = model_dim = 256
status token:         (B, 1, D)
context KV:           (B, 1025, D)            = 1024 + 1

# Diffusion
noisy_actions:        (B, 40, 2)              (accel, curvature), z-score normed
pred_noise:           (B, 40, 2)
t_batch:              (B,)                    integer timestep 0…T-1

# Output
dp_pred:              (B, N, 8, 3)            [x, y, heading] ego-relative
trajectory:           (B, 8, 3)               single selected proposal
interpolated_traj:    (B, 40, 3)              GT used for diffusion loss
```

---

## 7. Switching the Noise Scheduler

Currently the agent uses **DDPM** (`DDPMScheduler` from `diffusers`).  
Two upgrade paths were discussed; here is what each requires:

---

### 7.1 DDPM → DDIM (easy, ~15 min)

DDIM shares the **exact same training objective** as DDPM (epsilon-prediction MSE).  
Only inference changes — a non-Markovian shortcut path that allows ~10× fewer steps.

**`ad_config.py`**  
Add one field:
```python
num_inference_steps: int = 10   # DDIM inference steps (separate from training T)
```

**`ad_diffusion_head.py`**  
1. Change import: `from diffusers import DDIMScheduler`
2. Change scheduler init:
   ```python
   self.noise_scheduler = DDIMScheduler(
       num_train_timesteps=config.num_diffusion_timesteps,
       beta_start=0.0001,
       beta_end=0.02,
       beta_schedule="squaredcos_cap_v2",
       clip_sample=True,
       clip_sample_range=5.0,
       prediction_type="epsilon",
   )
   ```
3. In `compute_loss()`: **no changes** — training is identical.
4. In `forward()`, change one line:
   ```python
   # Before (DDPM):
   self.noise_scheduler.set_timesteps(cfg.num_diffusion_timesteps, device=device)
   # After (DDIM):
   self.noise_scheduler.set_timesteps(cfg.num_inference_steps, device=device)
   ```
   The `scheduler.step()` call inside the loop is identical — `DDIMScheduler` has the same API.

**Hydra YAML** — add `num_inference_steps: 10` under `config:`.

Net effect: training unchanged, inference 10× faster (10 steps vs 100), slight reduction in trajectory diversity.

---

### 7.2 DDPM → Flow Matching (deeper, ~80 lines)

Flow matching uses **straight probability paths** (linear interpolation between data and noise) instead of the Markov chain. The denoiser architecture (`_DenoisingTransformer`) is **completely unchanged**.

#### Training objective changes

DDPM trains a noise predictor: predict $\epsilon$ from $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$.

Flow matching trains a **velocity** predictor: given $x_t = (1-t) x_0 + t \epsilon$, predict $v = \epsilon - x_0$.

The continuous time variable $t \in [0,1]$ replaces the integer timestep index.

#### Changes required

**`ad_config.py`**
- Remove: `beta_start`, `beta_end`, `beta_schedule`, `variance_type`, `clip_sample` (scheduler fields, not present as explicit config fields but they're in the head init)
- Rename `num_diffusion_timesteps` → `num_flow_steps` (inference only, typically 20–50); or keep both (train T unused in FM, only N inference steps matters)

**`ad_diffusion_head.py`** — all changes concentrated here:

1. Remove `DDPMScheduler` import. No `diffusers` scheduler needed at all.

2. Rewrite `compute_loss()`:
   ```python
   def compute_loss(self, context, gt_traj_dense, features):
       B, device = context.shape[0], context.device
       
       # GT actions (same as before)
       gt_xyz, gt_rot = self._se2_to_xyz_rot(gt_traj_dense.float())
       hist_xyz, hist_rot = self._build_history(features)
       v0 = features["status_feature"][:, 4]
       x0 = self.action_space.traj_to_action(gt_xyz, gt_rot,
                                              hist_xyz, hist_rot, {"v": v0})  # (B,40,2)
       
       # Sample t uniformly in [0, 1], one value per sample
       t = torch.rand(B, device=device)                       # (B,)
       
       # Linear interpolation: x_t = (1-t)*x0 + t*noise
       noise = torch.randn_like(x0)
       t_exp = t[:, None, None]                               # (B,1,1) for broadcast
       x_t = (1 - t_exp) * x0 + t_exp * noise
       
       # Target velocity: v = noise - x0
       target_v = noise - x0
       
       # Predict velocity; timestep is now continuous float in [0,1]
       # _SinusoidalPosEmb works with integers; replace with _FourierEncoder for t
       pred_v = self.denoiser(x_t, t, context)  # t is (B,) float
       return F.mse_loss(pred_v, target_v)
   ```

3. Rewrite `forward()` (inference) — simple Euler ODE integration:
   ```python
   def forward(self, context, features):
       cfg = self.config
       B, N = context.shape[0], cfg.num_inference_proposals
       ctx = context.repeat_interleave(N, dim=0)
       # ...setup hist/v0 as before...
       
       x = torch.randn(B*N, cfg.internal_horizon, cfg.action_dim, ...)
       n_steps = cfg.num_flow_steps              # e.g. 20
       dt = 1.0 / n_steps
       for i in range(n_steps, 0, -1):
           t_val = i / n_steps
           t_b = torch.full((B*N,), t_val, device=device)
           v = self.denoiser(x, t_b, ctx)
           x = x - dt * v                        # Euler step: integrate from t=1 → 0
       pred_actions_norm = x
       # ...action_to_traj + downsample as before...
   ```

4. Timestep encoding — `_SinusoidalPosEmb` currently takes integer timesteps.  
   For flow matching, $t$ is a continuous float in [0,1].  
   The `_FourierEncoder` class already handles this (it multiplies by frequencies).  
   Change `_DenoisingTransformer` to use `_FourierEncoder(dim=model_dim)` instead of  
   `_SinusoidalPosEmb(dim=model_dim)` for the time token.

**`ad_model.py`, `ad_agent.py`, `ad_features.py`** — no changes needed.

#### Summary comparison

| | DDIM | Flow Matching |
|---|---|---|
| `compute_loss()` | unchanged | full rewrite (linear interp + velocity target) |
| Inference loop | 1 line (step count) | full rewrite (Euler ODE, no scheduler) |
| Timestep type | integer 0…T | continuous float 0…1 |
| `_SinusoidalPosEmb` | unchanged | replace with `_FourierEncoder` |
| `_DenoisingTransformer` | unchanged | unchanged |
| `ad_model.py` | unchanged | unchanged |
| Typical inference steps | 10–20 | 10–50 |
| Relative effort | ~15 min | ~1–2 hours |

Flow matching typically converges faster and produces straighter, more consistent trajectory rollouts. If training time matters more than code simplicity, it's worth doing.

---

## 8. Key Design Decisions & Gotchas

1. **VoVNet returns a Python `list`, not an OrderedDict.** Use `feat_list[-1]`, not dictionary access.

2. **timm backbone probe:** `PerceptionBackbone.__init__` runs a CPU dry-run with a zero tensor to detect the output channel count. This is a one-time cost but will appear during model instantiation.

3. **`model.train()` override:** PyTorch's `train()` is overridden in `ActionDiffusionModel` to force `backbone.eval()` when `freeze_backbone=True`. Without this, every Lightning `model.train()` call at epoch start would unfreeze frozen BatchNorm stats.

4. **`trajectory` is a zero placeholder during training.** NAVSIM's `AgentLightningModule` calls `forward()` then `compute_loss()`. The trajectory output is only used by the evaluator, never by the training loop. Returning zeros avoids wasting inference compute during training.

5. **`context_len + 1` in the denoiser.** `_DenoisingTransformer.cond_pos` is sized `(1, context_len+1, D)` because the sinusoidal time token is **prepended** to the context KV at every forward pass, making the effective sequence length `L+1`.

6. **Two checkpoint concepts:**
   - `config.pretrained_ckpt` → backbone warm-start only (strips diffusion_head keys)
   - `agent.checkpoint_path` → full agent restore (used in `initialize()`)

7. **Action normalisation stats** (`accel_mean/std`, `curv_mean/std`) were estimated on the mini training split. If you train on a significantly different data distribution, re-estimate them — a large mismatch will slow convergence.

8. **Inference proposal selection** is random (uniform over N proposals). For evaluation you may want to pick the trajectory closest to the ego's current speed/heading, or use a learned scoring head.
