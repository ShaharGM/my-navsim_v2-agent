# ActionDiffusionAgent — Implementation Notes

Complete reference for the `action_diffusion_agent` package.  
Intended audience: future sessions with an AI assistant, or any developer picking up this codebase.

---

## 1. Overview

`ActionDiffusionAgent` is a camera-based autonomous driving agent for the **NAVSIM** evaluation framework. It follows a diffusion-policy architecture:

```
front panoramic image (l0 + f0 + r0)  [+ rear view when use_back_view=True]
    │
    ▼
BackboneBase subclass  (PoolBackbone: timm/VoV  |  BEVBackbone: VoV+BEV fusion)
    │  features dict → (B, N_img, C)   N_img and C depend on backbone
    ▼
img_proj  (Linear: C → D)       status_encoder  (Linear+LN: 8 → D)
    │                                    │
    └──────────────┬─────────────────────┘
                   │  concat
                   ▼
        context KV  (B, N_img+1, D)       ← positional bias added
                   │
         ┌─────────┴─────────────────────────────┐
         │  TRAINING                             │  INFERENCE
         │                                       │
         │  DDPM: add noise at random int t      │  DDPM: sample N Gaussian noises
         │         predict ε (MSE)               │         reverse chain (T steps)
         │                                       │
         │  Flow: interpolate at random float t  │  Flow: sample N Gaussian noises
         │         predict velocity v (MSE)      │         Euler ODE t=1→0 (K steps)
         │                                       │
         └───────────────────────────────────────┘
                   │
     action_space.action_to_traj()   ← UnicycleAccelCurvature
                   │
        dense trajectory (40 × 3)
                   │
         ↓ downsample ×5
        sparse trajectory (8 × 3)   ← returned to NAVSIM evaluator
```

The noise mode is controlled by `config.noise_type` (`'ddpm'` or `'flow'`).

The agent operates in a **unicycle action space** — all diffusion is done on
(acceleration, curvature) pairs, not directly on positions.  
This gives the head a physically grounded, lower-dimensional signal to learn.

---

## 2. File Structure

```
action_diffusion_agent/
├── __init__.py               public API exports
├── ad_config.py              ActionDiffusionConfig dataclass (all hyperparams)
├── ad_diffusion_head.py      DDPM / Flow Matching head + DenoisingTransformer + physics
├── ad_model.py               ActionDiffusionModel (wires backbone→context→head)
├── ad_features.py            Feature/target builders (NAVSIM AbstractFeatureBuilder)
├── ad_agent.py               ActionDiffusionAgent (NAVSIM AbstractAgent)
├── ad_callback.py            TensorBoard visualisation callback
├── AGENT_NOTES.md            ← this file
└── backbones/
    ├── __init__.py           build_backbone() factory + __all__
    ├── base.py               BackboneBase abstract class (ABC + nn.Module)
    ├── pool_backbone.py      PoolBackbone — timm / VoV CNN + AdaptiveAvgPool2d
    └── bev_backbone.py       BEVBackbone — VoVNet + cross-attention BEV fusion
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
| `use_back_view` | True | Include rear panorama (l2+b0+r2) |
| `backbone_type` | `"timm"` | `"timm"`, `"vov"`, or `"bev"` |
| `timm_model_name` | `"resnet50"` | Any `timm.create_model` identifier (`"timm"` only) |
| `timm_pretrained` | True | Load ImageNet weights via timm hub (`"timm"` only) |
| `vov_ckpt` | `""` | Path to VoV checkpoint (`"vov"` and `"bev"` backbones) |
| `bev_ckpt` | `""` | Path to full BEVBackbone checkpoint (`"bev"` only) |
| `freeze_backbone` | True | Freeze backbone during training |
| `bev_fusion_layers` | 3 | TransformerEncoder depth inside BEV fusion (`"bev"` only) |
| `img_vert_anchors` | 16 | VoV/CNN pool height per camera — **all** backbone types; defaults match `gtrs_dp.ckpt` |
| `img_horz_anchors` | 64 | VoV/CNN pool width per camera — **all** backbone types |
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
| `noise_type` | `"ddpm"` | `"ddpm"` for DDPM or `"flow"` for flow matching |
| `num_flow_steps` | 20 | Euler integration steps at inference time (flow only) |
| `weight_decay` | 0.0 | Adam weight decay |
| `dp_loss_weight` | 10.0 | Multiplier on the MSE diffusion loss |

**Key distinction between two checkpoint fields:**
- `config.pretrained_ckpt` — loads only backbone + projection weights (strips diffusion head). Used to warm-start the visual encoder from a prior run.
- `agent.checkpoint_path` — loads the **complete** Lightning state dict for inference or training resume.

---

### 3.2 `backbones/` — Perception Backbone Package

The backbone is now a proper subpackage with an abstract base class and two concrete implementations. `ActionDiffusionModel` only ever calls `build_backbone(config)` and then uses the returned object through the `BackboneBase` interface — it has no knowledge of which backbone is active.

#### `base.py` — `BackboneBase` (ABC + nn.Module)

Every backbone must implement:

| Property / method | Type | Description |
|---|---|---|
| `num_tokens` | `int` (property) | Total context tokens returned per sample |
| `out_channels` | `int` (property) | Channel dimension of each token |
| `forward(features)` | `(B, num_tokens, out_channels)` | Receives the raw feature dict, returns flat token tensor |

The `features` dict is passed through unchanged, so each backbone can read whichever keys it needs.

#### `__init__.py` — `build_backbone(config)` factory

```python
if config.backbone_type in ("timm", "vov"):
    return PoolBackbone(config)
elif config.backbone_type == "bev":
    return BEVBackbone(config)
```

#### `pool_backbone.py` — `PoolBackbone`

Single CNN encoder + `AdaptiveAvgPool2d` spatial pooling.

```
Input keys: "camera_feature" (B,T,3,H,W)  +  optionally "camera_feature_back"
Output:     (B, N, C)
  N = img_vert_anchors × img_horz_anchors        (× 2 if use_back_view)
  C = encoder output channels
```

**timm path:**
```python
self._encoder = timm.create_model(name, pretrained=True, features_only=True, out_indices=(-1,))
feat = self._encoder(image)[-1]   # always a list; take last
```
Channel count is probed once at init with a CPU zero-tensor dry run.

**VoV path:**
```python
self._encoder = VoVNet("V-99-eSE", out_features=["stage4","stage5"], ...)
feat = self._encoder(image)[-1]   # plain list; 1024 channels
```

Both paths: `AdaptiveAvgPool2d((vert, horz))` → `flatten(2).permute(0,2,1)` → `(B, V×H, C)`.

#### `bev_backbone.py` — `BEVBackbone`

VoVNet encoder + cross-attention BEV fusion. Architecture matches `HydraBackboneBEV` from GTRS so pretrained GTRS backbone checkpoints can be loaded directly (attribute names are intentionally identical).

```
Input keys: "camera_feature"  (B,T,3,H,W)  — front stitched view
            "camera_feature_back"  (B,T,3,H,W)  — rear stitched view
Output:     (B, 64, 1024)   — 64 BEV query tokens (8×8 grid), 1024 channels
```

Internal pipeline:
1. VoV encoder → `AdaptiveAvgPool2d((img_vert_anchors, img_horz_anchors))` → `(B, P, 1024)` per camera, where `P = img_vert_anchors × img_horz_anchors`.
2. Concat front + back → `(B, 2P, 1024)`.
3. Concat with learnable BEV queries `(B, 64, 1024)` → `(B, 2P+64, 1024)`.
4. Add positional embeddings, run `TransformerEncoder` (`bev_fusion_layers` layers, 16 heads).
5. Return only the last 64 positions (BEV query outputs): `tokens[:, 2P:]`.

Checkpoint loading (`bev_ckpt`): strips `"agent.model._backbone."` / `"agent.model.backbone."` / `"model.backbone."` / `"backbone."` prefixes automatically.

---

### 3.3 `ad_diffusion_head.py` — `ActionDiffusionHead`

Supports two modes selected by `config.noise_type`:
- `'ddpm'` — classic DDPM epsilon-prediction (default)
- `'flow'` — flow matching with straight probability paths

The denoising transformer architecture is **shared and unchanged** between both modes.

#### Building blocks
| Class | Purpose |
|---|---|
| `_FourierEncoder` | Log-spaced sin/cos encoding of scalars. Used in `_PerWaypointEncoder` (dim=20) and as the flow matching time embedding (dim=model_dim) |
| `_RMSNorm` | Root-mean-square norm (no bias) |
| `_SinusoidalPosEmb` | Integer timestep → (B, D) sinusoidal embedding. Used as DDPM time embedding |
| `_build_mlp` | RMSNorm + SiLU depth-5 MLP |
| `_PerWaypointEncoder` | (B,T,A) + timestep → (B,T,D); per-dim Fourier + MLP + LN |
| `_DenoisingTransformer` | Pre-LN TransformerDecoder; action queries cross-attend context KV |

#### `_DenoisingTransformer`
- Input:  `noise_or_xt (B,T,A)`, `timestep (B,)`, `context (B,L,D)`
- `timestep` is an **integer** (DDPM) or a **continuous float in [0,1]** (flow matching)
- `noise_type` is passed at construction to select the time embedding module:
  - DDPM → `_SinusoidalPosEmb(d_model)`
  - Flow → `_FourierEncoder(dim=d_model)`
- Time token prepended to the context KV: `(B, L+1, D)`
- Learned positional biases: `action_pos (1,T,D)` on query side, `cond_pos (1,L+1,D)` on KV side
- Architecture: Pre-LN multi-head cross-attention → Pre-LN FFN (GeLU), N layers
- Output: predicted noise ε (DDPM) or predicted velocity v (flow): `(B, T, A)`

#### `_get_gt_actions()` — shared helper
Extracts normalised (accel, curvature) actions from the dense GT trajectory. Called by both `_ddpm_compute_loss` and `_flow_compute_loss` to avoid duplicating the trajectory-inversion logic.

#### `compute_loss()` — training dispatcher
Dispatches to `_ddpm_compute_loss` or `_flow_compute_loss` based on `config.noise_type`.

**DDPM (`_ddpm_compute_loss`)**:
```python
# 1. GT actions via _get_gt_actions() → (B, 40, 2)
# 2. DDPM forward process: add noise at a random integer timestep
noise = torch.randn_like(gt_actions)
timesteps = randint(0, T, (B,))  # integer
noisy_actions = noise_scheduler.add_noise(gt_actions, noise, timesteps)
# 3. Predict noise; MSE loss
pred_noise = denoiser(noisy_actions, timesteps, context)
return F.mse_loss(pred_noise, noise)
```

**Flow matching (`_flow_compute_loss`)**:
```python
# 1. GT actions via _get_gt_actions() → x0 (B, 40, 2)
# 2. Sample continuous time t ~ Uniform[0, 1], one per sample
t = torch.rand(B, device=device)          # float in [0, 1]
# 3. Linear interpolation: x_t = (1-t)*x0 + t*noise
noise = torch.randn_like(x0)
x_t = (1 - t[:,None,None]) * x0 + t[:,None,None] * noise
# 4. Ground-truth velocity
target_v = noise - x0
# 5. Predict velocity; MSE loss
pred_v = denoiser(x_t, t, context)        # t is (B,) float
return F.mse_loss(pred_v, target_v)
```

#### `forward()` — inference dispatcher
Dispatches to `_ddpm_forward` or `_flow_forward` based on `config.noise_type`.

**DDPM (`_ddpm_forward`)**:
```python
# 1. Start from pure Gaussian noise (B*N proposals)
noise = randn(B*N, 40, 2)
# 2. Full DDPM reverse chain
scheduler.set_timesteps(T, device)
for t in scheduler.timesteps:
    pred_eps = denoiser(noise, t.expand(B*N), ctx)
    noise = scheduler.step(pred_eps, t, noise).prev_sample
# 3. action_to_traj → dense trajectory → sparse trajectory
return {"dp_pred": (B,N,8,3), "pred_actions": (B,N,40,2)}
```

**Flow matching (`_flow_forward`)**:
```python
# 1. Start from pure Gaussian noise at t=1 (B*N proposals)
x = randn(B*N, 40, 2)
# 2. Euler integration from t=1 → t=0 (num_flow_steps uniform steps)
n_steps = cfg.num_flow_steps;  dt = 1.0 / n_steps
for i in range(n_steps, 0, -1):
    t_val = i / n_steps                         # float in (0, 1]
    t_b = torch.full((B*N,), t_val, ...)
    v = denoiser(x, t_b, ctx)                   # predicted velocity
    x = x - dt * v                               # Euler step toward t=0
# 3. action_to_traj → dense trajectory → sparse trajectory
return {"dp_pred": (B,N,8,3), "pred_actions": (B,N,40,2)}
```

#### History construction (`_build_history`)
`hist_status_feature` is a flat `(B, N_hist×7)` tensor.  
Layout per historical frame: `[vx, vy, ax, ay, px, py, heading]`.  
The last 3 values (px, py, heading) are extracted for each frame, and a zero current-frame placeholder is appended → `(B, N_hist+1, 3)` → converted to xyz/rotation tensors for the unicycle model.

Default `seq_len=1` gives 3 history frames + 1 current = `N_hist=3`, so `hist_status_feature` is `(B, 21)`.

---

### 3.4 `ad_model.py` — `ActionDiffusionModel`

Wires everything together. The model is **completely agnostic** to which backbone is active — it sizes all downstream layers from `backbone.num_tokens` and `backbone.out_channels` at init time.

```
context_len = backbone.num_tokens + 1

# Default (timm/vov, use_back_view=True):
#   16×64 × 2  +  1  =  2049
# BEV backbone:
#   64  +  1  =  65
```

**`build_backbone(config)`** (called at init) returns the correct `BackboneBase` subclass; the model stores it as `self.backbone` and never inspects it further.

**Context building (`_build_context`):**
1. `self.backbone(features)` → `(B, N, C)`   *(backbone owns all image extraction)*
2. `img_proj` (Linear: C → D) → `(B, N, D)`
3. `status_encoder(status)` → `(B, 1, D)`
4. `cat([img_tokens, status_tok], dim=1)` → `(B, N+1, D)`
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

# BEV backbone (VoVNet + cross-attention BEV fusion)
python navsim/planning/script/run_training.py \
  agent=action_diffusion_agent \
  agent.config.backbone_type=bev \
  agent.config.vov_ckpt=/path/to/dd3d_det_final.pth \
  agent.config.bev_ckpt=/path/to/gtrs_backbone.ckpt \
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

### 4.4 Flow matching mode

```bash
# Flow matching — faster convergence, cleaner trajectories
python navsim/planning/script/run_training.py \
  agent=action_diffusion_agent \
  agent.config.noise_type=flow \
  agent.config.num_flow_steps=20
```

Note: DDPM checkpoints and flow matching checkpoints are **not interchangeable** — the denoiser output is epsilon vs velocity, and the time embedding is different. Always train from scratch when switching `noise_type`.

### 4.5 Reducing inference cost

```yaml
agent.config.num_inference_proposals: 16   # was 64
agent.config.num_diffusion_timesteps: 50   # was 100 (also affects DDPM training)
agent.config.num_flow_steps: 10            # was 20 (flow only; does not affect training)
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

# Internal context  (defaults: timm/vov, use_back_view=True)
backbone tokens:      (B, N_img, C)           N_img = 16×64×2 = 2048  (or 64 for BEV)
img_proj output:      (B, N_img, D)           D = model_dim = 256
status token:         (B, 1, D)
context KV:           (B, 2049, D)            = 2048 + 1  (or 65 for BEV backbone)

# Diffusion (DDPM)
noisy_actions:        (B, 40, 2)              (accel, curvature), z-score normed
pred_noise:           (B, 40, 2)
t_batch:              (B,)                    integer timestep 0…T-1

# Diffusion (flow matching)
x_t:                  (B, 40, 2)              linearly interpolated actions
pred_velocity:        (B, 40, 2)              predicted velocity v = ε - x0
t_batch:              (B,)                    continuous float in [0, 1]

# Output
dp_pred:              (B, N, 8, 3)            [x, y, heading] ego-relative
trajectory:           (B, 8, 3)               single selected proposal
interpolated_traj:    (B, 40, 3)              GT used for diffusion loss
```

---

## 7. Noise Mode Reference

The head supports two modes, selected by `config.noise_type`.

---

### 7.1 DDPM (default)

Classic epsilon-prediction with a Markov noising chain.

| Aspect | Detail |
|---|---|
| Training target | Predict added noise ε |
| Training timestep | Integer t ~ Uniform{0, …, T-1}, `T = num_diffusion_timesteps` |
| Forward process | `x_t = sqrt(ᾱ_t)·x₀ + sqrt(1-ᾱ_t)·ε` (cosine schedule) |
| Inference | Full Markov reverse chain, T steps using `DDPMScheduler` |
| Time embedding | `_SinusoidalPosEmb` (integer input) |
| Scheduler object | `DDPMScheduler` from `diffusers` |
| Key config fields | `num_diffusion_timesteps` (default 100) |

### 7.2 Flow Matching (`noise_type='flow'`)

**Status: implemented.** Enabled by setting `config.noise_type = 'flow'`.

Straight probability paths (linear interpolation between data and noise) instead of the Markov chain. The denoiser architecture (`_DenoisingTransformer`) is **identical** between both modes — only the time embedding and the training/inference loops differ.

| Aspect | Detail |
|---|---|
| Training target | Predict velocity `v = ε - x₀` |
| Training timestep | Continuous t ~ Uniform[0, 1] |
| Forward process | `x_t = (1-t)·x₀ + t·ε` (linear interpolation) |
| Inference | Euler ODE integration from t=1 → t=0, `num_flow_steps` uniform steps |
| Time embedding | `_FourierEncoder(dim=model_dim)` (continuous float input) |
| Scheduler object | None (no `diffusers` dependency at all) |
| Key config fields | `num_flow_steps` (default 20) |

#### Why flow matching?
- Straight trajectories in probability space → faster convergence.
- No DDPM noise schedule to tune.
- Inference quality scales smoothly with step count; 10–20 steps is typically sufficient.

#### Checkpoint compatibility
DDPM and flow matching checkpoints are **not interchangeable** — the denoiser output semantics (ε vs v) and the time embedding differ. Always train from scratch when switching `noise_type`.

---

### 7.3 Future: DDIM (easy upgrade from DDPM, ~15 min)

DDIM shares the **exact same training objective** as DDPM (epsilon-prediction MSE).  
Only inference changes — a non-Markovian shortcut path that allows ~10× fewer steps.

**`ad_config.py`** — add:
```python
num_inference_steps: int = 10
```

**`ad_diffusion_head.py`**  
1. Change import: `from diffusers import DDIMScheduler`  
2. Change scheduler init to `DDIMScheduler(...)` with the same kwargs.  
3. `compute_loss()` — no changes.  
4. In `_ddpm_forward()` change one line: `scheduler.set_timesteps(cfg.num_inference_steps, device=device)`.  
   The `scheduler.step()` call inside the loop is identical — `DDIMScheduler` has the same API.

Net effect: training unchanged, inference 10× faster, slight reduction in trajectory diversity.

---

## 8. Key Design Decisions & Gotchas

1. **Backbone abstraction boundary.** `ActionDiffusionModel` never imports or inspects any concrete backbone class. It calls `build_backbone(config)` once at init, reads `backbone.num_tokens` and `backbone.out_channels` to size `img_proj` and `pos_embedding`, then calls `backbone(features)` during forward. Adding a new backbone type only requires implementing `BackboneBase` and registering it in `build_backbone()`.

2. **VoVNet returns a Python `list`, not an OrderedDict.** Use `feat_list[-1]`, not dictionary access. Both `PoolBackbone` and `BEVBackbone` handle this internally.

3. **timm backbone probe:** `PoolBackbone.__init__` runs a CPU dry-run with a zero tensor to detect the output channel count. This is a one-time cost but will appear during model instantiation.

4. **`model.train()` override:** PyTorch's `train()` is overridden in `ActionDiffusionModel` to force `backbone.eval()` when `freeze_backbone=True`. Without this, every Lightning `model.train()` call at epoch start would unfreeze frozen BatchNorm stats.

5. **`trajectory` is a zero placeholder during training.** NAVSIM's `AgentLightningModule` calls `forward()` then `compute_loss()`. The trajectory output is only used by the evaluator, never by the training loop. Returning zeros avoids wasting inference compute during training.

6. **`context_len + 1` in the denoiser.** `_DenoisingTransformer.cond_pos` is sized `(1, context_len+1, D)` because the sinusoidal time token is **prepended** to the context KV at every forward pass, making the effective sequence length `L+1`.

7. **Two checkpoint concepts:**
   - `config.pretrained_ckpt` → backbone warm-start only (strips diffusion_head keys)
   - `agent.checkpoint_path` → full agent restore (used in `initialize()`)

8. **Action normalisation stats** (`accel_mean/std`, `curv_mean/std`) were estimated on the mini training split. If you train on a significantly different data distribution, re-estimate them — a large mismatch will slow convergence.

9. **Inference proposal selection** is random (uniform over N proposals). For evaluation you may want to pick the trajectory closest to the ego's current speed/heading, or use a learned scoring head.

10. **Flow matching time embedding uses `_FourierEncoder`, not `_SinusoidalPosEmb`.** `_SinusoidalPosEmb` is designed for integer inputs and produces near-zero output for small floats in [0,1]. Always check that `_DenoisingTransformer` was constructed with `noise_type='flow'` when loading a flow matching checkpoint.

11. **`noise_type` is baked into the model at construction time** (it selects the time embedding class). A DDPM-trained checkpoint cannot be used with `noise_type='flow'` or vice versa — the `denoiser.time_emb` weights would be incompatible.
