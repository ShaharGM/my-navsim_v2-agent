"""
ActionDiffusionHead — trajectory diffusion head with DDPM and Flow Matching support.

Two modes are selected by ``ActionDiffusionConfig.noise_type``:

``'ddpm'`` (default)
    Classic DDPM (epsilon-prediction).

    Training
        1. Convert GT dense trajectory → normalised (accel, curvature) actions.
        2. Add DDPM noise at a random integer timestep t ∈ [0, T-1].
        3. Denoising transformer predicts the added noise ε.
        4. MSE loss: pred_ε vs actual ε.

    Inference
        1. Sample N parallel Gaussian noises.
        2. Run full DDPM Markov reverse chain (T steps).
        3. Convert denoised actions → dense trajectory → sparse trajectory.

``'flow'`` (flow matching)
    Straight probability paths (continuous-time flow matching).

    Training
        1. Convert GT dense trajectory → normalised actions x₀.
        2. Sample t ~ Uniform[0, 1].
        3. Linearly interpolate: x_t = (1-t)·x₀ + t·ε, ε ~ N(0, I).
        4. Denoising transformer predicts the velocity v = ε - x₀.
        5. MSE loss: pred_v vs (ε - x₀).

    Inference
        1. Sample N parallel Gaussian noises (starting at t=1).
        2. Euler integration from t=1 → t=0:
               x ← x - (dt)·v_θ(x, t)   for num_flow_steps uniform steps.
        3. Convert denoised actions → dense trajectory → sparse trajectory.

The denoising transformer architecture (``_DenoisingTransformer``) is
**identical** between both modes.  The only architectural difference is the
time embedding: DDPM uses a sinusoidal integer-timestep embedding; flow
matching uses a Fourier feature encoder for continuous t ∈ [0, 1].
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig
from navsim.agents.utils.action_space.unicycle_accel_curvature import (
    UnicycleAccelCurvatureActionSpace,
)


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------

class _FourierEncoder(nn.Module):
    """
    Logarithmically-spaced Fourier feature encoder for a scalar input.

    Maps a scalar x → [sin(freq_i * x), cos(freq_i * x)] of length `dim`.
    The scaling factor sqrt(2) keeps the expected L2 norm equal to 1.
    """

    def __init__(self, dim: int = 20, max_freq: float = 100.0) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        half = dim // 2
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        self.register_buffer("freqs", freqs[None, :])  # (1, half)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: arbitrary leading dims (...,)
        # output: (..., dim)
        arg = x[..., None] * self.freqs * 2.0 * math.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1) * math.sqrt(2)


class _RMSNorm(nn.Module):
    """Root-Mean-Square normalisation (no learnable bias)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.scale


class _SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep encoding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (...) integer timesteps — supports (B,) and (B, T)
        half = self.dim // 2
        freqs = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -freqs)  # (half,)
        emb = t[..., None].float() * freqs                               # (..., half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)                  # (..., dim)


def _build_mlp(in_dim: int, hidden: int, out_dim: int, depth: int = 4) -> nn.Sequential:
    """MLP with RMSNorm + SiLU activations."""
    layers: list = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(depth - 2):
        layers += [_RMSNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU()]
    layers += [_RMSNorm(hidden), nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Per-waypoint action encoder
# ---------------------------------------------------------------------------

class _PerWaypointEncoder(nn.Module):
    """
    Encodes a (B, T, A) noisy-action sequence + diffusion timestep
    into (B, T, d_model) embeddings.

    Each action dimension is encoded separately with Fourier features;
    the timestep is also Fourier-encoded and broadcast across all waypoints.
    All features are concatenated and passed through a shared MLP.

    This mirrors the PerWaypointActionInProjV2 approach used in the
    original action_dp_agent, but is self-contained here.
    """

    # Hyper-parameters kept constant for clarity; expose if you need to tune.
    _NUM_FOURIER = 20
    _MLP_HIDDEN = 1024
    _MLP_DEPTH = 5

    def __init__(self, action_dim: int, d_model: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        # One Fourier encoder per action dimension
        self.action_encoders = nn.ModuleList(
            [_FourierEncoder(self._NUM_FOURIER) for _ in range(action_dim)]
        )
        self.time_encoder = _FourierEncoder(self._NUM_FOURIER)

        in_dim = action_dim * self._NUM_FOURIER + self._NUM_FOURIER
        self.proj = _build_mlp(in_dim, self._MLP_HIDDEN, d_model, depth=self._MLP_DEPTH)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, actions: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions:  (B, T, A)      — noisy action sequence (normalised)
            timestep: (B,) or (B, T) — int timestep (DDPM) or float t (flow).
                      (B, T) is used when use_diffusion_forcing=True; each
                      waypoint gets its own independent noise level.

        Returns:
            (B, T, d_model)
        """
        B, T, _ = actions.shape
        # Action Fourier features: each dim encoded separately
        act_feats = torch.cat(
            [enc(actions[..., i]) for i, enc in enumerate(self.action_encoders)],
            dim=-1,
        )  # (B, T, A*F)

        # Timestep Fourier features: (B,) → broadcast to all T positions,
        # or (B, T) → per-token encoding (Diffusion Forcing).
        t_enc = self.time_encoder(timestep.float())   # (B, F) or (B, T, F)
        if t_enc.dim() == 2:                          # standard: same t for all tokens
            t_feat = t_enc[:, None, :].expand(-1, T, -1)   # → (B, T, F)
        else:                                         # diffusion forcing: per-token t
            t_feat = t_enc                                  # already (B, T, F)

        combined = torch.cat([act_feats, t_feat], dim=-1)  # (B, T, (A+1)*F)
        return self.norm(self.proj(combined))               # (B, T, d_model)


# ---------------------------------------------------------------------------
# Denoising transformer
# ---------------------------------------------------------------------------

class _DenoisingTransformer(nn.Module):
    """
    TransformerDecoder-based denoiser.

    The action embeddings play the role of query (tgt) and the scene context
    (image tokens + ego-status token, prepended by a sinusoidal time token)
    plays the role of key-value memory.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        num_layers: int,
        d_cond: int,                      # dimension of the incoming KV context tensor
        context_len: int,                 # number of context tokens (image + status tokens)
        action_dim: int,
        horizon: int,
        noise_type: str = "ddpm",         # 'ddpm' or 'flow'
        use_diffusion_forcing: bool = False,
    ) -> None:
        super().__init__()

        # Project context from d_cond → d_model (no-op if d_cond == d_model)
        self.cond_proj = nn.Linear(d_cond, d_model)

        # Time embedding:
        #   DDPM → sinusoidal encoding of integer timestep t ∈ [0, T-1]
        #   Flow → Fourier feature encoding of continuous t ∈ [0, 1]
        if noise_type == "flow":
            self.time_emb: nn.Module = _FourierEncoder(dim=d_model)
        else:
            self.time_emb = _SinusoidalPosEmb(d_model)

        self.action_encoder = _PerWaypointEncoder(action_dim, d_model)

        # Learnable positional biases
        self.action_pos = nn.Parameter(torch.zeros(1, horizon, d_model))
        # Standard mode:         1 time token  → cond_pos shape (1, context_len + 1,       d_model)
        # Diffusion Forcing mode: T time tokens → cond_pos shape (1, context_len + horizon, d_model)
        _n_time_slots = horizon if use_diffusion_forcing else 1
        self.cond_pos = nn.Parameter(torch.zeros(1, context_len + _n_time_slots, d_model))

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ffn_dim,
                dropout=0.0,
                batch_first=True,
                norm_first=True,    # Pre-LN for training stability
            ),
            num_layers=num_layers,
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, action_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.action_pos, std=0.02)
        nn.init.normal_(self.cond_pos, std=0.02)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_actions: (B, T, A)         — noisy (or interpolated) action sequence
            timestep:      (B,) or (B, T)     — int timestep (DDPM) or float t ∈ [0,1] (flow).
                           When use_diffusion_forcing=True, shape is (B, T): one noise
                           level per waypoint.  T time tokens are added to the KV.
            context:       (B, L, d_cond)    — scene KV (image + status tokens)

        Returns:
            predicted_noise / predicted_velocity: (B, T, A)
        """
        # --- Query: encode noisy actions ---
        act_emb = self.action_encoder(noisy_actions, timestep)   # (B, T, d_model)
        act_emb = act_emb + self.action_pos                       # add positional bias

        # --- Memory: time token(s) + projected context ---
        # Standard mode    (timestep: (B,))    → 1 global time token   → (B, 1,   d_model)
        # Diffusion Forcing (timestep: (B, T)) → T per-token time tokens → (B, T, d_model)
        # Both are handled identically downstream; cond_pos is pre-sized for the max.
        t_emb_raw = self.time_emb(timestep)                      # (B, d_model) or (B, T, d_model)
        t_emb = t_emb_raw.unsqueeze(1) if t_emb_raw.dim() == 2 else t_emb_raw  # (B, 1|T, d_model)
        ctx_emb = self.cond_proj(context)                        # (B, L, d_model)
        memory = torch.cat([t_emb, ctx_emb], dim=1)             # (B, L+1 or L+T, d_model)
        memory = memory + self.cond_pos[:, : memory.size(1), :]  # positional bias

        # --- Cross-attend ---
        out = self.transformer(tgt=act_emb, memory=memory)       # (B, T, d_model)
        return self.out_proj(self.out_norm(out))                  # (B, T, A)


# ---------------------------------------------------------------------------
# High-level diffusion head
# ---------------------------------------------------------------------------

class ActionDiffusionHead(nn.Module):
    """
    Full diffusion trajectory head — supports DDPM and flow matching.

    Mode is controlled by ``ActionDiffusionConfig.noise_type``:
      • ``'ddpm'`` — classic epsilon-prediction DDPM (default)
      • ``'flow'`` — flow matching with straight probability paths

    Combines:
      • DDPM noise schedule (DDPMScheduler from diffusers) — DDPM mode only
      • Denoising transformer (_DenoisingTransformer) — shared by both modes
      • Unicycle kinematic model for action ↔ trajectory conversion
        (UnicycleAccelCurvatureActionSpace)

    Training API
    ------------
    loss = head.compute_loss(context, gt_dense_traj, features)

    Inference API
    -------------
    predictions = head(context, features)
    # → { 'dp_pred': (B, N, 8, 3), 'pred_actions': (B, N, 40, 2) }
    """

    # Sparse output: (x, y, heading)
    _OUTPUT_DIM = 3

    def __init__(self, config: ActionDiffusionConfig, context_len: int) -> None:
        super().__init__()
        self.config = config

        # ── Unicycle physics model ────────────────────────────────────────────
        self.action_space = UnicycleAccelCurvatureActionSpace(
            dt=config.internal_dt,
            n_waypoints=config.internal_horizon,
            accel_mean=config.accel_mean,
            accel_std=config.accel_std,
            curvature_mean=config.curv_mean,
            curvature_std=config.curv_std,
        )

        # ── Noise schedule (DDPM only) ────────────────────────────────────────
        # Flow matching does not need a scheduler object — its training and
        # inference loops are implemented directly using simple arithmetic.
        if config.noise_type == "ddpm":
            self.noise_scheduler: Optional[DDPMScheduler] = DDPMScheduler(
                num_train_timesteps=config.num_diffusion_timesteps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                variance_type="fixed_small",
                clip_sample=True,
                clip_sample_range=5.0,
                prediction_type="epsilon",
            )
        else:
            self.noise_scheduler = None

        # ── Denoising transformer ─────────────────────────────────────────────
        self.denoiser = _DenoisingTransformer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_diffusion_layers,
            d_cond=config.model_dim,   # context is already projected to model_dim
            context_len=context_len,
            action_dim=config.action_dim,
            horizon=config.internal_horizon,
            noise_type=config.noise_type,
            use_diffusion_forcing=config.use_diffusion_forcing,
        )

    # ─────────────────────────────────────────────────────── helpers ──────────

    @staticmethod
    def _se2_to_xyz_rot(
        traj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a (..., T, 3) [x, y, heading] tensor into the
        (xyz, rotation_matrix) pair expected by the action space.

        Returns:
            traj_xyz: (..., T, 3)    — x, y, 0
            traj_rot: (..., T, 3, 3) — 2-D rotation embedded in 3-D
        """
        pos = traj[..., :2]
        heading = traj[..., 2]
        zeros = torch.zeros_like(pos[..., :1])
        traj_xyz = torch.cat([pos, zeros], dim=-1)

        c, s = torch.cos(heading), torch.sin(heading)
        z, o = torch.zeros_like(c), torch.ones_like(c)
        row0 = torch.stack([c, -s, z], dim=-1)
        row1 = torch.stack([s,  c, z], dim=-1)
        row2 = torch.stack([z,  z, o], dim=-1)
        traj_rot = torch.stack([row0, row1, row2], dim=-2)  # (..., T, 3, 3)

        return traj_xyz, traj_rot

    @staticmethod
    def _rot_to_heading(rot: torch.Tensor) -> torch.Tensor:
        """(..., T, 3, 3) → (..., T, 1) heading in radians."""
        return torch.atan2(rot[..., 1, 0], rot[..., 0, 0]).unsqueeze(-1)

    def _build_history(
        self, features: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Construct (hist_xyz, hist_rot) from the 'hist_status_feature' entry.

        `hist_status_feature` is a flat (B, N_hist * 7) tensor where each
        group of 7 values represents one historical frame:
            [vx, vy, ax, ay, px, py, heading]

        We append a current-frame placeholder (origin: all zeros) at the end,
        making the shape (B, N_hist + 1, 3) before converting to xyz/rot.
        """
        hist = features["hist_status_feature"]           # (B, N_hist * 7)
        B, device = hist.shape[0], hist.device
        past_se2 = hist.view(B, -1, 7)[..., -3:]         # (B, N_hist, 3): px, py, heading
        curr = torch.zeros(B, 1, 3, device=device)       # current state = origin
        full_hist = torch.cat([past_se2, curr], dim=1)   # (B, N_hist+1, 3)
        return self._se2_to_xyz_rot(full_hist)

    def _denormalize_actions(self, norm_actions: torch.Tensor) -> torch.Tensor:
        """Z-score de-normalise (accel, curvature) from the action tensor."""
        cfg = self.config
        accel = norm_actions[..., 0:1] * cfg.accel_std + cfg.accel_mean
        curv  = norm_actions[..., 1:2] * cfg.curv_std  + cfg.curv_mean
        return torch.cat([accel, curv], dim=-1)

    def _get_gt_actions(
        self,
        gt_traj_dense: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Convert a dense GT trajectory to normalised (accel, curvature) actions.

        Shared between the DDPM and flow matching training paths.

        Args:
            gt_traj_dense: (B, 40, 3) — dense 0.1 s GT [x, y, heading] ego-relative
            features:      raw feature dict from the feature builder

        Returns:
            gt_actions_norm: (B, 40, 2) — z-score normalised (accel, curvature)
        """
        v0 = features["status_feature"][:, 4]
        gt_xyz, gt_rot = self._se2_to_xyz_rot(gt_traj_dense.float())
        hist_xyz, hist_rot = self._build_history(features)
        return self.action_space.traj_to_action(
            traj_future_xyz=gt_xyz,
            traj_future_rot=gt_rot,
            t0_states={"v": v0},
            traj_history_xyz=hist_xyz,
            traj_history_rot=hist_rot,
        )  # (B, 40, 2)

    # ──────────────────────────────────────────────────────── training ────────

    def compute_loss(
        self,
        context: torch.Tensor,
        gt_traj_dense: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Dispatch to the appropriate training loss based on ``config.noise_type``.

        Args:
            context:       (B, L, D) — scene KV built by ActionDiffusionModel
            gt_traj_dense: (B, 40, 3) — dense 0.1 s GT [x, y, heading] ego-relative
            features:      raw feature dict from the feature builder

        Returns:
            scalar MSE loss
        """
        if self.config.noise_type == "flow":
            return self._flow_compute_loss(context, gt_traj_dense, features)
        return self._ddpm_compute_loss(context, gt_traj_dense, features)

    def _ddpm_compute_loss(
        self,
        context: torch.Tensor,
        gt_traj_dense: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        DDPM denoising loss (epsilon-prediction MSE).

        Forward process: x_t = sqrt(ᾱ_t)·x₀ + sqrt(1-ᾱ_t)·ε
        Objective:       predict ε from x_t
        """
        B, device = context.shape[0], context.device

        with torch.no_grad():
            gt_actions_norm = self._get_gt_actions(gt_traj_dense, features)

        # DDPM forward process: add noise at a random timestep
        noise = torch.randn_like(gt_actions_norm)
        T = gt_actions_norm.shape[1]
        if self.config.use_diffusion_forcing:
            # Diffusion Forcing: independent noise level per waypoint (B, T)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B, T),
                device=device,
            ).long()
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
            sqrt_alpha    = alphas_cumprod[timesteps].sqrt()[..., None]         # (B, T, 1)
            sqrt_1m_alpha = (1 - alphas_cumprod[timesteps]).sqrt()[..., None]   # (B, T, 1)
            noisy_actions = sqrt_alpha * gt_actions_norm + sqrt_1m_alpha * noise
        else:
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=device,
            ).long()
            noisy_actions = self.noise_scheduler.add_noise(gt_actions_norm, noise, timesteps)

        # Predict the noise
        pred_noise = self.denoiser(noisy_actions, timesteps, context)
        return F.mse_loss(pred_noise, noise)

    def _flow_compute_loss(
        self,
        context: torch.Tensor,
        gt_traj_dense: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Flow matching training loss (velocity-prediction MSE).

        Probability path: x_t = (1-t)·x₀ + t·ε,   t ~ Uniform[0, 1]
        Objective:        predict v = ε - x₀ from x_t and continuous t
        """
        B, device = context.shape[0], context.device

        with torch.no_grad():
            x0 = self._get_gt_actions(gt_traj_dense, features)   # (B, 40, 2)

        # Sample a continuous time value per training sample
        T = x0.shape[1]
        if self.config.use_diffusion_forcing:
            # Diffusion Forcing: independent t per waypoint (B, T) ∈ [0, 1]
            t = torch.rand(B, T, device=device)
            t_bcast = t[..., None]                              # (B, T, 1) for broadcasting
        else:
            t = torch.rand(B, device=device)                    # (B,) ∈ [0, 1]
            t_bcast = t[:, None, None]                          # (B, 1, 1) for broadcasting

        # Linear interpolation between clean actions and noise
        noise = torch.randn_like(x0)
        x_t = (1.0 - t_bcast) * x0 + t_bcast * noise           # (B, 40, 2)

        # Ground-truth velocity
        target_v = noise - x0                                   # (B, 40, 2)

        # Predict velocity (denoiser unchanged; t is (B,) or (B,T) float in [0,1])
        pred_v = self.denoiser(x_t, t, context)
        return F.mse_loss(pred_v, target_v)

    # ─────────────────────────────────────────────────────── inference ────────

    def forward(
        self,
        context: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Dispatch to the appropriate inference loop based on ``config.noise_type``.

        Returns:
            dp_pred:      (B, N, 8, 3)  — N sparse trajectory proposals [x,y,h]
            pred_actions: (B, N, 40, 2) — corresponding raw (de-normalised) actions
        """
        if self.config.noise_type == "flow":
            return self._flow_forward(context, features)
        return self._ddpm_forward(context, features)

    def _prepare_inference(
        self,
        context: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ):
        """
        Shared boilerplate for all inference paths.

        Returns:
            ctx       : (B*N, L, D)       — context expanded for N proposals
            hist_xyz  : (B*N, H+1, 3)
            hist_rot  : (B*N, H+1, 3, 3)
            t0_states : {'v': (B*N,)}
            B, N, device, dtype
        """
        cfg = self.config
        B, device, dtype = context.shape[0], context.device, context.dtype
        N = cfg.num_inference_proposals

        ctx = context.repeat_interleave(N, dim=0)                    # (B*N, L, D)
        hist_xyz, hist_rot = self._build_history(features)
        hist_xyz = hist_xyz.repeat_interleave(N, dim=0)
        hist_rot = hist_rot.repeat_interleave(N, dim=0)
        v0 = features["status_feature"][:, 4].repeat_interleave(N)  # (B*N,)
        t0_states = {"v": v0}

        return ctx, hist_xyz, hist_rot, t0_states, B, N, device, dtype

    def _actions_to_output(
        self,
        pred_actions_norm: torch.Tensor,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        t0_states: dict,
        B: int,
        N: int,
    ) -> Dict[str, torch.Tensor]:
        """Convert normalised action predictions to the standard output dict."""
        cfg = self.config
        pred_actions_denorm = self._denormalize_actions(pred_actions_norm)
        traj_xyz, traj_rot = self.action_space.action_to_traj(
            action=pred_actions_norm,
            t0_states=t0_states,
            traj_history_xyz=hist_xyz,
            traj_history_rot=hist_rot,
        )
        pos_xy      = traj_xyz[..., :2]
        heading     = self._rot_to_heading(traj_rot)
        traj_sparse = torch.cat([pos_xy, heading], dim=-1)[:, 4::5, :]  # (B*N, 8, 3)
        return {
            "dp_pred":      traj_sparse.view(B, N, 8, self._OUTPUT_DIM),
            "pred_actions": pred_actions_denorm.view(B, N, cfg.internal_horizon, cfg.action_dim),
        }

    @staticmethod
    def _build_pyramid_schedule(num_base_steps: int, num_tokens: int, offset: int) -> list:
        """
        Build the Diffusion Forcing scheduling matrix as a Python list-of-lists.

            level[m][i] = clip(num_base_steps + i*offset − m,  0,  num_base_steps)

        Two special cases:
          • offset == 0  →  flat "full-sequence" schedule: all tokens at the same
                            level per row.  This is standard (non-DF) sampling.
          • offset >= 1  →  pyramid schedule: token i lags token 0 by i*offset steps,
                            so near-future waypoints are denoised first.
        """
        total_steps = num_base_steps + (num_tokens - 1) * offset + 1
        return [
            [max(0, min(num_base_steps, num_base_steps + i * offset - m)) for i in range(num_tokens)]
            for m in range(total_steps)
        ]

    def _ddpm_forward(
        self,
        context: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        DDPM reverse process, unified for standard and Diffusion Forcing modes.

        Standard mode (``use_diffusion_forcing=False``):
            All waypoints share the same noise level at every step — a flat
            "full-sequence" scheduling matrix (offset=0).  This is equivalent
            to the standard DDPM reverse chain.

        Diffusion Forcing mode (``use_diffusion_forcing=True``):
            Near-future waypoints are denoised first; each subsequent waypoint
            lags by ``int(uncertainty_scale)`` DDPM timesteps — a pyramid
            scheduling matrix (offset >= 1).

        The per-token DDPM posterior is computed from ``alphas_cumprod`` buffers,
        which is equivalent to ``DDPMScheduler.step()`` when
        ``num_inference_steps == num_train_timesteps``.
        """
        cfg = self.config
        ctx, hist_xyz, hist_rot, t0_states, B, N, device, dtype = self._prepare_inference(context, features)

        T_way  = cfg.internal_horizon
        T_diff = self.noise_scheduler.config.num_train_timesteps

        # offset=0  →  flat schedule (standard);  offset>=1  →  pyramid (DF).
        offset = max(1, int(round(cfg.uncertainty_scale))) if cfg.use_diffusion_forcing else 0
        sched  = self._build_pyramid_schedule(T_diff, T_way, offset)

        # Precompute alpha-cumprod buffers for the per-token DDPM posterior step.
        acp      = self.noise_scheduler.alphas_cumprod.to(device)           # (T_diff,)
        acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]])      # ᾱ_{t-1}

        # Start from pure Gaussian noise.
        x = torch.randn(B * N, T_way, cfg.action_dim, dtype=dtype, device=device)

        for m in range(len(sched) - 1):
            from_lev = torch.tensor(sched[m],     dtype=torch.long, device=device)  # (T_way,)
            to_lev   = torch.tensor(sched[m + 1], dtype=torch.long, device=device)

            update_mask = from_lev > to_lev        # tokens whose level decreases this step
            if not update_mask.any():
                continue

            # DDPMScheduler timestep index = level − 1  (level 0 → token is clean).
            t_idx    = (from_lev - 1).clamp(min=0)                          # (T_way,)
            t_idx_bn = t_idx[None].expand(B * N, -1)                        # (B*N, T_way)

            # Standard mode: all tokens at the same level → pass scalar (B*N,) to
            # keep the denoiser on its 1-time-token path (cond_pos sized accordingly).
            t_in = t_idx_bn[:, 0] if not cfg.use_diffusion_forcing else t_idx_bn

            pred_eps = self.denoiser(x, t_in, ctx)                          # (B*N, T_way, A)

            # Per-token DDPM posterior  q(x_{t-1} | x_t, x̂_0).
            a_t    = acp[t_idx_bn]       # (B*N, T_way)  ᾱ_t
            a_prev = acp_prev[t_idx_bn]  # (B*N, T_way)  ᾱ_{t-1}
            beta_t = 1.0 - a_t / a_prev

            x0 = (x - (1.0 - a_t).sqrt()[..., None] * pred_eps) / a_t.sqrt()[..., None]
            x0 = x0.clamp(-5.0, 5.0)    # matches scheduler clip_sample_range

            post_mean = (
                a_prev.sqrt()[..., None] * beta_t[..., None] / (1.0 - a_t)[..., None] * x0
                + (a_t / a_prev).sqrt()[..., None] * (1.0 - a_prev)[..., None] / (1.0 - a_t)[..., None] * x
            )
            post_var = beta_t * (1.0 - a_prev) / (1.0 - a_t)
            x_new = post_mean + post_var.clamp(min=1e-20).sqrt()[..., None] * torch.randn_like(x)

            mask = update_mask[None, :, None].expand(B * N, -1, cfg.action_dim)
            x    = torch.where(mask, x_new, x)

        return self._actions_to_output(x, hist_xyz, hist_rot, t0_states, B, N)

    def _flow_forward(
        self,
        context: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Flow-matching inference via Euler integration, unified for standard and
        Diffusion Forcing modes.

        Both modes use the same pyramid scheduling matrix as the DDPM path
        (via ``_build_pyramid_schedule``), with integer levels divided by
        ``n_steps`` to map them into the continuous t ∈ [0, 1] range.

        Standard mode (``use_diffusion_forcing=False``, offset=0):
            Flat schedule — all tokens advance together from t=1 to t=0 in
            ``n_steps`` uniform Euler steps.  Equivalent to the original loop.
            The denoiser receives a scalar (B*N,) timestep.

            Example with n_steps=5, 4 tokens:
              m=0: t=[1.0, 1.0, 1.0, 1.0]
              m=1: t=[0.8, 0.8, 0.8, 0.8]
              ...
              m=5: t=[0.0, 0.0, 0.0, 0.0]

        Diffusion Forcing mode (``use_diffusion_forcing=True``, offset>=1):
            Pyramid schedule — token i lags token 0 by i*offset Euler steps.
            Near-future waypoints are fully denoised before far-future ones
            begin integrating.  t stays in [0, 1] at every step.
            The denoiser receives a per-token (B*N, T_way) timestep.

            Example with n_steps=5, 4 tokens, offset=1:
              m=0: t=[1.0, 1.0, 1.0, 1.0]
              m=1: t=[0.8, 1.0, 1.0, 1.0]
              m=2: t=[0.6, 0.8, 1.0, 1.0]
              m=3: t=[0.4, 0.6, 0.8, 1.0]
              m=4: t=[0.2, 0.4, 0.6, 0.8]
              m=5: t=[0.0, 0.2, 0.4, 0.6]
              m=6: t=[0.0, 0.0, 0.2, 0.4]
              m=7: t=[0.0, 0.0, 0.0, 0.2]
              m=8: t=[0.0, 0.0, 0.0, 0.0]
        """
        cfg = self.config
        ctx, hist_xyz, hist_rot, t0_states, B, N, device, dtype = self._prepare_inference(context, features)

        T_way   = cfg.internal_horizon
        n_steps = cfg.num_flow_steps

        # Build the scheduling matrix in integer "step-level" space [0, n_steps],
        # then convert to continuous t ∈ [0, 1] by dividing by n_steps.
        # offset=0 → flat (standard); offset>=1 → pyramid (DF).
        offset = max(1, int(round(cfg.uncertainty_scale))) if cfg.use_diffusion_forcing else 0
        sched  = self._build_pyramid_schedule(n_steps, T_way, offset)

        # Start from pure Gaussian noise.
        x = torch.randn(B * N, T_way, cfg.action_dim, dtype=dtype, device=device)

        for m in range(len(sched) - 1):
            from_lev = torch.tensor(sched[m],     dtype=dtype, device=device)  # (T_way,)
            to_lev   = torch.tensor(sched[m + 1], dtype=dtype, device=device)

            t_curr = from_lev / n_steps                           # (T_way,) ∈ [0, 1]
            t_next = to_lev   / n_steps

            update_mask = t_curr > t_next                         # tokens active this step
            if not update_mask.any():
                continue

            # Standard mode: all active tokens share the same t → scalar (B*N,)
            # DF mode: per-token t → (B*N, T_way)
            t_in = t_curr[0].expand(B * N) if not cfg.use_diffusion_forcing else t_curr[None].expand(B * N, -1)

            v    = self.denoiser(x, t_in, ctx)                    # (B*N, T_way, A)

            # Per-token Euler step: dt_i = t_curr[i] - t_next[i] (= 1/n_steps for
            # active tokens, 0 for inactive ones — so no explicit masking needed,
            # but we keep update_mask.any() above to skip the denoiser call).
            step = (t_curr - t_next)[None, :, None]               # (1, T_way, 1)
            x    = x - step * v

        return self._actions_to_output(x, hist_xyz, hist_rot, t0_states, B, N)
