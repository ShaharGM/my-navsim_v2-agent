import math
from typing import Dict, Optional, Any, Protocol

from navsim.agents.full_gtrs_dense.action_dp_agent.action_in_proj import PerWaypointActionInProjV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

from navsim.agents.full_gtrs_dense.action_dp_agent.dp_config import DPConfig
from navsim.agents.gtrs_dense.hydra_backbone_bev import HydraBackboneBEV
from navsim.agents.utils.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace

from dataclasses import dataclass

# --- CONSTANTS ---
INTERNAL_DT = 0.1
INTERNAL_HORIZON = 40
ACTION_DIM = 2 
OUTPUT_TRAJ_DIM = 3

# ACCEL_MEAN, ACCEL_STD = 0.0, 2.0
# CURV_MEAN, CURV_STD = 0.0, 0.1

# Calculated values from the 'mini' split
ACCEL_MEAN, ACCEL_STD = 0.05671, 1.12148
CURV_MEAN, CURV_STD   = 0.00180, 0.02297

# Normalization stats taken from alpamayo r1 repo 
# ACCEL_MEAN, ACCEL_STD = 0.02902694707164455, 0.6810426736454882
# CURV_MEAN, CURV_STD   = 0.0002692167976330542, 0.026148280660833106

class ScorerAgent(Protocol):
    """
    Interface for any agent that can score trajectory proposals.
    Using a Protocol prevents circular imports and provides IDE type hinting.
    """
    @property
    def model(self) -> Any: ...

    def evaluate_dp_proposals(
        self, 
        features: Dict[str, torch.Tensor], 
        dp_proposals: torch.Tensor, 
        dp_only_inference: bool, 
        topk: int
    ) -> Dict[str, torch.Tensor]:
        ...

@dataclass
class GuidanceConfig:
    agent: ScorerAgent  # The GTRS Agent
    scale: float = 0.0  # Strength (e.g. 1.0 to 10.0)
    top_k: int = 0  # 0 = Use all proposals

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionTransformer(nn.Module):
    """
    Diffusion transformer with integrated per-waypoint action encoding.
    
    Architecture:
    1. Fourier encode each action dimension + timestep separately
    2. MLP to combine features into transformer space
    3. Add learnable positional embeddings for waypoint sequence
    4. Cross-attend to BEV features via TransformerDecoder
    5. Project back to action space
    """
    
    def __init__(self, d_model: int, nhead: int, d_ffn: int, d_cond: int, 
                 num_layers: int, obs_len: int):
        super().__init__()
        
        # === Fourier Encoding Components ===
        self.num_fourier_feats = 20
        self.max_freq = 100.0
        
        # Separate Fourier encoder for each action dimension (accel, curvature)
        self.action_fourier_encoders = nn.ModuleList([
            self._make_fourier_encoder() for _ in range(ACTION_DIM)
        ])
        
        # Timestep encoder
        self.timestep_fourier_encoder = self._make_fourier_encoder()
        
        # === MLP to combine Fourier features ===
        fourier_dim = ACTION_DIM * self.num_fourier_feats + self.num_fourier_feats
        self.action_input_mlp = nn.Sequential(
            nn.Linear(fourier_dim, 1024),
            nn.SiLU(),
            self._rms_norm(1024),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            self._rms_norm(1024),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            self._rms_norm(1024),
            nn.Linear(1024, d_model),
            nn.LayerNorm(d_model)
        )
        
        # === Positional Embeddings ===
        # For action sequence (40 waypoints)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, INTERNAL_HORIZON, d_model))
        
        # For condition sequence (BEV features + ego status + time)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, obs_len + 1, d_model))
        
        # === Condition Processing ===
        self.cond_proj = nn.Linear(d_cond, d_model)
        self.time_emb = SinusoidalPosEmb(d_model)
        
        # === Transformer Cross-Attention ===
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_ffn,
                dropout=0.0,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers
        )
        
        # === Output Projection ===
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, ACTION_DIM)
        
        self._init_weights()
    
    def _make_fourier_encoder(self):
        """Create a Fourier feature encoder with log-spaced frequencies."""
        half_dim = self.num_fourier_feats // 2
        freqs = torch.logspace(0, math.log10(self.max_freq), steps=half_dim)
        
        class FourierEncoder(nn.Module):
            def __init__(self, freqs):
                super().__init__()
                self.register_buffer('freqs', freqs[None, :])
            
            def forward(self, x):
                # x: (...,) -> (..., num_fourier_feats)
                arg = x[..., None] * self.freqs * 2 * math.pi
                return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)
        
        return FourierEncoder(freqs)
    
    def _rms_norm(self, dim: int):
        """RMS normalization layer."""
        class RMSNorm(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(dim))
                self.eps = 1e-5
            
            def forward(self, x):
                norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                return x * norm * self.scale
        
        return RMSNorm(dim)
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Small init for positional embeddings
        nn.init.normal_(self.action_pos_emb, std=0.02)
        nn.init.normal_(self.cond_pos_emb, std=0.02)
    
    def encode_actions(self, actions: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Encode noisy actions into transformer input space.
        
        Args:
            actions: (B, T, ACTION_DIM) - noisy action sequence
            timestep: (B,) or scalar - diffusion timestep
        
        Returns:
            (B, T, d_model) - encoded action features
        """
        B, T, _ = actions.shape
        
        # Encode each action dimension with Fourier features
        action_feats = []
        for i, encoder in enumerate(self.action_fourier_encoders):
            feat = encoder(actions[:, :, i])  # (B, T, num_fourier_feats)
            action_feats.append(feat)
        action_feats = torch.cat(action_feats, dim=-1)  # (B, T, ACTION_DIM * num_fourier_feats)
        
        # Encode timestep and broadcast to all waypoints
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=actions.device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.expand(B)
        
        timestep_feat = self.timestep_fourier_encoder(timestep.float())  # (B, num_fourier_feats)
        timestep_feat = timestep_feat[:, None, :].expand(-1, T, -1)  # (B, T, num_fourier_feats)
        
        # Combine action and timestep features
        combined = torch.cat([action_feats, timestep_feat], dim=-1)  # (B, T, fourier_dim)
        
        # Project to transformer space
        encoded = self.action_input_mlp(combined)  # (B, T, d_model)
        
        return encoded
    
    def forward(self, noisy_actions: torch.Tensor, timestep: torch.Tensor, 
                condition: torch.Tensor) -> torch.Tensor:
        """
        Predict noise in the action sequence.
        
        Args:
            noisy_actions: (B, T, ACTION_DIM) - noisy action sequence
            timestep: (B,) or scalar - diffusion timestep
            condition: (B, L, d_cond) - BEV features + ego status
        
        Returns:
            (B, T, ACTION_DIM) - predicted noise
        """
        B, T, _ = noisy_actions.shape
        
        # === Encode Actions ===
        action_emb = self.encode_actions(noisy_actions, timestep)  # (B, T, d_model)
        action_emb = action_emb + self.action_pos_emb  # Add positional encoding
        
        # === Prepare Condition (Memory) ===
        # Time embedding
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=noisy_actions.device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.expand(B)
        time_emb = self.time_emb(timestep).unsqueeze(1)  # (B, 1, d_model)
        
        # Project condition
        cond_emb = self.cond_proj(condition)  # (B, L, d_model)
        
        # Concatenate time + condition
        memory = torch.cat([time_emb, cond_emb], dim=1)  # (B, L+1, d_model)
        memory = memory + self.cond_pos_emb[:, :memory.size(1), :]
        
        # === Transformer Cross-Attention ===
        output = self.transformer(tgt=action_emb, memory=memory)  # (B, T, d_model)
        
        # === Project to Action Space ===
        output = self.output_norm(output)
        noise_pred = self.output_proj(output)  # (B, T, ACTION_DIM)
        
        return noise_pred

class SimpleDiffusionTransformer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, d_cond, dp_nlayers, input_dim, obs_len):
        super().__init__()
        self.dp_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), dp_nlayers
        )
        self.input_emb = nn.Linear(input_dim, d_model)
        self.time_emb = SinusoidalPosEmb(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.output_emb = nn.Linear(d_model, input_dim)
        self.cond_emb = nn.Linear(d_cond, d_model)
        
        token_len = obs_len + 1
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, token_len, d_model))
        # self.pos_emb = nn.Parameter(torch.zeros(1, 1, d_model)) # for single token input
        self.pos_emb = nn.Parameter(torch.zeros(1, INTERNAL_HORIZON, d_model)) # for per-waypoint token input
        self.apply(self._init_weights)
        
        # Initiate action projection layers for DP Head
        # Project per-waypoint actions (unflattened) into the transformer space
        self.action_in_proj = PerWaypointActionInProjV2(
            in_dims=[INTERNAL_HORIZON, ACTION_DIM],
            out_dim=d_model,
        )
        self.action_out_proj = torch.nn.Linear(
            d_model,
            ACTION_DIM,
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, SinusoidalPosEmb, nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer, nn.TransformerEncoder, 
                        nn.TransformerDecoder, nn.ModuleList, nn.Mish, nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    # def forward(self, sample, timestep, cond):
    #     B, T, DIM = sample.shape
    #     sample = sample.view(B, -1).float()
    #     input_emb = self.input_emb(sample)

    #     timesteps = timestep
    #     if not torch.is_tensor(timesteps):
    #         timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
    #     timesteps = timesteps.expand(B)
    #     time_emb = self.time_emb(timesteps).unsqueeze(1)
        
    #     cond_embeddings = torch.cat([time_emb, cond], dim=1)
    #     tc = cond_embeddings.shape[1]
    #     x = cond_embeddings + self.cond_pos_emb[:, :tc, :]
    #     memory = x

    #     token_embeddings = input_emb.unsqueeze(1)
    #     t_len = token_embeddings.shape[1]
    #     x = token_embeddings + self.pos_emb[:, :t_len, :]
        
    #     x = self.dp_transformer(tgt=x, memory=memory)
    #     x = self.ln_f(x)
    #     x = self.output_emb(x)
    #     return x.squeeze(1).view(B, T, DIM)

    def forward(self, sample, timestep, cond):
        B, T, DIM = sample.shape
        sample = sample.float()  
        
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)

        timesteps = timesteps.expand(B)
        timesteps_emb_input = timesteps.view(B, *[1] * ACTION_DIM).expand(B, *[-1] * ACTION_DIM)
        input_emb = self.action_in_proj(sample, timesteps_emb_input)

        time_emb = self.time_emb(timesteps).unsqueeze(1)
        cond_emb = self.cond_emb(cond)
        
        cond_embeddings = torch.cat([time_emb, cond_emb], dim=1)
        tc = cond_embeddings.shape[1]
        memory = cond_embeddings + self.cond_pos_emb[:, :tc, :]

        # token_embeddings = input_emb.unsqueeze(1)
        # t_len = token_embeddings.shape[1]
        # x = token_embeddings + self.pos_emb[:, :t_len, :]
        
        t_len = input_emb.shape[1]
        x = input_emb + self.pos_emb[:, :t_len, :]

        x = self.dp_transformer(tgt=x, memory=memory)
        x = self.ln_f(x)
        # x = self.output_emb(x)
        # last_hidden = x[:, -1, :]
        x = self.action_out_proj(x)
        return x.view(B, T, DIM)

class DPHead(nn.Module):
    def __init__(self, d_ffn: int, d_model: int, nhead: int, nlayers: int, config: DPConfig = None):
        super().__init__()
        self.config = config
        
        self.action_space = UnicycleAccelCurvatureActionSpace(
            dt=INTERNAL_DT,
            n_waypoints=INTERNAL_HORIZON, 
            accel_std=ACCEL_STD, 
            curvature_std=CURV_STD
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.denoising_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            variance_type='fixed_small',
            clip_sample=True,
            clip_sample_range=5.0, # NOTE Changed due to different normalization technique (Z-Score vs (-1,1))
            prediction_type='epsilon'
        )
        img_num = 2 if config.use_back_view else 1

        self.transformer_dp = SimpleDiffusionTransformer(
            d_model, nhead, d_ffn, config.tf_d_model, config.dp_layers,
            input_dim=ACTION_DIM * INTERNAL_HORIZON, 
            obs_len=config.img_vert_anchors * config.img_horz_anchors * img_num + 1,
        )
        self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps

    # def _normalize_actions(self, actions):
    #     accel = (actions[..., 0:1] - ACCEL_MEAN) / ACCEL_STD
    #     curv = (actions[..., 1:2] - CURV_MEAN) / CURV_STD
    #     return torch.cat([accel, curv], dim=-1)

    def _denormalize_actions(self, norm_actions):
        accel = norm_actions[..., 0:1] * ACCEL_STD + ACCEL_MEAN
        curv = norm_actions[..., 1:2] * CURV_STD + CURV_MEAN
        return torch.cat([accel, curv], dim=-1)

    def _compute_guidance_gradient(
        self, 
        latents: torch.Tensor, 
        model_output: torch.Tensor, 
        t: torch.Tensor, 
        guidance: GuidanceConfig, 
        context: Dict
    ) -> torch.Tensor:
        """
        Calculates the gradient of the GTRS score w.r.t the noisy latents.
        """
        latents = latents.detach().requires_grad_(True)
        model_output = model_output.detach()

        with torch.enable_grad():
            step_out = self.noise_scheduler.step(model_output, t, latents)
            pred_clean_action = step_out.pred_original_sample 

            # 4. Differentiable Physics: Action -> Trajectory (XYZ, Rotation Matrix)
            traj_xyz, traj_rot = self.action_space.action_to_traj(
                action=pred_clean_action,
                t0_states=context['t0_states'],
                traj_history_xyz=context['hist_xyz'],
                traj_history_rot=context['hist_rot']
            )
            
            # A. Extract (X, Y)
            pos_xy = traj_xyz[..., :2] # Shape: (B*N, T, 2)

            # B. Extract Heading (Theta) from Rotation Matrix
            heading = self._rot_to_heading(traj_rot) # Shape: (B*N, T, 1)

            # C. Concatenate to form (X, Y, Heading)
            traj_se2 = torch.cat([pos_xy, heading], dim=-1) # Shape: (B*N, T, 3)

            # 5. Reshape for GTRS: (B*N) -> (B, N)
            B, N = context['batch_size'], context['num_proposals']
            
            # Now this tensor is truly (x, y, heading)
            traj_for_scorer = traj_se2.view(B, N, INTERNAL_HORIZON, 3)

            scorer_head = guidance.agent.model._trajectory_head

            scorer_out = scorer_head.eval_dp_proposals(
                bev_feature=context['gtrs_context']['keyval'],
                status_encoding=context['gtrs_context']['status_encoding'],
                dp_proposals=traj_for_scorer,
                topk=N if guidance.top_k == 0 else guidance.top_k,
                dp_only_inference=True
            )
            
            scores = scorer_out['overall_log_scores'].view(-1)
            grads = torch.autograd.grad(scores.sum(), latents)[0]
            
        return grads
    
    def _format_xyz_rot(self, traj_tensor):
        """Helpers to format (B, T, 3) into XYZ and Rotation Matrix tensors"""
        pos = traj_tensor[..., :2]
        
        heading = traj_tensor[..., 2] 
        
        zeros = torch.zeros_like(pos[..., :1])
        traj_xyz = torch.cat([pos, zeros], dim=-1)
        
        c, s = torch.cos(heading), torch.sin(heading)
        z, o = torch.zeros_like(c), torch.ones_like(c)
        
        row1 = torch.stack([c, -s, z], dim=-1)
        row2 = torch.stack([s, c, z], dim=-1)
        row3 = torch.stack([z, z, o], dim=-1)
        
        traj_rot = torch.stack([row1, row2, row3], dim=-2)
        
        return traj_xyz, traj_rot

    def _rot_to_heading(self, rot_matrix):
        # rot_matrix is (..., 3, 3)
        # sin is at [..., 1, 0], cos is at [..., 0, 0]
        sin_val = rot_matrix[..., 1, 0]
        cos_val = rot_matrix[..., 0, 0]
        return torch.atan2(sin_val, cos_val).unsqueeze(-1) # (..., 1)

    def _get_history(self, features):
        """
        Extracts history and automatically pads it with the current ego-pose (0,0,0).
        """
        hist_feat = features['hist_status_feature']
        B = hist_feat.shape[0]
        device = hist_feat.device

        # Reshape to (B, T, 7) and take last 3 dims (x, y, heading)
        past_state_local = hist_feat.view(B, -1, 7)[..., -3:] 

        # 2. Create Current State (Origin)
        # Shape: (B, 1, 3) -> [x=0, y=0, heading=0]
        curr_state_local = torch.zeros((B, 1, 3), device=device)
        
        # 3. Concatenate [Past, Current]
        full_history_local = torch.cat([past_state_local, curr_state_local], dim=1)

        # 4. Format everything at once using your helper
        # This handles turning the (0,0,0) state into the correct Identity Matrix
        return self._format_xyz_rot(full_history_local)

    def forward(self, kv, features: Dict[str, torch.Tensor], guidance: Optional[GuidanceConfig] = None) -> Dict[str, torch.Tensor]:
        B = kv.shape[0]
        result = {}
        
        if not self.training:
            # gtrs_context = None
            # if guidance is not None and guidance.scale > 0:
            #     with torch.no_grad():
            #         gtrs_model = guidance.agent.model
                    
            #         cam_curr = features["camera_feature"][:, -1]
            #         img_features = gtrs_model.img_feat_blc(cam_curr)
                    
            #         if gtrs_model._config.use_back_view:
            #             img_features_back = gtrs_model.img_feat_blc(features["camera_feature_back"])
            #             img_features = torch.cat([img_features, img_features_back], 1)
                    
            #         gtrs_keyval = img_features + gtrs_model._keyval_embedding.weight[None, ...]
                    
            #         status_feat = features["status_feature"]
            #         if gtrs_model._config.num_ego_status == 1 and status_feat.shape[1] == 32:
            #             gtrs_status_enc = gtrs_model._status_encoding(status_feat[:, :8])
            #         else:
            #             gtrs_status_enc = gtrs_model._status_encoding(status_feat)
                        
            #         # C. Pack for the loop
            #         gtrs_context = {
            #             'keyval': gtrs_keyval,
            #             'status_encoding': gtrs_status_enc
            #         }
            
            NUM_PROPOSALS = self.config.num_proposals
            condition = kv.repeat_interleave(NUM_PROPOSALS, dim=0)
            
            hist_xyz, hist_rot = self._get_history(features)
            hist_xyz_in = hist_xyz.repeat_interleave(NUM_PROPOSALS, dim=0)
            hist_rot_in = hist_rot.repeat_interleave(NUM_PROPOSALS, dim=0)
            v0 = features["status_feature"][:, 4]
            v0_expanded = v0.repeat_interleave(NUM_PROPOSALS)
            t0_states = {"v": v0_expanded}

            # # for the guidance
            # physics_context = {
            #     'batch_size': B,
            #     'num_proposals': NUM_PROPOSALS,
            #     't0_states': t0_states,
            #     # 't0_states': None,
            #     'hist_xyz': hist_xyz_in,
            #     'hist_rot': hist_rot_in,
            #     'gtrs_context': gtrs_context
            # }

            noise = torch.randn(
                size=(B * NUM_PROPOSALS, INTERNAL_HORIZON, ACTION_DIM),
                dtype=condition.dtype,
                device=condition.device,
            )

            self.noise_scheduler.set_timesteps(self.num_inference_steps, device=condition.device)

            for t in self.noise_scheduler.timesteps:
                model_output = self.transformer_dp(noise, t, condition)
                # if guidance is not None and guidance.scale > 0:
                #     guidance_grad = self._compute_guidance_gradient(
                #         noise, model_output, t, guidance, physics_context
                #     )
                #     # Apply Guidance: Shift predicted noise away from low-score regions
                #     # (Formula depends on prediction_type, assuming 'epsilon')
                #     model_output = model_output - (guidance.scale * guidance_grad)

                noise = self.noise_scheduler.step(model_output, t, noise).prev_sample
            
            pred_actions_normalized = noise
            pred_actions_dense = self._denormalize_actions(noise)
            

            traj_xyz, traj_rot = self.action_space.action_to_traj(
                action=pred_actions_normalized, 
                t0_states=t0_states,
                traj_history_xyz=hist_xyz_in,
                traj_history_rot=hist_rot_in
            )
            
            # print(f"action -> traj ouptut:\n{traj_xyz}")
            pos_xy = traj_xyz[..., :2]
            heading = self._rot_to_heading(traj_rot)
            final_traj_dense = torch.cat([pos_xy, heading], dim=-1)

            # 4. Downsample (40 -> 8 steps)
            final_traj_sparse = final_traj_dense[:, 4::5, :]
            # print(f"final returned trajectories:\n{final_traj_sparse}`")

            result['dp_pred'] = final_traj_sparse.view(B, NUM_PROPOSALS, 8, OUTPUT_TRAJ_DIM)
            result['pred_actions'] = pred_actions_dense.view(B, NUM_PROPOSALS, INTERNAL_HORIZON, ACTION_DIM)

        return result

    def get_dp_loss(self, kv, gt_trajectory_dense, features):
        B = kv.shape[0]
        device = kv.device
        gt_traj = gt_trajectory_dense.float()
        
        # Index 4 is vx (longitudinal velocity)
        v0 = features["status_feature"][:, 4]

        # print(f"using v0 = {v0}")
        # print(f"using historical statuse: {features['hist_status_feature']}")

        with torch.no_grad():
            gt_xyz, gt_rot = self._format_xyz_rot(gt_traj)
            
            # print(f"Future used for dynamic model: \n{gt_xyz}\n\n{gt_rot}")
            # Extract History (Now simple slicing)
            hist_xyz, hist_rot = self._get_history(features)
            
            t0_states = {"v": v0}
            
            # Inverse Dynamics to get GT Actions
            gt_actions_norm = self.action_space.traj_to_action(
                traj_future_xyz=gt_xyz,
                traj_future_rot=gt_rot,
                t0_states=t0_states,
                traj_history_xyz=hist_xyz, 
                traj_history_rot=hist_rot
            )

        # print(f"normalized gt traj actions: {gt_actions_norm}")
        # gt_actions_norm = torch.clamp(gt_actions_norm, min=-2.0, max=2.0)
        # Standard Diffusion Training
        noise = torch.randn(gt_actions_norm.shape, device=device, dtype=torch.float)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        # # make constant timesteps for stable training
        # timesteps = torch.full((B,), (self.noise_scheduler.config.num_train_timesteps - 1) / 2, device=device, dtype=torch.long)

        noisy_actions = self.noise_scheduler.add_noise(
            gt_actions_norm, noise, timesteps
        )
        # print(f"timestep: {timesteps}")
        # print(f"noised magnitude: {gt_actions_norm - noisy_actions}")

        pred_noise = self.transformer_dp(
            noisy_actions,
            timesteps,
            kv
        )
        return F.mse_loss(pred_noise, noise)


class DPModel(nn.Module):
    def __init__(self, config: DPConfig):
        super().__init__()
        self._config = config
        self._backbone = HydraBackboneBEV(config)

        kv_len = self._backbone.bev_w * self._backbone.bev_h
        emb_len = kv_len + 1 + (1 if self._config.use_hist_ego_status else 0)
        self._keyval_embedding = nn.Embedding(emb_len, config.tf_d_model)

        self.downscale_layer = nn.Linear(self._backbone.img_feat_c, config.tf_d_model)
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(config.bev_features_channels, config.bev_features_channels, (3, 3), 1, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.bev_features_channels, config.num_bev_classes, (1, 1), 1, 0),
            nn.Upsample(size=(config.lidar_resolution_height // 2, config.lidar_resolution_width), mode="bilinear", align_corners=False),
        )

        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)
        if self._config.use_hist_ego_status:
            self._hist_status_encoding = nn.Linear((2 + 2 + 3) * 3, config.tf_d_model)

        self._trajectory_head = DPHead(
            d_ffn=config.tf_d_ffn,
            d_model=config.dp_tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            config=config
        )
        if self._config.use_temporal_bev_kv:
            self.temporal_bev_fusion = nn.Conv2d(
                config.tf_d_model * 2,
                config.tf_d_model,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            )

        backbone_ckpt_path="/sci/labs/sagieb/dawndude/projects/navsim_workspace/navsim/weights/gtrs_dp.ckpt"
        self.load_backbone_from_ckpt(backbone_ckpt_path)

    def load_backbone_from_ckpt(self, ckpt_path):
        print(f"Loading backbone from {ckpt_path}...")
        
        # 1. Load File
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        raw_state = checkpoint.get('state_dict', checkpoint)

        # 2. Filter Keys (Strip prefix + Remove head)
        clean_state = {}
        for k, v in raw_state.items():
            # Strip "agent.model." or "model." if present
            name = k.replace("agent.model.", "").replace("model.", "")
            
            # Skip the trajectory head
            if "_trajectory_head" in name:
                continue
                
            clean_state[name] = v

        # 3. Load into self (Strict=False allows missing head weights)
        self.load_state_dict(clean_state, strict=False)
        print("Backbone weights loaded.")

        # 4. Freeze EVERYTHING
        for param in self.parameters():
            param.requires_grad = False

        # 5. Unfreeze ONLY Trajectory Head
        for param in self._trajectory_head.parameters():
            param.requires_grad = True
        print("Backbone frozen. Trajectory Head unfrozen.")
    

    def train(self, mode=True):
        """
        Safety Override: Ensures backbone stays in eval mode (frozen statistics)
        even when the main loop calls model.train()
        """
        super().train(mode)
        
        if mode:
            # Force frozen parts back to eval
            self._backbone.eval()
            self._keyval_embedding.eval()
            self.downscale_layer.eval()
            self._bev_semantic_head.eval()
            self._status_encoding.eval()
            if self._config.use_hist_ego_status:
                self._hist_status_encoding.eval()
            if self._config.use_temporal_bev_kv:
                self.temporal_bev_fusion.eval()

            # Ensure Head is Trainable
            self._trajectory_head.train()
    
    def forward(self, features: Dict[str, torch.Tensor], guidance: Optional[GuidanceConfig] = None) -> Dict[str, torch.Tensor]:
        camera_feature = features["camera_feature"]
        camera_feature_back = features["camera_feature_back"]
        status_feature = features["status_feature"]
        
        batch_size = status_feature.shape[0]
        camera_feature_curr = camera_feature[:, -1]
        camera_feature_back_curr = camera_feature_back[:, -1]

        img_tokens, bev_tokens, up_bev = self._backbone(camera_feature_curr, camera_feature_back_curr)
        keyval = self.downscale_layer(bev_tokens)
        
        if self._config.use_temporal_bev_kv:
            with torch.no_grad():
                camera_feature_prev = camera_feature[-2]
                camera_feature_back_prev = camera_feature_back[-2]
                _, bev_tokens_prev, _ = self._backbone(camera_feature_prev, camera_feature_back_prev)
                keyval_prev = self.downscale_layer(bev_tokens_prev)
            
            C = keyval.shape[-1]
            keyval = self.temporal_bev_fusion(
                torch.cat([
                    keyval.permute(0, 2, 1).view(batch_size, C, self._backbone.bev_h, self._backbone.bev_w),
                    keyval_prev.permute(0, 2, 1).view(batch_size, C, self._backbone.bev_h, self._backbone.bev_w)
                ], 1)
            ).view(batch_size, C, -1).permute(0, 2, 1).contiguous()

        bev_semantic_map = self._bev_semantic_head(up_bev)
        
        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([keyval, status_encoding[:, None]], dim=1)
        if self._config.use_hist_ego_status:
            hist_encoding = self._hist_status_encoding(features['hist_status_feature'])
            keyval = torch.concatenate([keyval, hist_encoding[:, None]], dim=1)

        keyval += self._keyval_embedding.weight[None, ...]

        output: Dict[str, torch.Tensor] = {}
        head_out = self._trajectory_head(keyval, features, guidance)
        
        selected_traj = torch.zeros(batch_size, 8, 3, device=keyval.device)
        if not self.training:
            num_samples = head_out['dp_pred'].shape[1] 
            random_idx = torch.randint(0, num_samples, (1,)).item()
            selected_traj = head_out['dp_pred'][:, random_idx]

        output.update(head_out)
        output['trajectory'] = selected_traj
        output['env_kv'] = keyval
        output['bev_semantic_map'] = bev_semantic_map

        return output