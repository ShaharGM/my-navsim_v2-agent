import math
import torch
import torch.nn as nn
from navsim.common.enums import StateSE2Index
from .dit import LightningDiT


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear beta schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine beta schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiTBlock(nn.Module):
    """Diffusion Transformer Block from DiT paper."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x


class DiffusionTrajectoryHead(nn.Module):
    """Diffusion-based trajectory prediction head."""

    def __init__(self, num_poses: int, d_model: int, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, beta_schedule: str = "linear"):
        """
        Initializes diffusion trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_model: input dimensionality (from trajectory query)
        :param timesteps: number of diffusion timesteps
        :param beta_start: starting beta for noise schedule
        :param beta_end: ending beta for noise schedule
        :param beta_schedule: type of beta schedule ("linear" or "cosine")
        """
        super(DiffusionTrajectoryHead, self).__init__()

        self.num_poses = num_poses
        self.pose_dim = StateSE2Index.size()  # x, y, heading
        self.d_model = d_model
        self.timesteps = timesteps

        # Create beta schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Noise schedule
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', torch.cat([torch.ones(1), self.alphas_cumprod[:-1]]))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / self.alphas_cumprod - 1))
        
        self.dit_model = LightningDiT(**config.diffusion_model_cfg)
        # DiT (Diffusion Transformer) architecture
        self.pose_embed = nn.Linear(self.pose_dim, d_model)  # Embed each pose to d_model
        self.time_embed = SinusoidalPositionEmbeddings(d_model)
        self.cond_embed = nn.Linear(d_model, d_model)
        
        # Position embeddings for trajectory sequence
        self.pos_embed = nn.Embedding(num_poses, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(d_model, num_heads=8, mlp_ratio=4.0)
            for _ in range(6)  # 6 transformer blocks
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, self.pose_dim)

    def normalize_poses(self, poses):
        """Normalize poses to [-1, 1] range using fixed statistics."""
        # NOTE Normalization values taken from recogdrive repo.
        x_norm = 2 * (poses[..., 0:1] + 1.57) / 66.74 - 1
        y_norm = 2 * (poses[..., 1:2] + 19.68) / 42 - 1
        heading_norm = 2 * (poses[..., 2:3] + 1.67) / 3.53 - 1
        return torch.cat([x_norm, y_norm, heading_norm], dim=-1)

    def denormalize_poses(self, normalized_poses):
        """Denormalize poses from [-1, 1] back to original coordinate space."""
        # NOTE De-normalization values taken from recogdrive repo.
        x = (normalized_poses[..., 0:1] + 1) / 2 * 66.74 - 1.57
        y = (normalized_poses[..., 1:2] + 1) / 2 * 42 - 19.68
        heading = (normalized_poses[..., 2:3] + 1) / 2 * 3.53 - 1.67
        return torch.cat([x, y, heading], dim=-1)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, pred_x_start):
        """Predict noise from predicted x_start."""
        return (
            (self.sqrt_recip_alphas_cumprod[t][:, None, None] * x_t - pred_x_start)
            / self.sqrt_recipm1_alphas_cumprod[t][:, None, None]
        )

    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_start from noise."""
        return (
            self.sqrt_recip_alphas_cumprod[t][:, None, None] * x_t -
            self.sqrt_recipm1_alphas_cumprod[t][:, None, None] * noise
        )

    def denoise_fn(self, x_t, t, condition):
        """Denoising function using DiT architecture."""
        batch_size, num_poses, pose_dim = x_t.shape
        
        # Embed each pose in the trajectory sequence
        pose_emb = self.pose_embed(x_t)  # [batch, num_poses, d_model]
        
        # TODO rethink how to incorporate conditions and timestep embeddings into the input. rn its just placeholder.

        # Add position embeddings
        pos_ids = torch.arange(num_poses, device=x_t.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(pos_ids)  # [batch, num_poses, d_model]
        x = pose_emb + pos_emb
        
        # Time embeddings (broadcast to sequence length)
        time_emb = self.time_embed(t.float())  # [batch, d_model]
        time_emb = time_emb.unsqueeze(1).expand(-1, num_poses, -1)  # [batch, num_poses, d_model]
        x = x + time_emb
        
        # Condition embeddings (broadcast to sequence length)
        cond_emb = self.cond_embed(condition)  # [batch, d_model]
        cond_emb = cond_emb.expand(-1, num_poses, -1)  # [batch, num_poses, d_model]
        x = x + cond_emb
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Output projection to predict noise
        pred_noise = self.output_proj(x)  # [batch, num_poses, pose_dim]
        
        return pred_noise

    def forward(self, trajectory_query, targets):
        """Training step - compute diffusion loss."""
        batch_size = trajectory_query.shape[0]
        device = trajectory_query.device
        
        # Extract trajectory from targets dict
        target_trajectory = targets["trajectory"]
        dtype = next(self.parameters()).dtype
        target_trajectory = target_trajectory.to(dtype=dtype)

        # Normalize target trajectory
        target_trajectory_norm = self.normalize_poses(target_trajectory)
        
        # Sample random timesteps - uniform sampling is standard and effective
        # Each timestep gets equal probability, which works well for most diffusion tasks
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # Add noise to normalized target
        noise = torch.randn_like(target_trajectory_norm)
        x_t = self.q_sample(target_trajectory_norm, t, noise)
        
        # # Predict noise
        # pred_noise = self.denoise_fn(x_t, t, trajectory_query)
        model_output = self.dit_model(x_t, cross_att_cond, adaln_cond, t)
        
        # Loss
        loss = nn.functional.mse_loss(pred_noise, noise)
        
        # NOTE need to add that if we want to use trajectory loss as well, commented for DDP training sake
        # # Reconstruct trajectory using predicted noise
        # pred_x0 = self.predict_start_from_noise(x_t, t, pred_noise)
        # trajectory = self.denormalize_poses(pred_x0)
        
        # return {"diffusion_loss": loss, "trajectory": trajectory}
        return {"diffusion_loss": loss}

    def sample(self, trajectory_query, x_t=None):
        """Sample trajectory using DDIM."""
        batch_size = trajectory_query.shape[0]
        device = trajectory_query.device
        
        # Start from pure noise
        if x_t is None:
            x_t = torch.randn(batch_size, self.num_poses, self.pose_dim, device=device)
        
        # DDIM sampling with eta=0 (deterministic)
        times = torch.linspace(self.timesteps - 1, 0, 50, dtype=torch.long, device=device)  # 50 steps
        
        for i in range(len(times) - 1):
            t = times[i]
            t_next = times[i + 1]
            
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            pred_noise = self.denoise_fn(x_t, t_tensor, trajectory_query)
            
            # DDIM update
            alpha_t = self.alphas_cumprod[t]
            alpha_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.ones_like(alpha_t)
            
            pred_x0 = self.predict_start_from_noise(x_t, t_tensor, pred_noise)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1. - alpha_t_next - 0.0**2) * pred_noise  # eta=0
            
            x_t = torch.sqrt(alpha_t_next) * pred_x0 + dir_xt
        
        # Final prediction
        poses = x_t
        
        # Denormalize poses back to original scale
        poses = self.denormalize_poses(poses)
        
        # Apply tanh to heading
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * math.pi
        
        return {"trajectory": poses}