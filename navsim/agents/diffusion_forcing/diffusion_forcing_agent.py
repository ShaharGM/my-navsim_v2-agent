from typing import Dict, List, Optional, Any
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.diffusion_forcing.models.diffusion import Diffusion


class DiffusionForcingFeatureBuilder(AbstractFeatureBuilder):
    """Feature builder for Diffusion Forcing agent."""

    def __init__(self, history_steps: int):
        self.history_steps = history_steps

    def get_unique_name(self) -> str:
        return "diffusion_forcing_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Extract historical trajectory poses."""
        # Stack poses: each pose is [x, y, heading]
        poses = []
        for e in agent_input.ego_statuses[-self.history_steps:]:
            poses.append(e.ego_pose)
        
        # Pad if necessary
        while len(poses) < self.history_steps:
            poses.insert(0, poses[0] if poses else [0, 0, 0])  # Pad with first or zeros
        
        hist_traj = torch.tensor(poses, dtype=torch.float32)
        return {"ego_status": hist_traj}


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    """Standard trajectory target builder."""

    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Extract future trajectory poses."""
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        future_poses = torch.tensor(future_trajectory.poses, dtype=torch.float32)
        return {"trajectory": future_poses}


class DiffusionForcingBackbone(nn.Module):
    """
    Adapted Diffusion Forcing backbone for NAVSIM trajectory prediction.
    Models future trajectory as a sequence with diffusion.
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        # NAVSIM specific
        self.future_steps = cfg.future_steps  # Number of future poses
        self.pose_dim = 3  # x, y, heading
        self.history_steps = cfg.history_steps
        
        # Diffusion forcing params
        self.unstacked_dim = self.pose_dim  # Each "frame" is a pose
        self.x_shape = (self.unstacked_dim,)
        self.pose_stack = cfg.pose_stack  # Number of poses per stacked token
        self.x_stacked_shape = (self.unstacked_dim * self.pose_stack,)
        self.n_tokens = self.future_steps // self.pose_stack  # Number of tokens in sequence
        
        self.guidance_scale = cfg.guidance_scale
        self.context_frames = cfg.context_frames
        self.chunk_size = cfg.chunk_size
        self.external_cond_dim = cfg.external_cond_dim  # Should be history_steps * pose_dim
        self.causal = cfg.causal
        
        self.uncertainty_scale = cfg.uncertainty_scale
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.clip_noise = cfg.diffusion.clip_noise
        
        self.cfg = cfg
        self.cfg.diffusion.cum_snr_decay = self.cfg.diffusion.cum_snr_decay ** self.pose_stack
        
        # load trajectory anchors
        plan_anchor = np.load(self.cfg.plan_anchor_path)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        print(f"Loaded anchors shape: {self.plan_anchor.shape}")

        self._build_model()
    
    def _normalize_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """Normalize poses to [-1, 1] using hardcoded stats (temporary)."""
        # poses: (..., 3) with [x, y, heading]
        x = poses[..., 0:1]
        y = poses[..., 1:2] 
        heading = poses[..., 2:3]
        
        # Hardcoded stats from DiffusionDrive (temporary)
        x_norm = 2 * (x + 1.2) / 56.9 - 1
        y_norm = 2 * (y + 20) / 46 - 1
        heading_norm = 2 * (heading + 2) / 3.9 - 1
        
        return torch.cat([x_norm, y_norm, heading_norm], dim=-1)
        # return poses
    
    def _denormalize_poses(self, poses_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize poses from [-1, 1] to original scale."""
        # poses_norm: (..., 3) with normalized [x, y, heading]
        x_norm = poses_norm[..., 0:1]
        y_norm = poses_norm[..., 1:2]
        heading_norm = poses_norm[..., 2:3]
        
        # Inverse of normalization
        x = (x_norm + 1) / 2 * 56.9 - 1.2
        y = (y_norm + 1) / 2 * 46 - 20
        heading = (heading_norm + 1) / 2 * 3.9 - 2
        
        return torch.cat([x, y, heading], dim=-1)
        # return poses_norm

    def _build_model(self):
        self.diffusion_model = Diffusion(
            x_shape=torch.Size(self.x_stacked_shape),
            external_cond_dim=self.external_cond_dim,
            is_causal=self.causal,
            cfg=self.cfg.diffusion
        )
    
    def _preprocess_batch(self, features, targets):
        """Preprocess batch for diffusion forcing."""
        # features: historical trajectory, targets: future trajectory
        hist_traj = features['ego_status']  # (batch, history_steps, pose_dim)
        future_traj = targets['trajectory']  # (batch, future_steps, pose_dim)
        
        # Normalize both
        normalized_hist = self._normalize_poses(hist_traj)
        normalized_future = self._normalize_poses(future_traj)
        
        # Concatenate for sequence input
        xs = torch.cat([normalized_hist, normalized_future], dim=1)  # (batch, total_steps, pose_dim)
        
        # No external conditioning
        conditions = None
        
        return xs, conditions, None
    
    def _generate_noise_levels(self, xs):
        """Generate noise levels for diffusion - per token for Diffusion Forcing."""
        batch_size, total_frames, *_ = xs.shape
        noise_levels = torch.zeros((batch_size, total_frames), dtype=torch.long, device=xs.device)
        # Add noise only to future steps
        noise_levels[:, self.history_steps:] = torch.randint(0, self.timesteps, (batch_size, self.future_steps), device=xs.device)
        return noise_levels
    
    def compute_loss(self, xs, conditions, noise_levels, masks=None):
        _, loss = self.diffusion_model(xs, conditions, noise_levels)
        # NOTE no need for masks for now since we do full sequence
        if masks is not None:
            loss = self.reweight_loss(loss, masks)
        return loss.mean()

    def reweight_loss(self, loss, masks):
        """Reweight loss based on masks."""
        return (loss * masks).sum() / masks.sum()
    
    def _generate_scheduling_matrix(self, horizon: int):
        # Match the base: hardcoded to pyramid for NAVSIM
        return self._generate_pyramid_scheduling_matrix(horizon, self.uncertainty_scale)

    def _generate_pyramid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m
        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)
    
    def sample_trajectory(self, features: torch.Tensor) -> torch.Tensor:
        """
        Sample future trajectory given historical trajectory using diffusion sampling.
        Handles batched inputs: features of the agent
        """
        device = next(self.parameters()).device
        hist_traj = features["ego_status"].to(device)
        
        # Handle both batched and single inputs
        if hist_traj.dim() == 2:  # (history_steps, pose_dim) -> add batch dim
            hist_traj = hist_traj.unsqueeze(0)  # (1, history_steps, pose_dim)
        
        batch_size = hist_traj.shape[0]
        
        # No external conditioning for now
        conditions = None
        
        # Start with hist + noise for future (video-like sequence)
        # base future noised trajectory on anchors (need to add traj. classifier as well to use all 20 anchors)
        if self.plan_anchor is not None:
            # Use anchor: select first mode (like a fixed prior), normalize, add noise, denormalize
            anchor = self.plan_anchor[0]  # (future_steps, 2)
            # Pad to 3D with zero heading
            anchor_3d = torch.cat([anchor, torch.zeros_like(anchor[..., :1])], dim=-1)  # (future_steps, 3)
            anchor_norm = self._normalize_poses(anchor_3d.unsqueeze(0)).squeeze(0)
            # print(f"og anchor 3d-ed: {anchor_3d}")
            anchor_denorm = self._denormalize_poses(anchor_norm.unsqueeze(0)).squeeze(0)

            # noise = 0.1 * torch.randn_like(anchor_norm)
            # if self.clip_noise:
            #     noise = torch.clamp(noise, -1, 1)
            # noisy_anchor_norm = anchor_norm + noise

            # Use q_sample to add controlled noise at timestep 8 (like DiffusionDrive)
            t = torch.full((anchor_norm.shape[0],), 8, device=device, dtype=torch.long)  # trunc_timesteps=8
            noisy_anchor_norm = self.diffusion_model.q_sample(anchor_norm.unsqueeze(0), t.unsqueeze(0)).squeeze(0)

            x_future = self._denormalize_poses(noisy_anchor_norm.unsqueeze(0)).squeeze(0)
            # print(f"noised future: {x_future}")
            # Repeat for batch
            x_future = x_future.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            x_future_norm = torch.randn((batch_size, self.future_steps, self.pose_dim), device=device)
            if self.clip_noise:
                x_future_norm = torch.clamp(x_future_norm, -1, 1)  # Clamp normalized noise
            x_future = self._denormalize_poses(x_future_norm)  # Denormalize to original scale
        
        # x_future_norm = torch.randn((batch_size, self.future_steps, self.pose_dim), device=device)
        # if self.clip_noise:
        #     x_future_norm = torch.clamp(x_future_norm, -1, 1)  # Clamp normalized noise
        # x_future = self._denormalize_poses(x_future_norm)  # Denormalize to original scale

        x = torch.cat([hist_traj, x_future], dim=1)  # (batch_size, total_steps, pose_dim)
        # print(f"Initial randomized trajectory: {x}")
        
        # 2D grid sampling with scheduling matrix (only for future steps)
        scheduling_matrix = self._generate_scheduling_matrix(self.future_steps)
        
        for m in range(scheduling_matrix.shape[0] - 1):
            # Noise levels for current step: zeros for history, scheduling_matrix[m] for future
            from_noise_levels = np.concatenate((np.zeros((self.history_steps,), dtype=np.int64), scheduling_matrix[m]))[None, :].repeat(batch_size, axis=0)
            from_noise_levels = torch.from_numpy(from_noise_levels).to(device)
            
            # Noise levels for next step
            to_noise_levels = np.concatenate((np.zeros((self.history_steps,), dtype=np.int64), scheduling_matrix[m + 1]))[None, :].repeat(batch_size, axis=0)
            to_noise_levels = torch.from_numpy(to_noise_levels).to(device)

            # print(f"current noise levels: {from_noise_levels}")
            # print(f"denoising to noise levels: {to_noise_levels}")
            
            # Use sample_step for proper DDIM update
            x = self.diffusion_model.sample_step(x, conditions, from_noise_levels, to_noise_levels)

            # Clamp by normalizing, clamping, denormalizing to prevent explosion
            x_norm = self._normalize_poses(x)
            x_norm = torch.clamp(x_norm, min=-1, max=1)
            x = self._denormalize_poses(x_norm)

            # print(f"denoised trajectory at step {m}: {x}")
        
        # Return only the future part
        future_traj = x[:, self.history_steps:]  # (batch_size, future_steps, pose_dim)
        
        return future_traj


class DiffusionForcingAgent(AbstractAgent):
    """
    NAVSIM agent wrapper for Diffusion Forcing trajectory prediction.
    """
    
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        history_steps: int = 4,
        pose_dim: int = 3,
        pose_stack: int = 1,
        guidance_scale: float = 1.0,
        context_frames: int = 1,
        chunk_size: int = 1,
        external_cond_dim: int = 0,
        causal: bool = False,
        uncertainty_scale: float = 1.0,
        lr: float = 1e-5,
        diffusion_timesteps: int = 1000,
        diffusion_sampling_timesteps: int = 100,
        diffusion_clip_noise: bool = True,
        diffusion_cum_snr_decay: float = 0.9,
        diffusion_beta_schedule: str = "cosine",
        diffusion_schedule_fn_kwargs: dict = None,
        diffusion_objective: str = "pred_noise",
        diffusion_use_fused_snr: bool = True,
        diffusion_snr_clip: float = 5.0,
        diffusion_ddim_sampling_eta: float = 0.0,
        diffusion_architecture: str = "transformer",
        diffusion_stabilization_level: int = 10,
        checkpoint_path: Optional[str] = None,
        plan_anchor_path: Optional[str] = None,  # Path to anchor numpy file
        diffusion_network_size: int = 128,
        diffusion_num_layers: int = 4,
        diffusion_attn_heads: int = 4,
        diffusion_dim_feedforward: int = 512,
    ):
        super().__init__(trajectory_sampling)
        
        # Build config from parameters
        if diffusion_schedule_fn_kwargs is None:
            diffusion_schedule_fn_kwargs = {}
        
        self.cfg = DictConfig({
            "history_steps": history_steps,
            "future_steps": trajectory_sampling.num_poses,  # Set dynamically
            "pose_dim": pose_dim,
            "pose_stack": pose_stack,
            "guidance_scale": guidance_scale,
            "context_frames": context_frames,
            "chunk_size": chunk_size,
            "external_cond_dim": external_cond_dim,
            "causal": causal,
            "uncertainty_scale": uncertainty_scale,
            "lr": lr,
            "plan_anchor_path": plan_anchor_path,
            "diffusion": {
                "timesteps": diffusion_timesteps,
                "sampling_timesteps": diffusion_sampling_timesteps,
                "clip_noise": diffusion_clip_noise,
                "cum_snr_decay": diffusion_cum_snr_decay,
                "beta_schedule": diffusion_beta_schedule,
                "schedule_fn_kwargs": diffusion_schedule_fn_kwargs,
                "objective": diffusion_objective,
                "use_fused_snr": diffusion_use_fused_snr,
                "snr_clip": diffusion_snr_clip,
                "ddim_sampling_eta": diffusion_ddim_sampling_eta,
                "architecture": {
                    "network_size": diffusion_network_size,
                    "num_layers": diffusion_num_layers,
                    "attn_heads": diffusion_attn_heads,
                    "dim_feedforward": diffusion_dim_feedforward,
                },
                "stabilization_level": diffusion_stabilization_level,
            }
        })
        print(f"Loading diffusion forcing agent with following data:\nhistory steps {history_steps}\nfuture steps {trajectory_sampling.num_poses}\nanchor path: {plan_anchor_path}")
        
        self.model = DiffusionForcingBackbone(self.cfg)
        self._checkpoint_path = checkpoint_path
    
    def name(self) -> str:
        return "DiffusionForcingAgent"
    
    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()
    
    def initialize(self) -> None:
        """Load checkpoint if provided."""
        if self._checkpoint_path is not None:
            if torch.cuda.is_available():
                state_dict = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
            self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # During training, predictions are not used for loss, so skip expensive inference
        if self.training:
            # Return dummy prediction with correct shape - match the batch size
            hist_traj = features.get("ego_status", torch.zeros(1, 10, 3))
            batch_size = hist_traj.shape[0] if len(hist_traj.shape) >= 2 else 1
            dummy_traj = torch.zeros((batch_size, self.cfg.future_steps, self.cfg.pose_dim), device=next(self.parameters()).device)
            return {"trajectory": dummy_traj}
        
        # During inference, do full trajectory sampling
        future_traj = self.model.sample_trajectory(features)
        return {"trajectory": future_traj}
    
    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [DiffusionForcingFeatureBuilder(self.cfg.history_steps)]
    
    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TrajectoryTargetBuilder(self._trajectory_sampling)]
    
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute diffusion loss for training."""
        xs, conditions, _ = self.model._preprocess_batch(features, targets)
        noise_levels = self.model._generate_noise_levels(xs)
        return self.model.compute_loss(xs, conditions, noise_levels)
    
    def get_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-4)