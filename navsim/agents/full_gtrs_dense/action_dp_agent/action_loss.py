import torch
import torch.nn as nn
import torch.nn.functional as F
from navsim.agents.utils.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
from navsim.agents.utils.action_space.utils import unwrap_angle

class DiffusionActionLoss(nn.Module):
    def __init__(self, dt=0.5, upsample_factor=5, device='cuda'):
        """
        Args:
            dt: The time delta of your input trajectory (e.g., 0.5s for 8 steps over 4s)
            upsample_factor: How much to interpolate before calc actions (5x -> 0.1s dt)
        """
        super().__init__()
        self.dt_input = dt
        self.upsample_factor = upsample_factor
        self.dt_calc = dt / upsample_factor # e.g. 0.1s
        
        # We assume 8 input steps * 5 = 40 calculation steps
        self.n_waypoints = 8 * upsample_factor 
        
        # Initialize the Physics Engine
        self.action_space = UnicycleAccelCurvatureActionSpace(
            dt=self.dt_calc,
            n_waypoints=self.n_waypoints,
            accel_std=2.0,   # Normalization stats (tune these based on your data!)
            curvature_std=0.1
        ).to(device)

    def _prepare_trajectory(self, traj_8step):
        """
        Upsamples (B, 8, 3) -> (B, 40, 3) and formats for ActionSpace.
        """
        B, T, D = traj_8step.shape
        
        # 1. Separate Pos and Heading
        pos = traj_8step[..., :2]
        heading = traj_8step[..., 2:3] # Keep dim for interpolation
        
        # 2. Upsample (Interpolate)
        # Permute to (B, C, T) for interpolate
        pos_up = F.interpolate(pos.permute(0, 2, 1), scale_factor=self.upsample_factor, mode='linear', align_corners=True)
        heading_up = F.interpolate(heading.permute(0, 2, 1), scale_factor=self.upsample_factor, mode='linear', align_corners=True)
        
        # Reshape back
        pos_up = pos_up.permute(0, 2, 1)      # (B, 40, 2)
        heading_up = heading_up.permute(0, 2, 1).squeeze(-1) # (B, 40)
        
        # 3. Format for ActionSpace (Needs 3D XYZ and Rotation Matrix)
        # Create Z zeros
        zeros = torch.zeros_like(pos_up[..., :1])
        traj_xyz = torch.cat([pos_up, zeros], dim=-1) # (B, 40, 3)
        
        # Convert Heading to Rotation Matrix
        c = torch.cos(heading_up)
        s = torch.sin(heading_up)
        z = torch.zeros_like(c)
        o = torch.ones_like(c)
        
        # Stack 3x3 rot matrix
        row1 = torch.stack([c, -s, z], dim=-1)
        row2 = torch.stack([s, c, z], dim=-1)
        row3 = torch.stack([z, z, o], dim=-1)
        traj_rot = torch.stack([row1, row2, row3], dim=-2) # (B, 40, 3, 3)
        
        return traj_xyz, traj_rot

    def forward(self, pred_traj, target_traj, current_velocity):
        """
        Args:
            pred_traj: (B, 8, 3) [x, y, heading]
            target_traj: (B, 8, 3) [x, y, heading]
            current_velocity: (B,) Scalar velocity at t=0 (Crucial for acceleration!)
        """
        # 1. Upsample and Prepare
        pred_xyz, pred_rot = self._prepare_trajectory(pred_traj)
        gt_xyz, gt_rot = self._prepare_trajectory(target_traj)
        
        # 2. Create Start State Dict (v0)
        # The ActionSpace needs v0 to calculate the first acceleration
        t0_states = {"v": current_velocity}
        
        # 3. Convert Prediction -> Actions (Differentiable!)
        # Returns (B, 40, 2) -> [Accel, Curvature]
        pred_actions = self.action_space.traj_to_action(
            traj_history_xyz=torch.zeros_like(pred_xyz), # Dummy history (not used if t0 provided)
            traj_history_rot=torch.zeros_like(pred_rot), 
            traj_future_xyz=pred_xyz,
            traj_future_rot=pred_rot,
            t0_states=t0_states
        )
        
        # 4. Convert Target -> Actions
        # We detach target actions so we don't backprop through the ground truth processing
        with torch.no_grad():
            gt_actions = self.action_space.traj_to_action(
                traj_history_xyz=torch.zeros_like(gt_xyz),
                traj_history_rot=torch.zeros_like(gt_rot),
                traj_future_xyz=gt_xyz,
                traj_future_rot=gt_rot,
                t0_states=t0_states
            )

        # 5. Compute Loss
        # We can weight Accel vs Curvature differently if needed
        loss = F.mse_loss(pred_actions, gt_actions)
        
        return loss