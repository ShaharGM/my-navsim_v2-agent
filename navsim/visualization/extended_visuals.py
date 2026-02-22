import os
import numpy as np
from navsim.visualization.config import TRAJECTORY_CONFIG

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from navsim.visualization.plots import configure_bev_ax, configure_ax
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax

from navsim.common.dataclasses import Trajectory, TrajectorySampling
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array

def make_bev_animation(scene, predicted_trajectory=None, output_dir="notebook_anim"):
    def convert_local_to_global(local_poses, reference_pose):
        global_poses = np.zeros_like(local_poses, dtype=np.float64)
        cos_h = np.cos(reference_pose[2])
        sin_h = np.sin(reference_pose[2])
        global_poses[:, 0] = reference_pose[0] + local_poses[:, 0] * cos_h - local_poses[:, 1] * sin_h
        global_poses[:, 1] = reference_pose[1] + local_poses[:, 0] * sin_h + local_poses[:, 1] * cos_h
        global_poses[:, 2] = (reference_pose[2] + local_poses[:, 2]) % (2 * np.pi)
        return global_poses
    
    # Create animation
    frame_indices = list(range(len(scene.frames)))  # All frames in the scene
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Compute the full future poses once
    start_idx = max(0, scene.scene_metadata.num_history_frames - 1)
    full_future_poses = [scene.frames[i].ego_status.ego_pose for i in range(start_idx, len(scene.frames))]
    
    def animate(frame_idx):
        try:
            ax.clear()
            add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
            # Get the remaining future poses from this frame
            future_poses = full_future_poses[max(0, frame_idx - start_idx):]
            future_poses = future_poses[:10]  # Limit to 10 future poses for visibility
            if len(future_poses) > 0:  # Use len for safety
                local_poses = convert_absolute_to_relative_se2_array(
                    StateSE2(*scene.frames[frame_idx].ego_status.ego_pose),
                    np.array(future_poses, dtype=np.float64),
                )
                human_trajectory = Trajectory(
                    local_poses.astype(np.float32),
                    TrajectorySampling(
                        num_poses=len(local_poses),
                        time_horizon=len(local_poses) * 0.5,
                        interval_length=0.5,
                    )
                )
                # Add the trajectory
                add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
                
            # Add predicted trajectory if provided
            if predicted_trajectory is not None:
                # Slice the predicted trajectory to remaining poses from current frame
                remaining_idx = max(0, frame_idx - start_idx)
                predicted_remaining = predicted_trajectory.poses[remaining_idx:]
                # Convert predicted trajectory (local to initial frame) to local to current frame
                global_pred = convert_local_to_global(predicted_remaining, scene.frames[start_idx].ego_status.ego_pose)
                # global_pred = relative_to_absolute_poses(
                #     predicted_trajectory.poses,
                #     StateSE2(*scene.frames[start_idx].ego_status.ego_pose)
                # )
                # global_pred_array = np.array([[p.x, p.y, p.heading] for p in global_pred])
                local_pred_current = convert_absolute_to_relative_se2_array(
                    StateSE2(*scene.frames[frame_idx].ego_status.ego_pose),
                    global_pred,
                )
                # local_pred_current = local_pred_current[remaining_idx:]  # Get the actual remaining poses
                local_pred_current = local_pred_current[:11]  # Limit to 11 poses
                if len(local_pred_current) > 0:
                    pred_traj = Trajectory(
                        local_pred_current.astype(np.float32),
                        TrajectorySampling(
                            num_poses=len(local_pred_current),
                            time_horizon=len(local_pred_current) * 0.5,
                            interval_length=0.5,
                        )
                    )
                    add_trajectory_to_bev_ax(ax, pred_traj, TRAJECTORY_CONFIG["agent"])
            configure_bev_ax(ax)
            configure_ax(ax)
        except Exception as e:
            print(f"Error in animate for frame {frame_idx}: {e}")
            ax.clear()
            ax.text(0.5, 0.5, f"Error at frame {frame_idx}: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            return  # Skip this frame
    
    anim = FuncAnimation(fig, animate, frames=frame_indices, interval=200, blit=False)  # 200ms per frame for ~5 FPS
    
    # Save as GIF
    gif_file = f"{output_dir}/{scene.scene_metadata.initial_token}_bev.gif"
    os.makedirs(os.path.dirname(gif_file), exist_ok=True)
    # try:
    anim.save(gif_file, writer='pillow', fps=5)
    print(f"GIF saved to {gif_file}")
    # except Exception as e:
    #     print(f"Failed to save GIF: {e}")