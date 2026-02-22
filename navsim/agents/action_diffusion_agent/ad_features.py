"""
ActionDiffusionFeatureBuilder  — builds all feature tensors for one scene.
ActionDiffusionTargetBuilder   — builds target tensors used for loss computation.

Feature layout
--------------
camera_feature       : (seq_len, 3, H, W)  front panoramic (l0 + f0 + r0)
camera_feature_back  : (seq_len, 3, H, W)  rear  panoramic (l2 + b0 + r2)
                        only built when config.use_back_view is True
status_feature       : (8,)  [driving_command(4), vx(1), vy(1), ax(1), ay(1)]
hist_status_feature  : (N_hist * 7,) flat, per historical frame:
                        [vx, vy, ax, ay, px, py, heading]

Target layout
-------------
trajectory           : (8, 3)  sparse 0.5 s ego-relative trajectory [x, y, h]
interpolated_traj    : (40, 3) dense  0.1 s ego-relative trajectory [x, y, h]
                       (used by the diffusion head as ground truth)
"""

from typing import Dict, List

import cv2
import numpy as np
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torchvision import transforms

from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig
from navsim.common.dataclasses import AgentInput, Scene
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

# Dense trajectory spec: 40 steps at 0.1 s = 4 s horizon
_DENSE_SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)


def _states_to_relative_traj(states: np.ndarray) -> List[List[float]]:
    """
    Convert an array of absolute SE2 states (from PDM StateIndex) to a list
    of relative [x, y, heading] poses expressed w.r.t. the first state.

    Args:
        states: (N, ≥3) array where columns StateIndex.STATE_SE2 = [x, y, h]

    Returns:
        list of [x, y, h] relative poses (length N-1, first pose excluded)
    """
    rel_poses = absolute_to_relative_poses(
        [StateSE2(*row) for row in states[:, StateIndex.STATE_SE2]]
    )
    return [pose.serialize() for pose in rel_poses[1:]]


class ActionDiffusionFeatureBuilder(AbstractFeatureBuilder):
    """Builds image and ego-status features from an AgentInput."""

    def __init__(self, config: ActionDiffusionConfig) -> None:
        super().__init__()
        self._config = config
        self._to_tensor = transforms.ToTensor()

    def get_unique_name(self) -> str:
        # return "action_diffusion_feature"
        return "transfuser_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}

        # -- Camera features --------------------------------------------------
        features["camera_feature"] = self._stitch_front(agent_input)
        if self._config.use_back_view:
            features["camera_feature_back"] = self._stitch_rear(agent_input)

        # -- Current ego status (most recent frame) ---------------------------
        # Layout: [driving_command(4), vx(1), vy(1), ax(1), ay(1)] = 8 dims
        current_status = agent_input.ego_statuses[-1]
        features["status_feature"] = torch.cat(
            [
                torch.tensor(current_status.driving_command, dtype=torch.float32),
                torch.tensor(current_status.ego_velocity,    dtype=torch.float32),
                torch.tensor(current_status.ego_acceleration, dtype=torch.float32),
            ]
        )

        # -- Historical ego status (all frames except the current one) --------
        # Layout per frame: [vx(1), vy(1), ax(1), ay(1), px(1), py(1), h(1)] = 7 dims
        hist_frames = agent_input.ego_statuses[:-1]
        hist_parts: List[torch.Tensor] = []
        for es in hist_frames:
            hist_parts.append(
                torch.cat(
                    [
                        torch.tensor(es.ego_velocity,     dtype=torch.float32),  # (2,)
                        torch.tensor(es.ego_acceleration, dtype=torch.float32),  # (2,)
                        torch.tensor(es.ego_pose,         dtype=torch.float32),  # (3,)
                    ]
                )
            )
        features["hist_status_feature"] = torch.cat(hist_parts)  # (N_hist * 7,)

        return features

    # ──────────────────────────────────── camera helpers ─────────────────────

    def _stitch_front(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Stitch the front panoramic image from l0 + f0 + r0 for each frame.

        Each side camera is cropped to remove the top and bottom 28 pixels
        and the 416 outermost columns on the joining side; the front camera
        loses only the 28-pixel top / bottom strip.  The result is resized to
        (camera_height × camera_width) and converted to a float32 tensor.
        """
        cfg = self._config
        seq = agent_input.cameras[-cfg.seq_len:]
        frames: List[torch.Tensor] = []
        for camera in seq:
            frames.append(
                self._stitch_trio(
                    camera.cam_l0.image,
                    camera.cam_f0.image,
                    camera.cam_r0.image,
                    cfg,
                )
            )
        return torch.stack(frames)   # (seq_len, 3, H, W)

    def _stitch_rear(self, agent_input: AgentInput) -> torch.Tensor:
        """Stitch the rear panoramic image from l2 + b0 + r2."""
        cfg = self._config
        seq = agent_input.cameras[-cfg.seq_len:]
        frames: List[torch.Tensor] = []
        for camera in seq:
            frames.append(
                self._stitch_trio(
                    camera.cam_l2.image,
                    camera.cam_b0.image,
                    camera.cam_r2.image,
                    cfg,
                )
            )
        return torch.stack(frames)

    def _stitch_trio(
        self,
        left_img: np.ndarray,
        center_img: np.ndarray,
        right_img: np.ndarray,
        cfg: ActionDiffusionConfig,
    ) -> torch.Tensor:
        """
        Crop + concatenate three camera images and resize the panorama.
        Falls back to a black frame if any image is empty/None.
        """
        if (
            left_img is None
            or left_img.size == 0
            or not np.any(left_img)
        ):
            # Return a black frame with the expected spatial resolution.
            return torch.zeros(3, cfg.camera_height, cfg.camera_width)

        left   = left_img[28:-28, 416:-416]
        center = center_img[28:-28]
        right  = right_img[28:-28, 416:-416]

        stitched = np.concatenate([left, center, right], axis=1)  # (H', W', 3)
        resized  = cv2.resize(stitched, (cfg.camera_width, cfg.camera_height))
        return self._to_tensor(resized)   # (3, H, W)


class ActionDiffusionTargetBuilder(AbstractTargetBuilder):
    """
    Builds target tensors for loss computation.

    Specifically produces:
      trajectory        — sparse 8-step future trajectory at 0.5 s intervals
      interpolated_traj — dense  40-step future trajectory at 0.1 s intervals
                          (used as the diffusion head ground-truth)
    """

    def __init__(self, config: ActionDiffusionConfig) -> None:
        super().__init__()
        self._config = config
        self._vehicle_params = get_pacifica_parameters()

    def get_unique_name(self) -> str:
        # return "action_diffusion_target"
        return "transfuser_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        cfg = self._config

        # Number of sparse trajectory frames = time_horizon / interval_length
        n_sparse = int(
            cfg.trajectory_sampling.time_horizon
            / cfg.trajectory_sampling.interval_length
        )
        future_traj = scene.get_future_trajectory(num_trajectory_frames=n_sparse)

        # Sparse trajectory as a plain tensor (for the output / evaluator)
        trajectory = torch.tensor(future_traj.poses, dtype=torch.float32)  # (8, 3)

        # ── Dense 40-step interpolation for the diffusion GT ──────────────────
        frame_idx = scene.scene_metadata.num_history_frames - 1
        frame = scene.frames[frame_idx]
        ego_pose = StateSE2(*frame.ego_status.ego_pose)

        ego_state = EgoState.build_from_rear_axle(
            rear_axle_pose=ego_pose,
            tire_steering_angle=0.0,
            vehicle_parameters=self._vehicle_params,
            time_point=TimePoint(frame.timestamp),
            rear_axle_velocity_2d=StateVector2D(*frame.ego_status.ego_velocity),
            rear_axle_acceleration_2d=StateVector2D(*frame.ego_status.ego_acceleration),
        )

        # transform_trajectory / get_trajectory_as_array upsample to 0.1 s
        trans_traj = transform_trajectory(future_traj, ego_state)
        dense_states = get_trajectory_as_array(
            trans_traj, _DENSE_SAMPLING, ego_state.time_point
        )  # (40, ≥3) where [:, StateIndex.STATE_SE2] = [x, y, heading]

        relative_poses = _states_to_relative_traj(dense_states)  # list of [x,y,h]
        interpolated_traj = torch.tensor(relative_poses, dtype=torch.float32)  # (40, 3)

        return {
            "trajectory":         trajectory,
            "interpolated_traj":  interpolated_traj,
        }
