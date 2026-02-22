from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, EgoStatus, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

from .recogdrive_agent import ReCogDriveAgent
from .weighted_ego_recogdrive_features import WeightedEgoRecogDriveFeatureBuilder, TrajectoryTargetBuilder


class WeightedEgoRecogDriveAgent(AbstractAgent):
    """Agent based on ReCogDrive with a trainable MLP for weighting future ego statuses."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        lr: float = 1e-4,
        checkpoint_path: Optional[str] = None,
        recogdrive_checkpoint_path: Optional[str] = None,
        num_future: int = 8,  # Number of future ego statuses to weight
        vlm_path: Optional[str] = None,
        vlm_type: str = 'internvl',
        dit_type: str = 'small',
        sampling_method: str = 'ddim',
        cache_hidden_state: bool = True,
        cache_mode: bool = True,
        vlm_size: Optional[str] = 'large', 
    ):
        """
        Initializes WeightedEgoRecogDriveAgent.
        :param lr: learning rate for MLP training
        :param checkpoint_path: path to pretrained ReCogDrive checkpoint
        :param trajectory_sampling: trajectory sampling specification
        :param num_future: number of future ego statuses to use for weighting
        :param vlm_path: path to VLM checkpoint (if needed)
        :param vlm_type: type of VLM
        :param dit_type: type of DiT model
        :param sampling_method: sampling method for diffusion
        :param cache_hidden_state: whether to cache hidden states
        """
        super().__init__(trajectory_sampling)

        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._num_future = num_future
        self._trajectory_sampling=trajectory_sampling

        self.feature_builder = WeightedEgoRecogDriveFeatureBuilder(
                                                cache_hidden_state=cache_hidden_state,
                                                model_type=vlm_type,
                                                checkpoint_path=vlm_path,
                                                cache_mode=cache_mode,
                                            )

        # Load ReCogDrive agent
        self._recogdrive_agent = ReCogDriveAgent(
            trajectory_sampling=trajectory_sampling,
            checkpoint_path=recogdrive_checkpoint_path,
            vlm_path=vlm_path,
            vlm_type=vlm_type,
            dit_type=dit_type,
            sampling_method=sampling_method,
            cache_hidden_state=cache_hidden_state,
            cache_mode=cache_mode,
            vlm_size=vlm_size,
        )

        # Freeze ReCogDrive parameters
        for param in self._recogdrive_agent.parameters():
            param.requires_grad = False

        # Set ReCogDrive to eval mode
        self._recogdrive_agent.eval()

        # Trainable MLP for ego status weighting
        ego_dim = 11  # pose(3) + velocity(2) + acceleration(2) + driving_command(4)
        self._mlp = nn.Sequential(
            nn.Linear(num_future * ego_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_future),
            nn.Softmax(dim=-1)
        )

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Initialize the ReCogDrive agent."""
        self._recogdrive_agent.initialize()
        # TODO add to also load checkpoint for weightedEgoRecogdrive as well if given

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass. Same as ReCogDrive."""
        return self._recogdrive_agent.get_sensor_config()

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass. Same as ReCogDrive."""
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass. Same as ReCogDrive."""
        return [self.feature_builder]

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """Compute trajectory with weighted ego status."""

        # get features
        feature_builder = self.get_feature_builders()[0]
        features = feature_builder.compute_features(agent_input)

        # Add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        self.eval()
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        return Trajectory(poses)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # """Forward pass through ReCogDrive agent."""
        # return self._recogdrive_agent.forward(features)
        """Forward pass: compute weighted ego from future in features, modify features, then call ReCogDrive."""
        if 'future_ego_statuses' in features:
            future_ego_tensors = features['future_ego_statuses'].squeeze(0)  # (num_future, ego_dim)
            # Compute weights
            future_flat = future_ego_tensors.flatten()  # (num_future * ego_dim,)
            print(f"ego weight mlp input: {future_flat}")
            weights = self._mlp(future_flat.unsqueeze(0)).squeeze(0)  # (num_future,)
            # Weighted average
            print(f"weights: {weights}")
            print(f"future ego statuses: {future_ego_tensors}")
            weighted = torch.sum(weights.unsqueeze(1) * future_ego_tensors, dim=0)  # (ego_dim,)
            print(f"weighted ego status: {weighted}")
            # Replace status_feature with weighted command, velocity, accel
            features['status_feature'] = torch.cat([weighted[:4], weighted[4:6], weighted[6:8]]).unsqueeze(0)
            # Remove future_ego_statuses
            del features['future_ego_statuses']
        # Call ReCogDrive
        return self._recogdrive_agent.forward(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Delegate loss computation to ReCogDrive agent."""
        return self._recogdrive_agent.compute_loss(features, targets, predictions)

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Only optimize MLP parameters."""
        return torch.optim.Adam(self._mlp.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Training callbacks. Same as ReCogDrive."""
        return []  # Adjust if needed