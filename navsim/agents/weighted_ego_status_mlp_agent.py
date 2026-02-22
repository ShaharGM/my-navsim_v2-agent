from typing import Any, Dict, List, Optional, Union

import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.ego_status_router import EgoStatusRouter


class WeightedEgoStatusFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder of EgoStatusMLP."""

    def __init__(self,
                num_statuses: int = 4,
                ego_dim: int = 8):
        """Initializes the feature builder."""
        self._num_statuses = num_statuses
        self._ego_dim = ego_dim

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "weighted_ego_status_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        ego_statuses = agent_input.ego_statuses

        # Compute future ego statuses
        ego_statuses_list = []
        for e in ego_statuses[-self._num_statuses:]:
            status_tensor = torch.cat([
                torch.tensor(e.ego_velocity, dtype=torch.float32),
                torch.tensor(e.ego_acceleration, dtype=torch.float32),
                torch.tensor(e.driving_command, dtype=torch.float32),
            ], dim=-1)
            ego_statuses_list.append(status_tensor)
        ego_statuses_tensor = torch.stack(ego_statuses_list) if ego_statuses_list else torch.empty(0, self._ego_dim)
        return {"ego_statuses": ego_statuses_tensor}


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    """Input feature builder of EgoStatusMLP."""

    def __init__(self, trajectory_sampling: TrajectorySampling):
        """
        Initializes the target builder.
        :param trajectory_sampling: trajectory sampling specification.
        """

        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}


class WeightedEgoStatusMLPAgent(AbstractAgent):
    """EgoStatMLP agent interface."""

    def __init__(
        self,
        hidden_layer_dim: int,
        mlp_checkpoint_path: str = None,
        lr: float = 1e-4,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        num_statuses: int = 4,
        ego_dim: int = 8,
    ):
        """
        Initializes the agent interface for EgoStatusMLP.
        :param hidden_layer_dim: dimensionality of hidden layer.
        :param lr: learning rate during training.
        :param checkpoint_path: optional checkpoint path as string, defaults to None
        :param trajectory_sampling: trajectory sampling specification.
        """
        super().__init__(trajectory_sampling)

        self._checkpoint_path = checkpoint_path
        self._lr = lr
        self._num_statuses = num_statuses
        self._ego_dim = ego_dim 
        self._mlp_checkpoint_path = mlp_checkpoint_path

        print(f"Initializing weighted ego mlp with: num_statuses {self._num_statuses}, ego_dim {self._ego_dim}, hidden_layer_dim {hidden_layer_dim}")

        self._status_router = EgoStatusRouter(
            lr=lr,
            checkpoint_path=self._checkpoint_path,
            num_statuses=self._num_statuses,
            ego_dim=self._ego_dim,
            )

        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(self._ego_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, self._trajectory_sampling.num_poses * 3),
        )
        
        # Freeze the MLP parameters since it's not trained
        for param in self._mlp.parameters():
            param.requires_grad = False
        # Ensure MLP is in eval mode for consistent inference
        self._mlp.eval()

        if self._checkpoint_path is None and self._mlp_checkpoint_path is not None:
            print(f"Initializing only base mlp from: {self._mlp_checkpoint_path}")
            self.initialize_base_mlp()

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def initialize_base_mlp(self) -> None:
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._mlp_checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._mlp_checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self._mlp.load_state_dict({k.replace("agent._mlp.", ""): v for k, v in state_dict.items()})
        print(f"Loaded MLP keys: {list(self._mlp.state_dict().keys())}")
        print(f"First layer weight sum: {self._mlp[0].weight.sum().item():.6f}")
        print(f"First layer weight mean: {self._mlp[0].weight.mean().item():.6f}")
        print(f"First layer bias sum: {self._mlp[0].bias.sum().item():.6f}")

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_no_sensors()

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [WeightedEgoStatusFeatureBuilder(num_statuses=self._num_statuses, ego_dim=self._ego_dim)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        weighted_status = self._status_router(features["ego_statuses"].to(torch.float32))
        # # Take last ego status instead
        # weighted_status = features["ego_statuses"][:, -1, :].to(torch.float32)
        poses: torch.Tensor = self._mlp(weighted_status)
        return {"trajectory": poses.reshape(-1, self._trajectory_sampling.num_poses, 3)}

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._status_router.parameters(), lr=self._lr)
