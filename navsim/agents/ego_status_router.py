from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.common.dataclasses import AgentInput, SensorConfig


class EgoStatusRouter(nn.Module):
    """Agent based on ReCogDrive with a trainable MLP for weighting future ego statuses."""

    def __init__(
        self,
        lr: float = 1e-4,
        checkpoint_path: Optional[str] = None,
        num_statuses: int = 4,  # Number of ego statuses to weight
        hidden_layer_dim: int = 128,
        ego_dim: int = 8,
    ):
        """
        Initializes WeightedEgoRecogDriveAgent.
        :param trajectory_sampling: the TrajectorySampling method to use
        :param lr: learning rate for MLP training
        :param checkpoint_path: path to pretrained ReCogDrive checkpoint
        :param num_statuses: number of future ego statuses to use for weighting
        """

        super().__init__()

        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self.num_statuses = num_statuses
        self.hidden_layer_dim = hidden_layer_dim
        self.ego_dim = ego_dim

        print(f"Initializing ego status router with: num_statuses {self.num_statuses}, ego_dim {self.ego_dim}, hidden_layer_dim {self.hidden_layer_dim}")

        # Trainable MLP for ego status weighting
        self._mlp = nn.Sequential(
            nn.Linear(self.num_statuses * self.ego_dim, self.hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_dim, num_statuses),
            nn.Softmax(dim=-1)
        )

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def forward(self, ego_statuses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass: compute weighted ego from a set of ego statuses"""
        # # Compute weights
        # egos_flat = ego_statuses.flatten()  # (num_statuses * ego_dim,)
        # weights = self._mlp(egos_flat.unsqueeze(0)).squeeze(0)  # (num_statuses,)
        # # Weighted average
        # weighted = torch.sum(weights.unsqueeze(1) * ego_statuses, dim=0)  # (ego_dim,)
        # # Replace status_feature with weighted command, velocity, accel
        # return torch.cat([weighted[:4], weighted[4:6], weighted[6:8]]).unsqueeze(0)
        batch_size = ego_statuses.shape[0]
        egos_flat = ego_statuses.view(batch_size, -1)  # (batch_size, num_statuses * ego_dim)
        weights = self._mlp(egos_flat)  # (batch_size, num_statuses * ego_dim)
        weighted = torch.sum(weights.unsqueeze(-1) * ego_statuses, dim=1)  # (batch_size, ego_dim)
        return weighted

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Delegate loss computation to ReCogDrive agent."""
        return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Only optimize MLP parameters."""
        return torch.optim.Adam(self._mlp.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Training callbacks. Same as ReCogDrive."""
        return []  # Adjust if needed



