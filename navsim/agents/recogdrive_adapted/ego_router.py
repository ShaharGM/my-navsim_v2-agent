import torch
from torch import nn
from timm.models.layers import Mlp

class EgoRouter(nn.Model):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        hidden_layer_dim: int = 128,
        lr: float = 1e-4,
        checkpoint_path: Optional[str] = None,
        ego_status_count: int = 1,
    ):
        """
        """
        super().__init__(trajectory_sampling)

        self._checkpoint_path = checkpoint_path

        self._lr = lr

        self._ego_status_count = ego_status_count

        # self._router = torch.nn.Sequential(
        #     torch.nn.Linear(8 * ego_status_count, hidden_layer_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_layer_dim, self.ego_status_count),
        # )

        self._router = Mlp(
            in_features=self._ego_status_count * 8,
            hidden_features=hidden_layer_dim,
            out_features=self._ego_status_count,
            norm_layer=nn.LayerNorm
        )

    def forward(self, ego_statuses: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        weights = self._router(ego_statuses)
        return weights