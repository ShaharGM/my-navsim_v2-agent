from typing import Any, Union, List, Dict
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

from navsim.agents.gtrs_dense.gtrs_agent import GTRSAgent
# from navsim.agents.full_gtrs_dense.action_dp_agent.dp_agent import DPAgent
from navsim.agents.dp.dp_agent import DPAgent

from navsim.agents.gtrs_dense.hydra_config import HydraConfig
# from navsim.agents.full_gtrs_dense.action_dp_agent.dp_config import DPConfig
from navsim.agents.dp.dp_config import DPConfig

# from navsim.agents.full_gtrs_dense.action_dp_agent.dp_features import DPFeatureBuilder, DPTargetBuilder
from navsim.agents.gtrs_dense.hydra_features import HydraFeatureBuilder, HydraTargetBuilder

class FullGTRSAgent(AbstractAgent):
    def __init__(
            self,
            dp_config: DPConfig,
            hydra_config: HydraConfig,
            lr: float,
            dp_checkpoint_path: str = None,
            scorer_checkpoint_path: str = None
    ):
        super().__init__(
            trajectory_sampling=hydra_config.trajectory_sampling
        )
        self._dp_config = dp_config
        self._scorer_config = hydra_config
        self._lr = lr

        self.dp_agent = DPAgent(
            dp_config,
            lr=lr,
            checkpoint_path=dp_checkpoint_path
        )
        self.scorer_agent = GTRSAgent(
            hydra_config,
            lr=lr,
            checkpoint_path=scorer_checkpoint_path
        )

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        print("Loading Full-GTRS model weights...")
        self.dp_agent.initialize()
        self.scorer_agent.initialize()
        print("Done!")

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[0, 1, 2, 3],
            cam_l0=[0, 1, 2, 3],
            cam_l1=[0, 1, 2, 3],
            cam_l2=[0, 1, 2, 3],
            cam_r0=[0, 1, 2, 3],
            cam_r1=[0, 1, 2, 3],
            cam_r2=[0, 1, 2, 3],
            cam_b0=[0, 1, 2, 3],
            lidar_pc=[],
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        # return [DPTargetBuilder(config=self._dp_config)]
        return [HydraTargetBuilder(config=self._dp_config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        # return [DPFeatureBuilder(config=self._dp_config)]
        return [HydraFeatureBuilder(config=self._dp_config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dp_prop = self.dp_agent(features)
        dp_pred = dp_prop['dp_pred'] # Shape: (B, N, 8, 3)
        
        # NOTE consider changing to the interpolation code used in GTRS
        # 1. Interpolate (8 -> 40)
        B, N, T, D = dp_pred.shape
        dp_pred = dp_pred.flatten(0, 1).permute(0, 2, 1) 
        
        dp_pred = F.interpolate(dp_pred, size=40, mode='linear', align_corners=True)
        
        # Reshape back to (B, N, 40, 3)
        dp_pred = dp_pred.permute(0, 2, 1).view(B, N, 40, D).contiguous()

        return self.scorer_agent.evaluate_dp_proposals(
            features, dp_pred, dp_only_inference=True, topk=1
        )

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
            tokens=None
    ):
        self.dp_agent.compute_loss(features, targets, predictions, tokens)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        self.dp_agent.get_optimizers()