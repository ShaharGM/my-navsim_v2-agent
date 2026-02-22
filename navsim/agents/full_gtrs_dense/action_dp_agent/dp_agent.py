import os
from typing import Any, Union, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    OneCycleLR,
    StepLR,
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
)

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.full_gtrs_dense.action_dp_agent.action_dp_callback import ActionDPCallbacks
from navsim.agents.full_gtrs_dense.action_dp_agent.dp_config import DPConfig
from navsim.agents.full_gtrs_dense.action_dp_agent.dp_model import DPModel
from navsim.agents.full_gtrs_dense.action_dp_agent.dp_features import DPFeatureBuilder, DPTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.agents.full_gtrs_dense.action_dp_agent.dp_model import GuidanceConfig


class DPAgent(AbstractAgent):
    def __init__(
            self,
            config: DPConfig,
            lr: float,
            checkpoint_path: str = None
    ):
        super().__init__(
            trajectory_sampling=config.trajectory_sampling
        )
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self.model = DPModel(config)
        self.backbone_wd = config.backbone_wd
        self.scheduler = config.scheduler
        self.guidance_config: Optional[GuidanceConfig] = None

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        print("Loading diffusion model weights...")
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"), weights_only=False)[
            "state_dict"]
        msg = self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})
        print("Diffusion model loaded", msg)

    def get_sensor_config(self) -> SensorConfig:
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
        return [DPTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [DPFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(features, guidance=self.guidance_config)

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
            tokens=None
    ):
        # Get Targets (Dense 40-step Trajectory)
        target_traj_dense = targets["interpolated_traj"]
        env_kv = predictions['env_kv'] 

        dp_loss = self.model._trajectory_head.get_dp_loss(
            predictions['env_kv'], 
            target_traj_dense, 
            features
        )
        return dp_loss * self._config.dp_loss_weight
        # bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
        # loss = (
        #     dp_loss * self._config.dp_loss_weight +
        #     bev_semantic_loss * self._config.bev_loss_weight
        # )
        # return loss

    # def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
    #     backbone_params_name = '_backbone.image_encoder'
    #     reference_transformer_name = '_trajectory_head.reference_transformer'
    #     ori_transformer_name = '_trajectory_head.ori_transformer'
    #     img_backbone_params = list(
    #         filter(lambda kv: backbone_params_name in kv[0], self.model.named_parameters()))
    #     default_params = list(filter(lambda kv:
    #                                  backbone_params_name not in kv[0] and
    #                                  reference_transformer_name not in kv[0] and
    #                                  ori_transformer_name not in kv[0], self.model.named_parameters()))
    #     params_lr_dict = [
    #         {'params': [tmp[1] for tmp in default_params]},
    #         {
    #             'params': [tmp[1] for tmp in img_backbone_params],
    #             'lr': self._lr * self._config.lr_mult_backbone,
    #             'weight_decay': self.backbone_wd
    #         }
    #     ]

    #     if self.scheduler == 'default':
    #         return torch.optim.Adam(params_lr_dict, lr=self._lr, weight_decay=self._config.weight_decay)
    #     elif self.scheduler == 'cycle':
    #         optim = torch.optim.Adam(params_lr_dict, lr=self._lr)
    #         return {
    #             "optimizer": optim,
    #             "lr_scheduler": OneCycleLR(
    #                 optim,
    #                 max_lr=0.01,
    #                 total_steps=100 * 202
    #             )
    #         }
    #     else:
    #         raise ValueError('Unsupported lr scheduler')

    def get_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.Adam(
            trainable_params, 
            lr=self._lr, 
            weight_decay=self._config.weight_decay
        )

        # If no scheduler is requested, just return the optimizer
        if self.scheduler is None:
            return optimizer

        lr_scheduler_config = {}

        if self.scheduler == 'default':
            return optimizer
        
        # 1. OneCycleLR (Updates every BATCH)
        if self.scheduler == 'cycle':
            lr_scheduler = OneCycleLR(
                optimizer,
                max_lr=0.01, # Note: This overrides self._lr
                total_steps=30000 * 4, # epochs * steps_per_epoch (assuming 4 batches/epoch in debug mode)
                pct_start=0.3,
                div_factor=25,
                final_div_factor=1000
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "step",  # <--- CRITICAL: Update every batch
                "frequency": 1
            }

        # 2. StepLR (Updates every EPOCH)
        elif self.scheduler == 'step':
            lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1
            }

        # 3. Cosine Annealing (Updates every EPOCH)
        elif self.scheduler == 'cosine':
            tmax = 30000
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=1e-6)
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1
            }

        # 4. Plateau (Updates every EPOCH, monitors validation loss)
        elif self.scheduler == 'plateau':
            lr_scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.1, 
                patience=5,
                verbose=True
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "monitor": "val_loss", # <--- CRITICAL: Must match your validation log key
                "frequency": 1
            }

        # Return the combined dictionary
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }

    def get_training_callbacks(self) -> List[pl.Callback]:
        ckpt_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val/loss_epoch",
            mode="min",
            dirpath=None,
            filename="epoch_{epoch:03d}-val_loss_{val/loss_epoch:.4f}",
            save_last=True,
            auto_insert_metric_name=False,
        )
        return [
            ckpt_callback,
            ActionDPCallbacks(config=self._config, num_samples=5)
        ]