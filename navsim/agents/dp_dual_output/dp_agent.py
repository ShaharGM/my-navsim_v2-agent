# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Union
from typing import Dict
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.dp_dual_output.dp_config import DPConfig
from navsim.agents.dp_dual_output.dp_model import DPModel
from navsim.agents.dp_dual_output.dp_features import HydraFeatureBuilder, HydraTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


def dp_loss_bev(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor],
        config: DPConfig, traj_head
):
    # B, 8 (4 secs, 0.5Hz), 3
    target_traj = targets["trajectory"]
    dp_loss = traj_head.get_dp_loss(predictions['env_kv'], target_traj.float())
    bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
    dp_loss = dp_loss * config.dp_loss_weight
    bev_semantic_loss = bev_semantic_loss * config.bev_loss_weight
    loss = (
            dp_loss +
            bev_semantic_loss
    )
    return loss, {
        'dp_loss': dp_loss,
        'bev_semantic_loss': bev_semantic_loss
    }


class DPAgent(AbstractAgent):
    def __init__(
            self,
            config: DPConfig,
            lr: float,
            checkpoint_path: str = None,
            pretrained_checkpoint: str = None
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

        if checkpoint_path is None and pretrained_checkpoint is not None:
            print(f"Loading pretrained base model from: {pretrained_checkpoint}")
            self.load_pretrained_weights_with_surgery(pretrained_checkpoint)
        
        # if checkpoint_path is None and pretrained_checkpoint is not None:
        #     self.load_pretrained_matching_weights(pretrained_checkpoint)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__ + " copy"

    def load_pretrained_weights_with_surgery(self, checkpoint_path):
        """
        Loads a model with a 4D-channel input/output embedder checkpoint (x,y,sin,cos) into an 8D-channel model (x,y,sin,cos,vx,vy,ax,ay).
        """
        # if checkpoint_path is None:
        #     checkpoint_path = self._checkpoint_path

        # 1. Load and Fix Prefix
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
        
        # Strip prefixes if necessary (e.g. "agent.")
        if list(ckpt.keys())[0].startswith("agent."):
            ckpt = {k.replace("agent.", ""): v for k, v in ckpt.items()}

        model_state = self.state_dict()
        new_state_dict = self.state_dict() # Start with current random init
        
        # 2. Iterate and Surgically Graft
        for name, old_param in ckpt.items():
            if name not in model_state: 
                print(f"{name} is not part of the model. Skipping...")
                continue
            
            # Check shapes to see if surgery is needed
            new_param = model_state[name]
            
            # --- CASE 1: Standard Layers (Matches perfectly) ---
            if old_param.shape == new_param.shape:
                new_state_dict[name] = old_param
                
            # --- CASE 2: Input Embedding Weight ---
            # We look for specific layer names to be safe and readable
            elif "input_emb.weight" in name:
                print(f"SURGERY: Interleaving Input Weights for {name}")
                
                # The weights are flattened [Hidden, horizon * 4]. 
                # We must reshape to [Hidden, horizon, 4] to separate the time horizon.
                horizon = 8
                hidden = old_param.shape[0] # e.g. 256
                
                # Old: [256, 32] -> [256, 8, 4]
                old_reshaped = old_param.view(hidden, horizon, 4)
                # New: [256, 64] -> [256, 8, 8]
                new_reshaped = new_state_dict[name].view(hidden, horizon, 8)
                
                # Copy old position weights (channels 0-4)
                new_reshaped[:, :, :4] = old_reshaped
                
                # Zero out new velocity weights (channels 4-8)
                # This ensures the model ignores the new inputs initially.
                torch.nn.init.zeros_(new_reshaped[:, :, 4:])
                
                # Flatten back to [256, 64]
                new_state_dict[name] = new_reshaped.reshape(hidden, horizon * 8)

            # --- CASE 3: Output Embedding Weight ---
            elif "output_emb.weight" in name:
                print(f"SURGERY: Interleaving Output Weights for {name}")
                
                horizon = 8
                hidden = old_param.shape[1] # e.g. 256
                
                # Old: [32, 256] -> [8, 4, 256]
                old_reshaped = old_param.view(horizon, 4, hidden)
                # New: [64, 256] -> [8, 8, 256]
                new_reshaped = new_state_dict[name].view(horizon, 8, hidden)
                
                # Copy old position weights
                new_reshaped[:, :4, :] = old_reshaped
                
                # Initialize new velocity weights to near-zero
                torch.nn.init.normal_(new_reshaped[:, 4:, :], mean=0, std=1e-5)
                
                # Flatten back
                new_state_dict[name] = new_reshaped.reshape(horizon * 8, hidden)

            # --- CASE 4: Output Embedding Bias ---
            elif "output_emb.bias" in name:
                print(f"SURGERY: Interleaving Output Bias for {name}")
                
                horizon = 8
                
                # Old: [32] -> [8, 4]
                old_reshaped = old_param.view(horizon, 4)
                new_reshaped = new_state_dict[name].view(horizon, 8)
                
                new_reshaped[:, :4] = old_reshaped
                torch.nn.init.zeros_(new_reshaped[:, 4:])
                
                new_state_dict[name] = new_reshaped.reshape(horizon * 8)
                
            else:
                print(f"WARNING: Unhandled mismatch for {name}. Kept random init.")

        # 3. Load
        self.load_state_dict(new_state_dict, strict=True)
        print("Success! Interleaved surgery complete.")
    
    def load_pretrained_matching_weights(self, checkpoint_path):
        """
        Loads weights from checkpoint_path. 
        automatically DROPS any layers where the shape doesn't match the current model.
        """
        # if checkpoint_path is None:
        #     checkpoint_path = self._checkpoint_path
        
        # 1. Load checkpoint (CPU to avoid VRAM spike)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle Lightning wrapper
        if "state_dict" in checkpoint:
            ckpt_dict = checkpoint["state_dict"]
        else:
            ckpt_dict = checkpoint

        # Print first 3 keys of each
        print("--- Checkpoint Keys ---")
        print(list(ckpt_dict.keys())[:3])

        print("\n--- Model Keys ---")
        print(list(self.state_dict().keys())[:3])

        # Strip the ckpt from the "agent." prefix
        ckpt_dict = {k.replace("agent.", ""): v for k, v in ckpt_dict.items()}

        print("--- Checkpoint Keys after prefix removal ---")
        print(list(ckpt_dict.keys())[:3])

        # 2. Get current model structure
        model_dict = self.state_dict()
        
        # 3. Filter: Keep only keys that exist AND have matching shapes
        filtered_dict = {}
        dropped_keys = []
        
        for k, v in ckpt_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    dropped_keys.append(f"{k} (Shape mismatch: {v.shape} vs {model_dict[k].shape})")
            # else: key is in ckpt but not model (ignored by strict=False anyway)

        # 4. Load with strict=False
        self.load_state_dict(filtered_dict, strict=False)
        
        # 5. Report
        print(f"Loaded {len(filtered_dict)} layers.")
        if dropped_keys:
            print(f"Dropped {len(dropped_keys)} layers due to shape mismatch:")
            for k in dropped_keys:
                print(f" - {k}")
        else:
            print("No layers dropped!")

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if self._checkpoint_path is not None:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"]
            self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})
        else:
            print(f"No agent checkpoint path give, unattended weights will be initialized randomly")
            

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
        return [HydraTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [HydraFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        model_output = self.model(features)
        # Clip the vx, vy, ax, ay dimensions when returning a trajectory
        # TODO move this to a new "compute_trajectory" function rather than here, and move training to be done through this forward method
        model_output['trajectory'] = model_output['trajectory'][:, :, :3]
        return model_output

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
            tokens=None
    ):
        loss, loss_dict = dp_loss_bev(targets, predictions, self._config, self.model._trajectory_head)
        return loss

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        backbone_params_name = '_backbone.image_encoder'
        reference_transformer_name = '_trajectory_head.reference_transformer'
        ori_transformer_name = '_trajectory_head.ori_transformer'
        img_backbone_params = list(
            filter(lambda kv: backbone_params_name in kv[0], self.model.named_parameters()))
        default_params = list(filter(lambda kv:
                                     backbone_params_name not in kv[0] and
                                     reference_transformer_name not in kv[0] and
                                     ori_transformer_name not in kv[0], self.model.named_parameters()))
        params_lr_dict = [
            {'params': [tmp[1] for tmp in default_params]},
            {
                'params': [tmp[1] for tmp in img_backbone_params],
                'lr': self._lr * self._config.lr_mult_backbone,
                'weight_decay': self.backbone_wd
            }
        ]

        if self.scheduler == 'default':
            return torch.optim.Adam(params_lr_dict, lr=self._lr, weight_decay=self._config.weight_decay)
        elif self.scheduler == 'cycle':
            optim = torch.optim.Adam(params_lr_dict, lr=self._lr)
            return {
                "optimizer": optim,
                "lr_scheduler": OneCycleLR(
                    optim,
                    max_lr=0.01,
                    total_steps=100 * 202
                )
            }
        else:
            raise ValueError('Unsupported lr scheduler')

    # def get_training_callbacks(self) -> List[pl.Callback]:
    #     ckpt_callback = ModelCheckpoint(
    #         save_top_k=100,
    #         monitor="val/loss_epoch",
    #         mode="min",
    #         dirpath=f"{os.environ.get('NAVSIM_EXP_ROOT')}/{self._config.ckpt_path}/",
    #         filename="{epoch:02d}-{step:04d}",
    #     )
    #     return [
    #         ckpt_callback
    #     ]
