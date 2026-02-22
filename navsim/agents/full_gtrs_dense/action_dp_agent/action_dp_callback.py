# import matplotlib.pyplot as plt
# import pytorch_lightning as pl
# import torch

# from navsim.agents.full_gtrs_dense.action_dp_agent.dp_config import DPConfig

# from navsim.visualization.config import AGENT_CONFIG, MAP_LAYER_CONFIG
# from nuplan.common.maps.abstract_map import SemanticMapLayer
# from PIL import ImageColor

# class ActionDPCallbacks(pl.Callback):
#     def __init__(self, config: DPConfig, num_samples=4):
#         super().__init__()
#         self.num_samples = num_samples
#         self._config = config

#     def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
#         # 1. Get a single batch from validation loader
#         try:
#             val_loader = trainer.val_dataloaders
#             if isinstance(val_loader, list): val_loader = val_loader[0]
#             batch = next(iter(val_loader))
#         except (StopIteration, TypeError):
#             return # Handle empty loaders gracefully

#         features, targets = batch
        
#         if trainer.current_epoch == 0:
#             self._log_reference_maps(trainer, targets)

#         # 2. Run Inference
#         device = pl_module.device
#         features = {k: v.to(device) for k, v in features.items()}
        
#         pl_module.eval()
#         with torch.no_grad():
#             predictions = pl_module.agent.forward(features)

#         # 3. Prepare Data (CPU/Numpy)
#         # GT: (B, 8, 3) -> [x, y, heading]
#         gt_traj = targets['trajectory'].cpu().numpy()
#         # Pred: (B, 8, 3) -> [x, y, heading]
#         pred_traj = predictions['dp_pred'].detach().cpu().numpy()

#         # 4. Plot
#         fig = self._plot_trajectories(gt_traj, pred_traj)
        
#         # 5. Log to TensorBoard
#         if trainer.logger:
#             trainer.logger.experiment.add_figure("Val_Preds", fig, global_step=trainer.current_epoch)
#         plt.close(fig)

#     def _plot_trajectories(self, gt_batch, pred_batch):
#         count = min(self.num_samples, gt_batch.shape[0])
        
#         # FIX 1: Adjust figsize to be taller (3x5 ratio) to match the vertical road aspect
#         fig, axes = plt.subplots(1, count, figsize=(3 * count, 5))
        
#         # Handle the edge case where count=1 (axes is not a list)
#         if count == 1: axes = [axes]

#         for i in range(count):
#             ax = axes[i]
#             gt = gt_batch[i]       # (8, 3)
#             preds = pred_batch[i]  # (NUM_PROP, 8, 3)

#             # 1. Plot Predictions (Red, dashed, transparent)
#             # We plot them first so GT sits on top
#             for k, pred in enumerate(preds):
#                 # Only label the first one to avoid "Pred, Pred, Pred..." in legend
#                 label = 'Pred' if k == 0 else None
#                 ax.plot(pred[:, 1], pred[:, 0], color='red', linestyle='--', 
#                         linewidth=1, alpha=0.5, label=label)
            
#             # 2. Plot Ground Truth (Green, solid, thick)
#             ax.plot(gt[:, 1], gt[:, 0], color='green', linewidth=3, label='GT')
            
#             # 3. Plot Ego Start (Black Triangle)
#             ax.plot(0, 0, marker='^', color='black', markersize=8, zorder=10)

#             # Formatting
#             ax.set_title(f"Sample {i}")
#             ax.set_aspect('equal') # This is the culprit, but we need it for physics!
#             ax.grid(True, alpha=0.3)
            
#             # Limits
#             ax.set_xlim(-20, 20)  # Width = 40
#             ax.set_ylim(-10, 50)  # Height = 60 (Increased slightly to give breathing room)
            
#             if i == 0: 
#                 ax.legend(loc='upper right')

#         # FIX 2: Use rect to prevent title clipping
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
#         return fig

#     def semantic_map_to_rgb(semantic_map: npt.NDArray[np.int64], config: TransfuserConfig) -> npt.NDArray[np.uint8]:
#         """
#         Convert semantic map to RGB image.
#         :param semantic_map: numpy array of segmentation map (multi-channel)
#         :param config: global config dataclass of TransFuser
#         :return: RGB image as numpy array
#         """

#         height, width = semantic_map.shape[:2]
#         rgb_map = np.ones((height, width, 3), dtype=np.uint8) * 255

#         for label in range(1, config.num_bev_classes):

#             if config.bev_semantic_classes[label][0] == "linestring":
#                 hex_color = MAP_LAYER_CONFIG[SemanticMapLayer.BASELINE_PATHS]["line_color"]
#             else:
#                 layer = config.bev_semantic_classes[label][-1][0]  # take color of first element
#                 hex_color = (
#                     AGENT_CONFIG[layer]["fill_color"]
#                     if layer in AGENT_CONFIG.keys()
#                     else MAP_LAYER_CONFIG[layer]["fill_color"]
#                 )

#             rgb_map[semantic_map == label] = ImageColor.getcolor(hex_color, "RGB")
#         return rgb_map[::-1, ::-1]

#     def _log_reference_maps(self, trainer, targets):
#         """
#         Logs GT Semantic Map + GT Trajectory using Config-defined colors and geometry.
#         """
#         bev_maps = targets['bev_semantic_map'].detach().cpu().numpy() # (B, H, W)

#         count = min(self.num_samples, bev_maps.shape[0])
#         cols = []
        
#         for i in range(count):
#             # A. Decode Map Colors using Config
#             rgb_map = self.semantic_map_to_rgb(bev_maps[i])
#             cols.append(rgb_map)

#         # Concatenate and Log
#         grid_image = np.concatenate(cols, axis=1)
#         grid_tensor = torch.tensor(grid_image).permute(2, 0, 1) # (C, H, W)
#         trainer.logger.experiment.add_image("Reference/GT_Semantic_Map", grid_tensor, global_step=0)


import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import numpy as np  # <--- FIXED: Added missing import
import numpy.typing as npt # <--- FIXED: Added missing import

from navsim.agents.full_gtrs_dense.action_dp_agent.dp_config import DPConfig
from navsim.visualization.config import AGENT_CONFIG, MAP_LAYER_CONFIG
from nuplan.common.maps.abstract_map import SemanticMapLayer
from PIL import ImageColor

class ActionDPCallbacks(pl.Callback):
    def __init__(self, config: DPConfig, num_samples=4):
        super().__init__()
        self.num_samples = num_samples
        self._config = config

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # 1. Get a single batch from validation loader
        try:
            val_loader = trainer.val_dataloaders
            if isinstance(val_loader, list): val_loader = val_loader[0]
            batch = next(iter(val_loader))
        except (StopIteration, TypeError):
            return 

        features, targets = batch
        
        # 2. Log Reference Map (Once at Epoch 0)
        # Note: This might run twice (Sanity Check + Real Epoch 0), but that is fine.
        if trainer.current_epoch == 0:
            self._log_reference_maps(trainer, targets)

        # 3. Run Inference
        device = pl_module.device
        features = {k: v.to(device) for k, v in features.items()}
        
        pl_module.eval()
        with torch.no_grad():
            predictions = pl_module.agent.forward(features)

        # 4. Prepare Data
        gt_traj = targets['trajectory'].cpu().numpy()
        # Ensure we detach predictions
        pred_traj = predictions['dp_pred'].detach().cpu().numpy()

        # 5. Plot Trajectories (Predictions vs GT)
        fig = self._plot_trajectories(gt_traj, pred_traj)
        
        if trainer.logger:
            trainer.logger.experiment.add_figure("Val_Preds", fig, global_step=trainer.current_epoch)
        plt.close(fig)

    def _plot_trajectories(self, gt_batch, pred_batch):
        count = min(self.num_samples, gt_batch.shape[0])
        fig, axes = plt.subplots(1, count, figsize=(3 * count, 5))
        if count == 1: axes = [axes]

        for i in range(count):
            ax = axes[i]
            gt = gt_batch[i]
            preds = pred_batch[i]

            for k, pred in enumerate(preds):
                label = 'Pred' if k == 0 else None
                ax.plot(pred[:, 1], pred[:, 0], color='red', linestyle='--', 
                        linewidth=1, alpha=0.5, label=label)
            
            ax.plot(gt[:, 1], gt[:, 0], color='green', linewidth=3, label='GT')
            ax.plot(0, 0, marker='^', color='black', markersize=8, zorder=10)

            ax.set_title(f"Sample {i}")
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-20, 20)
            ax.set_ylim(-10, 50)
            
            if i == 0: ax.legend(loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        return fig

    # --- FIXED METHOD SIGNATURE ---
    # 1. Added 'self'
    # 2. Removed 'config' arg (we use self._config)
    def semantic_map_to_rgb(self, semantic_map: npt.NDArray[np.int64]) -> npt.NDArray[np.uint8]:
        height, width = semantic_map.shape[:2]
        rgb_map = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Loop through classes defined in Config
        for label in range(1, self._config.num_bev_classes):
            # Check if this label is in our config mapping
            if label not in self._config.bev_semantic_classes:
                continue

            class_def = self._config.bev_semantic_classes[label]
            class_type = class_def[0]
            
            # FIXED: Handle list access carefully
            if class_type == "linestring":
                hex_color = MAP_LAYER_CONFIG[SemanticMapLayer.BASELINE_PATHS]["line_color"]
            else:
                # The config structure is ("box", [LayerEnum, ...])
                layers = class_def[-1]
                layer = layers[0] # Take color of first element
                
                if layer in AGENT_CONFIG:
                    hex_color = AGENT_CONFIG[layer]["fill_color"]
                elif layer in MAP_LAYER_CONFIG:
                    hex_color = MAP_LAYER_CONFIG[layer]["fill_color"]
                else:
                    hex_color = "#FFFFFF"

            rgb_map[semantic_map == label] = ImageColor.getcolor(hex_color, "RGB")
        
        # TransFuser flips image; keep this if your map looks upside down
        return rgb_map[::-1, ::-1]

    def _log_reference_maps(self, trainer, targets):
        bev_maps = targets['bev_semantic_map'].detach().cpu().numpy() # (B, H, W)

        count = min(self.num_samples, bev_maps.shape[0])
        cols = []
        
        for i in range(count):
            # FIXED: Call method correctly using self
            rgb_map = self.semantic_map_to_rgb(bev_maps[i])
            cols.append(rgb_map)

        grid_image = np.concatenate(cols, axis=1)
        # Convert HWC -> CHW for TensorBoard
        grid_tensor = torch.tensor(grid_image).permute(2, 0, 1) 
        trainer.logger.experiment.add_image("Reference/GT_Semantic_Map", grid_tensor, global_step=0)