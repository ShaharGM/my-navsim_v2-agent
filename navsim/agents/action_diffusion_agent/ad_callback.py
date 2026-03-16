"""
ActionDiffusionCallback — PyTorch Lightning callback for training visualisation.

At the end of each validation epoch this callback:
  1. Grabs one batch from the validation dataloader.
  2. Runs inference (DDPM reverse process) to generate trajectory proposals.
  3. Plots all proposals (red, dashed) against the ground-truth (green) for a
     small number of samples.
  4. Logs the figure to TensorBoard under "Val/Trajectories".

The callback is intentionally lightweight and does *not* depend on BEV maps
or any components outside of the core agent.
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig

logger = logging.getLogger(__name__)


class ActionDiffusionCallback(pl.Callback):
    """
    Visualisation callback that logs trajectory prediction plots to TensorBoard.

    Args:
        config:      ActionDiffusionConfig for the current run.
        num_samples: Number of scenes to plot per validation epoch.
    """

    def __init__(
        self,
        config: ActionDiffusionConfig,
        num_samples: int = 4,
    ) -> None:
        super().__init__()
        # Use a non-interactive backend only for training callback execution.
        # Keeping this here avoids changing matplotlib backend at import time.
        plt.switch_backend("Agg")
        self._config = config
        self.num_samples = num_samples

    # ─────────────────────────────────────────────────── hook ────────────────

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Fired after every validation epoch.  Generates a trajectory comparison
        figure and logs it to the TensorBoard experiment logger (if available).
        """
        # Guard: no logger, nothing to do
        if trainer.logger is None:
            return

        # --- Grab a validation batch -----------------------------------------
        try:
            val_loader = trainer.val_dataloaders
            if isinstance(val_loader, list):
                val_loader = val_loader[0]
            batch = next(iter(val_loader))
        except (StopIteration, TypeError, AttributeError):
            logger.warning("ActionDiffusionCallback: could not fetch a validation batch.")
            return

        features, targets = batch

        # --- Inference ---------------------------------------------------------
        device = pl_module.device
        features = {k: v.to(device) for k, v in features.items()}

        pl_module.eval()
        with torch.no_grad():
            predictions = pl_module.agent.forward(features)

        # --- Extract tensors --------------------------------------------------
        # GT sparse trajectory: (B, 8, 3)
        gt_traj: np.ndarray = targets["trajectory"].cpu().numpy()

        if "dp_pred" not in predictions:
            # Model was still in an unexpected training mode — skip
            return

        # Predicted proposals: (B, N, 8, 3)
        pred_traj: np.ndarray = predictions["dp_pred"].detach().cpu().numpy()

        # --- Plot & log -------------------------------------------------------
        epoch = trainer.current_epoch
        fig = self._plot_trajectories(gt_traj, pred_traj, epoch=epoch)
        trainer.logger.experiment.add_figure(
            "Val/Trajectories", fig, global_step=epoch
        )
        plt.close(fig)

    # ─────────────────────────────────────────────── plotting helper ──────────

    def _plot_trajectories(
        self, gt_batch: np.ndarray, pred_batch: np.ndarray, epoch: int = 0
    ) -> plt.Figure:
        """
        Create a side-by-side plot of trajectory proposals vs ground truth.

        Each sub-plot shows:
          • All N proposal trajectories (red, semi-transparent dashed lines).
          • The ground-truth trajectory (green, solid thick line).
          • A black triangle at the ego origin (0, 0).

        The x/y axes correspond to the ego-relative coordinate system
        (x = forward, y = lateral), but we plot y on the horizontal axis and
        x on the vertical axis so "forward" is up — the conventional bird's eye
        view orientation.

        Args:
            gt_batch:   (B, 8, 3) ground-truth trajectories
            pred_batch: (B, N, 8, 3) predicted trajectory proposals

        Returns:
            matplotlib Figure
        """
        count = min(self.num_samples, gt_batch.shape[0])
        fig, axes = plt.subplots(1, count, figsize=(3 * count, 5))
        if count == 1:
            axes = [axes]

        for i in range(count):
            ax = axes[i]
            gt    = gt_batch[i]    # (8, 3)
            preds = pred_batch[i]  # (N, 8, 3)

            # Draw all proposals
            for k, pred in enumerate(preds):
                ax.plot(
                    pred[:, 1], pred[:, 0],   # lateral vs forward
                    color="red",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.4,
                    label="Proposals" if k == 0 else None,
                )

            # Draw ground truth
            ax.plot(
                gt[:, 1], gt[:, 0],
                color="green",
                linewidth=2.5,
                label="GT",
            )

            # Ego origin
            ax.plot(0, 0, marker="^", color="black", markersize=8, zorder=10)

            ax.set_title(f"Sample {i}", fontsize=9)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-20, 20)
            ax.set_ylim(-10, 50)
            ax.set_xlabel("Lateral (m)")
            ax.set_ylabel("Longitudinal (m)")

            if i == 0:
                ax.legend(loc="upper right", fontsize=7)

        plt.suptitle(
            f"Epoch {epoch} — trajectory proposals vs GT",
            fontsize=10,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
