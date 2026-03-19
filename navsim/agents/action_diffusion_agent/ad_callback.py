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

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

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
        uncertainty_viz_mode: str = "gaussian",
    ) -> None:
        super().__init__()
        # Use a non-interactive backend only for training callback execution.
        # Keeping this here avoids changing matplotlib backend at import time.
        plt.switch_backend("Agg")
        self._config = config
        self.num_samples = num_samples
        supported_modes = {"gaussian", "kde", "boxplot"}
        mode = uncertainty_viz_mode.lower()
        if mode not in supported_modes:
            raise ValueError(
                f"Unsupported uncertainty_viz_mode='{uncertainty_viz_mode}'. "
                f"Expected one of {sorted(supported_modes)}."
            )
        self.uncertainty_viz_mode = mode

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
        Create per-sample visualizations of trajectory proposals and uncertainty.

        Top row (per sample) shows:
          - All N proposal trajectories (red, semi-transparent dashed lines).
          - The ground-truth trajectory (green, solid thick line).
          - A black triangle at the ego origin (0, 0).

        Bottom row (per sample) shows:
          - Configurable uncertainty view (gaussian, kde, or boxplot).
          - Default gaussian mode shows 1-sigma / 2-sigma confidence ellipses
            in trajectory space for a pyramid-like uncertainty layout.

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
        fig, axes = plt.subplots(
            2,
            count,
            figsize=(5.2 * count, 10.0),
            squeeze=False,
            gridspec_kw={"height_ratios": [3.0, 2.0]},
        )

        for i in range(count):
            ax = axes[0, i]
            ax_unc = axes[1, i]
            gt    = gt_batch[i]    # (8, 3)
            preds = pred_batch[i]  # (N, 8, 3)
            num_waypoints = preds.shape[1]

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

            if self.uncertainty_viz_mode == "gaussian":
                self._plot_uncertainty_gaussian(ax_unc=ax_unc, gt=gt, preds=preds)
            elif self.uncertainty_viz_mode == "kde":
                self._plot_uncertainty_kde(ax_unc=ax_unc, gt=gt, preds=preds)
            else:
                self._plot_uncertainty_boxplot(ax_unc=ax_unc, gt=gt, preds=preds)

        plt.suptitle(
            f"Epoch {epoch} — trajectory proposals and uncertainty ({self.uncertainty_viz_mode})",
            fontsize=10,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        return fig

    def _plot_uncertainty_gaussian(
        self, ax_unc: plt.Axes, gt: np.ndarray, preds: np.ndarray
    ) -> None:
        """Render 2D waypoint uncertainty using 1-sigma covariance ellipses."""
        num_waypoints = preds.shape[1]
        cmap = plt.get_cmap("turbo")

        for t in range(num_waypoints):
            color = cmap((t + 1) / max(1, num_waypoints))
            pts = np.stack([preds[:, t, 1], preds[:, t, 0]], axis=1)
            self._draw_covariance_ellipse(
                ax=ax_unc,
                points=pts,
                n_std=1.0,
                edgecolor=color,
                facecolor=color,
                alpha=0.35,
                linewidth=1.8,
            )

        ax_unc.plot(
            gt[:, 1],
            gt[:, 0],
            color="black",
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="GT trajectory",
            zorder=5,
        )

        for t in range(num_waypoints):
            ax_unc.text(
                gt[t, 1] + 0.12,
                gt[t, 0] + 0.20,
                str(t + 1),
                fontsize=7,
                color="#222222",
                alpha=0.85,
                zorder=6,
            )

        ax_unc.plot(0, 0, marker="^", color="black", markersize=8, zorder=10)
        ax_unc.set_title("2D uncertainty pyramid (1σ)", fontsize=9)
        ax_unc.set_aspect("equal")
        ax_unc.grid(True, alpha=0.25)
        ax_unc.set_xlabel("Lateral (m)")
        ax_unc.set_ylabel("Longitudinal (m)")
        self._set_uncertainty_axis_limits(ax_unc=ax_unc, gt=gt, preds=preds)

    def _plot_uncertainty_kde(
        self, ax_unc: plt.Axes, gt: np.ndarray, preds: np.ndarray
    ) -> None:
        """Render per-waypoint 2D density contours using hist2d bins."""
        num_waypoints = preds.shape[1]
        cmap = plt.get_cmap("turbo")

        for t in range(num_waypoints):
            color = cmap((t + 1) / max(1, num_waypoints))
            lat = preds[:, t, 1]
            lon = preds[:, t, 0]
            # Coarse density estimate in trajectory space (fast and dependency-free).
            hist, x_edges, y_edges = np.histogram2d(lat, lon, bins=20)
            if hist.max() <= 0:
                continue
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            levels = np.linspace(hist.max() * 0.30, hist.max() * 0.90, 3)
            ax_unc.contour(
                x_centers,
                y_centers,
                hist.T,
                levels=levels,
                colors=[color],
                linewidths=1.3,
                alpha=0.9,
            )

        ax_unc.plot(
            gt[:, 1],
            gt[:, 0],
            color="black",
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="GT trajectory",
            zorder=5,
        )
        ax_unc.plot(0, 0, marker="^", color="black", markersize=8, zorder=10)
        ax_unc.set_title("2D uncertainty contours", fontsize=9)
        ax_unc.set_aspect("equal")
        ax_unc.grid(True, alpha=0.25)
        ax_unc.set_xlabel("Lateral (m)")
        ax_unc.set_ylabel("Longitudinal (m)")
        self._set_uncertainty_axis_limits(ax_unc=ax_unc, gt=gt, preds=preds)

    def _plot_uncertainty_boxplot(
        self, ax_unc: plt.Axes, gt: np.ndarray, preds: np.ndarray
    ) -> None:
        """Render 1D waypoint-wise boxplots (longitudinal/lateral)."""
        num_waypoints = preds.shape[1]
        waypoint_ids = np.arange(1, num_waypoints + 1)
        long_data = [preds[:, t, 0] for t in range(num_waypoints)]
        lat_data = [preds[:, t, 1] for t in range(num_waypoints)]

        long_box = ax_unc.boxplot(
            long_data,
            positions=waypoint_ids - 0.18,
            widths=0.32,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#0B3D91", "linewidth": 1.2},
        )
        lat_box = ax_unc.boxplot(
            lat_data,
            positions=waypoint_ids + 0.18,
            widths=0.32,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#A34A00", "linewidth": 1.2},
        )

        for patch in long_box["boxes"]:
            patch.set_facecolor("#7AA6E3")
            patch.set_alpha(0.70)
        for patch in lat_box["boxes"]:
            patch.set_facecolor("#F2A65A")
            patch.set_alpha(0.70)

        ax_unc.plot(
            waypoint_ids - 0.18,
            gt[:num_waypoints, 0],
            color="#0B3D91",
            marker="o",
            markersize=3,
            linewidth=1.0,
            label="GT longitudinal",
        )
        ax_unc.plot(
            waypoint_ids + 0.18,
            gt[:num_waypoints, 1],
            color="#A34A00",
            marker="o",
            markersize=3,
            linewidth=1.0,
            label="GT lateral",
        )

        ax_unc.set_title("Waypoint spread (box + whiskers)", fontsize=9)
        ax_unc.set_xlabel("Waypoint index")
        ax_unc.set_ylabel("Coordinate value (m)")
        ax_unc.set_xticks(waypoint_ids)
        ax_unc.grid(True, alpha=0.25)

    @staticmethod
    def _set_uncertainty_axis_limits(
        ax_unc: plt.Axes, gt: np.ndarray, preds: np.ndarray
    ) -> None:
        """Fit axis limits to trajectory + proposal cloud with padding."""
        lat_vals = np.concatenate([preds[:, :, 1].reshape(-1), gt[:, 1]])
        lon_vals = np.concatenate([preds[:, :, 0].reshape(-1), gt[:, 0]])

        x_min, x_max = float(np.min(lat_vals)), float(np.max(lat_vals))
        y_min, y_max = float(np.min(lon_vals)), float(np.max(lon_vals))
        x_pad = max(2.0, 0.15 * (x_max - x_min + 1e-6))
        y_pad = max(2.0, 0.15 * (y_max - y_min + 1e-6))

        ax_unc.set_xlim(x_min - x_pad, x_max + x_pad)
        ax_unc.set_ylim(min(-2.0, y_min - y_pad), y_max + y_pad)

    @staticmethod
    def _draw_covariance_ellipse(
        ax: plt.Axes,
        points: np.ndarray,
        n_std: float,
        edgecolor,
        facecolor,
        alpha: float,
        linewidth: float,
    ) -> None:
        """Draw an n-sigma covariance ellipse for 2D points."""
        if points.shape[0] < 2:
            return

        mean = points.mean(axis=0)
        cov = np.cov(points, rowvar=False)
        # Stabilize near-singular covariance estimates.
        cov = cov + 1e-6 * np.eye(2)

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        width = 2.0 * n_std * np.sqrt(max(eigvals[0], 1e-9))
        height = 2.0 * n_std * np.sqrt(max(eigvals[1], 1e-9))
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        style = "--" if n_std > 1.0 else "-"
        ellipse = Ellipse(
            xy=(mean[0], mean[1]),
            width=width,
            height=height,
            angle=angle,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=style,
            zorder=2,
        )
        ax.add_patch(ellipse)
