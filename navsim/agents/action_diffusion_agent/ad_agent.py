"""
ActionDiffusionAgent — NAVSIM AbstractAgent implementation.

Data flow
---------
features  →  ActionDiffusionModel.forward()
              → context_kv  (scene representation)
              → trajectory  (placeholder during training, real proposal at eval)

Training
--------
compute_loss() uses context_kv + interpolated_traj ground truth to compute
the DDPM epsilon-prediction MSE loss via ActionDiffusionHead.compute_loss().

Inference
---------
The model runs the full DDPM reverse chain, generates `num_inference_proposals`
trajectory candidates, and returns one randomly selected proposal.
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.action_diffusion_agent.ad_callback import ActionDiffusionCallback
from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig
from navsim.agents.action_diffusion_agent.ad_features import (
    ActionDiffusionFeatureBuilder,
    ActionDiffusionTargetBuilder,
)
from navsim.agents.action_diffusion_agent.ad_model import ActionDiffusionModel
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

if TYPE_CHECKING:
    from navsim.agents.gtrs_dense.gtrs_agent import GTRSAgent

logger = logging.getLogger(__name__)


class ActionDiffusionAgent(AbstractAgent):
    """
    Diffusion-policy-based autonomous driving agent for NAVSIM.

    Args:
        config: ActionDiffusionConfig dataclass.
        lr: Base learning rate for the optimizer.
        checkpoint_path: Path to a Lightning checkpoint used to restore the
            full agent state at inference time (via ``initialize()``).  This is
            *not* the pretrained backbone checkpoint — that goes inside
            ``config.pretrained_ckpt``.
        scheduler: Which learning-rate scheduler to use for training.
            Options: None (constant LR), 'cosine', 'step', 'cycle', 'plateau'.
        total_train_steps: Required when scheduler='cycle'.  Provides the total
            number of optimisation steps for OneCycleLR.
    """

    def __init__(
        self,
        config: ActionDiffusionConfig,
        lr: float = 1e-4,
        checkpoint_path: Optional[str] = None,
        scheduler: Optional[str] = None,
        total_train_steps: int = 30000,         # used for the cosine and cycle LR schedulers
    ) -> None:
        super().__init__(trajectory_sampling=config.trajectory_sampling)

        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._scheduler = scheduler
        self._total_train_steps = total_train_steps

        self.model = ActionDiffusionModel(config)
        self._hydra_scorer_agent: Optional["GTRSAgent"] = self._build_hydra_scorer(config)
        self._initialize_hydra_scorer()

    def _build_hydra_scorer(self, config: ActionDiffusionConfig) -> Optional["GTRSAgent"]:
        """Create scorer only when stepwise guidance is enabled."""
        if not config.use_hydra_diffusion_guidance:
            return None

        if not config.hydra_scorer_checkpoint_path:
            raise ValueError(
                "use_hydra_diffusion_guidance=True requires hydra_scorer_checkpoint_path"
            )
        if not config.hydra_vocab_path:
            raise ValueError(
                "use_hydra_diffusion_guidance=True requires hydra_vocab_path"
            )

        from navsim.agents.gtrs_dense.gtrs_agent import GTRSAgent
        from navsim.agents.gtrs_dense.hydra_config import HydraConfig

        vov_ckpt = config.hydra_vov_ckpt or config.vov_ckpt
        hydra_cfg = HydraConfig(
            vocab_path=config.hydra_vocab_path,
            vocab_size=config.hydra_vocab_size,
            normalize_vocab_pos=config.hydra_normalize_vocab_pos,
            backbone_type=config.hydra_backbone_type,
            fusion_layers=config.hydra_fusion_layers,
            use_back_view=False,  # Hydra scorer only uses front view for efficiency; no back-view doubling in token count
            camera_width=config.camera_width,
            camera_height=config.camera_height,
            img_vert_anchors=config.img_vert_anchors,
            img_horz_anchors=config.img_horz_anchors,
            vov_ckpt=vov_ckpt,
        )
        return GTRSAgent(
            config=hydra_cfg,
            lr=self._lr,
            checkpoint_path=config.hydra_scorer_checkpoint_path,
        )

    def _initialize_hydra_scorer(self) -> None:
        """Load scorer checkpoint and freeze it for inference-time guidance."""
        if self._hydra_scorer_agent is None:
            return

        logger.info("Loading Hydra scorer for diffusion guidance ...")
        self._hydra_scorer_agent.initialize()
        self._hydra_scorer_agent.eval()
        for param in self._hydra_scorer_agent.parameters():
            param.requires_grad = False
        logger.info("Hydra scorer loaded.")

    @staticmethod
    def _prepare_scorer_features(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Hydra scorer expects rear camera tensor as (B, 3, H, W) at inference."""
        scorer_features = dict(features)
        if "camera_feature_back" in scorer_features:
            back = scorer_features["camera_feature_back"]
            if isinstance(back, torch.Tensor) and back.ndim == 5:
                scorer_features["camera_feature_back"] = back[:, -1]
        return scorer_features

    def _build_guidance_fn(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Optional[Callable[[torch.Tensor, int, int], torch.Tensor]]:
        """Return scorer callback: dense traj (B*N, 40, 3) -> score (B*N,)."""
        if self._hydra_scorer_agent is None:
            return None

        scorer_features = self._prepare_scorer_features(features)

        def _guidance_fn(dense_traj: torch.Tensor, B: int, N: int) -> torch.Tensor:
            dp_proposals = dense_traj.view(B, N, dense_traj.shape[1], dense_traj.shape[2])
            out = self._hydra_scorer_agent.evaluate_dp_proposals(
                scorer_features,
                dp_proposals,
                dp_only_inference=True,
                topk=1,
            )
            scores = out["overall_log_scores"].reshape(B * N)
            return scores
            # Optional guards against inf/nan values
            # return torch.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)

        return _guidance_fn

    # ──────────────────────────────────────────────── AbstractAgent API ───────

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        """
        Load the full agent state from a Lightning checkpoint.

        This is called during *inference* (e.g. evaluation / submission).
        It restores all model weights, including the diffusion head, from a
        previously saved training run.
        """
        if self._checkpoint_path is None:
            logger.warning(
                "ActionDiffusionAgent.initialize() called but checkpoint_path is None."
                " Skipping weight loading."
            )
            return

        logger.info(f"Loading ActionDiffusionAgent from {self._checkpoint_path} …")
        ckpt: Dict[str, Any] = torch.load(
            self._checkpoint_path, map_location="cpu", weights_only=False
        )
        state_dict: Dict[str, torch.Tensor] = ckpt.get("state_dict", ckpt)

        # Lightning prefixes the agent's parameters with "agent."
        cleaned = {
            k.replace("agent.", "", 1): v
            for k, v in state_dict.items()
        }
        msg = self.load_state_dict(cleaned, strict=False)
        logger.info(
            f"Loaded.  Missing: {len(msg.missing_keys)}, "
            f"Unexpected: {len(msg.unexpected_keys)}"
        )

    def get_sensor_config(self) -> SensorConfig:
        """
        Request all cameras needed for the stitched panoramic views.

        We always request 4 history frames (indices 0–3); the feature builder
        slices the most recent `seq_len` ones itself.  Only the cameras used
        for stitching are requested; unused cameras (l1, r1) are set to an
        empty list so the dataloader skips them.
        """
        all_frames = [0, 1, 2, 3]
        empty = []

        # Rear cameras are only needed when use_back_view=True
        back = all_frames if self._config.use_back_view else empty

        return SensorConfig(
            cam_f0=all_frames,
            cam_l0=all_frames,
            cam_l1=empty,           # not used in stitching
            cam_l2=back,
            cam_r0=all_frames,
            cam_r1=empty,           # not used in stitching
            cam_r2=back,
            cam_b0=back,
            lidar_pc=empty,         # this agent uses cameras only
        )

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ActionDiffusionFeatureBuilder(config=self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [ActionDiffusionTargetBuilder(config=self._config)]

    # ───────────────────────────────────────────────────────── forward ────────

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        guidance_fn = None
        if not self.training:
            guidance_fn = self._build_guidance_fn(features)
        return self.model(features, scorer_guidance_fn=guidance_fn)

    # ────────────────────────────────────────────────────────── loss ──────────

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Diffusion denoising loss — dispatches to DDPM epsilon-prediction MSE
        or flow-matching velocity-prediction MSE based on ``config.noise_type``.

        The scene context KV (`predictions['context_kv']`) is produced by the
        model forward pass; the dense 40-step ground-truth trajectory is taken
        from `targets['interpolated_traj']`.
        """
        context_kv = predictions["context_kv"]
        gt_traj_dense = targets["interpolated_traj"]

        raw_loss = self.model.diffusion_head.compute_loss(
            context=context_kv,
            gt_traj_dense=gt_traj_dense,
            features=features,
        )
        return raw_loss * self._config.dp_loss_weight

    # ─────────────────────────────────────────────────────── optimisers ───────

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Any]]:
        """
        Returns an Adam optimizer and optionally a learning-rate scheduler.

        Parameter groups
        ----------------
        If the backbone is *not* frozen, a lower learning rate is applied to
        the backbone only; all other parameters (including projection layers)
        use the full LR.  When the backbone is frozen, only the trainable
        parameters are included.
        """
        cfg = self._config

        if cfg.freeze_backbone:
            # Only include parameters marked as requiring gradients.
            params = [p for p in self.model.parameters() if p.requires_grad]
            param_groups = [{"params": params}]
        else:
            # Two groups: backbone (lower LR) vs all other params (full LR).
            backbone_ids = {id(p) for p in self.model.backbone.parameters()}
            backbone_params = [
                p for p in self.model.parameters() if id(p) in backbone_ids
            ]
            other_params = [
                p for p in self.model.parameters() if id(p) not in backbone_ids
            ]
            param_groups = [
                {"params": other_params},
                {
                    "params": backbone_params,
                    "lr": self._lr * 0.1,          # backbone gets 10× lower LR
                    "weight_decay": 0.0,
                },
            ]

        optimizer = torch.optim.Adam(
            param_groups,
            lr=self._lr,
            weight_decay=cfg.weight_decay,
        )

        if self._scheduler is None or self._scheduler == "default":
            return optimizer

        # ── Learning-rate schedulers ──────────────────────────────────────────
        if self._scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self._total_train_steps, eta_min=1e-7
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }

        if self._scheduler == "step":
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
            }

        if self._scheduler == "cycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self._lr * 10,
                total_steps=self._total_train_steps,
                pct_start=0.1,
                div_factor=10,
                final_div_factor=1000,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }

        if self._scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val/loss_epoch",
                    "frequency": 1,
                },
            }

        raise ValueError(
            f"Unknown scheduler={self._scheduler!r}. "
            "Valid options: None, 'cosine', 'step', 'cycle', 'plateau'."
        )

    # ─────────────────────────────────────────────── training callbacks ───────

    def get_training_callbacks(self) -> List[pl.Callback]:
        ckpt_callback = ModelCheckpoint(
            save_top_k=3,
            monitor="val/loss_epoch",
            mode="min",
            dirpath=None,
            filename="epoch_{epoch:03d}-val_loss_{val/loss_epoch:.4f}",
            save_last=True,
            auto_insert_metric_name=False,
        )
        viz_callback = ActionDiffusionCallback(config=self._config)
        return [ckpt_callback, viz_callback]
