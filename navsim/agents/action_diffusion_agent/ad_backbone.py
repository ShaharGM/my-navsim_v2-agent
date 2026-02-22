"""
PerceptionBackbone — a pluggable visual backbone that maps a single RGB image
to a flat sequence of feature tokens.

Supported backends
------------------
backbone_type='vov'
    VoVNet V-99-eSE (the same backbone used in the GTRS / HydraBackboneBEV
    pipeline).  Requires a pretrained checkpoint; set `vov_ckpt` in the
    config accordingly.

backbone_type='timm'
    Any model registered in the `timm` library (e.g. "resnet50",
    "convnext_small", "vit_base_patch16_224").  ImageNet weights are
    fetched automatically when `timm_pretrained=True`.

In both cases the last feature map is passed through an AdaptiveAvgPool2d
that collapses it to a fixed (img_vert_anchors, img_horz_anchors) grid,
then flattened and transposed to produce an (N, C) token sequence.
"""

import torch
import torch.nn as nn

from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig


class PerceptionBackbone(nn.Module):
    """
    Visual backbone + spatial pooling.

    Input  : (B, 3, H, W) — a single RGB image (already resized / stitched)
    Output : (B, N, C)    — N = img_vert_anchors × img_horz_anchors tokens
                            C = backbone output channel count

    Example
    -------
    >>> cfg = ActionDiffusionConfig(backbone_type="timm", timm_model_name="resnet50")
    >>> bb = PerceptionBackbone(cfg)
    >>> tokens = bb(torch.randn(2, 3, 512, 2048))
    >>> tokens.shape  # (2, 16*64, 2048)
    """

    def __init__(self, config: ActionDiffusionConfig) -> None:
        super().__init__()
        self.config = config

        # ── Build encoder ────────────────────────────────────────────────────
        if config.backbone_type == "vov":
            from navsim.agents.backbones.vov import VoVNet

            self._encoder = VoVNet(
                spec_name="V-99-eSE",
                out_features=["stage4", "stage5"],
                norm_eval=True,
                with_cp=True,
                init_cfg=dict(
                    type="Pretrained",
                    checkpoint=config.vov_ckpt,
                    prefix="img_backbone.",
                ),
            )
            self._encoder.init_weights()
            self._out_channels: int = 1024  # V-99-eSE stage5 output channels

        elif config.backbone_type == "timm":
            import timm

            self._encoder = timm.create_model(
                config.timm_model_name,
                pretrained=config.timm_pretrained,
                features_only=True,
                out_indices=(-1,),  # only the deepest / richest feature map
            )
            # Probe output channel count once on CPU (no GPU memory consumed).
            with torch.no_grad():
                probe = torch.zeros(1, 3, config.camera_height, config.camera_width)
                out_feats = self._encoder(probe)
                self._out_channels = out_feats[-1].shape[1]

        else:
            raise ValueError(
                f"Unsupported backbone_type={config.backbone_type!r}. "
                "Valid options: 'vov', 'timm'."
            )

        # ── Adaptive spatial pool → fixed token grid ─────────────────────────
        self._pool = nn.AdaptiveAvgPool2d(
            (config.img_vert_anchors, config.img_horz_anchors)
        )

    # ─────────────────────────────────────────────────────── properties ──────

    @property
    def out_channels(self) -> int:
        """Channel dimension of each output token."""
        return self._out_channels

    @property
    def num_tokens(self) -> int:
        """Number of spatial tokens produced per image."""
        return self.config.img_vert_anchors * self.config.img_horz_anchors

    # ─────────────────────────────────────────────────────── forward ─────────

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W)

        Returns:
            tokens: (B, N, C)  where N = img_vert_anchors × img_horz_anchors
        """
        feat = self._encoder(image)[-1]

        pooled = self._pool(feat)              # (B, C, V, H)
        B, C, V, H = pooled.shape
        tokens = pooled.flatten(2).permute(0, 2, 1).contiguous()  # (B, V*H, C)
        return tokens
