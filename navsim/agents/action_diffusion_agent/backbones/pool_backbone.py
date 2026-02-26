"""
PoolBackbone — perception backbone that maps RGB images to a flat token sequence
via a CNN encoder + AdaptiveAvgPool2d spatial pooling.

Supported encoder types (set via config.backbone_type):
    'timm'  →  any model from the timm library (e.g. "resnet50", "convnext_small")
    'vov'   →  VoVNet V-99-eSE (same as used in GTRS); requires config.vov_ckpt

Token layout:
    use_back_view=False  →  N = img_vert_anchors × img_horz_anchors
    use_back_view=True   →  N = 2 × img_vert_anchors × img_horz_anchors
                            (front tokens concatenated with back tokens)
"""

from typing import Dict

import torch
import torch.nn as nn

from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig
from navsim.agents.action_diffusion_agent.backbones.base import BackboneBase


class PoolBackbone(BackboneBase):
    """
    Single CNN encoder with adaptive spatial pooling.

    Reads "camera_feature" (and "camera_feature_back" when use_back_view=True)
    from the features dict, runs each through the encoder + pool, and
    concatenates the resulting token sequences.

    Input keys:
        "camera_feature"       (B, T, 3, H, W)
        "camera_feature_back"  (B, T, 3, H, W)  — only when use_back_view=True

    Output: (B, N, C)
        N = img_vert_anchors × img_horz_anchors  [× 2 if use_back_view]
        C = encoder output channels
    """

    def __init__(self, config: ActionDiffusionConfig) -> None:
        super().__init__()
        self.config = config

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
            self._out_channels: int = 1024

        elif config.backbone_type == "timm":
            import timm
            self._encoder = timm.create_model(
                config.timm_model_name,
                pretrained=config.timm_pretrained,
                features_only=True,
                out_indices=(-1,),
            )
            with torch.no_grad():
                probe = torch.zeros(1, 3, config.camera_height, config.camera_width)
                self._out_channels = self._encoder(probe)[-1].shape[1]

        else:
            raise ValueError(
                f"PoolBackbone does not support backbone_type={config.backbone_type!r}. "
                "Valid options: 'timm', 'vov'."
            )

        self._pool = nn.AdaptiveAvgPool2d(
            (config.img_vert_anchors, config.img_horz_anchors)
        )

    # ── BackboneBase interface ────────────────────────────────────────────────

    @property
    def num_tokens(self) -> int:
        n = self.config.img_vert_anchors * self.config.img_horz_anchors
        return n * 2 if self.config.use_back_view else n

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        front = features["camera_feature"][:, -1]        # (B, 3, H, W)
        tokens = self._encode(front)                     # (B, N, C)

        if self.config.use_back_view:
            back = features["camera_feature_back"][:, -1]
            tokens = torch.cat([tokens, self._encode(back)], dim=1)  # (B, 2N, C)

        return tokens

    # ── helpers ───────────────────────────────────────────────────────────────

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, img_vert_anchors × img_horz_anchors, C)"""
        feat   = self._encoder(image)[-1]                         # (B, C, h, w)
        pooled = self._pool(feat)                                  # (B, C, V, H)
        return pooled.flatten(2).permute(0, 2, 1).contiguous()    # (B, V*H, C)
