"""
BEVBackbone — VoVNet encoder + cross-attention BEV fusion backbone.

Implements the same architecture as HydraBackboneBEV (GTRS) from scratch,
using ActionDiffusionConfig only.  No HydraConfig or HydraBackboneBEV
dependency — the class is fully self-contained.

Architecture
------------
For each view (front, back):
    VoV V-99-eSE  →  AdaptiveAvgPool2d  →  (B, P, 1024) image tokens
    where P = bev_img_vert_anchors × bev_img_horz_anchors  (e.g. 8×16 = 128)

Concatenate front + back image tokens with learnable BEV queries:
    (B, 2P + Q, 1024)   where Q = bev_h × bev_w = 8×8 = 64

Add positional embeddings, run through a TransformerEncoder (bev_fusion_layers
layers), then return only the Q BEV query positions.

Output: (B, 64, 1024)

Token count note
----------------
bev_img_vert_anchors × bev_img_horz_anchors   —  intermediate image pool size
                                                 (input to fusion transformer)
8 × 8 = 64                                    —  BEV query output size
                                                 (what this backbone returns)

Attribute names (`image_encoder`, `avgpool_img`, `bev_queries`, `pos_emb`,
`fusion_encoder`) intentionally match HydraBackboneBEV so that pretrained
GTRS backbone checkpoints can be loaded directly.

Checkpoint loading
------------------
Set config.bev_ckpt to load the full BEVBackbone state from a saved checkpoint.
Common key prefixes (agent.model.backbone.*, model.backbone.*, backbone.*)
are stripped automatically.
"""

from typing import Dict

import torch
import torch.nn as nn

from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig
from navsim.agents.action_diffusion_agent.backbones.base import BackboneBase


class BEVBackbone(BackboneBase):
    """
    VoVNet + cross-attention BEV fusion backbone.

    Input keys:
        "camera_feature"       (B, T, 3, H, W)  — front stitched view
        "camera_feature_back"  (B, T, 3, H, W)  — rear  stitched view

    Output: (B, 64, 1024)  — 64 BEV query tokens, 1024 channels
    """

    # BEV output grid size — matches HydraBackboneBEV (hardcoded there too).
    # Changing these breaks compatibility with pretrained GTRS checkpoints.
    _BEV_H: int = 8
    _BEV_W: int = 8
    _VOV_CHANNELS: int = 1024  # VoV V-99-eSE stage5 output channels

    def __init__(self, config: ActionDiffusionConfig) -> None:
        super().__init__()
        self.config = config

        from navsim.agents.backbones.vov import VoVNet

        # ── VoV image encoder ─────────────────────────────────────────────────
        # Attribute name matches HydraBackboneBEV for checkpoint compatibility.
        self.image_encoder = VoVNet(
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
        self.image_encoder.init_weights()

        # ── Spatial pool: VoV feature map → fixed image token grid ────────────
        # P = bev_img_vert_anchors × bev_img_horz_anchors per camera view.
        # Attribute name matches HydraBackboneBEV for checkpoint compatibility.
        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (config.bev_img_vert_anchors, config.bev_img_horz_anchors)
        )
        tokens_per_view = config.bev_img_vert_anchors * config.bev_img_horz_anchors
        num_bev_queries = self._BEV_H * self._BEV_W  # 64

        # ── Learnable BEV query embeddings ────────────────────────────────────
        # Attribute name matches HydraBackboneBEV for checkpoint compatibility.
        self.bev_queries = nn.Embedding(num_bev_queries, self._VOV_CHANNELS)

        # ── Positional embeddings: 2P image slots + Q BEV slots ───────────────
        # Attribute name matches HydraBackboneBEV for checkpoint compatibility.
        self.pos_emb = nn.Embedding(
            tokens_per_view * 2 + num_bev_queries,
            self._VOV_CHANNELS,
        )

        # ── Cross-attention fusion TransformerEncoder ─────────────────────────
        # Attribute name matches HydraBackboneBEV for checkpoint compatibility.
        self.fusion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self._VOV_CHANNELS,
                nhead=16,
                dim_feedforward=self._VOV_CHANNELS * 4,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=config.bev_fusion_layers,
        )

        # ── Optional: restore full BEVBackbone state from a saved checkpoint ──
        if config.bev_ckpt:
            self._load_bev_ckpt(config.bev_ckpt)

    # ── BackboneBase interface ────────────────────────────────────────────────

    @property
    def num_tokens(self) -> int:
        """64 BEV output tokens (8×8 query grid)."""
        return self._BEV_H * self._BEV_W

    @property
    def out_channels(self) -> int:
        """1024 — VoV V-99-eSE stage5 channel count."""
        return self._VOV_CHANNELS

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: agent feature dict with:
                "camera_feature"       (B, T, 3, H, W)
                "camera_feature_back"  (B, T, 3, H, W)

        Returns:
            bev_tokens: (B, 64, 1024)
        """
        B = features["camera_feature"].shape[0]

        front_tokens = self._encode_img(features["camera_feature"][:, -1])       # (B, P, 1024)
        back_tokens  = self._encode_img(features["camera_feature_back"][:, -1])  # (B, P, 1024)

        img_tokens  = torch.cat([front_tokens, back_tokens], dim=1)              # (B, 2P, 1024)
        bev_queries = self.bev_queries.weight[None].expand(B, -1, -1)            # (B, 64, 1024)

        img_len = img_tokens.shape[1]                                             # 2P
        tokens = torch.cat([img_tokens, bev_queries], dim=1)                     # (B, 2P+64, 1024)
        tokens = self.fusion_encoder(
            tokens + self.pos_emb.weight[None].expand(B, -1, -1)
        )

        return tokens[:, img_len:]  # (B, 64, 1024) — BEV query outputs only

    # ── helpers ───────────────────────────────────────────────────────────────

    def _encode_img(self, img: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, bev_img_vert_anchors × bev_img_horz_anchors, 1024)"""
        feat   = self.image_encoder(img)[-1]
        pooled = self.avgpool_img(feat)
        return pooled.flatten(-2, -1).permute(0, 2, 1)             # (B, P, 1024)

    def _load_bev_ckpt(self, ckpt_path: str) -> None:
        """
        Load a pretrained BEVBackbone (or GTRS) checkpoint.

        Strips common key prefixes in order of specificity so that the
        resulting keys align with this module's own attribute names.
        """
        print(f"[BEVBackbone] Loading weights from {ckpt_path} …")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw: dict = ckpt.get("state_dict", ckpt)

        prefixes = [
            "agent.model.backbone.",
            "model.backbone.",
            "backbone.",
        ]
        cleaned: dict = {}
        for k, v in raw.items():
            name = k
            for pfx in prefixes:
                if k.startswith(pfx):
                    name = k[len(pfx):]
                    break
            cleaned[name] = v

        msg = self.load_state_dict(cleaned, strict=False)
        print(
            f"[BEVBackbone] Loaded. "
            f"Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}"
        )
