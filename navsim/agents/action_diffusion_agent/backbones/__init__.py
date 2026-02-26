"""
Backbone package for ActionDiffusionAgent.

Use `build_backbone(config)` to get the correct backbone for the given config.
All backbones implement BackboneBase:
    forward(features)  →  (B, num_tokens, out_channels)
    num_tokens         →  int
    out_channels       →  int

Supported backbone_type values in ActionDiffusionConfig:
    'timm'  →  PoolBackbone with a timm model encoder
    'vov'   →  PoolBackbone with VoVNet V-99-eSE encoder
    'bev'   →  BEVBackbone  with VoVNet + cross-attention BEV fusion
"""

import torch.nn as nn

from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig
from navsim.agents.action_diffusion_agent.backbones.base import BackboneBase
from navsim.agents.action_diffusion_agent.backbones.pool_backbone import PoolBackbone
from navsim.agents.action_diffusion_agent.backbones.bev_backbone import BEVBackbone


def build_backbone(config: ActionDiffusionConfig) -> BackboneBase:
    """
    Factory function — returns the correct BackboneBase subclass instance.

    Args:
        config: ActionDiffusionConfig with backbone_type set to one of
                'timm', 'vov', 'bev'.

    Returns:
        A constructed backbone ready to be used as self.backbone in
        ActionDiffusionModel.
    """
    if config.backbone_type in ("timm", "vov"):
        return PoolBackbone(config)
    elif config.backbone_type == "bev":
        return BEVBackbone(config)
    else:
        raise ValueError(
            f"Unknown backbone_type={config.backbone_type!r}. "
            "Valid options: 'timm', 'vov', 'bev'."
        )


__all__ = ["BackboneBase", "PoolBackbone", "BEVBackbone", "build_backbone"]
