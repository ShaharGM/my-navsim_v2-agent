"""
BackboneBase — abstract base class for all perception backbones used by
ActionDiffusionModel.

Every concrete backbone must implement:
    forward(features)  →  (B, num_tokens, out_channels)
    num_tokens         →  int  (total context tokens returned per sample)
    out_channels       →  int  (channel dimension of each token)

The `features` dict is the raw agent feature dict as produced by the feature
builder.  Required keys depend on the backbone implementation; the two
standard ones are:
    "camera_feature"       (B, T, 3, H, W)  — front stitched panoramic view
    "camera_feature_back"  (B, T, 3, H, W)  — rear  stitched panoramic view

The model is completely agnostic to which backbone is active — it just calls
`backbone(features)` and receives a flat token tensor.
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BackboneBase(ABC, nn.Module):
    """Abstract base class for ActionDiffusion perception backbones."""

    @property
    @abstractmethod
    def num_tokens(self) -> int:
        """Total number of context tokens returned per sample."""

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Channel dimension of each returned token."""

    @abstractmethod
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: agent feature dict (see module docstring for keys).

        Returns:
            tokens: (B, num_tokens, out_channels)
        """
