"""
ActionDiffusionModel — main nn.Module that wires together:

    BackboneBase subclass   →  img_proj  →  image tokens (B, N_img, D)
    STATUS ENCODER                        →  status token (B,   1,   D)
    concatenate + positional bias         →  context KV   (B, N_img+1, D)
    ActionDiffusionHead                   →  trajectories / loss

The active backbone is selected by `config.backbone_type`:
  'timm' / 'vov'  —  PoolBackbone (CNN + adaptive pooling)
  'bev'           —  BEVBackbone  (VoVNet + cross-attention BEV fusion)

The backbone owns all image extraction, so the model is agnostic to
how many cameras are used or how tokens are produced.

Freezing
--------
`freeze_backbone=True` freezes only the backbone encoder.
All other parameters (img_proj, status_encoder, diffusion_head) are always
trainable.

Checkpoint loading
------------------
When `pretrained_ckpt` is set, weights are loaded for any key that matches
(diffusion_head is skipped so it always trains from scratch).  Freezing is
applied afterwards based solely on `freeze_backbone`.
"""

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig
from navsim.agents.action_diffusion_agent.backbones import build_backbone
from navsim.agents.action_diffusion_agent.ad_diffusion_head import ActionDiffusionHead


class ActionDiffusionModel(nn.Module):

    def __init__(self, config: ActionDiffusionConfig) -> None:
        super().__init__()
        self.config = config

        # ── Perception backbone ───────────────────────────────────────────────
        self.backbone = build_backbone(config)

        # num_tokens already accounts for back-view doubling (PoolBackbone)
        # or BEV grid size (BEVBackbone) — no manual adjustment needed here.
        num_img_tokens = self.backbone.num_tokens

        # ── Feature projection: backbone channels → model_dim ────────────────
        self.img_proj = nn.Linear(self.backbone.out_channels, config.model_dim)

        # ── Ego-status encoder ────────────────────────────────────────────────
        # [driving_command(4), vx(1), vy(1), ax(1), ay(1)] → (B, 1, D)
        self.status_encoder = nn.Sequential(
            nn.Linear(config.ego_status_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
        )

        # ── Learnable positional embedding ────────────────────────────────────
        self._context_len = num_img_tokens + 1
        self.pos_embedding = nn.Embedding(self._context_len, config.model_dim)

        # ── Diffusion head ────────────────────────────────────────────────────
        self.diffusion_head = ActionDiffusionHead(
            config=config, context_len=self._context_len
        )

        # ── Optional weight loading ───────────────────────────────────────────
        if config.pretrained_ckpt:
            self._load_pretrained(config.pretrained_ckpt)

        # ── Backbone freezing (independent of checkpoint) ─────────────────────
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[ActionDiffusionModel] Backbone frozen.")

    # ─────────────────────────────────────────────── checkpoint loading ───────

    def _load_pretrained(self, ckpt_path: str) -> None:
        """Load matching weights from a checkpoint. Diffusion head is always skipped."""
        print(f"[ActionDiffusionModel] Loading pretrained weights from {ckpt_path} …")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_state: dict = ckpt.get("state_dict", ckpt)

        cleaned: dict = {}
        for k, v in raw_state.items():
            name = k.replace("agent.model.", "").replace("model.", "")
            if "diffusion_head" in name:
                continue
            cleaned[name] = v

        msg = self.load_state_dict(cleaned, strict=False)
        print(
            f"[ActionDiffusionModel] Loaded. "
            f"Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}"
        )

    # ─────────────────────────────────────────────── train() override ────────

    def train(self, mode: bool = True) -> "ActionDiffusionModel":
        """Keep the frozen backbone in eval mode so BN stats don't drift."""
        super().train(mode)
        if mode and self.config.freeze_backbone:
            self.backbone.eval()
        return self

    # ────────────────────────────────────────────────── context builder ───────

    def _build_context(
        self, features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Build the (B, context_len, D) context KV tensor consumed by the
        diffusion head.

        Steps:
          1. Backbone extracts images from the features dict and returns tokens.
          2. Project backbone tokens to model_dim.
          3. Encode ego-status as a single token.
          4. Concatenate image tokens + status token.
          5. Add learnable positional bias.
        """
        cfg = self.config

        # The backbone owns all image extraction; it receives the full
        # features dict and returns (B, num_tokens, out_channels).
        tokens = self.img_proj(self.backbone(features))           # (B, N, D)

        # --- Ego-status token ---
        status = features["status_feature"][:, : cfg.ego_status_dim]  # (B, 8)
        status_tok = self.status_encoder(status).unsqueeze(1)          # (B, 1, D)

        # --- Assemble context ---
        context = torch.cat([tokens, status_tok], dim=1)          # (B, N+1, D)
        context = context + self.pos_embedding.weight[None, :]    # broadcast positional bias
        return context

    # ──────────────────────────────────────────────────────── forward ────────

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        scorer_guidance_fn: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Training
        --------
        Returns {'context_kv': (B, L, D), 'trajectory': zeros placeholder}.
        The actual diffusion loss is computed separately in ActionDiffusionAgent
        via `diffusion_head.compute_loss`.

        Inference
        ---------
        Returns {
            'context_kv':   (B, L, D),
            'dp_pred':      (B, N, 8, 3),   – N trajectory proposals
            'pred_actions': (B, N, 40, 2),
            'trajectory':   (B, 8, 3),      – one randomly selected proposal
        }
        """
        B = features["status_feature"].shape[0]
        device = features["status_feature"].device

        context = self._build_context(features)  # (B, L, D)
        out: Dict[str, torch.Tensor] = {"context_kv": context}

        if self.training:
            # During training the caller computes the loss externally.
            # Return a zero placeholder so the framework doesn't break.
            out["trajectory"] = torch.zeros(B, 8, 3, device=device)
        else:
            head_out = self.diffusion_head(
                context,
                features,
                scorer_guidance_fn=scorer_guidance_fn,
            )
            out.update(head_out)

            # Pick one proposal at random as the "committed" trajectory.
            N = head_out["dp_pred"].shape[1]
            idx = torch.randint(0, N, (1,)).item()
            out["trajectory"] = head_out["dp_pred"][:, idx]  # (B, 8, 3)

        return out
